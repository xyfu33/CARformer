# downstream/test_cls.py
# Evaluation for SupCon-trained classifier.

import argparse, os, json, math, re, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dataset.multimodal_cls_dataset import get_datasets_cls
from model.uni_unet import UniUnet_SC
from monai.data import DataLoader
from trainer_cls import _ensure_class_dim1, _coerce_to_2d_logits, _extract_logits_and_feat, val_epoch_cls


def _r(x):
    try:
        if isinstance(x, list):
            return [_r(v) for v in x]
        if isinstance(x, float):
            return None if not math.isfinite(x) else round(x, 6)
        return x
    except Exception:
        return x


def _get_case_id_from_batch(batch, i: int):
    def _to_str(x):
        try:
            return str(x)
        except Exception:
            return None

    def _extract_sub_id(p: str):
        if not p:
            return None
        m = re.search(r"(sub-[^/]+)", p)
        return m.group(1) if m else None

    def _normalize_path(p):
        ps = _to_str(p)
        return ps

    for k in ["case_id", "subject_id", "sub_id", "image_path", "img_path", "path", "filepath", "file", "filename"]:
        if k in batch:
            v = batch[k]
            try:
                p = v[i]
            except Exception:
                p = v
            p = _normalize_path(p)
            sid = _extract_sub_id(p)
            return sid if sid else (os.path.basename(p) if p else f"idx={i}")

    if "image" in batch:
        v = batch["image"]
        try:
            candidate = v[i]
            if isinstance(candidate, (list, tuple)) and len(candidate) > 0:
                candidate = candidate[0]
            if isinstance(candidate, (str, os.PathLike)):
                p = _normalize_path(candidate)
                sid = _extract_sub_id(p)
                return sid if sid else os.path.basename(p)
        except Exception:
            pass

    for mk in ["image_meta_dict", "meta_dict"]:
        if mk in batch and isinstance(batch[mk], dict):
            md = batch[mk]
            for fk in ["filename_or_obj", "filename", "path", "filepath"]:
                if fk in md:
                    fv = md[fk]
                    try:
                        p = fv[i]
                    except Exception:
                        p = fv
                    if isinstance(p, (list, tuple)) and len(p) > 0:
                        p0 = p[0]
                    else:
                        p0 = p
                    p0 = _normalize_path(p0)
                    sid = _extract_sub_id(p0)
                    return sid if sid else (os.path.basename(p0) if p0 else f"idx={i}")

    return f"idx={i}"


def _build_model_from_ckpt(args):
    ck = torch.load(args.checkpoint, map_location='cpu')
    state = ck.get('state_dict', ck)
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}

    has_boq = any(
        k.startswith(("boq_", "boq.", "boq_queries", "boq_tok_norm", "boq_q_norm", "boq_attn", "boq_norm", "boq_cls_head"))
        for k in state.keys()
    )
    has_boq_diag = any(k.startswith("boq_cls_head.") for k in state.keys())
    hparams = ck.get('hparams', {}) if isinstance(ck, dict) else {}
    ms_from_keys = any(k.startswith('ms_out.') for k in state.keys())
    ms_from_hp = hparams.get('multi_scale', None)
    if ms_from_hp is None:
        ms_from_hp = True
    ms_from_hp = bool(ms_from_hp or ms_from_keys)

    use_boq = bool(getattr(args, "use_boq", False) or has_boq)
    boq_num_queries = getattr(args, "boq_num_queries", None)
    if boq_num_queries is None and "boq_queries" in state:
        boq_num_queries = int(state["boq_queries"].shape[0])
    boq_use_diag_head = getattr(args, "boq_use_diag_head", None)
    if boq_use_diag_head is None:
        boq_use_diag_head = has_boq_diag if use_boq else True

    proj_dim = args.supcon_proj_dim
    model = UniUnet_SC(
        input_shape=args.patch_shape,
        in_channels=args.in_channels,
        out_channels=args.num_classes,
        multi_scale=ms_from_hp,
        segmentation=False,
        supcon_proj_dim=proj_dim,
        use_boq=use_boq,
        boq_num_queries=boq_num_queries,
        boq_heads=int(getattr(args, "boq_heads", 4)),
        boq_dropout=float(getattr(args, "boq_dropout", 0.0)),
        boq_class_queries=bool(getattr(args, "boq_class_queries", False)),
        boq_use_diag_head=bool(boq_use_diag_head),
        boq_blend_alpha=float(getattr(args, "boq_blend_alpha", 0.3)),
    ).cuda()
    mstate = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in mstate}
    missing = [k for k in mstate.keys() if k not in filtered]
    unexpected = [k for k in state.keys() if k not in mstate]
    if missing or unexpected:
        print(f"[ckpt] Warning: missing={len(missing)} unexpected={len(unexpected)}. Verify test args match training.")
    model.load_state_dict(filtered, strict=False)
    print(f"[build] multi_scale={ms_from_hp} use_boq={use_boq} boq_num_queries={boq_num_queries}")
    return model, ck


def _json_path(args):
    p = Path(args.json_file)
    if p.is_file():
        return p
    return Path(args.data_root) / args.json_file


def _init_eval_loss_state(args):
    try:
        with open(_json_path(args), "r") as f:
            j = json.load(f)
        train_labels = [int(it["label"]) for it in j.get("training", [])]
        if not train_labels:
            return
        counts = np.bincount(train_labels, minlength=args.num_classes).astype(np.float64)
        counts_safe = np.maximum(counts, 1.0)
        args._class_counts = counts.tolist()
        args._log_prior_tensor = torch.log(torch.tensor(counts_safe, dtype=torch.float32))
        print(f"[loss-priors] class_counts={counts.tolist()}")
    except Exception as e:
        print(f"[loss-priors] skip (reason: {e})")


def main():
    parser = argparse.ArgumentParser(description="SupCon classification testing")
    parser.add_argument('--data_root', default='', type=str)
    parser.add_argument('--json_file', default='dataset_cls.json', type=str)
    parser.add_argument('--mix_template', action='store_true')
    parser.add_argument('--template_dir', default='', type=str)
    parser.add_argument('--use_cl', action='store_true')

    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--devices', default='0', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--patch_shape', default=96, type=int)
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--num_classes', default=3, type=int)
    parser.add_argument('--supcon_proj_dim', default=128, type=int)
    parser.add_argument('--loss', type=str, default='balanced_softmax', choices=['balanced_softmax'])
    parser.add_argument('--bs_tau', type=float, default=1.0)
    parser.add_argument('--use_boq', action='store_true',
                        help='Enable BoQ pooling; auto-enabled when checkpoint contains BoQ weights.')
    parser.add_argument('--boq_num_queries', type=int, default=None)
    parser.add_argument('--boq_heads', type=int, default=4)
    parser.add_argument('--boq_dropout', type=float, default=0.0)
    parser.add_argument('--boq_class_queries', action='store_true', default=False)
    parser.add_argument('--boq_use_diag_head', dest='boq_use_diag_head', action='store_true', default=None)
    parser.add_argument('--boq_no_diag_head', dest='boq_use_diag_head', action='store_false')
    parser.add_argument('--boq_blend_alpha', type=float, default=0.3)

    parser.add_argument('--dump_json', default='', type=str)
    parser.add_argument('--log_out_csv', default='', type=str)
    parser.add_argument('--log_per_case', action='store_true')
    parser.add_argument('--log_topk', default=3, type=int)
    parser.add_argument('--print_batch_keys_once', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    _init_eval_loss_state(args)

    print("Building model ...")
    model, ck = _build_model_from_ckpt(args)
    print(f"Loaded checkpoint: {args.checkpoint}")

    _, _, test_ds = get_datasets_cls(args)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=torch.cuda.is_available())
    test_metrics = val_epoch_cls(test_loader, model, args, args.num_classes)

    per_case_rows = []
    if args.log_per_case or args.log_out_csv or args.print_batch_keys_once:
        model.eval()
        printed_keys = False
        with torch.no_grad():
            for batch in test_loader:
                if args.print_batch_keys_once and (not printed_keys):
                    print("[DEBUG] batch keys:", list(batch.keys()))
                    if "image_meta_dict" in batch and isinstance(batch["image_meta_dict"], dict):
                        print("[DEBUG] image_meta_dict keys:", list(batch["image_meta_dict"].keys()))
                    printed_keys = True

                if not (args.log_per_case or args.log_out_csv):
                    continue

                x = batch['image'].float().cuda()
                out = model(x)
                logits, _ = _extract_logits_and_feat(out)
                logits = _ensure_class_dim1(logits, args.num_classes)
                logits = _coerce_to_2d_logits(logits, args.num_classes)
                p_cpu = F.softmax(logits, dim=1).detach().cpu().float()
                y_cpu = batch['label'].detach().cpu().long()
                B, C = p_cpu.shape
                topk = min(max(int(args.log_topk), 1), C)
                topv, topi = torch.topk(p_cpu, k=topk, dim=1)
                for i in range(B):
                    case_id = _get_case_id_from_batch(batch, i)
                    gt = int(y_cpu[i].item())
                    pred = int(torch.argmax(p_cpu[i]).item())
                    pred_prob = float(p_cpu[i, pred].item())
                    correct = int(pred == gt)
                    row = {
                        "case_id": case_id,
                        "gt": gt,
                        "pred": pred,
                        "pred_prob": pred_prob,
                        "correct": correct,
                        "topk_idx": [int(v) for v in topi[i].tolist()],
                        "topk_prob": [float(v) for v in topv[i].tolist()],
                        "probs": [float(v) for v in p_cpu[i].tolist()],
                    }
                    per_case_rows.append(row)
                    if args.log_per_case:
                        topk_str = ", ".join([f"{ci}:{cp:.4f}" for ci, cp in zip(row["topk_idx"], row["topk_prob"])])
                        print(f"[TEST] {case_id} | gt={gt} pred={pred} pred_prob={pred_prob:.4f} correct={correct} | top{topk}={topk_str}")

    results = {
        "checkpoint": args.checkpoint,
        "split": "test",
        "loss": test_metrics["loss"],
        "macro_auc": test_metrics["macro_auc"],
        "macro_auprc": test_metrics["macro_auprc"],
        "auc_per_class": test_metrics["auc_per_class"],
        "auprc_per_class": test_metrics["auprc_per_class"],
        "support_per_class": test_metrics["support_per_class"],
    }

    print("\n=== Test Metrics ===")
    print(f"Loss: {_r(test_metrics['loss'])}")
    print(f"Macro AUC:   {_r(test_metrics['macro_auc'])}")
    print(f"Macro AUPRC: {_r(test_metrics['macro_auprc'])}")
    print(f"AUC per class:   {_r(test_metrics['auc_per_class'])}")
    print(f"AUPRC per class: {_r(test_metrics['auprc_per_class'])}")
    print(f"Support per class: {_r(test_metrics['support_per_class'])}")

    if args.log_out_csv:
        import csv
        with open(args.log_out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "case_id", "gt", "pred", "pred_prob", "correct", "topk_idx", "topk_prob", "probs"
            ])
            w.writeheader()
            for r in per_case_rows:
                w.writerow({
                    "case_id": r["case_id"],
                    "gt": r["gt"],
                    "pred": r["pred"],
                    "pred_prob": r["pred_prob"],
                    "correct": r["correct"],
                    "topk_idx": json.dumps(r["topk_idx"]),
                    "topk_prob": json.dumps(r["topk_prob"]),
                    "probs": json.dumps(r["probs"]),
                })
        print(f"\nSaved per-case predictions CSV to: {args.log_out_csv}")

    if args.dump_json:
        with open(args.dump_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics JSON to: {args.dump_json}")


if __name__ == "__main__":
    main()
