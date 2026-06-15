# Classification with optional SupCon (separate entrypoint)

from __future__ import annotations

import argparse, json, os, sys, random
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
from monai.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from train_utils import init_config, set_seed
from dataset.multimodal_cls_dataset import get_datasets_cls
from model.uni_unet import UniUnet_SC
from trainer_cls import train_epoch_cls, val_epoch_cls

def build_argparser():
    p = argparse.ArgumentParser(description='BrainMVP downstream classification with SupCon (T1-only)')
    p.add_argument('--start_epoch', default=0, type=int)
    p.add_argument('--max_epochs', default=200, type=int)
    p.add_argument('--batch_size', default=8, type=int)
    p.add_argument('--patch_shape', default=96, type=int)
    p.add_argument('--eval_interval', default=1, type=int)
    p.add_argument('--accum_steps', default=8, type=int)
    p.add_argument('--in_channels', default=1, type=int)
    p.add_argument('--num_classes', default=3, type=int)
    p.add_argument('--resume', default='', type=str)
    p.add_argument('--pretrained', default='', type=str)
    p.add_argument('--mix_template', action='store_true', default=False)
    p.add_argument('--template_dir', default='templates', type=str)
    p.add_argument('--use_cl', action='store_true', default=False)
    p.add_argument('--cl_weight', default=0.1, type=float)
    p.add_argument('--lr', default=3e-4, type=float)
    p.add_argument('--wd', default=1e-4, type=float)
    p.add_argument('--eta_min', default=0.0, type=float)
    p.add_argument('--workers', default=6, type=int)
    p.add_argument('--devices', default='0', type=str)
    p.add_argument('--random_seed', type=int, default=42)
    p.add_argument('--dataset', type=str, default='custom')
    p.add_argument('--data_root', default='', type=str)
    p.add_argument('--json_file', default='dataset_cls.json', type=str)
    p.add_argument('--pixdim', type=float, nargs=3, default=(1.0, 1.0, 1.0))
    p.add_argument('--experiment', default='debug', type=str)
    p.add_argument('--output_dir', default='debug', type=str)
    p.add_argument('--cfg', type=str, default="configs/config.yaml")
    p.add_argument('--loss', type=str, default='balanced_softmax', choices=['balanced_softmax'])
    p.add_argument('--bs_tau', type=float, default=1.0, help='Temperature multiplier for balanced softmax logits.')
    p.add_argument('--save_top_k', type=int, default=3,
                   help='Keep the top-k checkpoints by validation macroAUC. 0 disables validation-AUC top-k saves.')
    # SupCon
    p.add_argument('--use_supcon', action='store_true', default=False)
    p.add_argument('--supcon_lambda', type=float, default=0.05)
    p.add_argument('--supcon_proj_dim', type=int, default=128)
    p.add_argument('--supcon_temp', type=float, default=0.07)
    p.add_argument('--supcon_queue_size', type=int, default=512)
    p.add_argument('--supcon_weight_cap', type=float, default=3.0,
                   help='Cap for class-aware SupCon anchor weights.')
    # BoQ (Bag of Queries) pooling
    p.add_argument('--use_boq', action='store_true', default=False,
                   help='Enable bag-of-learnable-queries pooling.')
    p.add_argument('--boq_num_queries', type=int, default=None,
                   help='Number of learnable queries (default: num_classes).')
    p.add_argument('--boq_heads', type=int, default=4,
                   help='Attention heads for BoQ cross-attention.')
    p.add_argument('--boq_dropout', type=float, default=0.0,
                   help='Dropout for BoQ attention.')
    p.add_argument('--boq_class_queries', action='store_true', default=False,
                   help='Treat queries as class-aware (M=K).')
    p.add_argument('--boq_use_diag_head', dest='boq_use_diag_head', action='store_true', default=True,
                   help='Use diagonal per-class head when class queries are enabled.')
    p.add_argument('--boq_no_diag_head', dest='boq_use_diag_head', action='store_false',
                   help='Disable diagonal per-class head even when class queries are enabled.')
    p.add_argument('--boq_lambda_div', type=float, default=1e-3,
                   help='Weight for BoQ query diversity loss.')
    p.add_argument('--boq_blend_alpha', type=float, default=0.3,
                   help='Blend weight for BoQ logits: logits = logits_gap + alpha * logits_boq.')
    # CAC (Confusion-aware SupCon)
    p.add_argument('--cac_use', action='store_true', default=False,
                   help='Enable confusion-aware weighting for SupCon negatives (uses val confusion).')
    p.add_argument('--cac_decay', type=float, default=0.5,
                   help='Decay for confusion weights across evals.')
    p.add_argument('--cac_max_w', type=float, default=2.0,
                   help='Cap for confusion-based negative weights.')
    return p

def _ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def _save_templates_if_available(state_dict, template_dir: str):
    if not template_dir: return
    rep = state_dict.get('rep_template', None)
    if rep is None: return
    _ensure_dir(template_dir)
    if isinstance(rep, torch.Tensor): rep = rep.cpu()
    tem_list = ['flair','t1','t1c','t2','mra','pd','dwi','adc']
    for i in range(rep.shape[0]):
        data_np = rep[i].numpy()
        nii = nib.Nifti1Image(data_np, affine=np.eye(4))
        name = tem_list[i].upper() if i < len(tem_list) else f"TEMP_{i}"
        outp = os.path.join(template_dir, f"{name}.nii.gz")
        nib.save(nii, outp)
        print(f"[template] saved: {outp}")

def main(args):
    args.rank = 0
    queue_state = None
    _opt_state = None
    _sch_state = None
    ck_epoch = None
    if getattr(args, "devices", None):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    init_config(args)
    print(args)
    set_seed(args)

    exp_dir = getattr(args, "ckpt_save_dir", None)
    exp_dir = str(exp_dir) if exp_dir else os.path.join(args.output_dir, args.experiment)
    _ensure_dir(exp_dir)
    train_log_path = os.path.join(exp_dir, f"train_log_{args.experiment}.jsonl")
    print("[build] creating UniUnet_SC (classification head + SupCon head)")
    model = UniUnet_SC(
        input_shape=args.patch_shape,
        in_channels=args.in_channels,
        out_channels=args.num_classes,
        multi_scale=True,
        segmentation=False,
        supcon_proj_dim=args.supcon_proj_dim,
        use_boq=getattr(args, "use_boq", False),
        boq_num_queries=getattr(args, "boq_num_queries", None),
        boq_heads=int(getattr(args, "boq_heads", 4)),
        boq_dropout=float(getattr(args, "boq_dropout", 0.0)),
        boq_class_queries=bool(getattr(args, "boq_class_queries", False)),
        boq_use_diag_head=bool(getattr(args, "boq_use_diag_head", True)),
        boq_blend_alpha=float(getattr(args, "boq_blend_alpha", 0.3)),
    ).cuda()

    if args.resume:
        print(f"[ckpt] resume from {args.resume}")
        ck = torch.load(args.resume, map_location='cpu')
        queue_state = ck.get("queue_state", None) if isinstance(ck, dict) else None
        if isinstance(ck, dict) and "cac_weight_matrix" in ck:
            args._cac_weight_matrix = ck.get("cac_weight_matrix", None)
        _opt_state = ck.get("optimizer", None) if isinstance(ck, dict) else None
        _sch_state = ck.get("scheduler", None) if isinstance(ck, dict) else None
        ck_epoch = ck.get("epoch", None) if isinstance(ck, dict) else None
        state = ck['state_dict'] if 'state_dict' in ck else ck
        model.load_state_dict(state, strict=False)
        del ck, state
    elif args.pretrained:
        print(f"[ckpt] load pretrained from {args.pretrained}")
        ck = torch.load(args.pretrained, map_location='cpu')
        state = ck.get('state_dict', ck)
        cleaned = {}
        for k, v in state.items():
            nk = k[7:] if k.startswith('module.') else k
            cleaned[nk] = v
        for dk in [
            'encoder.patch_embed1.proj.weight',
            'encoder.patch_embed1.proj.bias',
            'decoder.proj1.proj.weight',
            'decoder.proj1.proj.bias',
        ]:
            if dk in cleaned: del cleaned[dk]
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"[ckpt] loaded with strict=False | missing: {len(missing)} | unexpected: {len(unexpected)}")
        try:
            _save_templates_if_available(cleaned, args.template_dir)
        except Exception as e:
            print(f"[template] skipping save (reason: {e})")
        del ck, state, cleaned, missing, unexpected
    else:
        print("[ckpt] training from scratch (not recommended)")

    if not getattr(args, "use_supcon", False):
        queue_state = None

    train_ds, val_ds, _ = get_datasets_cls(args)

    # ---- class counts (avoid running heavy transforms just to read labels) ----
    train_labels = None
    try:
        json_path = Path(args.json_file) if Path(args.json_file).is_file() else Path(args.data_root) / args.json_file
        with open(json_path, "r") as f:
            j = json.load(f)
        train_labels = [int(it["label"]) for it in j.get("training", [])]
        if len(train_labels) != len(train_ds):
            raise RuntimeError(f"train_labels ({len(train_labels)}) != train_ds ({len(train_ds)})")
    except Exception:
        try:
            train_labels = [int(train_ds[i]["label"]) for i in range(len(train_ds))]
        except Exception as e2:
            print(f"[labels] failed to build train_labels (reason: {e2})")
            train_labels = None

    counts = None
    if train_labels is not None:
        counts = np.bincount(train_labels, minlength=args.num_classes).astype(np.float64)

    # deterministic DataLoader seeding
    g = torch.Generator()
    g.manual_seed(args.random_seed + args.rank)

    def _seed_worker(worker_id):
        base = args.random_seed + args.rank
        np.random.seed(base + worker_id)
        torch.manual_seed(base + worker_id)
        random.seed(base + worker_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=g,
    )
    print(f"[data] train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    # ---- store class counts and log-priors for balanced softmax ----
    try:
        if counts is None:
            raise RuntimeError("no train_labels for class counts")
        counts_safe = np.maximum(counts, 1.0)
        args._class_counts = counts.tolist()
        args._log_prior_tensor = torch.log(torch.tensor(counts_safe, dtype=torch.float32))
        print(f"[loss-priors] class_counts={counts.tolist()} | log_prior(min,max)=({float(np.log(counts_safe).min()):.3f},{float(np.log(counts_safe).max()):.3f})")
    except Exception as e:
        print(f"[loss-priors] skip (reason: {e})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, eta_min=args.eta_min)

    if _opt_state is not None:
        try:
            optimizer.load_state_dict(_opt_state)
            print("[ckpt] optimizer state restored.")
        except Exception as e:
            print(f"[ckpt] optimizer state load failed (reason: {e})")
    if _sch_state is not None:
        try:
            scheduler.load_state_dict(_sch_state)
            print("[ckpt] scheduler state restored.")
        except Exception as e:
            print(f"[ckpt] scheduler state load failed (reason: {e})")
    if ck_epoch is not None and args.start_epoch == 0:
        args.start_epoch = int(ck_epoch) + 1
        print(f"[ckpt] resuming from epoch {ck_epoch}, start_epoch set to {args.start_epoch}")
    best_auc = None
    top_auc_ckpts = []
    top_k_auc = max(0, int(getattr(args, "save_top_k", 0)))
    top_auc_manifest_path = os.path.join(exp_dir, "top_val_auc_checkpoints.json")
    if top_k_auc > 0 and os.path.exists(top_auc_manifest_path):
        try:
            with open(top_auc_manifest_path, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    auc_path = item.get("path") or os.path.join(exp_dir, item.get("checkpoint", ""))
                    if auc_path and os.path.exists(auc_path):
                        top_auc_ckpts.append((float(item["val_macro_auc"]), int(item["epoch"]), auc_path))
            top_auc_ckpts.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            top_auc_ckpts = top_auc_ckpts[:top_k_auc]
            if top_auc_ckpts:
                print(f"[ckpt] restored {len(top_auc_ckpts)} validation-AUC checkpoint records from manifest.")
        except Exception as e:
            print(f"[ckpt] could not restore validation-AUC checkpoint manifest (reason: {e})")
    def _write_top_auc_manifest():
        if top_k_auc <= 0:
            return
        manifest = []
        for rank, (auc_value, auc_epoch, auc_path) in enumerate(top_auc_ckpts, start=1):
            manifest.append(dict(
                rank=rank,
                epoch=int(auc_epoch),
                val_macro_auc=float(auc_value),
                checkpoint=os.path.basename(auc_path),
                path=str(auc_path),
            ))
        with open(top_auc_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    for epoch in range(args.start_epoch, args.max_epochs):
        train_loss, train_metrics, queue_state = train_epoch_cls(
            epoch, model, train_loader, optimizer, scheduler, args, queue_state=queue_state
        )

        row_tr = {"experiment": args.experiment, "epoch": epoch, "split": "train", "loss": float(train_loss)}
        if isinstance(train_metrics, dict):
            row_tr.update(train_metrics)
        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row_tr) + "\n")

        do_eval = (epoch % max(1, args.eval_interval) == 0) or ((epoch + 1) == args.max_epochs)
        if do_eval:
            metrics = val_epoch_cls(val_loader, model, args, args.num_classes)
            print(f"[Val] Epoch {epoch:03d} | loss {metrics['loss']:.4f} | "
                  f"AUC {metrics['macro_auc']:.3f} | AUPRC {metrics['macro_auprc']:.3f}")
            row_val = {
                "experiment": args.experiment,
                "epoch": epoch,
                "split": "val",
                "loss": float(metrics["loss"]),
                "macro_auc": metrics.get("macro_auc"),
                "macro_auprc": metrics.get("macro_auprc"),
                "auc_per_class": metrics.get("auc_per_class"),
                "auprc_per_class": metrics.get("auprc_per_class"),
            }
            with open(train_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row_val) + "\n")

            # Update confusion-aware weights from validation confusion (if enabled)
            if getattr(args, "cac_use", False):
                cm = metrics.get("confusion", None)
                try:
                    cm_arr = np.asarray(cm, dtype=np.float32) if cm is not None else None
                    if cm_arr is not None and cm_arr.ndim == 2:
                        row_sum = cm_arr.sum(axis=1, keepdims=True)
                        conf_norm = np.divide(cm_arr, np.maximum(row_sum, 1.0))
                        w_new = 1.0 + conf_norm  # baseline 1, up to 1+conf_norm
                        w_new = np.clip(w_new, 1.0, float(getattr(args, "cac_max_w", 2.0)))
                        np.fill_diagonal(w_new, 1.0)
                        if hasattr(args, "_cac_weight_matrix"):
                            w_old = np.asarray(getattr(args, "_cac_weight_matrix"), dtype=np.float32)
                            if w_old.shape == w_new.shape:
                                decay = float(getattr(args, "cac_decay", 0.5))
                                w_new = decay * w_old + (1.0 - decay) * w_new
                        args._cac_weight_matrix = w_new.tolist()
                        try:
                            w_max, w_mean = float(w_new.max()), float(w_new.mean())
                            print(f"[cac] updated confusion weights | max={w_max:.3f} mean={w_mean:.3f}")
                        except Exception:
                            pass
                    else:
                        print("[cac] skip update (no confusion matrix available)")
                except Exception as e:
                    print(f"[cac] failed to update weights (reason: {e})")

            macro_auc_val = metrics.get("macro_auc", None)
            if macro_auc_val is not None and ((best_auc is None) or (macro_auc_val > best_auc)):
                best_auc = macro_auc_val

            if top_k_auc > 0 and macro_auc_val is not None and np.isfinite(float(macro_auc_val)):
                auc_value = float(macro_auc_val)
                should_save_auc = len(top_auc_ckpts) < top_k_auc
                if not should_save_auc:
                    worst_auc = min(item[0] for item in top_auc_ckpts)
                    should_save_auc = auc_value > worst_auc
                if should_save_auc:
                    auc_path = os.path.join(exp_dir, f"model_top{top_k_auc}_val_auc_epoch{epoch}_auc{auc_value:.6f}.pth.tar")
                    print(f"[ckpt] saving top-{top_k_auc} checkpoint by val macroAUC: {auc_value:.6f}")
                    torch.save(dict(epoch=epoch, state_dict=model.state_dict(),
                                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                    queue_state=queue_state,
                                    cac_weight_matrix=getattr(args, "_cac_weight_matrix", None),
                                    val_macro_auc=auc_value),
                               auc_path)
                    top_auc_ckpts.append((auc_value, epoch, auc_path))
                    top_auc_ckpts.sort(key=lambda item: (item[0], -item[1]), reverse=True)
                    while len(top_auc_ckpts) > top_k_auc:
                        removed_auc, removed_epoch, removed_path = top_auc_ckpts.pop()
                        if os.path.exists(removed_path):
                            try:
                                os.remove(removed_path)
                                print(f"[ckpt] removed val-AUC checkpoint outside top-{top_k_auc}: "
                                      f"epoch={removed_epoch} auc={removed_auc:.6f}")
                            except OSError as e:
                                print(f"[ckpt] failed to remove old val-AUC checkpoint {removed_path} (reason: {e})")
                    _write_top_auc_manifest()

    print(f"[done] Best Val macroAUC (observed): {best_auc if best_auc is not None else 'N/A'}")

if __name__ == "__main__":
    parser = build_argparser()
    a = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = a.devices
    main(a)
