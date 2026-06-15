import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from train_utils import AverageMeter


def _ensure_class_dim1(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    if logits.dim() < 2:
        return logits
    shape = list(logits.shape)
    cls_axes = [i for i, s in enumerate(shape) if s == num_classes and i != 0]
    if not cls_axes:
        return logits
    cls_axis = cls_axes[0]
    if cls_axis == 1:
        return logits
    order = [0, cls_axis] + [i for i in range(1, logits.dim()) if i != cls_axis]
    return logits.permute(*order).contiguous()


def _coerce_to_2d_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    logits = _ensure_class_dim1(logits, num_classes)
    if logits.dim() == 2:
        return logits
    if logits.dim() >= 3:
        spatial_dims = tuple(range(2, logits.dim()))
        return logits.mean(dim=spatial_dims)
    return logits.view(logits.size(0), -1)


def _apply_log_prior_logits(logits: torch.Tensor, log_prior: Optional[torch.Tensor], tau: float = 1.0) -> torch.Tensor:
    if log_prior is None:
        return logits
    lp = log_prior.to(logits.device, dtype=logits.dtype)
    return logits + float(tau) * lp


def _classification_loss(logits: torch.Tensor, y: torch.Tensor, args) -> torch.Tensor:
    log_prior = getattr(args, "_log_prior_tensor", None)
    tau = float(getattr(args, "bs_tau", 1.0))
    logits = _apply_log_prior_logits(logits, log_prior, tau=tau)
    return F.cross_entropy(logits, y)


def _extract_logits_and_feat(model_out) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    feat = None
    out = model_out
    if isinstance(out, tuple):
        if len(out) == 2:
            preds, feat = out
        elif len(out) >= 3:
            preds = out[0]
            feat = out[1] if torch.is_tensor(out[1]) else None
        else:
            preds = out
    else:
        preds = out

    if isinstance(preds, (list, tuple)):
        flat: List[torch.Tensor] = []
        def _collect(x):
            if torch.is_tensor(x):
                flat.append(x)
            elif isinstance(x, (list, tuple)):
                for xx in x:
                    _collect(xx)
        _collect(preds)
        if len(flat) == 0:
            raise TypeError("Model returned no tensor predictions.")
        logits = flat[-1]
    else:
        logits = preds

    if not torch.is_tensor(logits):
        raise TypeError(f"Expected Tensor logits, got {type(logits)}")
    return logits, feat


def _per_class_auc(y_true_np: np.ndarray, probs_np: np.ndarray, num_classes: int) -> (List[Optional[float]], Optional[float]):
    auc_per_class: List[Optional[float]] = [None] * num_classes
    macro_auc: Optional[float] = None
    try:
        from sklearn.metrics import roc_auc_score
        vals = []
        for c in range(num_classes):
            y_bin = (y_true_np == c).astype(np.int32)
            if y_bin.sum() > 0 and (len(y_bin) - y_bin.sum()) > 0:
                auc_c = float(roc_auc_score(y_bin, probs_np[:, c]))
                auc_per_class[c] = auc_c
                vals.append(auc_c)
        if len(vals) > 0:
            macro_auc = float(np.mean(vals))
    except Exception:
        pass
    return auc_per_class, macro_auc


def _per_class_auprc(y_true_np: np.ndarray, probs_np: np.ndarray, num_classes: int) -> (List[Optional[float]], Optional[float]):
    auprc_per_class: List[Optional[float]] = [None] * num_classes
    macro_auprc: Optional[float] = None
    try:
        from sklearn.metrics import average_precision_score
        vals = []
        for c in range(num_classes):
            y_bin = (y_true_np == c).astype(np.int32)
            if y_bin.sum() > 0:
                ap_c = float(average_precision_score(y_bin, probs_np[:, c]))
                auprc_per_class[c] = ap_c
                vals.append(ap_c)
        if len(vals) > 0:
            macro_auprc = float(np.mean(vals))
    except Exception:
        pass
    return auprc_per_class, macro_auprc

def _supcon_loss_with_queue(feat: torch.Tensor, labels: torch.Tensor,
                            q_feat: Optional[torch.Tensor], q_lab: Optional[torch.Tensor],
                            temp: float = 0.07):
    """
    SupCon over batch anchors vs (batch + queue) positives/negatives.
    Returns (loss, pos_mean, pos_min).
    """
    device = feat.device
    B = feat.size(0)
    if q_feat is not None and q_lab is not None and q_feat.numel() > 0:
        all_feat = torch.cat([feat, q_feat.to(device)], dim=0)
        all_lab = torch.cat([labels, q_lab.to(device)], dim=0)
    else:
        all_feat = feat
        all_lab = labels
    all_feat = F.normalize(all_feat, dim=1)

    # logits: anchors (batch) x all
    logits = torch.matmul(F.normalize(feat, dim=1), all_feat.t()) / temp  # [B, B+Q]
    # mask of positives
    mask = labels.view(-1,1) == all_lab.view(1,-1)  # [B, B+Q]
    # remove self-contrast for batch positions
    eye = torch.eye(B, device=device, dtype=torch.bool)
    mask[:, :B] = mask[:, :B] & (~eye)

    exp_logits = torch.exp(logits)
    # denominator: exclude self for batch part; queue has no self
    logits_mask = torch.ones_like(mask, dtype=exp_logits.dtype)
    logits_mask[:, :B] = logits_mask[:, :B] * (~eye)

    # optional confusion-aware negative weights
    w_neg_full = None
    conf_w = getattr(_supcon_loss_with_queue, "_conf_weight_matrix", None)
    if conf_w is not None:
        try:
            conf_w = conf_w.to(device=device, dtype=feat.dtype)
            all_lab_t = all_lab.to(device=device)
            w_mat = conf_w.index_select(0, labels)  # [B, num_classes]
            w_neg_full = w_mat.index_select(1, all_lab_t)  # [B, B+Q]
            w_neg_full = torch.where(mask, torch.ones_like(w_neg_full), w_neg_full)
        except Exception:
            w_neg_full = None

    if w_neg_full is not None:
        exp_logits = exp_logits * w_neg_full

    exp_logits = exp_logits * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    pos_counts = mask.sum(1)
    mean_log_prob_pos = (mask * log_prob).sum(1) / torch.clamp(pos_counts, min=1e-12)
    # ----- class-aware weighting (optional; mean-normalised to avoid scale drift) -----
    loss_weight = None
    if hasattr(_supcon_loss_with_queue, "_class_counts_tensor"):
        w_raw = 1.0 / torch.clamp(_supcon_loss_with_queue._class_counts_tensor.to(device=device, dtype=feat.dtype), min=1.0)
        cap = getattr(_supcon_loss_with_queue, "_weight_cap", 3.0)
        if cap is not None:
            w_raw = torch.clamp(w_raw, max=float(cap))
        loss_weight = w_raw / (w_raw.mean() + 1e-12)
    if loss_weight is not None:
        anchor_w = loss_weight.index_select(0, labels)
        loss = -(anchor_w * mean_log_prob_pos).mean()
    else:
        loss = -mean_log_prob_pos.mean()
    pos_mean = float(pos_counts.float().mean().item())
    pos_min = float(pos_counts.min().item()) if pos_counts.numel() > 0 else 0.0
    return loss, pos_mean, pos_min

def train_epoch_cls(epoch: int,
                    model: torch.nn.Module,
                    loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                    args,
                    queue_state=None) -> tuple[float, dict, Optional[dict]]:
    model.train()
    loss_meter = AverageMeter('Loss', ':.4e')
    ce_meter   = AverageMeter('CE', ':.4e')
    supcon_meter = AverageMeter('SupCon', ':.4e')
    supcon_pos_min_meter = AverageMeter('SupConPosMin', ':.4e')
    mse_meter  = AverageMeter('MSE',  ':.4e')

    _ys, _probs = [], []
    accum = max(1, int(getattr(args, "accum_steps", 1)))
    optimizer.zero_grad(set_to_none=True)
    queue_size = max(0, int(getattr(args, "supcon_queue_size", 0)))
    num_classes = int(getattr(args, "num_classes", 1))
    per_class_cap = max(1, queue_size // num_classes) if queue_size > 0 else 0
    q_feat = {}
    q_ptr = {}
    q_len = {}
    # restore queue state if provided
    if queue_state and queue_size > 0:
        try:
            per_class_cap = max(1, min(per_class_cap if per_class_cap > 0 else queue_size, int(queue_state.get("per_class_cap", per_class_cap or 1))))
            q_ptr_loaded = queue_state.get("q_ptr", {})
            q_len_loaded = queue_state.get("q_len", {})
            for cls, buf in queue_state.get("q_feat", {}).items():
                c = int(cls)
                buf_dev = buf.to(dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                cap = min(per_class_cap, buf_dev.shape[0])
                q_feat[c] = buf_dev[:cap].clone()
                q_ptr[c] = int(q_ptr_loaded.get(str(cls), q_ptr_loaded.get(c, 0))) % max(1, per_class_cap)
                q_len[c] = min(cap, int(q_len_loaded.get(str(cls), q_len_loaded.get(c, cap))))
        except Exception:
            q_feat, q_ptr, q_len = {}, {}, {}
    # stash class-counts for optional class-aware SupCon weighting
    if getattr(args, "use_supcon", False):
        cc = getattr(args, "_class_counts", None)
        if cc is not None:
            try:
                cc_tensor = torch.tensor(np.maximum(np.asarray(cc, dtype=np.float32), 1.0), dtype=torch.float32)
                _supcon_loss_with_queue._class_counts_tensor = cc_tensor
                _supcon_loss_with_queue._weight_cap = float(getattr(args, "supcon_weight_cap", 3.0))
            except Exception:
                pass
    if getattr(args, "cac_use", False):
        w = getattr(args, "_cac_weight_matrix", None)
        try:
            if w is not None:
                w_arr = np.asarray(w, dtype=np.float32)
                _supcon_loss_with_queue._conf_weight_matrix = torch.tensor(w_arr, dtype=torch.float32)
            else:
                _supcon_loss_with_queue._conf_weight_matrix = None
        except Exception:
            _supcon_loss_with_queue._conf_weight_matrix = None
    else:
        _supcon_loss_with_queue._conf_weight_matrix = None

    t0 = time.perf_counter()
    for it, batch in enumerate(loader):
        x = batch['image'].float().cuda(non_blocking=True)
        y = batch['label'].long().cuda(non_blocking=True)

        aux1 = {}
        feat_proj = None
        if getattr(args, "use_supcon", False) and hasattr(model, "forward_features"):
            out_pack = model.forward_features(x)
            if isinstance(out_pack, tuple):
                out1 = out_pack[0]
                feat_enc = out_pack[1] if len(out_pack) >= 2 else None
                feat_proj = out_pack[2] if len(out_pack) >= 3 else None
                if len(out_pack) >= 5 and isinstance(out_pack[4], dict):
                    aux1 = out_pack[4]
            else:
                out1 = out_pack
                feat_enc = None
            logits1, _ = _extract_logits_and_feat((out1, feat_enc))
        else:
            out1 = model(x)
            feat_enc = None
            if isinstance(out1, tuple):
                feat_enc = out1[1] if len(out1) >= 2 and torch.is_tensor(out1[1]) else None
                if len(out1) >= 5 and isinstance(out1[4], dict):
                    aux1 = out1[4]
            logits1, feat_enc = _extract_logits_and_feat(out1)
        logits1 = _ensure_class_dim1(logits1, args.num_classes)
        logits1 = _coerce_to_2d_logits(logits1, args.num_classes)

        loss_main = _classification_loss(logits1, y, args)
        loss = loss_main
        ce_meter.update(loss_main.item(), x.size(0))

        # BoQ diversity auxiliary loss (only from first view)
        if aux1.get("boq_div_loss", None) is not None:
            lam_div = float(getattr(args, "boq_lambda_div", 1e-3))
            if lam_div > 0:
                loss = loss + lam_div * aux1["boq_div_loss"]

        if getattr(args, "use_cl", False) and ('image2' in batch):
            x2 = batch['image2'].float().cuda(non_blocking=True)
            if getattr(args, "use_supcon", False) and hasattr(model, "forward_features"):
                out_pack2 = model.forward_features(x2)
                if isinstance(out_pack2, tuple):
                    out2 = out_pack2[0]
                    feat2 = out_pack2[1] if len(out_pack2) >= 2 else None
                else:
                    out2 = out_pack2
                    feat2 = None
                logits2, feat2 = _extract_logits_and_feat((out2, feat2))
            else:
                out2 = model(x2)
                logits2, feat2 = _extract_logits_and_feat(out2)
            logits2 = _ensure_class_dim1(logits2, args.num_classes) 
            logits2 = _coerce_to_2d_logits(logits2, args.num_classes)
            ce2 = _classification_loss(logits2, y, args)
            # for consistency here, just compare logits to avoid shape mismatch
            base1 = logits1
            base2 = logits2
            lcons = F.mse_loss(base1, base2)
            loss = loss + ce2 + float(getattr(args, "cl_weight", 0.1)) * lcons
            mse_meter.update(lcons.item())

        if getattr(args, "use_supcon", False) and (feat_proj is not None):
            z = F.normalize(feat_proj, dim=1)
            # build class-balanced queue
            def _gather_queue(qf, ql):
                feat_list, lab_list = [], []
                total = 0
                for cls, buf in qf.items():
                    L = int(ql.get(cls, 0))
                    if L <= 0:
                        continue
                    feat_list.append(buf[:L])
                    lab_list.append(torch.full((L,), cls, device=buf.device, dtype=torch.long))
                    total += L
                if total == 0:
                    return None, None, 0
                return torch.cat(feat_list, dim=0), torch.cat(lab_list, dim=0), total

            q_feat_cat, q_lab_cat, q_total = _gather_queue(q_feat, q_len) if queue_size > 0 else (None, None, 0)
            if q_total > 0:
                supcon, _, pos_min = _supcon_loss_with_queue(
                    z, y, q_feat_cat, q_lab_cat, temp=float(getattr(args, "supcon_temp", 0.07))
                )
            else:
                supcon, _, pos_min = _supcon_loss_with_queue(
                    z, y, None, None, temp=float(getattr(args, "supcon_temp", 0.07))
                )
            loss = loss + float(getattr(args, "supcon_lambda", 0.05)) * supcon
            supcon_meter.update(supcon.item(), x.size(0))
            supcon_pos_min_meter.update(pos_min, x.size(0))
            # enqueue
            if queue_size > 0:
                with torch.no_grad():
                    bs = z.size(0)
                    for i in range(bs):
                        cls = int(y[i].item())
                        if cls not in q_feat:
                            cap_cls = per_class_cap
                            q_feat[cls] = torch.zeros((cap_cls, z.size(1)), device=z.device, dtype=z.dtype)
                            q_ptr[cls] = 0
                            q_len[cls] = 0
                        cap_cls = q_feat[cls].shape[0]
                        ptr = q_ptr.get(cls, 0) % cap_cls
                        q_feat[cls][ptr] = z[i].detach()
                        ptr = (ptr + 1) % cap_cls
                        q_ptr[cls] = ptr
                        q_len[cls] = min(cap_cls, q_len.get(cls, 0) + 1)

        do_step = ((it + 1) % accum == 0) or ((it + 1) == len(loader))
        (loss / accum).backward()
        if do_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(args, "grad_clip_norm", 5.0))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_meter.update(loss.item(), x.size(0))
        _ys.append(y.detach().cpu())
        _probs.append(torch.softmax(logits1.detach(), dim=1).cpu())

        dt = time.perf_counter() - t0
        t0 = time.perf_counter()
        print(f"[Train][Ep {epoch:03d}][{it:04d}/{len(loader):04d}] "
              f"loss {loss_meter.avg:.4f} "
              f"ce {ce_meter.avg:.4f} "
              f"supcon {supcon_meter.avg:.3e} "
              f"supcon_pos_min {supcon_pos_min_meter.avg:.3e} "
              f"mse {mse_meter.avg:.3e} "
              f"lr {optimizer.param_groups[0]['lr']:.3e} "
              f"dt {dt:.2f}s")

    # Step LR scheduler once per epoch (matches segmentation pipeline intent)
    if scheduler is not None:
        try:
            scheduler.step()
        except Exception:
            pass

    train_metrics = {}
    try:
        y_np = torch.cat(_ys, dim=0).numpy()
        p_np = torch.cat(_probs, dim=0).numpy()
        auc_pc, macro_auc = _per_class_auc(y_np, p_np, args.num_classes)
        auprc_pc, macro_apr = _per_class_auprc(y_np, p_np, args.num_classes)
        train_metrics = {
            "macro_auc": macro_auc,
            "macro_auprc": macro_apr,
            "auc_per_class": [float(x) if x is not None else None for x in auc_pc],
            "auprc_per_class": [float(x) if x is not None else None for x in auprc_pc],
        }
    except Exception:
        pass

    # stash queue state for checkpoint/resume
    queue_state_out = None
    if queue_size > 0 and getattr(args, "use_supcon", False):
        try:
            q_cpu = {int(c): buf[:q_len.get(c, 0)].detach().cpu() for c, buf in q_feat.items() if q_len.get(c, 0) > 0}
            queue_state_out = {
                "per_class_cap": per_class_cap,
                "q_ptr": {int(k): int(v) for k, v in q_ptr.items()},
                "q_len": {int(k): int(v) for k, v in q_len.items()},
                "q_feat": q_cpu,
            }
        except Exception:
            queue_state_out = None

    return float(loss_meter.avg), train_metrics, queue_state_out

@torch.no_grad()
def val_epoch_cls(loader, model, args, num_classes):
    model.eval()
    ys, probs = [], []
    loss_meter = AverageMeter('ValLoss', ':.4e')

    for batch in loader:
        x = batch['image'].float().cuda(non_blocking=True)
        y = batch['label'].long().cuda(non_blocking=True)

        out = model(x)
        logits, _ = _extract_logits_and_feat(out)
        logits = _ensure_class_dim1(logits, num_classes)
        logits = _coerce_to_2d_logits(logits, num_classes)

        loss = _classification_loss(logits, y, args)
        loss_meter.update(loss.item(), x.size(0))

        p = F.softmax(logits, dim=1)
        probs.append(p.detach().cpu())
        ys.append(y.detach().cpu())

    y_np = torch.cat(ys, dim=0).numpy()
    p_np = torch.cat(probs, dim=0).numpy()

    auc_pc, macro_auc = _per_class_auc(y_np, p_np, num_classes)
    auprc_pc, macro_auprc = _per_class_auprc(y_np, p_np, num_classes)
    pred_labels = p_np.argmax(1)
    support_per_class = np.bincount(y_np, minlength=num_classes).astype(int).tolist()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_np, pred_labels):
        cm[int(t), int(p)] += 1

    return dict(
        loss=float(loss_meter.avg),
        auc_per_class=[float(x) if x is not None else None for x in auc_pc],
        macro_auc=float(macro_auc) if macro_auc is not None else None,
        auprc_per_class=[float(x) if x is not None else None for x in auprc_pc],
        macro_auprc=float(macro_auprc) if macro_auprc is not None else None,
        support_per_class=support_per_class,
        confusion=cm.tolist(),
    )
