"""UDA Stage 1 pseudo-label losses (confidence-weighted Dice + BCE).

Pure tensor module -- no model, no teacher logic. The teacher
(`model/uda/cfmc_teacher.py`) supplies p_stab / conf_map / per-frame stats;
the student's own (grad-carrying) logits are stacked as [B, T, 2, H, W].

Frame validity: a target frame contributes to L_self only if its pseudo mask
has >= min_pseudo_px foreground pixels AND its mean confidence passes
frame_conf_thr. An all-invalid batch returns a (graph-connected) zero so
backward/DDP stay well-defined -- never NaN.
"""
from typing import Dict

import torch
import torch.nn.functional as F

EPS = 1e-6


def pseudo_label_loss(logits_bt2hw: torch.Tensor,
                      p_stab: torch.Tensor,
                      conf_map: torch.Tensor,
                      frame_mean_conf: torch.Tensor,
                      frame_fg_px: torch.Tensor,
                      conf_thr: float = 0.9,
                      conf_mode: str = 'hard',
                      frame_conf_thr: float = 0.85,
                      min_pseudo_px: float = 50.0,
                      dice_weight: float = 1.0,
                      bce_weight: float = 1.0) -> (torch.Tensor, Dict):
    """Confidence-weighted pseudo-label loss.

    logits_bt2hw : student output, [B, T, 2, H, W] (grad-carrying)
    p_stab       : teacher stabilized fg prob, [B, T, H, W] (no grad)
    conf_map     : teacher pixel confidence in [0,1], [B, T, H, W]
    frame_mean_conf / frame_fg_px : [B, T] teacher frame stats
    conf_mode    : 'hard' -> binary weights (w = 1[conf >= conf_thr]);
                   'soft' -> continuous margin weights
    """
    assert conf_mode in ('hard', 'soft')
    B, T = p_stab.shape[:2]
    student_p = torch.softmax(logits_bt2hw.float(), dim=2)[:, :, 1]  # [B,T,H,W]

    with torch.no_grad():
        pseudo_mask = (p_stab > 0.5).float()
        # continuous margin weight w = clip(2*max(p,1-p) - 0.5, 0, 1)
        w_cont = torch.clamp(2.0 * torch.max(p_stab, 1.0 - p_stab) - 0.5, 0.0, 1.0)
        if conf_mode == 'hard':
            w = (w_cont >= conf_thr).float()
        else:
            w = w_cont
        frame_valid = (frame_fg_px >= min_pseudo_px) & \
                      (frame_mean_conf >= frame_conf_thr)

    # zero-out invalid frames before reduction
    fv = frame_valid.float().unsqueeze(-1).unsqueeze(-1)  # [B,T,1,1]
    w = w * fv
    pseudo_mask = pseudo_mask * fv

    n_valid = int(frame_valid.sum().item())
    stats = {
        'uda_valid_ratio': n_valid / max(B * T, 1),
        'uda_fg_ratio': float((pseudo_mask.sum() /
                               max(fv.sum().item() * pseudo_mask.shape[-1] *
                                   pseudo_mask.shape[-2], 1.0)).item()),
        'uda_mean_conf': float((conf_map * fv).sum().item() /
                               max(fv.expand_as(conf_map).sum().item(), 1.0)),
        'uda_n_valid_frames': n_valid,
    }

    if n_valid == 0:
        # all-invalid batch: graph-connected zero, no NaN
        zero = student_p.sum() * 0.0
        stats['uda_dice'] = 0.0
        stats['uda_bce'] = 0.0
        return zero, stats

    # ---- confidence-weighted BCE (per-frame mean over weighted pixels) ----
    bce_map = F.binary_cross_entropy(student_p.clamp(EPS, 1 - EPS),
                                     pseudo_mask, reduction='none')
    wsum = w.sum(dim=(2, 3)).clamp(min=1.0)                    # [B,T]
    bce_frame = (bce_map * w).sum(dim=(2, 3)) / wsum           # [B,T]
    bce_frame = bce_frame * frame_valid.float()
    bce_loss = bce_frame.sum() / n_valid

    # ---- Dice on valid frames (binary pseudo mask vs student prob) ----
    inter = (student_p * pseudo_mask * w).sum(dim=(2, 3))
    card = (student_p * w).sum(dim=(2, 3)) + pseudo_mask.sum(dim=(2, 3))
    dice_frame = 1.0 - (2.0 * inter + EPS) / (card + EPS)      # [B,T]
    dice_frame = dice_frame * frame_valid.float()
    dice_loss = dice_frame.sum() / n_valid

    loss = dice_weight * dice_loss + bce_weight * bce_loss
    stats['uda_dice'] = float(dice_loss.item())
    stats['uda_bce'] = float(bce_loss.item())
    return loss, stats
