"""UDA Stage 2: phase-aware area-curve temporal consistency.

Lightweight counterpart of GraphEcho's temporal cycle consistency: instead of
a TGCN over features, we regularize the SECOND DIFFERENCE of the student's
own soft area curve, A_t = sum_hw p_t, on target windows -- and only on
phase-stable triples (t-1, t, t+1):

  - the three frames fall in the same or adjacent area-phase bins
    (window-level running min/max normalization, 4 bins -- same spirit as
    the CFMC area-bin cache), OR
  - both consecutive area jumps are small (|dA|/A < jump_thr),

so fast-filling/ejection transients are never smoothed (the over-smoothing
lesson from the CFMC ablations). Pure tensor module, no model/teacher deps.
"""
from typing import Dict

import torch

EPS = 1e-6


def area_smooth_loss(logits_bt2hw: torch.Tensor,
                     area_bins: int = 4,
                     jump_thr: float = 0.15,
                     area_mode: str = 'sum',
                     min_area_px: float = 300.0) -> (torch.Tensor, Dict):
    """Second-difference area smoothness on phase-stable triples.

    logits_bt2hw : student output [B, T, 2, H, W] (grad-carrying)
    area_mode    : 'sum' -> A_t = sum of fg prob (pixel area scale)
                   'mean' -> A_t = mean of fg prob
    The second difference is NORMALIZED by the window's mean area
    (no-grad) so the loss is dimension-free and the suggested
    lambda_tc grid {0.01, 0.03, 0.05} is meaningful; with raw pixel-area
    scale lambda_tc=0.03 dominates the total loss and rewards degenerate
    constant-area solutions. Windows whose mean area < min_area_px are
    skipped (no LV / collapsed prediction -- never reward empty masks).
    Returns (loss, stats); graph-connected zero when no valid triple.
    """
    assert area_mode in ('sum', 'mean')
    p = torch.softmax(logits_bt2hw.float(), dim=2)[:, :, 1]     # [B,T,H,W]
    reduce = torch.sum if area_mode == 'sum' else torch.mean
    A = reduce(p, dim=(2, 3))                                  # [B,T] grad

    with torch.no_grad():
        a = A.detach()
        B, T = a.shape
        win_mean = a.mean(dim=1, keepdim=True)                 # [B,1]
        # window-level phase bins (running min/max normalization)
        a_min = a.min(dim=1, keepdim=True).values
        a_max = a.max(dim=1, keepdim=True).values
        norm = (a - a_min) / (a_max - a_min).clamp(min=EPS)
        bins = torch.clamp((norm * area_bins).long(), max=area_bins - 1)
        # gate: same/adjacent bins across the triple, OR small jumps on both
        # sides (never smooth fast contraction/filling transients)
        b0, b1, b2 = bins[:, :-2], bins[:, 1:-1], bins[:, 2:]
        bin_ok = (torch.stack([b0, b1, b2], dim=0).max(dim=0).values -
                  torch.stack([b0, b1, b2], dim=0).min(dim=0).values) <= 1
        j0 = (a[:, 1:-1] - a[:, :-2]).abs() / a[:, :-2].clamp(min=1.0)
        j1 = (a[:, 2:] - a[:, 1:-1]).abs() / a[:, 1:-1].clamp(min=1.0)
        jump_ok = (j0 < jump_thr) & (j1 < jump_thr)
        valid = (bin_ok | jump_ok) & (win_mean > min_area_px)  # [B,T-2]

    second = (A[:, 2:] - 2.0 * A[:, 1:-1] + A[:, :-2]).abs()   # [B,T-2]
    n = int(valid.sum().item())
    stats = {
        'tc_valid_ratio': n / max(valid.numel(), 1),
        'tc_second_diff': float((second / win_mean).detach().mean().item())
                          if second.numel() else 0.0,
    }
    if n == 0:
        return A.sum() * 0.0, stats
    loss = (second * valid.float()).sum() / (n * win_mean)     # relative
    loss = loss.mean()
    stats['tc_second_diff_valid'] = float(loss.item())
    return loss, stats
