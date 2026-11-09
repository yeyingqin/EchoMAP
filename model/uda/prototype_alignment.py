"""UDA Stage 3: structure-aware prototype alignment.

GraphEcho aligns cardiac structure across domains with a graph module over
multiple chambers; LV segmentation needs only one structure, so we align
three LV-specific feature prototypes between source (GT masks) and target
(CFMC pseudo masks):

    fg  : inside the LV mask
    bd  : boundary band of the mask (dilate - erode)
    bg  : near-background ring outside the mask (dilate(r_out) - dilate(r_in))

Each prototype is a masked-average-pooled feature vector, L2-normalized,
computed over the whole batch (source and target windows are independent
samples, so there is no per-frame pairing). Source prototypes are computed
no-grad: the source side is the GT-anchored anchor -- aligning target to a
drifting source would risk source forgetting. Loss per class is the cosine
distance between the (normalized) source and target prototypes; classes
with too few valid pixels are skipped; if all are missing the loss is a
graph-connected zero (never NaN).

Pure PyTorch morphology via max_pool2d:
    dilate(m, r) = max_pool2d(m, 2r+1, stride=1, padding=r)
    erode(m, r)  = 1 - dilate(1 - m, r)

Masks are resized to the feature resolution BEFORE morphology (radii are
in feature-map pixels).
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    k = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return 1.0 - _dilate(1.0 - mask, radius)


def boundary_band(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """mask [B,T,H,W] (or [B,H,W]) float 0/1 -> boundary band 0/1."""
    return (_dilate(mask, radius) - _erode(mask, radius)).clamp(0, 1)


def near_background_ring(mask: torch.Tensor, r_in: int, r_out: int) -> torch.Tensor:
    """mask [B,T,H,W] (or [B,H,W]) -> ring outside the mask 0/1."""
    return (_dilate(mask, r_out) - _dilate(mask, r_in)).clamp(0, 1)


def _resize_like(mask_bt_hw: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """[B,T,H,W] binary -> [B,T,h,w] binary (nearest keeps it binary)."""
    B, T, H, W = mask_bt_hw.shape
    flat = mask_bt_hw.view(B * T, 1, H, W)
    return F.interpolate(flat, size=size, mode='nearest').view(B, T, *size)


def _masked_pool(feat_btchw: torch.Tensor, region_bthw: torch.Tensor,
                 min_pixels: int) -> Tuple[Optional[torch.Tensor], int]:
    """Masked average pooling -> one prototype per batch.

    feat_btchw : [B,T,C,h,w]
    region_bthw: [B,T,h,w] float 0/1
    Returns (proto[C] or None, n_valid_pixels).
    """
    m = region_bthw.unsqueeze(2)                       # [B,T,1,h,w]
    n = int(m.sum().item())
    if n < min_pixels:
        return None, n
    proto = (feat_btchw * m).sum(dim=(0, 1, 3, 4))     # [C]
    return proto / max(n, 1), n


def prototype_alignment_loss(
    feat_s: torch.Tensor, feat_t: torch.Tensor,
    mask_s: torch.Tensor, mask_t: torch.Tensor,
    conf_t: Optional[torch.Tensor] = None,
    proto_conf_thr: float = 0.9,
    boundary_radius: int = 2,
    bg_inner_radius: int = 2,
    bg_outer_radius: int = 6,
    min_pixels: int = 20,
    regions: Tuple[str, ...] = ('fg', 'bd', 'bg'),
    region_weights: Optional[Dict[str, float]] = None,
) -> (torch.Tensor, Dict):
    """Structure-aware prototype alignment (cosine).

    feat_s/feat_t : [B,T,C,h,w] (grad-carrying; source is pooled no-grad
                    internally as a stable GT anchor)
    mask_s        : [B,T,H,W] source GT binary masks
    mask_t        : [B,T,H,W] target pseudo binary masks
    conf_t        : [B,T,H,W] target pseudo confidence (optional); target
                    regions keep only pixels with conf > proto_conf_thr

    Returns (loss, stats). stats:
        proto_{fg,bd,bg}_pixels, proto_valid_ratio (over selected),
        cosine_{fg,bd,bg} (nan when class skipped or not selected),
        proto_{fg,bd,bg}_loss (nan when class skipped or not selected)
    """
    assert all(r in ('fg', 'bd', 'bg') for r in regions) and regions
    device = feat_s.device
    _, _, _, h, w = feat_s.shape

    def regions_fn(mask_bt_hw: torch.Tensor, conf: Optional[torch.Tensor]):
        m = _resize_like(mask_bt_hw.float(), (h, w))
        out = {}
        if 'fg' in regions:
            out['fg'] = m
        if 'bd' in regions:
            out['bd'] = boundary_band(m, boundary_radius)
        if 'bg' in regions:
            out['bg'] = near_background_ring(m, bg_inner_radius, bg_outer_radius)
        if conf is not None:
            c = _resize_like((conf > proto_conf_thr).float(), (h, w))
            out = {k: v * c for k, v in out.items()}
        return out

    with torch.no_grad():
        regs_s = regions_fn(mask_s, None)
        protos_s, px_s = zip(*(
            _masked_pool(feat_s.detach().float(), r, min_pixels)
            for r in regs_s.values()))
    regs_t = regions_fn(mask_t, conf_t)
    protos_t, px_t = zip(*(
        _masked_pool(feat_t.float(), r, min_pixels)
        for r in regs_t.values()))

    if region_weights is None:
        region_weights = {}
    assert all(float(region_weights.get(r, 1.0)) >= 0.0 for r in regions)

    stats: Dict = {}
    loss_items: list = []          # (weight, 1-cos) over valid regions
    cosines = []
    for name, ps, pt, n_s, n_t in zip(regions, protos_s, protos_t, px_s, px_t):
        stats[f'proto_{name}_pixels'] = float(min(n_s, n_t))
        if ps is None or pt is None:
            stats[f'cosine_{name}'] = float('nan')
            stats[f'proto_{name}_loss'] = float('nan')
            continue
        ns = F.normalize(ps, dim=0)
        nt = F.normalize(pt, dim=0)
        cos = (ns * nt).sum().clamp(-1.0, 1.0)
        loss_items.append((float(region_weights.get(name, 1.0)), 1.0 - cos))
        cosines.append(cos)
        stats[f'cosine_{name}'] = float(cos.item())
        stats[f'proto_{name}_loss'] = float((1.0 - cos).item())
    for name in ('fg', 'bd', 'bg'):        # keep log keys present
        stats.setdefault(f'cosine_{name}', float('nan'))
        stats.setdefault(f'proto_{name}_loss', float('nan'))

    stats['proto_valid_ratio'] = len(loss_items) / len(regions)
    stats['proto_norm_s'] = float(torch.stack(
        [p.norm() for p in protos_s if p is not None]).mean().item()) \
        if any(p is not None for p in protos_s) else 0.0
    stats['proto_norm_t'] = float(torch.stack(
        [p.norm() for p in protos_t if p is not None]).mean().item()) \
        if any(p is not None for p in protos_t) else 0.0

    if not loss_items:
        return feat_s.sum() * 0.0, stats
    weights = torch.tensor([w for w, _ in loss_items], device=device)
    if float(weights.sum()) <= 0.0:      # degenerate config: all weights 0
        return feat_s.sum() * 0.0, stats
    loss = (weights * torch.stack([l for _, l in loss_items])).sum() / weights.sum()
    return loss, stats
