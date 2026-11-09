"""CDPA: Contrast-Decomposed Prototype Alignment (UDA, config-gated).

Replaces the raw-feature fg prototype cosine (prototype_alignment.py,
regions=['fg']) with prototypes computed in a *contrast space*:

    d = x - mu_local          (center-surround, k x k local mean)

Motivated by the feature-level diagnostic (EchoNet target, A0 ckpt):
signed center-surround margin gt-vs-rand +0.946 vs raw +0.423, k=7
(stride-4 -> ~28 px image domain = cavity scale), per-sample channel-wise
z-score normalization. Signed > magnitude > raw.

Design (frozen for the first experiment):
    kernel_size=7, mode='signed', normalize='spatial_z',
    use_background=False (fg-only, same as C2), bg_weight=0.25 (unused
    unless use_background).

Model-agnostic: operates on [N, C, h, w] feature maps + [N, H, W] masks.
No trainable parameters; inference unchanged.
"""
import torch
import torch.nn.functional as F

__all__ = ['contrast_transform', 'prototype_from_mask',
           'contrast_proto_alignment_loss']


def contrast_transform(feat, kernel_size=7, mode='signed',
                       normalize='spatial_z', eps=1e-6):
    """Decompose features into a local-contrast space.

    feat: [N, C, H, W] (any float dtype; returned in float32).
    mode:
      'signed'  -> d = x - mu            (C channels; DEFAULT)
      'concat'  -> cat([relu(d), relu(-d)], dim=1)   (2C channels)
      'pos'     -> relu(d)
      'neg'     -> relu(-d)
      'mag'     -> |d|
    normalize:
      'spatial_z' -> per-sample per-channel (x - mean) / std over H,W
      'l2'        -> channel-wise L2 per pixel
      'none'      -> no normalization
    """
    feat = feat.float()
    mu = F.avg_pool2d(feat, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, count_include_pad=False)
    d = feat - mu
    if mode == 'signed':
        out = d
    elif mode == 'concat':
        out = torch.cat([F.relu(d), F.relu(-d)], dim=1)
    elif mode == 'pos':
        out = F.relu(d)
    elif mode == 'neg':
        out = F.relu(-d)
    elif mode == 'mag':
        out = d.abs()
    else:
        raise ValueError(f'unknown contrast mode: {mode}')

    if normalize == 'spatial_z':
        mean = out.mean(dim=(2, 3), keepdim=True)
        std = out.std(dim=(2, 3), keepdim=True)
        out = (out - mean) / (std + eps)
    elif normalize == 'l2':
        out = F.normalize(out, dim=1, eps=eps)
    elif normalize != 'none':
        raise ValueError(f'unknown normalize: {normalize}')
    return out


def prototype_from_mask(feat, mask, conf=None, eps=1e-6, min_pixels=16):
    """Confidence-weighted GAP prototype in the given feature space.

    feat: [N, C, h, w]; mask/conf: [N, H, W] or [N, 1, H, W] (resized to
    (h, w) with nearest if shapes differ). weight = mask * conf.
    Returns (proto [N, C] L2-normalized per sample or None, n_pixels
    [N] weight sums). Samples below min_pixels get a zero-weight proto
    row; the caller decides validity.
    """
    n, c, h, w = feat.shape
    mask = mask.float()
    if mask.dim() == 4:
        mask = mask[:, 0]
    if mask.shape[-2:] != (h, w):
        mask = F.interpolate(mask.unsqueeze(1), size=(h, w),
                             mode='nearest').squeeze(1)
    weight = mask
    if conf is not None:
        conf = conf.float()
        if conf.dim() == 4:
            conf = conf[:, 0]
        if conf.shape[-2:] != (h, w):
            conf = F.interpolate(conf.unsqueeze(1), size=(h, w),
                                 mode='nearest').squeeze(1)
        weight = weight * conf
    n_px = weight.sum(dim=(1, 2))                       # [N]
    weight = weight.reshape(n, 1, h, w)
    proto = (feat * weight).sum(dim=(2, 3)) / (weight.sum(dim=(2, 3)) + eps)
    proto = F.normalize(proto, dim=1, eps=eps)
    return proto, n_px


def contrast_proto_alignment_loss(src_feat, src_mask, tgt_feat, tgt_pseudo,
                                  tgt_conf=None, kernel_size=7,
                                  contrast_mode='signed', normalize='spatial_z',
                                  min_pixels=16, use_background=False,
                                  bg_weight=0.25, eps=1e-6):
    """CDPA loss: align target contrast-space fg prototypes to the source
    GT-anchor contrast prototypes (cosine, 1 - cos).

    src_feat/tgt_feat: [N, C, h, w] (tgt carries grad; src is a frozen
        GT anchor -- computed under no_grad).
    src_mask: [N, H, W] source GT (1 = LV fg).
    tgt_pseudo: [N, H, W] target pseudo mask (0/1).
    tgt_conf: [N, H, W] confidence map or None.

    Returns (loss scalar, stats dict). Zero-grad loss when no valid
    prototype exists in the batch.
    """
    # ---- source anchor: no grad, GT mask, no conf gating ----
    with torch.no_grad():
        src_c = contrast_transform(src_feat, kernel_size, contrast_mode,
                                   normalize, eps)
        src_proto, src_px = prototype_from_mask(
            src_c, src_mask, None, eps, min_pixels)
        if use_background:
            src_proto_bg, _ = prototype_from_mask(
                src_c, 1.0 - src_mask, None, eps, min_pixels)

    # ---- target: grad flows through the contrast transform ----
    tgt_c = contrast_transform(tgt_feat, kernel_size, contrast_mode,
                               normalize, eps)
    tgt_proto, tgt_px = prototype_from_mask(
        tgt_c, tgt_pseudo, tgt_conf, eps, min_pixels)

    valid = (src_proto is not None) and (tgt_proto is not None)
    if use_background:
        valid = valid and (src_proto_bg is not None)
    if not valid:
        zero = tgt_feat.float().sum() * 0.0
        stats = {'cdpa_valid': 0.0, 'cdpa_mode': contrast_mode,
                 'cdpa_kernel': float(kernel_size),
                 'cdpa_normalize': normalize,
                 'cdpa_src_fg_pixels': 0.0, 'cdpa_tgt_fg_pixels': 0.0,
                 'cdpa_loss_fg': 0.0}
        return zero, stats

    ok = (src_px >= min_pixels) & (tgt_px >= min_pixels)      # [N]
    if not ok.any():
        zero = tgt_feat.float().sum() * 0.0
        stats = {'cdpa_valid': 0.0, 'cdpa_mode': contrast_mode,
                 'cdpa_kernel': float(kernel_size),
                 'cdpa_normalize': normalize,
                 'cdpa_src_fg_pixels': float(src_px.sum()),
                 'cdpa_tgt_fg_pixels': float(tgt_px.sum()),
                 'cdpa_loss_fg': 0.0}
        return zero, stats

    cos = F.cosine_similarity(src_proto, tgt_proto, dim=1)    # [N]
    loss_fg = (1.0 - cos[ok]).mean()

    loss = loss_fg
    stats = {'cdpa_valid': 1.0, 'cdpa_mode': contrast_mode,
             'cdpa_kernel': float(kernel_size),
             'cdpa_normalize': normalize,
             'cdpa_src_fg_pixels': float(src_px[ok].sum()),
             'cdpa_tgt_fg_pixels': float(tgt_px[ok].sum()),
             'cdpa_loss_fg': float(loss_fg.detach())}
    if use_background:
        tgt_proto_bg, tgt_px_bg = prototype_from_mask(
            tgt_c, 1.0 - tgt_pseudo, None, eps, min_pixels)
        if tgt_proto_bg is not None:
            ok_bg = (tgt_px_bg >= min_pixels)
            if ok_bg.any():
                cos_bg = F.cosine_similarity(src_proto_bg, tgt_proto_bg, dim=1)
                loss_bg = (1.0 - cos_bg[ok_bg]).mean()
                loss = loss + bg_weight * loss_bg
                stats['cdpa_loss_bg'] = float(loss_bg.detach())
    return loss, stats
