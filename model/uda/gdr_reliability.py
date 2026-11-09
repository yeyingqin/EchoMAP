"""Reliability-aware GDR: per-frame target reliability from GDR dynamics.

Estimates a per-frame reliability score r_t from the GDR memory readout /
state statistics returned by the backbone under return_features=True, and
uses it to (a) reweight the UDA pseudo-label loss per frame and (b) scale
the target prototype pooling weight. Pure tensor module -- no parameters,
no GDR update-rule changes, inference untouched.

    delta_t = mean_abs(R_t - R_{t-1}) / (mean_abs(R_{t-1}) + eps)
    r_t     = exp(-gamma * delta_t)

t = 0 has no predecessor -> delta_0 = 0 -> r_0 = 1. All statistics are
detached: r_t is a constant weight, never a gradient path into the GDR.
"""
import torch

from model.uda.uda_losses import pseudo_label_loss

EPS = 1e-6


def gdr_reliability_scores(feats: dict,
                           source: str = 'gdr_readout',
                           gamma: float = 2.0,
                           normalize: bool = True,
                           clamp_min: float = 0.3,
                           clamp_max: float = 1.0,
                           eps: float = 1e-6,
                           motion: torch.Tensor | None = None,
                           lambda_motion: float = 0.0) -> torch.Tensor:
    """Per-frame reliability r [B, T] in [clamp_min, clamp_max] (detached).

    source:
      'gdr_readout' -- delta on the readout maps [B,T,V,H,W]
      'state_norm'  -- delta on the state L2 norm sequence [B,T]
      'beta_alpha'  -- delta on the beta/alpha (mean,std) curves [B,T,4]
    normalize: per-sample min-max over T to [0,1] before clamping (keeps
    only the within-window ordering; the window mean weight is preserved
    by the caller's weight normalization).
    motion / lambda_motion: phase-aware correction (mode='phase_corrected').
    motion m_t in [0, m_max] is a normalized motion indicator (LV prob-fg
    area jump, computed by the caller from the same window). With
    lambda_motion > 0 the decay denominator becomes (1 + lambda_motion*m_t):

        r_phase_t = exp(-gamma * delta_t / (1 + lambda_motion * m_t))

    so genuine fast motion phases (large m_t) are protected from being
    down-weighted. lambda_motion = 0 recovers the raw score exactly.
    """
    assert source in ('gdr_readout', 'state_norm', 'beta_alpha'), source
    if source == 'gdr_readout':
        seq = feats['gdr_readout'].float().detach()          # [B,T,V,H,W]
        seq = seq.flatten(2)                                 # [B,T,D]
    elif source == 'state_norm':
        seq = feats['gdr_state_norm'].float().detach()[..., None]
    else:
        parts = [feats['gdr_beta'][k].float().detach()[..., None]
                 for k in ('mean', 'std')] + \
                [feats['gdr_alpha'][k].float().detach()[..., None]
                 for k in ('mean', 'std')]
        seq = torch.cat(parts, dim=-1)                       # [B,T,4]
        seq = torch.nan_to_num(seq, nan=0.0)                 # t=0 placeholders

    prev = torch.cat([seq[:, :1], seq[:, :-1]], dim=1)       # shift by one
    num = (seq - prev).abs().mean(dim=2)                     # [B,T]
    den = prev.abs().mean(dim=2) + eps
    delta = num / den
    delta[:, 0] = 0.0                                        # no predecessor

    if lambda_motion > 0.0 and motion is not None:
        delta = delta / (1.0 + lambda_motion * motion.to(delta.device))
    r = torch.exp(-gamma * delta)
    if normalize:
        lo = r.min(dim=1, keepdim=True).values
        hi = r.max(dim=1, keepdim=True).values
        r = (r - lo) / (hi - lo + eps)
    return r.clamp(min=clamp_min, max=clamp_max)


def reliability_weighted_pseudo_loss(logits_stack, p_stab, conf, fmc, ffpx,
                                     r, **pseudo_kwargs):
    """Pseudo-label loss with per-frame reliability weights.

    Recomputes the (unmodified) pseudo_label_loss on single frames and
    combines them as sum(w_t * l_t) / sum(w_t) over frame-valid (b, t),
    w = r / mean(r_valid) so the overall loss scale matches the baseline
    (lambda semantics unchanged -- single-variable w.r.t. the weights).
    Invalid frames contribute neither loss nor weight, exactly like the
    baseline's n_valid reduction.
    """
    B, T = r.shape[:2]
    frame_valid = ((ffpx >= pseudo_kwargs.get('min_pseudo_px', 50.0)) &
                   (fmc >= pseudo_kwargs.get('frame_conf_thr', 0.85))).float()
    w = r.float()
    w = w / w[frame_valid > 0].mean().clamp(min=EPS)         # mean weight = 1

    per_loss, per_w, per_dice, per_bce = [], [], [], []
    for b in range(B):
        for t in range(T):
            l, st = pseudo_label_loss(
                logits_stack[b:b + 1, t:t + 1],
                p_stab[b:b + 1, t:t + 1],
                conf[b:b + 1, t:t + 1],
                fmc[b:b + 1, t:t + 1],
                ffpx[b:b + 1, t:t + 1],
                **pseudo_kwargs)
            per_loss.append(l)
            per_w.append(w[b, t] * frame_valid[b, t])
            per_dice.append(st['uda_dice'])
            per_bce.append(st['uda_bce'])
    per_loss = torch.stack(per_loss)
    per_w = torch.stack(per_w)
    wsum = per_w.sum()
    if wsum.item() <= 0:
        zero = per_loss.sum() * 0.0
        return zero, {'uda_dice': 0.0, 'uda_bce': 0.0,
                      'rel_mean': float(r.mean().item()),
                      'rel_min': float(r.min().item())}
    loss = (per_loss * per_w).sum() / wsum
    stats = {
        'uda_dice': float((torch.tensor(per_dice) *
                           per_w.cpu()).sum().item() / wsum.item()),
        'uda_bce': float((torch.tensor(per_bce) *
                          per_w.cpu()).sum().item() / wsum.item()),
        'rel_mean': float(r.mean().item()),
        'rel_min': float(r.min().item()),
    }
    return loss, stats
