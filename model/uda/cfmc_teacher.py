"""CFMC no-grad teacher for UDA Stage 1 (CFMC-guided pseudo-label
self-training).

This is a THIN WRAPPER over `shape_cache.ShapeLogitsCache` -- the CFMC core
(descriptor, gating, area-bin organization, retrieval, fusion) lives only in
`shape_cache.py` and is NOT duplicated here. The teacher:

  - runs fully under torch.no_grad(), on CPU numpy probability maps;
  - consumes DETACHED per-frame foreground probabilities (softmax fg channel
    of the student's logits, identical to the inference-time probs the
    offline sweep caches);
  - keeps one ShapeLogitsCache per video id (LRU-capped) so a long target
    sequence processed in windows behaves like a single continuous stream
    (stateful mode), or resets per window (stateless mode);
  - defaults to the D8b_ab configuration: gate=entropy_area (max),
    retrieval=prob_area_temporal, cache_org=area_bin, fixed-interval
    writing. Reliable-Write is deliberately NOT exposed as a default.

RW-soft ablation (config-gated, default OFF): teacher.write_policy='rw_soft'
keeps the Area-Bin organization AND the fixed interval=5 write cadence, but
additionally requires a frame to be reliable before it may enter the memory:
    write iff (t+1) % 5 == 0  AND  entropy <= p95  AND  area_jump <= 0.15
                              AND  area >= rw_min_area
The reliability check runs HERE (shape_cache.py is frozen): a rejected frame
is denied by temporarily raising the cache's interval so the fixed-interval
check in ShapeLogitsCache._maybe_store naturally fails; the cache's internal
state (entropy history, running area min/max, prev descriptors) still updates
every frame. RW-soft does NOT touch retrieval or beta -- the entropy-area
gate keeps controlling only the fusion strength, as before.

Component-level output cleanup (config-gated, default OFF; engineering
validation): teacher.cfmc_component_cleanup=true scores the connected
components of the FUSED foreground mask and keeps only the most LV-like one
(area + mean confidence + centroid consistency + optional memory-shape
consistency, minus a tiny-component penalty). It applies to the teacher
OUTPUT ONLY (p_stab / conf_map consumed by the pseudo-label loss and the
prototype alignment) -- memory writes happen inside process_frame on the RAW
student probs and are therefore never affected. cfmc_cc_clean_write=true
(option, default false) additionally feeds cleaned raw probs into the fusion
stream, which DOES change memory content. All cleanup runs on numpy under
torch.no_grad(); no gradients are introduced.
"""
import os
import sys
from collections import OrderedDict, deque

import numpy as np
import torch
from scipy.ndimage import label as _scipy_label

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from shape_cache import ShapeLogitsCache  # noqa: E402


def frame_entropy(p: np.ndarray) -> float:
    """Mean binary entropy of one prob map (same formula as
    shape_cache_sweep.frame_entropy)."""
    q = np.clip(p, 1e-4, 1 - 1e-4)
    return float((-(q * np.log(q) + (1 - q) * np.log(1 - q))).mean())


# --------------------------------------------------------------------------
# Component-level cleanup helpers (CFMC output-only; no new module name --
# part of the CFMC teacher's pseudo-label generation stage).
# --------------------------------------------------------------------------
def _connected_components_2d(mask: np.ndarray):
    """Label 8-connected foreground components. Returns (label_map, n)."""
    return _scipy_label(mask)


def _confidence_weighted_centroid(weights: np.ndarray):
    """Centroid (row, col) of a non-negative weight map; None if empty."""
    tot = float(weights.sum())
    if tot <= 1e-8:
        return None
    H, W = weights.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    return (float((weights * yy).sum() / tot),
            float((weights * xx).sum() / tot))


def _cleanup_frame_components(p_t: np.ndarray, conf_t: np.ndarray, *,
                              thr: float, min_area: float,
                              w_area: float, w_conf: float,
                              w_centroid: float, w_shape: float,
                              w_tiny: float, ref_area: float = None):
    """Keep the most LV-like connected component of one prob map.

    p_t   : [H, W] fused foreground probability (soft, NOT binarized out)
    conf_t: [H, W] confidence map (same semantics as p_t's conf)
    ref_area: optional memory reference area (CFMC cache descriptor);
              when None the shape term contributes 0 for every component,
              leaving the ranking to area/conf/centroid.

    Returns (p_clean, gate, info):
      p_clean = p_t * gate (soft -- never hard-binarized),
      gate    : float32 [H, W] in {0, 1},
      info    : debug dict (num_components, selected_area, removed_area, ...).
    Empty input (no fg) or a single component returns (p_t, ones, info).
    """
    info = {'num_components': 0, 'selected_area': 0.0, 'removed_area': 0.0,
            'removed_area_ratio': 0.0, 'mean_conf_selected': 0.0,
            'mean_conf_removed': 0.0, 'changed': False}
    binary = p_t > thr
    if not binary.any():
        return p_t, np.ones_like(p_t, dtype=np.float32), info
    lab, n = _connected_components_2d(binary)
    info['num_components'] = int(n)
    if n <= 1:
        return p_t, np.ones_like(p_t, dtype=np.float32), info

    H, W = p_t.shape
    diag = float(np.hypot(H, W))
    # current prediction center: soft-prob-weighted centroid of the frame
    g_cent = _confidence_weighted_centroid(p_t.astype(np.float64))
    areas, mean_confs, cents = [], [], []
    for c in range(1, n + 1):
        sel = lab == c
        a = float(sel.sum())
        areas.append(a)
        mean_confs.append(float(conf_t[sel].mean()) if a else 0.0)
        cents.append(_confidence_weighted_centroid(
            np.where(sel, p_t, 0.0).astype(np.float64)))
    total_area = float(sum(areas))
    max_area = max(areas)

    scores = []
    for c in range(n):
        a = areas[c]
        area_score = a / max(total_area, 1.0)
        conf_score = mean_confs[c]
        if g_cent is not None and cents[c] is not None:
            d = float(np.hypot(cents[c][0] - g_cent[0],
                               cents[c][1] - g_cent[1])) / diag
            centroid_score = float(np.exp(-d / 0.25))
        else:
            centroid_score = 0.0
        if ref_area is not None and ref_area > 0:
            # log-scale area agreement with the memory reference
            shape_score = float(np.exp(
                -abs(np.log(max(a, 1.0) / max(ref_area, 1.0))) / 0.5))
        else:
            shape_score = 0.0
        tiny = 1.0 if a < min_area else 0.0
        scores.append(w_area * area_score + w_conf * conf_score
                      + w_centroid * centroid_score + w_shape * shape_score
                      - w_tiny * tiny)
    sel_c = int(np.argmax(scores)) + 1
    gate = (lab == sel_c).astype(np.float32)
    sel_area = areas[sel_c - 1]
    removed = total_area - sel_area
    info.update({
        'selected_area': sel_area,
        'removed_area': removed,
        'removed_area_ratio': removed / max(total_area, 1.0),
        'mean_conf_selected': mean_confs[sel_c - 1],
        'mean_conf_removed':
            (float(np.mean([mean_confs[c] for c in range(n)
                            if c != sel_c - 1]))
             if n > 1 else 0.0),
        'changed': bool(removed > 0),
    })
    return p_t * gate, gate, info

# D8b_ab = BASE + D8B + AB from shape_cache_sweep.py (entropy thresholds are
# filled by calibrate() from the target-domain entropy distribution).
D8B_AB_DEFAULTS = dict(
    interval=5, topk=2, tau=0.25, beta_max=0.2,
    gate='entropy_area', combine='max',
    jump_low=0.03, jump_high=0.15,
    retrieval_mode='prob_area_temporal',
    cache_org='area_bin', area_bins=4, bin_slots=3,
    candidate_bin_radius=1, bin_replace_policy='fifo',
    max_slots=12,                       # area_bins * bin_slots
    cache_write_policy='fixed_interval',
)


class CFMCTeacher:
    """Stateful no-grad teacher producing stabilized probability maps,
    hard pseudo masks, confidence maps and per-window statistics."""

    def __init__(self, cfg: dict):
        tcfg = dict(cfg or {})
        self.stateful = bool(tcfg.get('stateful_cache', False))
        self.video_cache_cap = int(tcfg.get('video_cache_cap', 32))
        self.mask_thr = float(tcfg.get('mask_thr', 0.5))
        # merge explicit cfg overrides on top of D8b_ab defaults
        kw = dict(D8B_AB_DEFAULTS)
        for k in kw:
            if k in tcfg:
                kw[k] = tcfg[k]
        assert kw['cache_write_policy'] == 'fixed_interval', \
            'Stage 1 teacher must use fixed-interval writing (no Reliable-Write)'
        # --- RW-soft ablation switch (default: plain fixed_interval) ---
        self.write_policy = str(tcfg.get('write_policy', 'fixed_interval'))
        assert self.write_policy in ('fixed_interval', 'rw_soft'), \
            f'unknown write_policy: {self.write_policy}'
        self.rw_jump_thr = float(tcfg.get('rw_jump_thr', 0.15))
        self.rw_min_area = float(tcfg.get('rw_min_area', 300.0))
        # --- component-level output cleanup (default OFF: existing configs
        # keep bit-identical behaviour; enable via cfmc_component_cleanup) ---
        self.cc_enable = bool(tcfg.get('cfmc_component_cleanup', False))
        self.cc_thr = float(tcfg.get('cfmc_cc_thr', 0.5))
        self.cc_min_area = float(tcfg.get('cfmc_cc_min_area', 64.0))
        self.cc_w = dict(
            w_area=float(tcfg.get('cfmc_cc_w_area', 0.45)),
            w_conf=float(tcfg.get('cfmc_cc_w_conf', 0.30)),
            w_centroid=float(tcfg.get('cfmc_cc_w_centroid', 0.20)),
            w_shape=float(tcfg.get('cfmc_cc_w_shape', 0.05)),
            w_tiny=float(tcfg.get('cfmc_cc_w_tiny', 0.10)),
        )
        # option: also feed cleaned probs into the fusion stream (CHANGES
        # memory content); default false -- v1 validates output-side only
        self.cc_clean_write = bool(tcfg.get('cfmc_cc_clean_write', False))
        self.cc_debug_video = str(tcfg.get('cfmc_cc_debug_video', '') or '')
        self.cc_stats = {'n_frames': 0, 'n_multi': 0, 'n_changed': 0,
                         'removed_px': 0.0, 'fg_px': 0.0}
        self.cc_debug_records = []     # drained by the trainer
        n_bins = int(kw.get('area_bins', 4))
        self.rw_stats = {'eligible': 0, 'written': 0, 'rej_entropy': 0,
                         'rej_jump': 0, 'rej_area': 0,
                         'occ_sum': [0] * n_bins, 'occ_n': 0}
        self.cache_cfg = kw
        # entropy gate thresholds (dataset percentiles over target windows);
        # calibrate() fills them, otherwise fall back to cfg / sweep defaults
        self.entropy_low = float(tcfg.get('entropy_low', 0.0))
        self.entropy_high = float(tcfg.get('entropy_high', 1.0))
        self.calibrated = False
        self._caches = OrderedDict()   # video_id -> ShapeLogitsCache (LRU)
        self.stats = dict(n_windows=0, n_frames=0, n_fused=0,
                          beta_sum=0.0, ent_sum=0.0, fg_px=0, px=0)

    # ------------------------------------------------------------- calibration
    @torch.no_grad()
    def calibrate(self, entropies) -> tuple:
        """Set entropy gate thresholds from a pool of target-frame entropies
        (dataset percentile mode, matching shape_cache_sweep.py p50/p95)."""
        ent = np.asarray(entropies, dtype=np.float64)
        self.entropy_low = float(np.percentile(ent, 50))
        self.entropy_high = float(np.percentile(ent, 95))
        self.calibrated = True
        return self.entropy_low, self.entropy_high

    # ------------------------------------------------------------- cache mgmt
    def _cache_kwargs(self) -> dict:
        kw = dict(self.cache_cfg)
        kw['entropy_low'] = self.entropy_low
        kw['entropy_high'] = self.entropy_high
        return kw

    def _cc_kwargs(self) -> dict:
        """Keyword args for _cleanup_frame_components from the cc_* config."""
        return dict(self.cc_w, thr=self.cc_thr, min_area=self.cc_min_area)

    def _get_cache(self, video_id: str) -> ShapeLogitsCache:
        if not self.stateful:
            return ShapeLogitsCache(**self._cache_kwargs())
        if video_id in self._caches:
            self._caches.move_to_end(video_id)
            return self._caches[video_id]
        sc = ShapeLogitsCache(**self._cache_kwargs())
        self._caches[video_id] = sc
        while len(self._caches) > self.video_cache_cap:
            self._caches.popitem(last=False)
        return sc

    def reset(self):
        """Drop all per-video caches (e.g. at epoch boundaries)."""
        self._caches.clear()

    # ---------------------------------------------------------------- stream
    @torch.no_grad()
    def process_window(self, video_id: str, probs_thw: np.ndarray,
                       frame_offset: int = 0, total_frames: int = None) -> dict:
        """Stabilize one window of foreground probabilities.

        probs_thw: [T, H, W] numpy fg probabilities (already detached).
        frame_offset: absolute index of probs_thw[0] inside the full video
            (keeps fixed-interval writes and temporal distances continuous).
        total_frames: full-video frame count for temporal normalization.
        Returns dict with p_stab [T,H,W], pseudo_mask, conf_map, frame stats.
        """
        assert self.calibrated, 'call calibrate() before process_window()'
        assert probs_thw.ndim == 3
        T, H, W = probs_thw.shape
        # keep the caller's dtype (float32) so numerics match the offline
        # sweep exactly -- do NOT upcast to float64
        probs = np.ascontiguousarray(probs_thw)
        n_frames = int(total_frames) if total_frames else frame_offset + T
        sc = self._get_cache(video_id)

        fused = np.empty_like(probs)
        interval = int(self.cache_cfg['interval'])
        ref_areas = [None] * T   # per-frame memory/stream reference areas
        for i in range(T):
            t_abs = frame_offset + i
            ref_d = getattr(sc, 'prev_desc', None)
            if ref_d is not None:
                ref_areas[i] = float(ref_d.get('area', 0.0)) or None
            pin = probs[i]
            if self.cc_enable and self.cc_clean_write:
                # optional (default false): clean the probs BEFORE they enter
                # the fusion stream -- this DOES change memory content
                pin, _, _ = _cleanup_frame_components(
                    probs[i], np.clip(2.0 * np.abs(probs[i] - 0.5), 0.0, 1.0),
                    **self._cc_kwargs())
            if self.write_policy == 'rw_soft' and (t_abs + 1) % interval == 0:
                # write-eligible frame: apply the RW-soft reliability gate
                self.rw_stats['eligible'] += 1
                area = float((probs[i] > 0.5).sum())
                ent = frame_entropy(probs[i])
                pa = sc.prev_area_raw   # raw area of the previous frame
                jump = (abs(area - pa) / (pa + 1e-6)) if pa else 0.0
                deny = None
                if ent > self.entropy_high:
                    deny = 'entropy'
                elif jump > self.rw_jump_thr:
                    deny = 'jump'
                elif area < self.rw_min_area:
                    deny = 'area'
                if deny is not None:
                    self.rw_stats[f'rej_{deny}'] += 1
                    # deny the write WITHOUT touching shape_cache.py: raise
                    # interval so the fixed-interval check fails for this
                    # frame only; all internal cache state still updates
                    saved = sc.interval
                    sc.interval = 10 ** 9
                    try:
                        fused[i], _ = sc.process_frame(t_abs, probs[i], n_frames)
                    finally:
                        sc.interval = saved
                    continue
                self.rw_stats['written'] += 1
            fused[i], _ = sc.process_frame(t_abs, probs[i], n_frames)

        if self.write_policy == 'rw_soft':
            # per-bin occupancy snapshot at window end
            occ = [0] * len(self.rw_stats['occ_sum'])
            for e in sc.entries:
                if 'bin' in e:
                    occ[e['bin']] += 1
            for b in range(len(occ)):
                self.rw_stats['occ_sum'][b] += occ[b]
            self.rw_stats['occ_n'] += 1

        # ---- training-time component cleanup (off by default) ------------
        # suppress isolated false-positive blobs in the OUTPUT pseudo labels
        # only; memory content written above is untouched unless cc_clean_write
        fused_raw, conf_raw, cc_debug = None, None, []
        if self.cc_enable:
            fused_raw = fused.copy()  # real copy: the loop below edits in place
            conf_raw = np.clip(2.0 * np.abs(fused - 0.5), 0.0, 1.0)
            cc_kwargs = self._cc_kwargs()
            infos = []
            for i in range(T):
                fused[i], _, inf = _cleanup_frame_components(
                    fused[i], conf_raw[i], ref_area=ref_areas[i], **cc_kwargs)
                infos.append(inf)
                self.cc_stats['n_frames'] += 1
                if inf['num_components'] > 1:
                    self.cc_stats['n_multi'] += 1
                if inf['changed']:
                    self.cc_stats['n_changed'] += 1
                self.cc_stats['removed_px'] += float(inf['removed_area'])
                self.cc_stats['fg_px'] += float((fused[i] > self.mask_thr).sum())
                if self.cc_debug_video and self.cc_debug_video in str(video_id):
                    rec = dict(frame=int(frame_offset + i), **inf)
                    cc_debug.append(rec)
                    self.cc_debug_records.append(rec)

        # confidence = margin to the 0.5 decision boundary, in [0, 1]
        conf = np.clip(2.0 * np.abs(fused - 0.5), 0.0, 1.0)
        pseudo_mask = (fused > self.mask_thr).astype(np.float32)
        fg = pseudo_mask.sum(axis=(1, 2))
        ent = np.array([frame_entropy(fused[i]) for i in range(T)])
        n_fused = sum(1 for r in sc.log_rows
                      if frame_offset <= r['frame_idx'] < frame_offset + T)
        betas = [float(r['beta_t']) for r in sc.log_rows
                 if frame_offset <= r['frame_idx'] < frame_offset + T]

        self.stats['n_windows'] += 1
        self.stats['n_frames'] += T
        self.stats['n_fused'] += n_fused
        self.stats['beta_sum'] += float(np.sum(betas)) if betas else 0.0
        self.stats['ent_sum'] += float(ent.sum())
        self.stats['fg_px'] += float(pseudo_mask.sum())
        self.stats['px'] += float(pseudo_mask.size)

        out = {
            'p_stab': fused.astype(np.float32),
            'pseudo_mask': pseudo_mask,
            'conf_map': conf.astype(np.float32),
            'frame_fg_px': fg,
            'frame_entropy': ent,
            'frame_mean_conf': conf.mean(axis=(1, 2)),
            'n_fused_frames': n_fused,
            'mean_beta': float(np.mean(betas)) if betas else 0.0,
        }
        if self.cc_enable:
            # pre-cleanup copies + post-cleanup binary mask + debug records;
            # downstream losses keep using p_stab/conf_map -- keys unchanged
            out['p_stab_raw'] = fused_raw.astype(np.float32)
            out['conf_map_raw'] = conf_raw.astype(np.float32)
            out['pseudo_mask_clean'] = pseudo_mask
            out['cc_debug'] = cc_debug
        return out

    # ------------------------------------------------------------- batch API
    @torch.no_grad()
    def process_batch(self, video_ids, probs_bthw: np.ndarray,
                      frame_offsets=None, total_frames=None) -> dict:
        """Process a batch of windows: probs_bthw [B, T, H, W] numpy,
        video_ids list[str] length B. Returns stacked [B, T, H, W] outputs."""
        B, T, H, W = probs_bthw.shape
        outs = [self.process_window(str(v), probs_bthw[b],
                                    frame_offsets[b] if frame_offsets is not None else 0,
                                    total_frames[b] if total_frames is not None else None)
                for b, v in enumerate(video_ids)]
        stack = lambda k: np.stack([o[k] for o in outs], axis=0)
        out = {
            'p_stab': stack('p_stab'),
            'pseudo_mask': stack('pseudo_mask'),
            'conf_map': stack('conf_map'),
            'frame_fg_px': stack('frame_fg_px'),
            'frame_entropy': stack('frame_entropy'),
            'frame_mean_conf': stack('frame_mean_conf'),
            'n_fused_frames': [o['n_fused_frames'] for o in outs],
            'mean_beta': [o['mean_beta'] for o in outs],
        }
        if self.cc_enable:
            out['p_stab_raw'] = stack('p_stab_raw')
            out['conf_map_raw'] = stack('conf_map_raw')
            out['pseudo_mask_clean'] = stack('pseudo_mask_clean')
            out['cc_debug'] = [d for o in outs for d in o['cc_debug']]
        return out

    def mean_stats(self) -> dict:
        s = self.stats
        n = max(s['n_frames'], 1)
        return {
            'teacher_fused_ratio': s['n_fused'] / n,
            'teacher_mean_beta': s['beta_sum'] / max(s['n_fused'], 1),
            'teacher_mean_entropy': s['ent_sum'] / n,
            'teacher_fg_ratio': s['fg_px'] / max(s['px'], 1),
        }

    def rw_summary(self) -> dict:
        """RW-soft write statistics: acceptance ratio and mean per-bin
        occupancy (None stats when write_policy is fixed_interval)."""
        if self.write_policy != 'rw_soft':
            return {}
        s = self.rw_stats
        el = max(s['eligible'], 1)
        occ = [o / max(s['occ_n'], 1) for o in s['occ_sum']]
        out = {
            'rw_eligible': s['eligible'],
            'rw_written': s['written'],
            'rw_acceptance': s['written'] / el,
            'rw_rej_entropy': s['rej_entropy'],
            'rw_rej_jump': s['rej_jump'],
            'rw_rej_area': s['rej_area'],
            'rw_mean_occupancy': float(sum(occ)),
        }
        for b, o in enumerate(occ):
            out[f'rw_bin{b}_occupancy'] = o
        return out

    def cc_summary(self) -> dict:
        """Component-cleanup statistics (empty when the cleanup is off)."""
        if not self.cc_enable:
            return {}
        s = self.cc_stats
        n = max(s['n_frames'], 1)
        return {
            'cc_frames': s['n_frames'],
            'cc_multi_ratio': s['n_multi'] / n,
            'cc_changed_ratio': s['n_changed'] / n,
            'cc_removed_ratio': s['removed_px'] / max(s['fg_px'], 1.0),
        }
