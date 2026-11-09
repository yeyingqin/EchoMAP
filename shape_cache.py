"""Shape-aware Logits Memory Cache (inference-time / post-training).

Fuses the current frame's logits with cached snapshots of past logits that
are retrieved by SHAPE similarity (pooled probability + area + centroid +
temporal distance) instead of learned key routing. Operates purely on the
per-frame probability maps produced by any segmenter -- no model weights,
no GDR state, no retraining.

Fusion (per frame t with at least one cache hit):
    logits_final = (1 - beta_t) * logits_t + beta_t * sum_i gamma_i * logits_i
    gamma = softmax(-score / tau) over the top-k nearest cache entries
    score_i = lambda_p * d_prob + lambda_a * d_area + lambda_d * d_temporal
              [+ lambda_centroid * d_centroid]        (mode-dependent)

beta_t pipeline (Part A):
    1. gate:    none | entropy | area_jump | entropy_area (combine max/mean/product)
                beta_t = beta_max * g
    2. retrieval confidence (optional): beta_t *= conf(gamma_max)
    3. distance decay (optional):       beta_t *= exp(-d_time_mean / decay_scale)
    4. safety clamp:                    beta_t = min(beta_t, beta_max); 0 when
                                        the cache is still empty.
"""
import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-4
LN2 = float(np.log(2.0))

RETRIEVAL_MODES = ('recent_only', 'prob_only', 'area_only', 'prob_area',
                   'prob_area_temporal', 'prob_area_temporal_centroid')
GATES = ('none', 'entropy', 'area_jump', 'entropy_area')
WRITE_POLICIES = ('fixed_interval', 'reliable_novel')
CACHE_ORGS = ('fifo', 'area_bin')
BIN_REPLACE_POLICIES = ('fifo', 'max_entropy')


def to_logit(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _pool16(x: np.ndarray, low_res: int = 16) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(x))[None, None].float()
    return F.adaptive_avg_pool2d(t, (low_res, low_res))[0, 0].numpy()


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def temporal_metrics(probs: np.ndarray, outlier_thr: float = 0.1) -> dict:
    """Temporal consistency of a [T, H, W] probability/mask sequence."""
    areas = (probs > 0.5).sum(axis=(1, 2)).astype(np.float64)
    mean_area = max(areas.mean(), 1.0)
    jumps = np.abs(np.diff(areas)) / np.maximum(areas[:-1], 1.0)
    out = {'area_jitter_rel': float(np.abs(np.diff(areas)).mean() / mean_area),
           'n_outliers': int((jumps > outlier_thr).sum())}
    if len(areas) >= 3:
        out['curve_smoothness_rel'] = float(
            np.abs(np.diff(areas, n=2)).mean() / mean_area)
    else:
        out['curve_smoothness_rel'] = 0.0
    return out


def worst_frame_stats(hd95_list) -> dict:
    """max / p95 / top-10%-mean of a per-video HD95 list."""
    a = np.asarray(hd95_list, dtype=np.float64)
    if a.size == 0:
        return {'hd_max': 0.0, 'hd_p95': 0.0, 'hd_top10': 0.0}
    k = max(1, int(np.ceil(0.1 * a.size)))
    return {'hd_max': float(a.max()),
            'hd_p95': float(np.percentile(a, 95)),
            'hd_top10': float(np.sort(a)[-k:].mean())}


class ShapeLogitsCache:
    def __init__(self,
                 interval: int = 5, topk: int = 2, tau: float = 0.25,
                 gate: str = 'none', beta_max: float = 0.2,
                 combine: str = 'max',
                 jump_low: float = 0.03, jump_high: float = 0.15,
                 entropy_low: float = 0.2, entropy_high: float = 0.6,
                 use_retrieval_conf_gate: bool = False,
                 gamma_low: float = 0.5, gamma_high: float = 0.8,
                 use_distance_decay: bool = False, decay_scale: float = 10.0,
                 use_fused_prev_area: bool = True,
                 retrieval_mode: str = 'prob_area_temporal',
                 lambda_p: float = 1.0, lambda_a: float = 1.0,
                 lambda_d: float = 0.5, lambda_centroid: float = 1.0,
                 max_slots: int = 5, low_res: int = 16,
                 # --- Reliable-Write (adaptive snapshot writing) ---
                 cache_write_policy: str = 'fixed_interval',
                 write_min_interval: int = 5,
                 write_entropy_thr: float = 0.5,
                 write_entropy_percentile: int = 70,
                 write_entropy_mode: str = 'dataset',  # dataset | running
                 write_running_min_hist: int = 10,
                 write_area_jump_thr: float = 0.15,
                 write_novelty_thr: float = 0.05,
                 # --- Area-Bin (phase-organized cache) ---
                 cache_org: str = 'fifo',
                 area_bins: int = 4,
                 bin_slots: int = 3,
                 candidate_bin_radius: int = 1,
                 bin_replace_policy: str = 'fifo'):
        assert gate in GATES and retrieval_mode in RETRIEVAL_MODES
        assert topk >= 1 and interval >= 1 and max_slots >= 1
        assert cache_write_policy in WRITE_POLICIES
        assert cache_org in CACHE_ORGS
        assert bin_replace_policy in BIN_REPLACE_POLICIES
        assert write_entropy_mode in ('dataset', 'running')
        self.interval, self.topk, self.tau = interval, topk, tau
        self.gate, self.beta_max, self.combine = gate, beta_max, combine
        self.jump_low, self.jump_high = jump_low, jump_high
        self.entropy_low, self.entropy_high = entropy_low, entropy_high
        self.use_retrieval_conf_gate = use_retrieval_conf_gate
        self.gamma_low, self.gamma_high = gamma_low, gamma_high
        self.use_distance_decay, self.decay_scale = use_distance_decay, decay_scale
        self.use_fused_prev_area = use_fused_prev_area
        self.retrieval_mode = retrieval_mode
        self.lambdas = dict(p=lambda_p, a=lambda_a, d=lambda_d, c=lambda_centroid)
        self.max_slots = max_slots
        self.low_res = low_res
        self.cache_write_policy = cache_write_policy
        self.write_min_interval = write_min_interval
        self.write_entropy_thr = write_entropy_thr
        self.write_entropy_percentile = write_entropy_percentile
        self.write_entropy_mode = write_entropy_mode
        self.write_running_min_hist = write_running_min_hist
        self.write_area_jump_thr = write_area_jump_thr
        self.write_novelty_thr = write_novelty_thr
        self.cache_org = cache_org
        self.area_bins = area_bins
        self.bin_slots = bin_slots
        self.candidate_bin_radius = candidate_bin_radius
        self.bin_replace_policy = bin_replace_policy
        self.reset()

    def reset(self, video_id: str = ''):
        self.entries = []          # FIFO of raw-output cache snapshots
        self.prev_desc = None      # descriptor of the previous fused output
        self.prev_area_raw = None  # raw-model area of the previous frame
        self.video_id = video_id
        self.log_rows = []
        # --- write-policy / cache-org state ---
        self._last_write = -10 ** 9
        self._entropy_hist = []    # running entropy history (running percentile)
        self._area_min = None      # running area min/max (area_norm binning)
        self._area_max = None
        self.write_stats = {
            'writes': 0, 'rej_entropy': 0, 'rej_jump': 0,
            'rej_novelty': 0, 'rej_interval': 0,
            'occupancy_sum': 0, 'occupancy_n': 0,
            'bin_writes': [0] * self.area_bins,
        }

    # ------------------------------------------------------------- descriptor
    def _descriptor(self, prob: np.ndarray) -> dict:
        p = np.clip(prob, EPS, 1.0 - EPS)
        mask = prob > 0.5
        if mask.any():
            ys, xs = np.nonzero(mask)
            centroid = (float(ys.mean() / prob.shape[0]),
                        float(xs.mean() / prob.shape[1]))
        else:
            centroid = (0.5, 0.5)
        entropy = float((- (p * np.log(p) + (1.0 - p) * np.log(1.0 - p))).mean())
        return {
            'prob_low': _pool16(prob, self.low_res),
            'area': float(mask.sum()),
            'centroid': centroid,
            'entropy': entropy,
        }

    # ------------------------------------------------------------ write policy
    def _bin_of(self, area: float) -> int:
        """Phase bin from running-min/max normalized area (0=small/ES-like,
        bins-1=large/ED-like)."""
        if self._area_max is None or self._area_max <= self._area_min:
            return self.area_bins // 2
        norm = (area - self._area_min) / (self._area_max - self._area_min)
        norm = _clip01(norm)
        return min(int(norm * self.area_bins), self.area_bins - 1)

    def _entropy_write_ok(self, entropy: float) -> bool:
        if self.write_entropy_mode == 'running' and \
                len(self._entropy_hist) >= self.write_running_min_hist:
            thr = float(np.percentile(self._entropy_hist,
                                      self.write_entropy_percentile))
        else:
            # dataset mode, or running mode with too-short history: fall back
            # to the (dataset percentile) absolute threshold
            thr = self.write_entropy_thr
        return entropy <= thr

    def _novelty_ok(self, t: int, desc: dict, n_frames: int) -> bool:
        """True if the candidate snapshot is sufficiently different from
        every cached entry (min shape-mode score > novelty threshold)."""
        if not self.entries:
            return True
        scores = []
        for e in self.entries:
            dp, da, dc, dt = self._distances(desc, e, t, n_frames)
            scores.append(self._mode_score(dp, da, dc, dt))
        return min(scores) > self.write_novelty_thr

    def _maybe_store(self, t: int, prob: np.ndarray, desc: dict,
                     jump: float, n_frames: int):
        """Cache the RAW model output (never fused) per the write policy."""
        # update running stats every frame regardless of write decision
        self._entropy_hist.append(desc['entropy'])
        a = desc['area']
        self._area_min = a if self._area_min is None else min(self._area_min, a)
        self._area_max = a if self._area_max is None else max(self._area_max, a)

        if self.cache_write_policy == 'fixed_interval':
            ok = (t + 1) % self.interval == 0
        else:  # reliable_novel
            ok = True
            if self.entries and (t - self._last_write) < self.write_min_interval:
                self.write_stats['rej_interval'] += 1
                ok = False
            if ok and not self._entropy_write_ok(desc['entropy']):
                self.write_stats['rej_entropy'] += 1
                ok = False
            # area-jump check needs a previous frame; skip on the very first
            if ok and self.prev_desc is not None and \
                    jump > self.write_area_jump_thr:
                self.write_stats['rej_jump'] += 1
                ok = False
            # empty cache: allow the first reliable snapshot (skip novelty)
            if ok and not self._novelty_ok(t, desc, n_frames):
                self.write_stats['rej_novelty'] += 1
                ok = False
        if not ok:
            return

        entry = {'frame_idx': t, 'logits_full': to_logit(prob), **desc}
        if self.cache_org == 'area_bin':
            b = self._bin_of(a)
            entry['bin'] = b
            bucket = [e for e in self.entries if e.get('bin') == b]
            if len(bucket) >= self.bin_slots:
                if self.bin_replace_policy == 'max_entropy':
                    worst = max(bucket, key=lambda e: e['entropy'])
                    self.entries.remove(worst)
                else:  # fifo: evict oldest in this bin
                    self.entries.remove(bucket[0])
            self.write_stats['bin_writes'][b] += 1
        else:
            if len(self.entries) > self.max_slots:
                self.entries.pop(0)
        self.entries.append(entry)
        self._last_write = t
        self.write_stats['writes'] += 1

    # --------------------------------------------------------------- retrieval
    def _distances(self, prev, e, t, n_frames):
        d_prob = float(((prev['prob_low'] - e['prob_low']) ** 2).mean())
        d_area = abs(prev['area'] - e['area']) / e['logits_full'].size
        dy, dx = prev['centroid'][0] - e['centroid'][0], prev['centroid'][1] - e['centroid'][1]
        d_cent = float(np.hypot(dy, dx))
        d_time = (t - e['frame_idx']) / max(n_frames, 1)
        return d_prob, d_area, d_cent, d_time

    def _mode_score(self, dp, da, dc, dt):
        lam = self.lambdas
        if self.retrieval_mode == 'recent_only':
            return dt
        if self.retrieval_mode == 'prob_only':
            return lam['p'] * dp
        if self.retrieval_mode == 'area_only':
            return lam['a'] * da
        if self.retrieval_mode == 'prob_area':
            return lam['p'] * dp + lam['a'] * da
        if self.retrieval_mode == 'prob_area_temporal':
            return lam['p'] * dp + lam['a'] * da + lam['d'] * dt
        return lam['p'] * dp + lam['a'] * da + lam['d'] * dt + lam['c'] * dc

    # ---------------------------------------------------------------- process
    def process_frame(self, t: int, prob: np.ndarray, n_frames: int):
        """Fuse one frame. Returns (fused_prob, log_row or None)."""
        desc = self._descriptor(prob)
        prev_area_for_jump = (self.prev_desc['area'] if self.use_fused_prev_area
                              and self.prev_desc is not None else self.prev_area_raw)
        jump = (abs(desc['area'] - prev_area_for_jump) / (prev_area_for_jump + 1e-6)
                if prev_area_for_jump else 0.0)
        g_entropy = _clip01((desc['entropy'] - self.entropy_low) /
                            max(self.entropy_high - self.entropy_low, 1e-9))
        g_jump = _clip01((jump - self.jump_low) /
                         max(self.jump_high - self.jump_low, 1e-9))
        if self.gate == 'none':
            g = 1.0
        elif self.gate == 'entropy':
            g = g_entropy
        elif self.gate == 'area_jump':
            g = g_jump
        else:  # entropy_area
            if self.combine == 'mean':
                g = (g_entropy + g_jump) / 2
            elif self.combine == 'product':
                g = g_entropy * g_jump
            else:
                g = max(g_entropy, g_jump)
        beta_t = self.beta_max * g

        fused, row = prob, None
        if self.prev_desc is not None and self.entries and beta_t > 0:
            hw = self.entries[-1]['logits_full'].size
            if self.cache_org == 'area_bin':
                qb = self._bin_of(desc['area'])
                cand = [e for e in self.entries
                        if abs(e.get('bin', qb) - qb) <= self.candidate_bin_radius]
                if not cand:  # fallback: all bins
                    cand = self.entries
            else:
                cand = self.entries
            scored = []
            for e in cand:
                dp, da, dc, dt = self._distances(self.prev_desc, e, t, n_frames)
                scored.append((self._mode_score(dp, da, dc, dt), e,
                               (dp, da, dc, dt)))
            scored.sort(key=lambda x: x[0])
            picked = scored[:self.topk]
            scores = np.array([s for s, _, _ in picked], dtype=np.float64)
            gamma = np.exp(-scores / self.tau)
            gamma = gamma / gamma.sum()
            gamma_max = float(gamma.max())
            conf = _clip01((gamma_max - self.gamma_low) /
                           max(self.gamma_high - self.gamma_low, 1e-9)) \
                if self.use_retrieval_conf_gate else 1.0
            dts = [c[3] for c in (p[2] for p in picked)]
            if self.use_distance_decay:
                beta_t *= float(np.exp(-np.mean(dts) / self.decay_scale))
            beta_t = min(beta_t, self.beta_max)

            logit_t = to_logit(prob)
            mix = np.zeros_like(logit_t)
            for g_, (_, e, _) in zip(gamma, picked):
                mix += g_ * e['logits_full']
            fused = 1.0 / (1.0 + np.exp(-((1.0 - beta_t) * logit_t + beta_t * mix)))
            row = {
                'video_id': self.video_id, 'frame_idx': t,
                'gate_type': self.gate,
                'beta_t': f'{beta_t:.6f}',
                'selected_cache_frames': ';'.join(str(e['frame_idx']) for _, e, _ in picked),
                'gamma': ';'.join(f'{g_:.6f}' for g_ in gamma),
                'gamma_max': f'{gamma_max:.6f}',
                'retrieval_confidence': f'{conf:.6f}',
                'd_prob': ';'.join(f'{p_[2][0]:.6f}' for p_ in picked),
                'd_area': ';'.join(f'{p_[2][1]:.6f}' for p_ in picked),
                'd_centroid': ';'.join(f'{p_[2][2]:.6f}' for p_ in picked),
                'd_temporal': ';'.join(f'{p_[2][3]:.6f}' for p_ in picked),
                'score_total': ';'.join(f'{s:.6f}' for s, _, _ in picked),
                'selected_distance_mean': f'{np.mean(dts):.6f}',
                'selected_distance_min': f'{np.min(dts):.6f}',
                'score': ';'.join(f'{s:.6f}' for s, _, _ in picked),
                'area_current': f"{desc['area']:.0f}",
                'area_selected': ';'.join(f"{e['area']:.0f}" for _, e, _ in picked),
                'entropy': f"{desc['entropy']:.6f}",
                'area_jump': f'{jump:.6f}',
                'cache_size': str(len(self.entries)),
                'query_bin': str(self._bin_of(desc['area'])
                                 if self.cache_org == 'area_bin' else -1),
                'g_entropy': f'{g_entropy:.6f}',
                'g_jump': f'{g_jump:.6f}',
                'uncertainty': f"{desc['entropy']:.6f}",
            }
            self.log_rows.append(row)

        # cache stores the RAW model output; prev tracks fused (or raw) output
        self._maybe_store(t, prob, desc, jump, n_frames)
        self.write_stats['occupancy_sum'] += len(self.entries)
        self.write_stats['occupancy_n'] += 1
        self.prev_desc = self._descriptor(fused)
        self.prev_area_raw = desc['area']
        return fused, row
