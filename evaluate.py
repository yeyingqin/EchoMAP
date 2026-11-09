"""Video-level evaluation for Backbone: per-frame Dice + HD95, averaged within each
video, then mean±std across videos. Optional Shape-aware Logits Memory Cache
(post-processing, see shape_cache.py) evaluated in the same pass.

  python evaluate.py --weights outputs/.../xxx.pth \
      --data ./data/camus_png256x256_10f_20250709 --split test_data \
      --out outputs/eval_camus_test [--shape_cache_enable ...]

HD95 convention (nnU-Net): 95th percentile of the surface distances between the
predicted and GT boundaries, measured in both directions and pooled. Distances
are in pixels of the 256x256 resized images.
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer import infer_sequence, load_model, load_sequence_from_dataset  # noqa: E402
from shape_cache import ShapeLogitsCache, temporal_metrics, worst_frame_stats  # noqa: E402

IMG_DIAG = float(np.hypot(256, 256))            # max possible surface distance


def boundary(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
    return (mask > 0) & (eroded == 0)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th percentile of pooled pred->gt and gt->pred boundary distances."""
    pb, gb = boundary(pred), boundary(gt)
    if not pb.any() or not gb.any():
        return IMG_DIAG                          # degenerate: treat as max distance
    d_to_gt = distance_transform_edt(~gb)[pb]    # pred boundary -> nearest gt boundary
    d_to_pred = distance_transform_edt(~pb)[gb]  # gt boundary -> nearest pred boundary
    return float(np.percentile(np.concatenate([d_to_gt, d_to_pred]), 95))


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    denom = pred.sum() + gt.sum()
    return float('nan') if denom == 0 else 2.0 * (pred & gt).sum() / denom


def assd(pred: np.ndarray, gt: np.ndarray) -> float:
    """Average symmetric surface distance between pred and gt boundaries."""
    pb, gb = boundary(pred), boundary(gt)
    if not pb.any() or not gb.any():
        return IMG_DIAG
    d_to_gt = distance_transform_edt(~gb)[pb]
    d_to_pred = distance_transform_edt(~pb)[gb]
    return float((d_to_gt.mean() + d_to_pred.mean()) / 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--split', default='test_data',
                    help='key in the split json (train_data/val_data/test_data)')
    ap.add_argument('--limit', type=int, default=0,
                    help='only the first N sequences (0 = all)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--carry-state', action='store_true',
                    help='continue GDR memory across 10-frame windows')
    ap.add_argument('--use_memory_cache', action='store_true',
                    help='enable Fixed-window Memory Cache + Softmax Memory Soup')
    ap.add_argument('--cache_interval', type=int, default=10)
    ap.add_argument('--detach_memory_cache', action='store_true', default=True)
    ap.add_argument('--memory_cache_context', default='gap_key',
                    choices=['gap_key', 'spatial_key_4x4'])
    ap.add_argument('--dump_memory_cache_routing', action='store_true',
                    help='log per-frame routing gamma stats to routing_log.csv')
    # ---- shape-aware logits memory cache (post-processing, no retraining) ----
    ap.add_argument('--shape_cache_enable', action='store_true')
    ap.add_argument('--shape_cache_interval', type=int, default=5)
    ap.add_argument('--shape_cache_topk', type=int, default=2)
    ap.add_argument('--shape_cache_tau', type=float, default=0.25)
    ap.add_argument('--shape_cache_gate', default='none',
                    choices=['none', 'entropy', 'area_jump', 'entropy_area'])
    ap.add_argument('--shape_cache_beta_max', type=float, default=0.2)
    ap.add_argument('--shape_cache_combine', default='max',
                    choices=['max', 'mean', 'product'])
    ap.add_argument('--shape_cache_jump_low', type=float, default=0.03)
    ap.add_argument('--shape_cache_jump_high', type=float, default=0.15)
    ap.add_argument('--shape_cache_entropy_low', type=float, default=0.2)
    ap.add_argument('--shape_cache_entropy_high', type=float, default=0.6)
    ap.add_argument('--shape_cache_use_retrieval_conf_gate', action='store_true')
    ap.add_argument('--shape_cache_gamma_low', type=float, default=0.5)
    ap.add_argument('--shape_cache_gamma_high', type=float, default=0.8)
    ap.add_argument('--shape_cache_use_distance_decay', action='store_true')
    ap.add_argument('--shape_cache_decay_scale', type=float, default=10.0)
    ap.add_argument('--shape_cache_use_fused_prev_area', action='store_true',
                    default=True)
    ap.add_argument('--shape_cache_retrieval_mode', default='prob_area_temporal',
                    choices=['recent_only', 'prob_only', 'area_only', 'prob_area',
                             'prob_area_temporal', 'prob_area_temporal_centroid'])
    ap.add_argument('--shape_cache_max_slots', type=int, default=5)
    ap.add_argument('--shape_cache_lambda_p', type=float, default=1.0)
    ap.add_argument('--shape_cache_lambda_a', type=float, default=1.0)
    ap.add_argument('--shape_cache_lambda_d', type=float, default=0.5)
    ap.add_argument('--shape_cache_lambda_centroid', type=float, default=1.0)
    ap.add_argument('--temporal_outlier_jump_threshold', type=float, default=0.1)
    args = ap.parse_args()

    print(f"use_memory_cache={args.use_memory_cache}, "
          f"cache_interval={args.cache_interval}, "
          f"memory_cache_context={args.memory_cache_context}")
    if args.use_memory_cache and not args.carry_state:
        print('[note] memory cache needs --carry-state to chain windows; '
              'enabling it automatically')
        args.carry_state = True

    with open(os.path.join(args.data, 'camus_public_datasplit_20250706.json')) as f:
        sids = json.load(f)[args.split]
    if args.limit:
        sids = sids[:args.limit]

    model = load_model(args.weights,
                       use_memory_cache=args.use_memory_cache,
                       cache_interval=args.cache_interval,
                       detach_memory_cache=args.detach_memory_cache,
                       memory_cache_context=args.memory_cache_context)
    model.dump_routing = args.dump_memory_cache_routing
    torch.set_grad_enabled(False)
    os.makedirs(args.out, exist_ok=True)

    sc = (ShapeLogitsCache(
        interval=args.shape_cache_interval, topk=args.shape_cache_topk,
        tau=args.shape_cache_tau, gate=args.shape_cache_gate,
        beta_max=args.shape_cache_beta_max, combine=args.shape_cache_combine,
        jump_low=args.shape_cache_jump_low, jump_high=args.shape_cache_jump_high,
        entropy_low=args.shape_cache_entropy_low,
        entropy_high=args.shape_cache_entropy_high,
        use_retrieval_conf_gate=args.shape_cache_use_retrieval_conf_gate,
        gamma_low=args.shape_cache_gamma_low, gamma_high=args.shape_cache_gamma_high,
        use_distance_decay=args.shape_cache_use_distance_decay,
        decay_scale=args.shape_cache_decay_scale,
        use_fused_prev_area=args.shape_cache_use_fused_prev_area,
        retrieval_mode=args.shape_cache_retrieval_mode,
        lambda_p=args.shape_cache_lambda_p, lambda_a=args.shape_cache_lambda_a,
        lambda_d=args.shape_cache_lambda_d,
        lambda_centroid=args.shape_cache_lambda_centroid,
        max_slots=args.shape_cache_max_slots)
        if args.shape_cache_enable else None)
    if sc:
        print(f'shape cache: interval={sc.interval} topk={sc.topk} tau={sc.tau} '
              f'gate={sc.gate} beta_max={sc.beta_max} mode={sc.retrieval_mode} '
              f'slots={sc.max_slots}')

    rows, retrieval_rows, temporal_rows = [], [], []
    worst_rows, delta_rows, frame_rows = [], [], []
    routing_rows = []

    for k, sid in enumerate(sids):
        frames, gts = load_sequence_from_dataset(args.data, sid)
        recs = [] if args.dump_memory_cache_routing else None
        probs = infer_sequence(model, frames, carry_state=args.carry_state,
                               routing_records=recs)
        for r in recs or []:
            routing_rows.append([os.path.basename(os.path.normpath(args.data)),
                                 sid, r['frame_idx'], r['num_cache'],
                                 r['cache_frame_indices'], r['gamma_values'],
                                 r['gamma_max'], r['gamma_argmax'],
                                 r['gamma_entropy'], r['gamma_norm_entropy'],
                                 r['score_range'], r['score_std'],
                                 r['selected_cache_frame'],
                                 r['distance_to_selected_cache']])

        fused = probs
        if sc is not None:
            sc.reset(sid)
            fused = np.stack([sc.process_frame(t, probs[t], len(probs))[0]
                              for t in range(len(probs))])
            retrieval_rows.extend(sc.log_rows)

        def _metrics(pp):
            d_list, h_list, a_list, fid = [], [], [], []
            for t, p in enumerate(pp):
                gt = gts[t] > 0
                if gt.sum() == 0:
                    continue
                pm = p > 0.5
                d_list.append(dice(pm, gt))
                h_list.append(hd95(pm.astype(np.uint8), gt.astype(np.uint8)))
                a_list.append(assd(pm.astype(np.uint8), gt.astype(np.uint8)))
                fid.append((float(pm.sum()), float(gt.sum())))
            return d_list, h_list, a_list, fid

        d_raw, h_raw, a_raw, fid_raw = _metrics(probs)
        d_fus, h_fus, a_fus, fid_fus = (_metrics(fused) if sc is not None
                                        else (d_raw, h_raw, a_raw, fid_raw))

        tm_raw = temporal_metrics(probs, args.temporal_outlier_jump_threshold)
        tm_fus = temporal_metrics(fused, args.temporal_outlier_jump_threshold)

        vd, vh = float(np.nanmean(d_raw)), float(np.mean(h_raw))
        va = float(np.mean(a_raw))
        vd_f, vh_f = float(np.nanmean(d_fus)), float(np.mean(h_fus))
        rows.append([sid, len(d_raw), round(vd, 4), round(vh, 2),
                     round(va, 2), round(vd_f, 4), round(vh_f, 2)])
        for t, (d, h, ad) in enumerate(zip(d_raw, h_raw, a_raw)):
            frame_rows.append([sid, t, round(d, 4), round(h, 2), round(ad, 2)])

        wf_r, wf_f = worst_frame_stats(h_raw), worst_frame_stats(h_fus)
        worst_rows.append([sid, f'{wf_r["hd_max"]:.2f}', f'{wf_r["hd_p95"]:.2f}',
                           f'{wf_r["hd_top10"]:.2f}', f'{wf_f["hd_max"]:.2f}',
                           f'{wf_f["hd_p95"]:.2f}', f'{wf_f["hd_top10"]:.2f}'])

        ae_raw = float(np.mean([abs(a - g) for a, g in fid_raw])) if fid_raw else 0.0
        ae_fus = float(np.mean([abs(a - g) for a, g in fid_fus])) if fid_fus else 0.0
        delta_rows.append([sid, f'{vd_f - vd:+.4f}', f'{vh_f - vh:+.2f}',
                           len(d_raw), f'{ae_raw:.1f}', f'{ae_fus:.1f}'])

        temporal_rows.append([sid, f'{tm_raw["area_jitter_rel"]:.6f}',
                              f'{tm_fus["area_jitter_rel"]:.6f}',
                              f'{tm_raw["curve_smoothness_rel"]:.6f}',
                              f'{tm_fus["curve_smoothness_rel"]:.6f}',
                              tm_raw['n_outliers'], tm_fus['n_outliers']])

        print(f'[{k + 1}/{len(sids)}] {sid}: frames={len(d_raw)} '
              f'Dice={vd:.4f} HD95={vh:.2f}px'
              + (f' | fused Dice={vd_f:.4f} HD95={vh_f:.2f}px' if sc else ''),
              flush=True)

    def _write(name, header, body):
        with open(os.path.join(args.out, name), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(body)

    _write('video_report.csv',
           ['sequence', 'n_frames', 'dice_mean', 'hd95_mean_px', 'assd_mean_px',
            'fused_dice_mean', 'fused_hd95_mean_px'], rows)
    _write('frame_report.csv',
           ['sequence', 'frame', 'dice', 'hd95_px', 'assd_px'], frame_rows)
    if routing_rows:
        _write('routing_log.csv',
               ['dataset', 'video_id', 'frame_idx', 'num_cache',
                'cache_frame_indices', 'gamma_values', 'gamma_max',
                'gamma_argmax', 'gamma_entropy', 'gamma_norm_entropy',
                'score_range', 'score_std', 'selected_cache_frame',
                'distance_to_selected_cache'], routing_rows)
        print(f'routing log: {os.path.join(args.out, "routing_log.csv")} '
              f'({len(routing_rows)} frames)')

    vd = np.array([r[2] for r in rows])
    vh = np.array([r[3] for r in rows])
    nf = np.array([r[1] for r in rows])
    print(f'\n== {args.split}: {len(rows)} videos, {nf.sum()} annotated frames ==')
    print(f'Dice : {vd.mean():.4f} ± {vd.std(ddof=1):.4f} (video-level)   '
          f'median={np.median(vd):.4f}')
    print(f'HD95 : {vh.mean():.2f} ± {vh.std(ddof=1):.2f} px (video-level)  '
          f'median={np.median(vh):.2f}')

    va = np.array([r[4] for r in rows])
    print(f'ASSD : {va.mean():.2f} ± {va.std(ddof=1):.2f} px (video-level)  '
          f'median={np.median(va):.2f}')

    if sc is not None:
        vdf = np.array([r[4] for r in rows])
        vhf = np.array([r[5] for r in rows])
        imp = sum(1 for r in rows if r[4] > r[2] + 1e-9)
        deg1 = sum(1 for r in rows if r[2] - r[4] > 0.001)
        deg5 = sum(1 for r in rows if r[2] - r[4] > 0.005)
        deltas = np.array([r[4] - r[2] for r in rows])
        print(f'\n== shape cache fused (gate={sc.gate}, beta_max={sc.beta_max}, '
              f'mode={sc.retrieval_mode}) ==')
        print(f'Fused Dice : {vdf.mean():.4f} ± {vdf.std(ddof=1):.4f} '
              f'(raw {vd.mean():.4f} ± {vd.std(ddof=1):.4f})  '
              f'delta mean={deltas.mean():+.5f} median={np.median(deltas):+.5f}')
        print(f'Fused HD95 : {vhf.mean():.2f} ± {vhf.std(ddof=1):.2f} px '
              f'(raw {vh.mean():.2f} ± {vh.std(ddof=1):.2f})')
        print(f'per-video: improved={imp}, degraded>0.001={deg1} '
              f'({100*deg1/len(rows):.0f}%), degraded>0.005={deg5} '
              f'({100*deg5/len(rows):.0f}%)')
        aj_r = np.array([float(r[1]) for r in temporal_rows])
        aj_f = np.array([float(r[2]) for r in temporal_rows])
        cs_r = np.array([float(r[3]) for r in temporal_rows])
        cs_f = np.array([float(r[4]) for r in temporal_rows])
        ol_r = sum(int(r[5]) for r in temporal_rows)
        ol_f = sum(int(r[6]) for r in temporal_rows)
        print(f'temporal: jitter {aj_r.mean():.5f}->{aj_f.mean():.5f} '
              f'({(aj_f.mean()/max(aj_r.mean(),1e-12)-1)*100:+.1f}%), '
              f'smoothness {cs_r.mean():.5f}->{cs_f.mean():.5f} '
              f'({(cs_f.mean()/max(cs_r.mean(),1e-12)-1)*100:+.1f}%), '
              f'outliers {ol_r}->{ol_f}')
        _write('temporal_report.csv',
               ['sequence', 'area_jitter_raw', 'area_jitter_fused',
                'curve_smoothness_raw', 'curve_smoothness_fused',
                'n_outliers_raw', 'n_outliers_fused'], temporal_rows)
        _write('worst_frame_report.csv',
               ['sequence', 'raw_hd95_max', 'raw_hd95_p95', 'raw_hd95_top10',
                'fused_hd95_max', 'fused_hd95_p95', 'fused_hd95_top10'], worst_rows)
        _write('per_video_delta_report.csv',
               ['sequence', 'delta_dice', 'delta_hd95', 'n_gt_frames',
                'raw_area_err_vs_gt', 'fused_area_err_vs_gt'], delta_rows)
        _write('retrieval_log.csv',
               ['video_id', 'frame_idx', 'gate_type', 'beta_t',
                'selected_cache_frames', 'gamma', 'gamma_max',
                'retrieval_confidence', 'd_prob', 'd_area', 'd_centroid',
                'd_temporal', 'score_total', 'selected_distance_mean',
                'selected_distance_min', 'score', 'area_current',
                'area_selected', 'entropy', 'area_jump', 'g_entropy', 'g_jump',
                'uncertainty'], retrieval_rows)
        print(f'retrieval log: {os.path.join(args.out, "retrieval_log.csv")} '
              f'({len(retrieval_rows)} frames)')


if __name__ == '__main__':
    main()
