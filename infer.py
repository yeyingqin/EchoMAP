"""
Standalone Backbone inference: run a trained checkpoint over full sequences
(longer than the 10-frame training window), export per-frame LV masks as
overlay images / video, and a DICE report.

Modes:
  1. Backbone-format dataset (img/ + gt_lv/ + split json):
       python infer.py --weights outputs/.../xxx.pth \
                       --data ~/data/camus_png256x256_10f_20250709 \
                       --split test --out outputs/infer_camus [--limit 5] [--video]
  2. Single NIfTI case (image + optional label):
       python infer.py --weights ... --nifti case_image.nii.gz \
                       --label case_label.nii.gz --lv-label 1 --out outputs/infer_case [--video]

Notes:
  - The model never consumes ground-truth masks (training zeroes ff_gt), so
    inference is fully automatic: frames in, LV masks out.
  - Sequences longer than 10 frames are processed in 10-frame windows. By
    default each window is an independent cold start (matches training); pass
    --carry-state to continue the GDR memory across windows.
"""
import argparse
import csv
import json
import math
import os
import re
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.backbone import Backbone  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW = 10


# ----------------------------------------------------------------- data

def frame_key(name: str) -> int:
    m = re.search(r'(\d+)\.png$', name)
    return int(m.group(1)) if m else -1


def load_sequence_from_dataset(root: str, sample_id: str):
    img_dir = os.path.join(root, 'img', sample_id)
    gt_dir = os.path.join(root, 'gt_lv', sample_id)
    names = sorted((n for n in os.listdir(img_dir) if frame_key(n) >= 0),
                   key=frame_key)                     # skip sync temp files etc.
    frames = [cv2.imread(os.path.join(img_dir, n), cv2.IMREAD_GRAYSCALE) for n in names]
    gts = None
    if os.path.isdir(gt_dir):
        gts = [cv2.imread(os.path.join(gt_dir, n), cv2.IMREAD_GRAYSCALE) for n in names]
    pairs = [(f, (g if gts is None else g)) for f, g in zip(frames, gts or frames)
             if f is not None and (gts is None or g is not None)]
    frames = [p[0] for p in pairs]
    gts = None if gts is None else [p[1] for p in pairs]
    return np.stack(frames), (np.stack(gts) if gts else None)


def load_sequence_from_nifti(image_path: str, label_path: str | None, lv_label: int = 1):
    import SimpleITK as sitk
    vol = sitk.GetArrayFromImage(sitk.ReadImage(image_path))  # (T, H, W)
    lo, hi = np.percentile(vol, (1, 99))
    vol = np.clip(vol, lo, hi)
    vol = ((vol - lo) / max(hi - lo, 1e-8) * 255).round().astype(np.uint8)
    frames = np.stack([cv2.resize(f, (256, 256), interpolation=cv2.INTER_AREA)
                       for f in vol])
    gts = None
    if label_path:
        lab = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        gts = np.stack([(cv2.resize((l == lv_label).astype(np.uint8), (256, 256),
                                    interpolation=cv2.INTER_NEAREST))
                        for l in lab])
    return frames, gts


# ----------------------------------------------------------------- model

def load_model(weights: str, use_memory_cache: bool = False,
               cache_interval: int = 10,
               detach_memory_cache: bool = True,
               memory_cache_context: str = 'gap_key',
               kpff_enhancer: dict | None = None,
               kpff_wavelet: dict | None = None,
               kpff_joint: dict | None = None,
               kpff_context: dict | None = None) -> Backbone:
    sd = torch.load(weights, map_location='cpu')
    # a checkpoint carrying kpff_enhancer weights must NOT be silently loaded
    # into a model without the enhancer (strict=False would drop them);
    # structure (hidden_dim, rank) is inferred from the weight shapes
    if kpff_enhancer is None and any(k.startswith('kpff_enhancer.') for k in sd):
        h = sd['kpff_enhancer.g_in.weight'].shape[0]
        r = sd['kpff_enhancer.affinity.weight'].shape[0]
        kpff_enhancer = {'enabled': True, 'hidden_dim': h, 'rank': r}
        print(f'[load_model] checkpoint contains kpff_enhancer weights -> '
              f'auto-enabling (hidden_dim={h}, rank={r})')
    # same for the wavelet gate: structure (hidden_dim) inferred from shapes
    if kpff_wavelet is None and any(k.startswith('KPFF.wavelet_gate.') for k in sd):
        h = sd['KPFF.wavelet_gate.freq_gate.0.weight'].shape[0]
        kpff_wavelet = {'enabled': True, 'hidden_dim': h}
        print(f'[load_model] checkpoint contains kpff_wavelet weights -> '
              f'auto-enabling (hidden_dim={h})')
    # same for the joint gate: a checkpoint carrying KPFF.p_gate.* must not
    # be silently dropped; C2 checkpoints have no such key and stay baseline
    if kpff_joint is None and any(k.startswith('KPFF.p_gate.') for k in sd):
        kpff_joint = {'enabled': True}
        print('[load_model] checkpoint contains KPFF.p_gate weights -> '
              'auto-enabling kpff_joint')
    model = Backbone(use_memory_cache=use_memory_cache,
                  cache_interval=cache_interval,
                  detach_memory_cache=detach_memory_cache,
                  memory_cache_context=memory_cache_context,
                  kpff_enhancer=kpff_enhancer,
                  kpff_wavelet=kpff_wavelet,
                  kpff_joint=kpff_joint,
                  kpff_context=kpff_context)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # kpff_context has NO state_dict signature (global/local3 share identical
    # params) -- it can never be auto-detected; always echo what is in effect
    print(f'[load_model] KPFF context mode = {model.KPFF.context_mode}'
          + (' (explicit)' if kpff_context else ' (default global)'))
    if missing or unexpected:
        print(f'[warn] missing keys: {missing}\n[warn] unexpected keys: {unexpected}')
    if model.kpff_enhancer is not None and \
            not any(k.startswith('kpff_enhancer.') for k in sd):
        print('[warn] kpff_enhancer enabled but checkpoint has no enhancer '
              'weights -- enhancer runs at random init')
    if model.KPFF.wavelet_gate is not None and \
            not any(k.startswith('KPFF.wavelet_gate.') for k in sd):
        print('[warn] kpff_wavelet enabled but checkpoint has no wavelet gate '
              'weights -- freq_gate runs at init (last layer zero => delta 0)')
    if model.KPFF.p_gate is not None and \
            not any(k.startswith('KPFF.p_gate.') for k in sd):
        print('[warn] kpff_joint enabled but checkpoint has no p_gate '
              'weights -- p_gate runs at zero init (z_joint = 0)')
    return model.to(DEV).eval()


def softmax_key(key_BCHW: torch.Tensor) -> torch.Tensor:
    kmax = torch.max(key_BCHW, dim=1, keepdim=True).values
    return (key_BCHW - kmax).softmax(dim=1)


def run_window(model: Backbone, frames: np.ndarray, init=None, frame_offset: int = 0,
               routing_records: list | None = None):
    """Run one <=10-frame window. init=(state, sensory, obj_sum[, cached_memories,
    ctx_sum_BC, ctx_count, cache_frame_indices]) carries memory from the previous
    window. Returns (masks list[T,256,256], new state tuple)."""
    B, T = 1, len(frames)
    images = torch.from_numpy(frames).float().div(255)[None, :, None].to(DEV)  # [1,T,1,H,W]

    mc = getattr(model, 'use_memory_cache', False)
    mcc = getattr(model, 'memory_cache_context', 'gap_key')

    ms_feats = model.encode_image(images.view(B * T, *images.shape[2:]))
    feat = ms_feats[0]
    pixfeat_DCHW = model.pix_projector(feat)
    pixfeat_BTCHW = pixfeat_DCHW.view(B, T, *pixfeat_DCHW.shape[1:])
    key_DCHW = model.KPFF(model.key_projector(feat), pixfeat_DCHW)
    key_DCHW = model.enhance_key(key_DCHW, ms_feats)
    key_BTCHW = key_DCHW.view(B, T, *key_DCHW.shape[1:])

    N, Cs = 1, model.sensory_dim
    Hk, Wk = key_BTCHW.shape[-2:]
    if init is None:
        first_mask = torch.zeros(B, N, *images.shape[-2:], device=DEV)
        sensory = torch.zeros(B, N, Cs, Hk, Wk, device=DEV)
        value, sensory, obj_mem = model.encode_mask(
            images[:, 0], pixfeat_BTCHW[:, 0], first_mask, sensory, deep_update=True)
        obj_sum = obj_mem.clone()
        state = torch.einsum('bkhw,bnvhw->bnkv', softmax_key(key_BTCHW[:, 0]), value)
        last_masks = first_mask
        cached_memories, cache_frames = [], []
        ctx_sum_BC, ctx_count = None, 0
    else:
        state, sensory, obj_sum = init[0], init[1], init[2]
        cached_memories = list(init[3]) if (mc and len(init) > 3) else []
        ctx_sum_BC = init[4] if (mc and len(init) > 4) else None
        ctx_count = init[5] if (mc and len(init) > 5) else 0
        cache_frames = list(init[6]) if len(init) > 6 else []
        last_masks = None

    out_masks = []
    for i in range(T):
        if i > 0:
            # ---- Gated Delta Rule memory update (mirrors backbone.forward) ----
            state_t_1 = state.clone()
            this_key = softmax_key(key_BTCHW[:, i])
            v_old = torch.einsum('bkhw,bnkv->bnvhw', this_key, state_t_1).contiguous()
            v_k = torch.einsum('bkhw,bnvhw->bnkv', this_key, v_old)
            beta = model.b_proj(state_t_1).sigmoid()
            beta_exp = beta.repeat_interleave(model.value_dim // model.num_heads, dim=3)
            eraser = torch.einsum('bnkv,bnkv->bnkv', beta_exp, v_k).contiguous()

            value, sensory, obj_mem = model.encode_mask(
                images[:, i], pixfeat_BTCHW[:, i - 1], last_masks, sensory, deep_update=True)
            vk_t = torch.einsum('bkhw,bnvhw->bnkv', this_key, value).contiguous()
            new = torch.einsum('bnkv,bnkv->bnkv', beta_exp, vk_t).contiguous()

            old = state_t_1 - eraser
            alpha = -model.A_log.float().exp() * F.softplus(
                model.a_proj(state_t_1).float() + model.dt_bias)
            alpha_exp = alpha.repeat(1, 1, 1, model.value_dim // model.num_heads)
            old = torch.einsum('bnkv,bnkv->bnkv', alpha_exp, old).contiguous()
            state = old + new
            obj_sum = obj_sum + obj_mem

        this_key = softmax_key(key_BTCHW[:, i])

        # ---- memory cache routing (mirrors backbone.forward) ----
        if mc:
            if mcc == 'gap_key':
                key_ctx_BC = model._memory_cache_gap(this_key)
                if ctx_sum_BC is None:
                    ctx_sum_BC = torch.zeros_like(key_ctx_BC)
                    ctx_count = 0
                ctx_sum_BC = ctx_sum_BC + key_ctx_BC
                ctx_count += 1
                online_context_BC = ctx_sum_BC / max(ctx_count, 1)
            else:
                online_context_BC = model._memory_cache_context(this_key)
            state_for_read = model._route_and_mix_state(
                online_state_BNCC=state,
                online_context_BC=online_context_BC,
                cached_memories=cached_memories,
                query_key_BCHW=this_key,
            )
            if routing_records is not None and getattr(model, 'last_routing', None):
                import math
                rec = model.last_routing
                model.last_routing = None       # avoid stale reuse
                if rec['num_cache'] == len(cached_memories):   # skip stale entries
                    g = rec['gamma'][0]                      # B=1
                    M = g.numel()
                    ent = float(-(g * g.clamp_min(1e-12).log()).sum())
                    am = int(g.argmax())
                    sel = cache_frames[am - 1] if am > 0 else None
                    routing_records.append({
                        'frame_idx': frame_offset + i,
                        'num_cache': rec['num_cache'],
                        'cache_frame_indices': ';'.join(map(str, cache_frames)),
                        'gamma_values': ';'.join(f'{v:.6f}' for v in g.tolist()),
                        'gamma_max': f'{g.max():.6f}',
                        'gamma_argmax': am,                  # 0 = online state
                        'gamma_entropy': f'{ent:.6f}',
                        'gamma_norm_entropy': f'{ent / max(math.log(M), 1e-12):.6f}',
                        'score_range': f"{rec.get('score_range', 0.0):.6f}",
                        'score_std': f"{rec.get('score_std', 0.0):.6f}",
                        'selected_cache_frame': sel if sel is not None else '',
                        'distance_to_selected_cache': (frame_offset + i - sel)
                            if sel is not None else '',
                    })
        else:
            state_for_read = state

        readout = torch.einsum('bkhw,bnkv->bnvhw', this_key, state_for_read).contiguous()
        this_ms = [f.view(B, T, *f.shape[1:])[:, i] for f in ms_feats]
        _, masks_full, sensory, _ = model.segment(
            this_ms, readout, pixfeat_BTCHW[:, i],
            last_masks if last_masks is not None else torch.zeros(
                B, N, *images.shape[-2:], device=DEV),
            sensory, obj_sum, update_sensory=True)
        fg = masks_full[:, 1:]                     # [B,N,H,W] prob
        out_masks.append(fg[0, 0].cpu().numpy())
        last_masks = fg

        if mc and ((i + 1) % model.cache_interval == 0):
            cached_state = state.detach() if model.detach_memory_cache else state
            cached_context = (online_context_BC.detach() if model.detach_memory_cache
                              else online_context_BC)
            cached_memories.append({
                "state": cached_state,
                "ctx": cached_context,
                "frame_idx": frame_offset + i,
            })
            cache_frames.append(frame_offset + i)
            ctx_sum_BC = None
            ctx_count = 0

    return out_masks, (state, sensory, obj_sum, cached_memories,
                       ctx_sum_BC, ctx_count, cache_frames)


def infer_sequence(model: Backbone, frames: np.ndarray, carry_state: bool,
                   routing_records: list | None = None):
    T = len(frames)
    masks, state = [], None
    for w in range(math.ceil(T / WINDOW)):
        win = frames[w * WINDOW:(w + 1) * WINDOW]
        mw, state = run_window(model, win,
                               init=state if (carry_state and w > 0) else None,
                               frame_offset=w * WINDOW,
                               routing_records=routing_records)
        masks.extend(mw)
    return np.stack(masks)


# ----------------------------------------------------------------- output

def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred > 0.5, gt > 0
    denom = p.sum() + g.sum()
    return float('nan') if denom == 0 else 2.0 * (p & g).sum() / denom


def overlay(frame: np.ndarray, prob: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    m = prob > 0.5
    bgr[m] = 0.45 * bgr[m] + 0.55 * np.array([0, 0, 255], np.float32)
    edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    bgr[edge > 0] = (0, 255, 255)
    return bgr.astype(np.uint8)


def save_video(path: str, frames: np.ndarray, probs: np.ndarray, fps: int = 10):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (256, 256))
    for f, p in zip(frames, probs):
        vw.write(overlay(f, p))
    vw.release()


def make_grid(path: str, frames: np.ndarray, probs: np.ndarray, cols: int = 6):
    rows = math.ceil(len(frames) / cols)
    grid = np.full((rows * 256, cols * 256, 3), 20, np.uint8)
    for i, (f, p) in enumerate(zip(frames, probs)):
        r, c = divmod(i, cols)
        grid[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256] = overlay(f, p)
    cv2.imwrite(path, grid)


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--data', help='Backbone-format dataset root (img/ gt_lv/ json)')
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--nifti', help='single NIfTI case image')
    ap.add_argument('--label', help='optional NIfTI label for DICE')
    ap.add_argument('--lv-label', type=int, default=1, help='LV value in label volume')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0, help='only first N sequences (0=all)')
    ap.add_argument('--video', action='store_true', help='export mp4 per sequence')
    ap.add_argument('--carry-state', action='store_true',
                    help='continue GDR memory across 10-frame windows')
    ap.add_argument('--use_memory_cache', action='store_true',
                    help='enable Fixed-window Memory Cache + Softmax Memory Soup')
    ap.add_argument('--cache_interval', type=int, default=10,
                    help='cache a state snapshot every N frames')
    ap.add_argument('--detach_memory_cache', action='store_true', default=True,
                    help='detach cached states from the graph (default on)')
    args = ap.parse_args()

    print(f"use_memory_cache={args.use_memory_cache}, "
          f"cache_interval={args.cache_interval}, "
          f"detach_memory_cache={args.detach_memory_cache}")
    if args.use_memory_cache and not args.carry_state:
        print('[note] memory cache only carries across windows with '
              '--carry-state; within one 10-frame window it stays inert')

    # gather sequences: list of (name, frames, gts|None)
    seqs = []
    if args.data:
        j = json.load(open(os.path.join(args.data, 'camus_public_datasplit_20250706.json')))
        ids = j[args.split + '_data']
        if args.limit:
            ids = ids[:args.limit]
        for sid in ids:
            f, g = load_sequence_from_dataset(args.data, sid)
            seqs.append((sid, f, g))
    elif args.nifti:
        f, g = load_sequence_from_nifti(args.nifti, args.label, args.lv_label)
        name = os.path.basename(args.nifti).replace('.nii.gz', '')
        seqs.append((name, f, g))
    else:
        ap.error('--data or --nifti required')

    model = load_model(args.weights,
                       use_memory_cache=args.use_memory_cache,
                       cache_interval=args.cache_interval,
                       detach_memory_cache=args.detach_memory_cache)
    torch.set_grad_enabled(False)
    os.makedirs(args.out, exist_ok=True)

    report_rows, all_dice = [], []
    for sid, frames, gts in seqs:
        probs = infer_sequence(model, frames, args.carry_state)
        make_grid(os.path.join(args.out, f'{sid}_overlay.png'), frames, probs)
        if args.video:
            save_video(os.path.join(args.out, f'{sid}.mp4'), frames, probs)
        if gts is not None:
            for t, (p, g) in enumerate(zip(probs, gts)):
                if g.max() == 0:
                    continue  # un-annotated frame: skip
                d = dice(p, g)
                report_rows.append([sid, t, f'{d:.4f}'])
                all_dice.append(d)
        print(f'{sid}: {len(frames)} frames'
              + (f', DICE={np.nanmean([dice(p, g) for p, g in zip(probs, gts) if g.max() > 0]):.4f}'
                 if gts is not None else ''))

    if report_rows:
        with open(os.path.join(args.out, 'dice_report.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['sequence', 'frame', 'dice'])
            w.writerows(report_rows)
        print(f'\n== Overall DICE: {np.nanmean(all_dice):.4f} '
              f'({len(all_dice)} annotated frames, {len(seqs)} sequences) ==')
        print(f'report: {os.path.join(args.out, "dice_report.csv")}')


if __name__ == '__main__':
    main()
