"""
Inference + visualization on echo AVI clips.

  python infer_video.py --weights outputs/.../xxx.pth \
      --videos /path/a4c.avi /path/a2c.avi --out outputs/infer_video [--video]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer import infer_sequence, load_model, make_grid, overlay, save_video  # noqa: E402


def load_video_sequence(path: str, size: int = 256) -> np.ndarray:
    """AVI -> uint8 gray [T, H, W] resized to size x size."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA))
    cap.release()
    return np.stack(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--videos', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--carry-state', action='store_true')
    args = ap.parse_args()

    model = load_model(args.weights)
    torch.set_grad_enabled(False)
    os.makedirs(args.out, exist_ok=True)

    for path in args.videos:
        sid = os.path.splitext(os.path.basename(path))[0]
        frames = load_video_sequence(path)
        print(f'{sid}: {len(frames)} frames')
        probs = infer_sequence(model, frames, args.carry_state)
        np.save(os.path.join(args.out, f'{sid}_probs.npy'), probs)
        make_grid(os.path.join(args.out, f'{sid}_overlay_grid.png'), frames, probs)
        if args.video:
            save_video(os.path.join(args.out, f'{sid}.mp4'), frames, probs)
        areas = np.array([(p > 0.5).sum() for p in probs])
        print(f'  LV area px per frame: min={areas.min()}, max={areas.max()}, '
              f'mean={areas.mean():.0f}')


if __name__ == '__main__':
    main()
