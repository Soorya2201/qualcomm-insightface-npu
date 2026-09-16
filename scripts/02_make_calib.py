#!/usr/bin/env python3
"""
Step 2: Build a calibration set for post-training quantization.

Quantization accuracy depends entirely on the calibration data matching the
real input distribution. For ArcFace that means aligned 112x112 face crops
preprocessed exactly the way the runtime will preprocess them:
    (BGR - 127.5) / 127.5  ->  NCHW float32

Point --images at a directory of real aligned faces. Without it the script
falls back to synthetic noise, which is enough to make the pipeline run but
NOT enough for production accuracy.

Usage:
  python 02_make_calib.py --images ./calib --shape 1,3,112,112 --n 128 \
      --out models/prepared/calib_w600k.npz
"""
import argparse
import glob
import os
import sys

import numpy as np


def load_real(paths, shape, n):
    import cv2
    _, c, h, w = shape
    batch = []
    for p in paths[:n]:
        img = cv2.imread(p, cv2.IMREAD_COLOR)  # BGR, matching InsightFace
        if img is None:
            print(f"[warn] unreadable, skipping: {p}")
            continue
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        arr = (img.astype(np.float32) - 127.5) / 127.5
        batch.append(arr.transpose(2, 0, 1))  # HWC -> CHW
    return batch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="directory of aligned face crops")
    ap.add_argument("--shape", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--input-name", default="input")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    shape = [int(x) for x in args.shape.split(",")]

    samples = []
    if args.images:
        paths = sorted(
            p for ext in ("jpg", "jpeg", "png", "bmp", "webp")
            for p in glob.glob(os.path.join(args.images, f"**/*.{ext}"), recursive=True)
        )
        print(f"[cal] found {len(paths)} images under {args.images}")
        samples = load_real(paths, shape, args.n)

    if not samples:
        print("[cal] WARNING: no real images -> synthetic calibration data.")
        print("[cal] The pipeline will run, but quantized accuracy will be unrepresentative.")
        print("[cal] Supply ~100-500 real aligned faces via --images before you trust the numbers.")
        samples = [np.random.randn(*shape[1:]).astype(np.float32) * 0.5 for _ in range(args.n)]

    # AI Hub expects a list of per-sample batches, one entry per calibration step.
    data = {args.input_name: [s[None, ...].astype(np.float32) for s in samples]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **{args.input_name: np.concatenate(data[args.input_name], axis=0)})

    print(f"[cal] {len(samples)} samples -> {args.out}")
    print(f"[cal] input name '{args.input_name}', per-sample shape {shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
