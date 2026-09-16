#!/usr/bin/env python3
"""
Step 5: Run InsightFace on the Snapdragon NPU. THIS SCRIPT RUNS ON THE DEVICE
(Windows on ARM64 / Snapdragon X), not on the Mac.

Loads the compiled QNN artifact through ONNX Runtime's QNN Execution Provider
and produces 512-d ArcFace embeddings. This is the runtime path for a CNN --
Genie/GenieX is a text-generation runtime and cannot load this model.

Setup on the device (Windows ARM64, native arm64 Python -- not x64):
    pip install onnxruntime-qnn numpy opencv-python

Usage:
    python 05_run_on_device.py --model w600k_r50_qnn.onnx --images faces/
    python 05_run_on_device.py --model w600k_r50_qnn.onnx --compare a.jpg b.jpg
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

# Cosine threshold for "same person" with buffalo_l ArcFace embeddings.
# Tune on your own data; this is the commonly used starting point.
SAME_PERSON_THRESHOLD = 0.36


def make_session(model_path: str, backend: str):
    import onnxruntime as ort

    providers, opts = [], []
    if backend != "cpu":
        # QnnHtp = Hexagon NPU. QnnCpu is the reference fallback and is slow;
        # it exists to prove correctness, not for production.
        providers.append("QNNExecutionProvider")
        opts.append({
            "backend_path": "QnnHtp.dll",
            "htp_performance_mode": "high_performance",
            "htp_graph_finalization_optimization_mode": "3",
        })
    providers.append("CPUExecutionProvider")
    opts.append({})

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(model_path, sess_options=so,
                                providers=providers, provider_options=opts)

    active = sess.get_providers()
    print(f"[ep ] active providers: {active}")
    if backend != "cpu" and "QNNExecutionProvider" not in active:
        print("[!! ] QNN EP did NOT load -- you are running on CPU.", file=sys.stderr)
        print("[!! ] Check: arm64 Python (not x64), onnxruntime-qnn installed, "
              "QnnHtp.dll on PATH.", file=sys.stderr)
    return sess


def preprocess(path: str, size: int = 112) -> np.ndarray:
    """InsightFace preprocessing. Must match what calibration used exactly."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise SystemExit(f"cannot read image: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    arr = (img.astype(np.float32) - 127.5) / 127.5
    return arr.transpose(2, 0, 1)[None, ...]  # NCHW


def embed(sess, batch: np.ndarray) -> np.ndarray:
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: batch})[0]
    # L2-normalize: ArcFace embeddings are only meaningful on the unit sphere.
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", help="directory of aligned face crops to embed")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--backend", default="npu", choices=["npu", "cpu"])
    ap.add_argument("--bench", type=int, default=0, help="run N timed iterations")
    args = ap.parse_args()

    sess = make_session(args.model, args.backend)

    if args.compare:
        a, b = (embed(sess, preprocess(p)) for p in args.compare)
        cos = float((a @ b.T).item())
        verdict = "SAME person" if cos >= SAME_PERSON_THRESHOLD else "DIFFERENT people"
        print(f"[cmp] cosine {cos:.4f} (threshold {SAME_PERSON_THRESHOLD}) -> {verdict}")

    if args.images:
        paths = sorted(p for ext in ("jpg", "jpeg", "png")
                       for p in glob.glob(os.path.join(args.images, f"**/*.{ext}"), recursive=True))
        print(f"[emb] {len(paths)} images")
        embs = {}
        for p in paths:
            embs[os.path.basename(p)] = embed(sess, preprocess(p))[0]
        np.savez("embeddings.npz", **embs)
        print(f"[emb] wrote embeddings.npz ({len(embs)} x 512)")

    if args.bench:
        x = np.random.rand(1, 3, 112, 112).astype(np.float32)
        name = sess.get_inputs()[0].name
        for _ in range(10):  # warmup: first calls include graph finalization
            sess.run(None, {name: x})
        t0 = time.perf_counter()
        for _ in range(args.bench):
            sess.run(None, {name: x})
        dt = (time.perf_counter() - t0) / args.bench * 1000
        print(f"[bch] {dt:.2f} ms/inference  ({1000 / dt:.0f} FPS)  backend={args.backend}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
