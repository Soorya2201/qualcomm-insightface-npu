#!/usr/bin/env python3
"""
Step 4: Verify the compiled NPU model against the float ONNX golden output.

For a face-recognition embedding, raw per-element error is not the number that
matters -- cosine similarity between the fp32 and quantized embeddings is.
Below ~0.99 cosine, expect measurable degradation in verification accuracy.

Usage:
  python 04_verify.py --target models/compiled/w600k_r50 \
      --golden models/prepared/w600k_r50_static_golden.npz \
      --device "Snapdragon X Elite CRD"
"""
import argparse
import sys

import numpy as np
import qai_hub as hub


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="compiled artifact path or AI Hub model id")
    ap.add_argument("--golden", required=True)
    ap.add_argument("--device", default="Snapdragon X Elite CRD")
    ap.add_argument("--input-name", default="input")
    args = ap.parse_args()

    g = np.load(args.golden)
    x, y_ref = g["x"], g["y0"]

    job = hub.submit_inference_job(
        model=args.target,
        device=hub.Device(args.device),
        inputs={args.input_name: [x.astype(np.float32)]},
        name="insightface-verify",
    )
    print(f"[inf] {job.url}")
    out = job.download_output_data()
    if out is None:
        print("[inf] FAILED — see job URL.", file=sys.stderr)
        return 1

    y_dev = np.array(next(iter(out.values()))[0])
    print(f"[inf] device output {y_dev.shape}, reference {y_ref.shape}")

    cos = cosine(y_ref, y_dev)
    mae = float(np.abs(y_ref.ravel() - y_dev.ravel()).mean())
    print(f"[cmp] cosine similarity : {cos:.6f}")
    print(f"[cmp] mean abs error    : {mae:.6f}")
    print("[ok ] PASS" if cos >= 0.99 else "[!! ] BELOW 0.99 — recalibrate with real faces, or move to w8a16/fp16")
    return 0 if cos >= 0.99 else 2


if __name__ == "__main__":
    sys.exit(main())
