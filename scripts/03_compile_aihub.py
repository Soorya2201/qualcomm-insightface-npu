#!/usr/bin/env python3
"""
Step 3: Quantize + compile a prepared ONNX model for the Snapdragon NPU.

Produces a QNN context binary (or a precompiled QNN ONNX wrapper) targeting
the Hexagon NPU on a Snapdragon X series part. Compilation runs on Qualcomm
AI Hub against real silicon, so it works from macOS.

Requires: `qai-hub configure --api_token <token>` (free token from
https://app.aihub.qualcomm.com/account/).

Usage:
  python 03_compile_aihub.py \
      --model models/prepared/w600k_r50_static.onnx \
      --calib models/prepared/calib_w600k.npz \
      --device "Snapdragon X Elite CRD" \
      --runtime qnn_context_binary \
      --outdir models/compiled
"""
import argparse
import os
import sys

import numpy as np
import qai_hub as hub

# w8a16 is the recommended default on Hexagon: INT8 weights keep the model
# small, 16-bit activations preserve accuracy for embedding models like
# ArcFace, where cosine distance is sensitive to activation clipping.
DTYPES = {
    "w8a8": (hub.QuantizeDtype.INT8, hub.QuantizeDtype.INT8),
    "w8a16": (hub.QuantizeDtype.INT8, hub.QuantizeDtype.INT16),
    "w4a16": (hub.QuantizeDtype.INT4, hub.QuantizeDtype.INT16),
}


def load_calib(path: str) -> dict[str, list[np.ndarray]]:
    """npz of stacked samples -> AI Hub's {input_name: [sample, sample, ...]}."""
    z = np.load(path)
    entries = {}
    for name in z.files:
        arr = z[name]
        entries[name] = [arr[i : i + 1] for i in range(arr.shape[0])]
    n = len(next(iter(entries.values())))
    print(f"[cal] {path}: inputs={list(entries)} samples={n}")
    return entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--device", default="Snapdragon X Elite CRD")
    ap.add_argument("--precision", default="w8a16", choices=list(DTYPES))
    ap.add_argument("--runtime", default="qnn_context_binary",
                    choices=["qnn_context_binary", "precompiled_qnn_onnx", "onnx", "tflite"])
    ap.add_argument("--outdir", default="models/compiled")
    ap.add_argument("--skip-quantize", action="store_true",
                    help="compile the model as-is (fp16 on NPU); much simpler, larger and slower")
    ap.add_argument("--profile", action="store_true", help="also run an on-device profile job")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = hub.Device(args.device)
    base = os.path.splitext(os.path.basename(args.model))[0]
    print(f"[dev] {device.name}")

    model_for_compile = args.model
    calib = load_calib(args.calib)

    if not args.skip_quantize:
        w, a = DTYPES[args.precision]
        print(f"[qnt] submitting quantize job ({args.precision}) ...")
        qjob = hub.submit_quantize_job(
            model=args.model,
            calibration_data=calib,
            weights_dtype=w,
            activations_dtype=a,
            name=f"{base}-{args.precision}",
        )
        print(f"[qnt] {qjob.url}")
        model_for_compile = qjob.get_target_model()
        if model_for_compile is None:
            print("[qnt] FAILED — see the job URL above for the op-level reason.", file=sys.stderr)
            return 1
        print("[qnt] done")

    print(f"[cmp] submitting compile job (--target_runtime {args.runtime}) ...")
    cjob = hub.submit_compile_job(
        model=model_for_compile,
        device=device,
        options=f"--target_runtime {args.runtime}",
        name=f"{base}-{args.runtime}",
    )
    print(f"[cmp] {cjob.url}")
    target = cjob.get_target_model()
    if target is None:
        print("[cmp] FAILED — see the job URL above.", file=sys.stderr)
        return 1

    out = os.path.join(args.outdir, base)
    path = target.download(out)
    size = os.path.getsize(path) if os.path.isfile(path) else sum(
        os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs
    )
    print(f"[cmp] artifact -> {path}  ({size / 1e6:.1f} MB)")

    if args.profile:
        pjob = hub.submit_profile_job(model=target, device=device, name=f"{base}-profile")
        print(f"[prf] {pjob.url}")
        prof = pjob.download_profile()
        ex = prof["execution_summary"]
        print(f"[prf] inference  {ex['estimated_inference_time'] / 1000:.2f} ms")
        print(f"[prf] peak mem   {ex.get('inference_memory_peak_range', '?')}")
        layers = prof["execution_detail"]
        npu = sum(1 for l in layers if l.get("compute_unit") == "NPU")
        print(f"[prf] layers on NPU: {npu}/{len(layers)}  (anything not NPU is a fallback to CPU/GPU)")

    print("[ok ] compiled artifact is SoC-specific: deploy it to the same Snapdragon part you targeted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
