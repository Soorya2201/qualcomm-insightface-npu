#!/usr/bin/env python3
"""
Step 1: Prepare an InsightFace ONNX model for Qualcomm NPU compilation.

The Hexagon NPU needs a fully static graph: fixed batch, fixed spatial dims,
no dynamic shape operators. This script pins every input dimension, folds
constants, and reports the resulting I/O signature.

Usage:
  python 01_prepare_onnx.py models/onnx/models/buffalo_l/w600k_r50.onnx \
      --shape 1,3,112,112 --out models/prepared/w600k_r50_static.onnx
"""
import argparse
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference


def pin_input_shape(model: onnx.ModelProto, shape: list[int]) -> onnx.ModelProto:
    """Rewrite the graph's single input to a fully static shape."""
    if len(model.graph.input) != 1:
        names = [i.name for i in model.graph.input]
        raise SystemExit(f"expected 1 graph input, found {len(names)}: {names}")

    inp = model.graph.input[0]
    dims = inp.type.tensor_type.shape.dim
    if len(dims) != len(shape):
        raise SystemExit(f"rank mismatch: graph input has rank {len(dims)}, --shape has rank {len(shape)}")

    for dim, value in zip(dims, shape):
        dim.ClearField("dim_param")
        dim.dim_value = value

    # Downstream value_info may carry stale symbolic dims; drop and re-infer.
    del model.graph.value_info[:]
    for out in model.graph.output:
        for dim in out.type.tensor_type.shape.dim:
            if dim.HasField("dim_param"):
                dim.ClearField("dim_param")
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--shape", required=True, help="static input shape, e.g. 1,3,112,112")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-simplify", action="store_true")
    args = ap.parse_args()

    shape = [int(x) for x in args.shape.split(",")]
    model = onnx.load(args.model)

    print(f"[in ] {args.model}")
    print(f"[in ] opset={model.opset_import[0].version} ir={model.ir_version}")

    model = pin_input_shape(model, shape)
    model = shape_inference.infer_shapes(model)

    if not args.no_simplify:
        from onnxsim import simplify
        model, ok = simplify(model, overwrite_input_shapes={model.graph.input[0].name: shape})
        print(f"[sim] simplified={ok}")

    onnx.checker.check_model(model)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    onnx.save(model, args.out)

    # Confirm the static graph actually runs, and capture the golden output
    # that step 4 compares the on-device result against.
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.rand(*shape).astype(np.float32)
    outs = sess.run(None, {name: x})

    print(f"[out] {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    print(f"[out] input  {name} {sess.get_inputs()[0].shape}")
    for o, arr in zip(sess.get_outputs(), outs):
        print(f"[out] output {o.name} {list(arr.shape)} {arr.dtype}")

    np.savez(args.out.replace(".onnx", "_golden.npz"), x=x, **{f"y{i}": a for i, a in enumerate(outs)})
    print(f"[out] golden reference saved alongside model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
