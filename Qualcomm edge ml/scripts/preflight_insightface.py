"""Answer one question: will InsightFace w600k_r50 actually work with this config?

Every check here is a silent-failure mode -- a wrong value produces embeddings that
look statistically normal (cosine scores in a believable range) but match the wrong
people. None of them raise on their own. Run this before trusting a result.

    python scripts/preflight_insightface.py --config config.toml
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

# What the Snapdragon X Elite W8A16 context binary was calibrated and compiled for.
REQUIRED = {
    "channel_order": "bgr",          # InsightFace trained on BGR, not RGB
    "image_mean": [0.5, 0.5, 0.5],   # with /255 first, this is (x-127.5)/127.5
    "image_std": [0.5, 0.5, 0.5],
    "align_landmarks": True,         # 5-point ArcFace warp, not a plain crop
    "embedding_dimension": 512,
    "input_width": 112,
    "input_height": 112,
}

PASS, FAIL, WARN = "[ok  ]", "[FAIL]", "[warn]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.toml")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        print(f"{FAIL} config not found: {config_path}")
        print("       This file is gitignored, so a fresh clone has no config at all.")
        return 1

    raw = tomllib.load(config_path.open("rb"))
    emb = raw.get("models", {}).get("embedder", {})
    db = raw.get("database", {})
    runtime = raw.get("runtime", {})
    failures = 0

    print(f"config: {config_path}\n")

    # --- preprocessing contract -------------------------------------------------
    print("preprocessing (wrong values => plausible but wrong matches)")
    for key, want in REQUIRED.items():
        got = emb.get(key, "<default>")
        if isinstance(want, list):
            ok = isinstance(got, list) and len(got) == 3 and all(
                abs(float(g) - w) < 1e-9 for g, w in zip(got, want)
            )
        else:
            ok = got == want
        print(f"  {PASS if ok else FAIL} {key:<20} want={want!r:<18} got={got!r}")
        failures += not ok

    # --- the model itself -------------------------------------------------------
    print("\nmodel artifact")
    model_path = emb.get("path")
    if not model_path:
        print(f"  {FAIL} models.embedder.path is not set")
        failures += 1
    else:
        resolved = (config_path.parent / model_path).resolve()
        exists = resolved.is_file()
        print(f"  {PASS if exists else FAIL} onnx exists         {resolved}")
        failures += not exists
        if exists:
            # The 333-byte wrapper is useless without the context binary next to it.
            companion = resolved.with_name("model.bin")
            has_bin = companion.is_file()
            small = resolved.stat().st_size < 10_000
            if small:
                print(f"  {PASS if has_bin else FAIL} model.bin beside it {companion}")
                failures += not has_bin
                if has_bin:
                    mb = companion.stat().st_size / 1e6
                    print(f"  {PASS if mb > 30 else WARN} model.bin size      {mb:.1f} MB (expect ~42)")

    # --- database ---------------------------------------------------------------
    print("\nembedding database")
    db_path = db.get("embeddings_path")
    if not db_path:
        print(f"  {FAIL} database.embeddings_path is not set")
        failures += 1
    else:
        resolved_db = (config_path.parent / db_path).resolve()
        if not resolved_db.is_file():
            print(f"  {FAIL} not found: {resolved_db}")
            print("       Run: python -m access_vision.enroll --config " + config_path.name)
            failures += 1
        else:
            data = json.load(resolved_db.open(encoding="utf-8"))
            dims, total = set(), 0
            for templates in data.values():
                rows = templates if isinstance(templates[0], list) else [templates]
                total += len(rows)
                dims.update(len(r) for r in rows)
            ok_dim = dims == {512}
            print(f"  {PASS if ok_dim else FAIL} dimension           {dims} (want {{512}})")
            failures += not ok_dim
            print(f"  {PASS} identities={len(data)} templates={total}")

            # CavaFace is ALSO 512-d, so the dimension guard cannot catch a stale
            # database enrolled with a different model. Only the filename can.
            model_id = str(emb.get("model_id", ""))
            tagged = model_id and model_id in resolved_db.name
            print(f"  {PASS if tagged else WARN} db name tagged      {resolved_db.name}")
            if not tagged:
                print(f"       WARNING: name does not contain {model_id!r}.")
                print("       CavaFace embeddings are also 512-d, so a stale database")
                print("       passes every check and silently matches the wrong people.")
                print("       Re-enroll if you are not certain which model wrote this.")

    # --- threshold / runtime ----------------------------------------------------
    print("\nthreshold and runtime")
    thr = float(db.get("cosine_threshold", 0.60))
    sane = 0.2 <= thr <= 0.6
    print(f"  {PASS if sane else WARN} cosine_threshold    {thr} (0.36 is the buffalo_l starting point)")
    if abs(thr - 0.60) < 1e-9:
        print("       0.60 is the MobileFaceNet default -- almost certainly not calibrated for this model.")
    print(f"  {PASS if runtime.get('require_npu', True) else WARN} require_npu         {runtime.get('require_npu', True)}")
    if not runtime.get("require_npu", True):
        print("       With require_npu=false a CPU-only session is accepted silently.")

    print(f"\n{'PASS - configuration matches the compiled model' if not failures else f'{failures} BLOCKING PROBLEM(S) - see [FAIL] above'}")
    print("\nNote: this validates configuration only. It cannot tell you whether the")
    print("threshold is right for your cameras -- that needs genuine/impostor pairs.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
