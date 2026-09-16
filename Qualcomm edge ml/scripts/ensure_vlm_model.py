"""Check whether the GenieX VLM bundle is present; fetch it from Qualcomm AI
Hub if not.

Run this before starting access_vision with [vlm].enabled = true, or just
run access_vision -- server.py does not call this automatically, on purpose:
a multi-gigabyte download should never start as a side effect of launching
the app.

    python scripts/ensure_vlm_model.py

What this actually does
------------------------
1. If GenieX's local server is already running and reports the model in its
   `/v1/models` list, nothing to do.
2. Otherwise, checks whether `models/vlm/` already has fetched content.
3. Otherwise, shells out to the documented AI Hub fetch command:

       qai-hub-models fetch Qwen2.5-VL-7B-Instruct \
           --runtime geniex_qairt --precision w4a16 -o models/vlm

   This needs the `qai-hub-models` package and a free Qualcomm AI Hub token
   (`qai-hub configure --api_token ...`), the same account already used for
   the InsightFace pipeline in this repo's root scripts/.

What is NOT verified here, and needs your actual Windows ARM64 + GenieX +
Snapdragon NPU machine to confirm
-----------------------------------------------------------------------------
Qualcomm's own docs, at the time this was written, do not fully specify how
a `qai-hub-models fetch --runtime geniex_qairt` bundle in an arbitrary output
directory becomes something `geniex serve` will actually serve under the
model name access_vision sends (vlm.MODEL_ID / [vlm].model in config). Two
things to check once, on device, before relying on this:

    1. Run the fetch command above, then `geniex serve`, then:
           curl http://127.0.0.1:18181/v1/models
       Confirm "Qwen2.5-VL-7B-Instruct" (or whatever string appears) is
       listed, and that string exactly matches [vlm].model in your config.
    2. If it is NOT listed, the documented alternative is pulling by the same
       identifier directly through GenieX's own CLI instead of qai-hub-models:
           geniex pull Qwen2.5-VL-7B-Instruct
       (this project's uno-q-board submodule already uses this exact
       `geniex pull <model>` then `geniex serve` pattern successfully for an
       LLM -- see x_elite/client.py -- so it is the safer fallback if the
       qai-hub-models path above does not line up.)

This script tries the qai-hub-models path (it is the one with a fully
documented precision flag) and tells you plainly if it could not confirm the
result, rather than reporting success it did not actually check.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "Qwen2.5-VL-7B-Instruct"
DEFAULT_PRECISION = "w4a16"
DEFAULT_BASE_URL = "http://127.0.0.1:18181/v1"


def model_listed_by_running_server(base_url: str, model: str, timeout: float = 3.0) -> bool | None:
    """True/False if GenieX answered; None if it is not running at all --
    a caller needs to tell "not present" apart from "can't tell"."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{base_url.rstrip('/')}/models", timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    ids = {entry.get("id") for entry in body.get("data", []) if isinstance(entry, dict)}
    return model in ids


def output_dir_has_content(output_dir: Path) -> bool:
    return output_dir.is_dir() and any(output_dir.iterdir())


def fetch_via_qai_hub_models(model: str, precision: str, output_dir: Path) -> int:
    if shutil.which("qai-hub-models") is None:
        print(
            "qai-hub-models is not on PATH. Install it (same environment as the rest of\n"
            "this app) and configure an AI Hub token first:\n"
            "    pip install qai-hub-models\n"
            "    qai-hub configure --api_token <your token>\n",
            file=sys.stderr,
        )
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "qai-hub-models", "fetch", model,
        "--runtime", "geniex_qairt", "--precision", precision,
        "-o", str(output_dir),
    ]
    print(f"[fetch] {' '.join(command)}")
    result = subprocess.run(command)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"AI Hub / GenieX model identifier (default: {DEFAULT_MODEL})")
    parser.add_argument("--precision", default=DEFAULT_PRECISION,
                        help=f"qai-hub-models --precision value (default: {DEFAULT_PRECISION}; "
                             "Qualcomm does not publish an int8/w8a8 bundle for this model -- "
                             "see vlm.py's module docstring before changing this)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="GenieX server URL to check first, if it happens to be running")
    parser.add_argument("--output-dir", default="models/vlm",
                        help="Where qai-hub-models fetch writes the bundle (default: models/vlm)")
    parser.add_argument("--force", action="store_true",
                        help="Fetch even if models/vlm already has content")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    listed = model_listed_by_running_server(args.base_url, args.model)
    if listed is True:
        print(f"[ok] GenieX at {args.base_url} already reports '{args.model}'. Nothing to do.")
        return 0
    if listed is False:
        print(f"[warn] GenieX at {args.base_url} is running but does NOT list '{args.model}' yet.")
    else:
        print(f"[info] GenieX is not reachable at {args.base_url} (not running right now).")

    if not args.force and output_dir_has_content(output_dir):
        print(f"[ok] {output_dir} already has fetched content; skipping download.")
        print("[note] this does not confirm GenieX will serve it under the right model name --")
        print("       see the verification steps in this script's own module docstring.")
        return 0

    code = fetch_via_qai_hub_models(args.model, args.precision, output_dir)
    if code != 0:
        print(f"[FAIL] qai-hub-models fetch exited {code}.", file=sys.stderr)
        print(
            "       Documented fallback: pull the same identifier directly through GenieX:\n"
            f"           geniex pull {args.model}\n"
            "       (this is the pattern already used successfully for an LLM in this repo's\n"
            "       uno-q-board submodule -- see x_elite/client.py)",
            file=sys.stderr,
        )
        return code

    print(f"[ok] fetched into {output_dir}.")
    print("[IMPORTANT] this does not confirm `geniex serve` will resolve model=\"%s\"." % args.model)
    print("            Start `geniex serve`, then run:")
    print(f"                curl {args.base_url}/models")
    print(f"            and confirm '{args.model}' is listed, exactly matching [vlm].model")
    print("            in your config. See this script's module docstring if it is not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
