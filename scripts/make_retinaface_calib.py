from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _load_image(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize(size, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a RetinaFace calibration NPZ with input name 'data'."
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/prepared/calib_retinaface_serengil_640.npz"),
    )
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()

    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(args.images.rglob(pattern))
    paths = sorted(paths)[: args.limit]

    if not paths:
        raise FileNotFoundError(f"No calibration images found in {args.images}")

    batch = np.stack(
        [_load_image(path, (args.width, args.height)) for path in paths],
        axis=0,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, data=batch)
    print(f"Wrote {batch.shape} calibration tensor to {args.out}")


if __name__ == "__main__":
    main()
