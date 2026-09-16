from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _resize_pad_rgb(image: Image.Image, width: int, height: int) -> np.ndarray:
    image = image.convert("RGB")
    original_w, original_h = image.size
    scale = min(width / original_w, height / original_h)
    resized_w = max(1, min(width, round(original_w * scale)))
    resized_h = max(1, min(height, round(original_h * scale)))
    resized = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    pad_left = (width - resized_w) // 2
    pad_top = (height - resized_h) // 2
    canvas[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = np.asarray(
        resized, dtype=np.uint8
    )
    return canvas


def _load_image(path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(path) as image:
        image_rgb = _resize_pad_rgb(image, width, height)
    image_bgr = image_rgb[..., ::-1].astype(np.float32)
    image_bgr -= np.asarray((104.0, 117.0, 123.0), dtype=np.float32)
    return np.transpose(image_bgr, (2, 0, 1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build calibration NPZ for yakhyo RetinaFace ONNX input."
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/prepared/calib_retinaface_yakhyo_mv1_025_640.npz"),
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
        [_load_image(path, args.width, args.height) for path in paths],
        axis=0,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, input=batch)
    print(f"Wrote {batch.shape} calibration tensor to {args.out}")


if __name__ == "__main__":
    main()
