from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path

import numpy as np

from access_vision.config import load_config
from access_vision.runtime import QnnSession
from access_vision.vision import FaceDetector


def read_bmp(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if data[:2] != b"BM":
        raise ValueError(f"Not a BMP file: {path}")
    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    width = struct.unpack_from("<i", data, 18)[0]
    signed_height = struct.unpack_from("<i", data, 22)[0]
    bits_per_pixel = struct.unpack_from("<H", data, 28)[0]
    compression = struct.unpack_from("<I", data, 30)[0]
    if width <= 0 or signed_height == 0 or bits_per_pixel not in (24, 32) or compression != 0:
        raise ValueError(f"Unsupported BMP layout in {path}")

    height = abs(signed_height)
    channels = bits_per_pixel // 8
    row_stride = ((width * bits_per_pixel + 31) // 32) * 4
    pixels = np.frombuffer(data, dtype=np.uint8, offset=pixel_offset)
    rows = pixels[: height * row_stride].reshape(height, row_stride)
    image = rows[:, : width * channels].reshape(height, width, channels)[..., :3]
    if signed_height > 0:
        image = image[::-1]
    return image[..., ::-1].copy()


def write_png(path: Path, image_rgb: np.ndarray) -> None:
    height, width = image_rgb.shape[:2]

    def chunk(name: bytes, payload: bytes) -> bytes:
        body = name + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    scanlines = b"".join(b"\0" + row.tobytes() for row in image_rgb)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(scanlines, level=6))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def draw_boxes(image: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    result = image.copy()
    height, width = result.shape[:2]
    thickness = max(3, min(width, height) // 250)
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
        y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
        result[y1 : min(height, y1 + thickness), x1 : x2 + 1] = (0, 255, 0)
        result[max(0, y2 - thickness + 1) : y2 + 1, x1 : x2 + 1] = (0, 255, 0)
        result[y1 : y2 + 1, x1 : min(width, x1 + thickness)] = (0, 255, 0)
        result[y1 : y2 + 1, max(0, x2 - thickness + 1) : x2 + 1] = (0, 255, 0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the configured NPU face detector on BMP images")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics/face_detection"))
    args = parser.parse_args()

    config = load_config(args.config)
    detector = FaceDetector(QnnSession(config.detector.path, config.runtime), config.detector)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for path in args.images:
        image = read_bmp(path)
        faces = detector.detect(image)
        output = args.output_dir / f"{path.stem}_detected.png"
        write_png(output, draw_boxes(image, [face.xyxy for face in faces]))
        item = {
            "image": path.name,
            "maximum_detector_score": round(detector.last_max_score, 6),
            "required_score": config.detector.score_threshold,
            "faces": [
                {"bbox": list(face.xyxy), "score": round(face.score, 6)}
                for face in faces
            ],
            "output": str(output.resolve()),
        }
        results.append(item)
        print(json.dumps(item))

    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
