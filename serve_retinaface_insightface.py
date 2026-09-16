from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from access_vision.config import load_config
from access_vision.matching import AllowList
from access_vision.runtime import QnnSession
from access_vision.vision import FaceDetector, FaceEmbedder, prepare_face


LOGGER = logging.getLogger("retinaface_insightface_server")
MAX_FRAME_BYTES = 32 * 1024 * 1024


def _decode_rgba(body: bytes, width: int, height: int) -> np.ndarray:
    expected = width * height * 4
    if width <= 0 or height <= 0 or len(body) != expected:
        raise ValueError(
            f"Expected {expected} RGBA bytes for {width}x{height}, received {len(body)}"
        )
    return np.frombuffer(body, dtype=np.uint8).reshape(height, width, 4)[..., :3].copy()


class RetinaFaceInsightFaceService:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config = load_config(config_path)
        if self.config.detector.model_id != "retinaface_yakhyo":
            raise ValueError(
                "This serving file is pinned to RetinaFace. Use a config with "
                'models.detector.model_id = "retinaface_yakhyo".'
            )
        if self.config.embedder.model_id != "insightface_w600k_r50":
            raise ValueError(
                "This serving file is pinned to InsightFace. Use a config with "
                'models.embedder.model_id = "insightface_w600k_r50".'
            )

        LOGGER.info("Loading RetinaFace detector: %s", self.config.detector.path)
        self.detector = FaceDetector(
            QnnSession(self.config.detector.path, self.config.runtime),
            self.config.detector,
        )
        LOGGER.info("Loading InsightFace embedder: %s", self.config.embedder.path)
        self.embedder = FaceEmbedder(
            QnnSession(self.config.embedder.path, self.config.runtime),
            self.config.embedder,
        )
        self.allow_list = AllowList.load(
            self.config.database.embeddings_path,
            self.config.database.cosine_threshold,
            self.config.embedder.embedding_dimension,
        )

    def analyze(self, camera_id: str, frame_rgb: np.ndarray) -> dict:
        started = time.perf_counter()
        faces = self.detector.detect(frame_rgb)
        detector_ms = (time.perf_counter() - started) * 1000
        embedding_ms = 0.0
        results = []

        for face in faces:
            x1, y1, x2, y2 = face.xyxy
            if min(x2 - x1, y2 - y1) < self.config.processing.min_face_size:
                continue
            face_rgb = prepare_face(frame_rgb, face, self.config.embedder)
            if not face_rgb.size:
                continue
            embedding_started = time.perf_counter()
            embedding = self.embedder.embed(face_rgb)
            embedding_ms += (time.perf_counter() - embedding_started) * 1000
            match = self.allow_list.match(embedding)
            results.append(
                {
                    "bbox": list(face.xyxy),
                    "landmarks": (
                        [list(point) for point in face.landmarks]
                        if face.landmarks is not None
                        else []
                    ),
                    "allowed": match.allowed,
                    "person_id": match.person_id,
                    "cosine_similarity": round(match.cosine_similarity, 4),
                }
            )

        if not results:
            status = "no_face"
        elif self.config.processing.decision_mode == "all_faces_allowed":
            status = "allowed" if all(item["allowed"] for item in results) else "unauthorized"
        else:
            status = "allowed" if any(item["allowed"] for item in results) else "unauthorized"

        total_ms = (time.perf_counter() - started) * 1000
        return {
            "status": status,
            "camera": camera_id,
            "detector": self.config.detector.model_id,
            "embedder": self.config.embedder.model_id,
            "faces": results,
            "performance": {
                "total_ms": round(total_ms, 2),
                "detector_ms": round(detector_ms, 2),
                "embedding_ms": round(embedding_ms, 2),
            },
        }


def make_handler(service: RetinaFaceInsightFaceService):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, value: dict, status: int = 200) -> None:
            payload = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._json(
                    {
                        "status": "ok",
                        "detector": service.config.detector.model_id,
                        "embedder": service.config.embedder.model_id,
                        "config": str(service.config_path),
                    }
                )
                return
            self._json({"error": "not_found"}, 404)

        def do_POST(self) -> None:
            try:
                parsed = urlparse(self.path)
                if parsed.path not in {"/api/frame", "/v1/face/analyze"}:
                    self._json({"error": "not_found"}, 404)
                    return

                query = parse_qs(parsed.query)
                length = int(self.headers.get("Content-Length", "0"))
                if length > MAX_FRAME_BYTES:
                    raise ValueError("Frame is too large")

                width = int(query["width"][0])
                height = int(query["height"][0])
                camera_id = query.get("camera", ["camera"])[0]
                frame = _decode_rgba(self.rfile.read(length), width, height)
                self._json(service.analyze(camera_id, frame))
            except Exception as exc:
                LOGGER.exception("Request failed")
                self._json({"error": str(exc)}, 400)

        def log_message(self, format: str, *args) -> None:
            LOGGER.debug(format, *args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone RetinaFace + InsightFace HTTP serving sidecar. "
            "This does not modify or start the main Access Vision app."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.retinaface.toml",
        help="Config containing retinaface_yakhyo detector and insightface_w600k_r50 embedder.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18182)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    service = RetinaFaceInsightFaceService(args.config.resolve())
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    server.daemon_threads = True
    LOGGER.info("RetinaFace + InsightFace serving at http://%s:%d", args.host, args.port)
    LOGGER.info("POST raw RGBA frames to /api/frame?camera=<id>&width=<w>&height=<h>")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
