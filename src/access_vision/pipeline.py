from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone

import numpy as np

from .matching import AllowList, Match
from .vision import Face, prepare_face

LOGGER = logging.getLogger(__name__)


class JsonEventSink:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    def emit(self, payload: dict) -> None:
        with self._lock:
            print(json.dumps(payload, separators=(",", ":")), flush=True)


class FrameProcessor:
    def __init__(self, config, detector, embedder, allow_list: AllowList, sink=None) -> None:
        self.config = config
        self.detector = detector
        self.embedder = embedder
        self.allow_list = allow_list
        self.sink = sink or JsonEventSink()
        self.last_alert: dict[str, float] = {}
        self._stats: dict[str, dict[str, float]] = {}

    def _record_stats(
        self,
        camera_id: str,
        status: str,
        faces: int,
        detector_ms: float,
        embedding_ms: float,
        total_ms: float,
    ) -> None:
        now = time.monotonic()
        stats = self._stats.setdefault(
            camera_id,
            {
                "started": now,
                "frames": 0.0,
                "faces": 0.0,
                "detector_ms": 0.0,
                "embedding_ms": 0.0,
                "total_ms": 0.0,
            },
        )
        stats["frames"] += 1
        stats["faces"] += faces
        stats["detector_ms"] += detector_ms
        stats["embedding_ms"] += embedding_ms
        stats["total_ms"] += total_ms
        elapsed = now - stats["started"]
        LOGGER.debug(
            "Frame camera=%s status=%s faces=%d total_ms=%.2f detector_ms=%.2f embedding_ms=%.2f",
            camera_id,
            status,
            faces,
            total_ms,
            detector_ms,
            embedding_ms,
        )
        if elapsed < self.config.processing.log_interval_seconds:
            return
        frames = stats["frames"]
        LOGGER.info(
            "Performance camera=%s processed_fps=%.2f frames=%d faces=%d avg_total_ms=%.2f avg_detector_ms=%.2f avg_embedding_ms=%.2f",
            camera_id,
            frames / max(elapsed, 1e-9),
            int(frames),
            int(stats["faces"]),
            stats["total_ms"] / frames,
            stats["detector_ms"] / frames,
            stats["embedding_ms"] / frames,
        )
        self._stats[camera_id] = {
            "started": now,
            "frames": 0.0,
            "faces": 0.0,
            "detector_ms": 0.0,
            "embedding_ms": 0.0,
            "total_ms": 0.0,
        }

    def process(self, camera_id: str, frame_rgb: np.ndarray) -> dict:
        frame_started = time.perf_counter()
        faces = self.detector.detect(frame_rgb)
        detector_ms = (time.perf_counter() - frame_started) * 1000
        embedding_ms = 0.0
        matches: list[tuple[Match, Face]] = []
        for face in faces:
            x1, y1, x2, y2 = face.xyxy
            if min(x2 - x1, y2 - y1) < self.config.processing.min_face_size:
                continue
            crop = prepare_face(frame_rgb, face, self.config.embedder)
            if crop.size:
                embedding_started = time.perf_counter()
                embedding = self.embedder.embed(crop)
                embedding_ms += (time.perf_counter() - embedding_started) * 1000
                matches.append((self.allow_list.match(embedding), face))

        if not matches:
            total_ms = (time.perf_counter() - frame_started) * 1000
            self._record_stats(
                camera_id, "no_face", 0, detector_ms, embedding_ms, total_ms
            )
            return {
                "status": "no_face",
                "camera": camera_id,
                "faces": [],
                "performance": {
                    "total_ms": round(total_ms, 2),
                    "detector_ms": round(detector_ms, 2),
                    "embedding_ms": round(embedding_ms, 2),
                },
            }
        allowed = (
            any(match.allowed for match, _ in matches)
            if self.config.processing.decision_mode == "any_allowed"
            else all(match.allowed for match, _ in matches)
        )
        result = {
            "status": "allowed" if allowed else "unauthorized",
            "camera": camera_id,
            "faces": [
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
                for match, face in matches
            ],
        }
        total_ms = (time.perf_counter() - frame_started) * 1000
        result["performance"] = {
            "total_ms": round(total_ms, 2),
            "detector_ms": round(detector_ms, 2),
            "embedding_ms": round(embedding_ms, 2),
        }
        self._record_stats(
            camera_id, result["status"], len(matches), detector_ms, embedding_ms, total_ms
        )
        now = time.monotonic()
        if not allowed and now - self.last_alert.get(camera_id, 0.0) >= self.config.processing.alert_cooldown_seconds:
            self.last_alert[camera_id] = now
            self.sink.emit({
                "camera": camera_id,
                "tag": "unauthorized",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "faces": result["faces"],
            })
        return result
