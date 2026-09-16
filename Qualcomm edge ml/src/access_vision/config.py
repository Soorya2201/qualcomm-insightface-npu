from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import os
import tomllib


@dataclass(frozen=True)
class CameraConfig:
    id: str
    source: str = "webcam"
    url: str | None = None


@dataclass(frozen=True)
class DetectorConfig:
    path: Path
    model_id: str = "face_det_lite"
    landmark_path: Path | None = None
    score_threshold: float = 0.55
    nms_iou_threshold: float = 0.30


@dataclass(frozen=True)
class EmbedderConfig:
    path: Path
    model_id: str = "mobile_facenet"
    embedding_dimension: int = 128
    input_width: int = 112
    input_height: int = 112
    output_name: str | None = None
    image_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    image_std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    channel_order: str = "rgb"
    align_landmarks: bool = False


@dataclass(frozen=True)
class DatabaseConfig:
    embeddings_path: Path
    cosine_threshold: float = 0.60


@dataclass(frozen=True)
class EnrollmentConfig:
    minimum_images_per_person: int = 3


@dataclass(frozen=True)
class RuntimeConfig:
    provider: str = "QNNExecutionProvider"
    backend_path: str = "QnnHtp.dll"
    performance_mode: str = "burst"
    require_npu: bool = True
    context_cache: bool = True


@dataclass(frozen=True)
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    frame_width: int = 640
    frame_height: int = 480
    frame_interval_ms: int = 300
    open_browser: bool = True


@dataclass(frozen=True)
class ProcessingConfig:
    alert_cooldown_seconds: float = 5.0
    decision_mode: str = "any_allowed"
    min_face_size: int = 32
    log_interval_seconds: float = 5.0


@dataclass(frozen=True)
class BoardConfig:
    """Arduino Uno Q status light. Disabled by default; absent tooling is not an error."""

    enabled: bool = False
    transport: str = "http"          # "http" board listens; "ntfy" relay via internet; "ssh" legacy push
    url: str = "http://SCL-UNOQ05.local:8770"
    token: str = ""
    ntfy_topic: str = ""             # long random string, not a guessable name
    ntfy_base_url: str = "https://ntfy.sh"
    scripts_dir: Path | None = None
    heartbeat_seconds: float = 30.0  # accepted for older configs; cadence is min_interval_seconds
    min_interval_seconds: float = 0.5  # heartbeat cadence and minimum gap between sends
    ntfy_budget_burst: int = 50        # ntfy.sh allows ~60; margin for other clients on the same IP
    ntfy_budget_refill_seconds: float = 5.0
    ntfy_budget_reserve: int = 10      # tokens heartbeats may not spend, kept for changes
    max_queue: int = 100               # pending changes held before the oldest is dropped
    ntfy_quota_backoff_seconds: float = 600.0  # pause after ntfy.sh reports its daily quota used up


@dataclass(frozen=True)
class AppConfig:
    cameras: tuple[CameraConfig, ...]
    detector: DetectorConfig
    embedder: EmbedderConfig
    database: DatabaseConfig
    enrollment: EnrollmentConfig
    runtime: RuntimeConfig
    web: WebConfig
    processing: ProcessingConfig
    board: BoardConfig


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)

    camera_items: list[CameraConfig] = []
    for item in raw.get("cameras", []):
        source = str(item.get("source", "webcam"))
        endpoint = str(item["url"]) if item.get("url") else None
        if source not in {"webcam", "mjpeg"}:
            raise ValueError("Camera source must be webcam or mjpeg")
        if source == "mjpeg":
            parsed_endpoint = urlparse(endpoint or "")
            if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.netloc:
                raise ValueError("An MJPEG camera requires a valid http(s) URL")
        camera_items.append(CameraConfig(id=str(item["id"]), source=source, url=endpoint))
    cameras = tuple(camera_items)
    if not cameras:
        raise ValueError("At least one camera must be configured")
    if len({camera.id for camera in cameras}) != len(cameras):
        raise ValueError("Camera ids must be unique")

    models = raw["models"]
    det = models["detector"]
    emb = models["embedder"]
    database = raw["database"]
    enrollment = raw.get("enrollment", {})
    runtime = raw.get("runtime", {})
    web = raw.get("web", {})
    processing = raw.get("processing", {})
    board = raw.get("board", {})
    if str(board.get("transport", "http")).lower() not in {"http", "ntfy", "ssh"}:
        raise ValueError("board.transport must be http, ntfy, or ssh")
    if int(board.get("ntfy_budget_burst", 50)) < 1:
        raise ValueError("board.ntfy_budget_burst must be at least 1")
    if float(board.get("ntfy_budget_refill_seconds", 5.0)) <= 0:
        raise ValueError("board.ntfy_budget_refill_seconds must be positive")
    if not 0 <= int(board.get("ntfy_budget_reserve", 10)) < int(board.get("ntfy_budget_burst", 50)):
        raise ValueError("board.ntfy_budget_reserve must be >= 0 and below ntfy_budget_burst")
    if int(board.get("max_queue", 100)) < 1:
        raise ValueError("board.max_queue must be at least 1")
    embedding_dimension = int(emb.get("embedding_dimension", 128))
    if embedding_dimension <= 0:
        raise ValueError("models.embedder.embedding_dimension must be positive")
    input_width = int(emb.get("input_width", 112))
    input_height = int(emb.get("input_height", 112))
    if input_width <= 0 or input_height <= 0:
        raise ValueError("models.embedder input dimensions must be positive")
    image_mean = tuple(float(value) for value in emb.get("image_mean", [0, 0, 0]))
    image_std = tuple(float(value) for value in emb.get("image_std", [1, 1, 1]))
    if len(image_mean) != 3 or len(image_std) != 3 or any(value <= 0 for value in image_std):
        raise ValueError("models.embedder image_mean/image_std must contain three valid values")
    channel_order = str(emb.get("channel_order", "rgb")).lower()
    if channel_order not in {"rgb", "bgr"}:
        raise ValueError("models.embedder.channel_order must be rgb or bgr")
    decision_mode = processing.get("decision_mode", "any_allowed")
    if decision_mode not in {"any_allowed", "all_faces_allowed"}:
        raise ValueError("decision_mode must be any_allowed or all_faces_allowed")

    return AppConfig(
        cameras=cameras,
        detector=DetectorConfig(
            path=_resolve(config_path.parent, det["path"]),
            model_id=str(det.get("model_id", "face_det_lite")),
            landmark_path=(
                _resolve(config_path.parent, str(det["landmark_path"]))
                if det.get("landmark_path")
                else None
            ),
            score_threshold=float(det.get("score_threshold", 0.55)),
            nms_iou_threshold=float(det.get("nms_iou_threshold", 0.30)),
        ),
        embedder=EmbedderConfig(
            path=_resolve(config_path.parent, emb["path"]),
            model_id=str(emb.get("model_id", "mobile_facenet")),
            embedding_dimension=embedding_dimension,
            input_width=input_width,
            input_height=input_height,
            output_name=str(emb["output_name"]) if emb.get("output_name") else None,
            image_mean=image_mean,
            image_std=image_std,
            channel_order=channel_order,
            align_landmarks=bool(emb.get("align_landmarks", False)),
        ),
        database=DatabaseConfig(
            embeddings_path=_resolve(config_path.parent, database["embeddings_path"]),
            cosine_threshold=float(database.get("cosine_threshold", 0.60)),
        ),
        enrollment=EnrollmentConfig(
            minimum_images_per_person=int(enrollment.get("minimum_images_per_person", 3)),
        ),
        runtime=RuntimeConfig(**runtime),
        web=WebConfig(**web),
        processing=ProcessingConfig(**processing),
        board=BoardConfig(
            enabled=bool(board.get("enabled", False)),
            transport=str(board.get("transport", "http")).lower(),
            url=str(board.get("url", "http://SCL-UNOQ05.local:8770")),
            token=str(os.environ.get("VERDICT_TOKEN", board.get("token", ""))),
            ntfy_topic=str(os.environ.get("NTFY_TOPIC", board.get("ntfy_topic", ""))),
            ntfy_base_url=str(board.get("ntfy_base_url", "https://ntfy.sh")),
            scripts_dir=(
                _resolve(config_path.parent, str(board["scripts_dir"]))
                if board.get("scripts_dir")
                else None
            ),
            heartbeat_seconds=float(board.get("heartbeat_seconds", 30.0)),
            min_interval_seconds=float(board.get("min_interval_seconds", 0.5)),
            ntfy_budget_burst=int(board.get("ntfy_budget_burst", 50)),
            ntfy_budget_refill_seconds=float(board.get("ntfy_budget_refill_seconds", 5.0)),
            ntfy_budget_reserve=int(board.get("ntfy_budget_reserve", 10)),
            max_queue=int(board.get("max_queue", 100)),
            ntfy_quota_backoff_seconds=float(board.get("ntfy_quota_backoff_seconds", 600.0)),
        ),
    )
