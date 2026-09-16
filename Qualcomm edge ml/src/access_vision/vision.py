from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DetectorConfig


@dataclass(frozen=True)
class Face:
    xyxy: tuple[int, int, int, int]
    score: float
    landmarks: tuple[tuple[int, int], ...] | None = None


def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinear HWC resize implemented with NumPy only."""
    source_h, source_w = image.shape[:2]
    if (source_w, source_h) == (width, height):
        return image.copy()
    y = np.linspace(0, source_h - 1, height, dtype=np.float32)
    x = np.linspace(0, source_w - 1, width, dtype=np.float32)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.minimum(y0 + 1, source_h - 1)
    x1 = np.minimum(x0 + 1, source_w - 1)
    wy = (y - y0)[:, None, None]
    wx = (x - x0)[None, :, None]
    top = image[y0[:, None], x0[None, :]] * (1.0 - wx) + image[y0[:, None], x1[None, :]] * wx
    bottom = image[y1[:, None], x0[None, :]] * (1.0 - wx) + image[y1[:, None], x1[None, :]] * wx
    return np.clip(top * (1.0 - wy) + bottom * wy, 0, 255).astype(image.dtype)


def _local_maximum(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, 1, mode="edge")
    neighborhoods = [
        padded[dy : dy + values.shape[0], dx : dx + values.shape[1]]
        for dy in range(3)
        for dx in range(3)
    ]
    return values == np.maximum.reduce(neighborhoods)


def _input_hw(shape: list[int | str | None], default: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 4:
        raise ValueError(f"Expected a 4D image input, got {shape}")
    numeric = [int(value) if isinstance(value, int) else 0 for value in shape]
    if numeric[1] in (1, 3):
        return (numeric[2] or default[0], numeric[3] or default[1])
    return (numeric[1] or default[0], numeric[2] or default[1])


def _for_layout(image_hwc: np.ndarray, shape: list[int | str | None]) -> np.ndarray:
    batch = image_hwc[None, ...].astype(np.float32)
    if len(shape) == 4 and shape[1] in (1, 3):
        return np.transpose(batch, (0, 3, 1, 2))
    return batch


def _nchw(array: np.ndarray, channels: int) -> np.ndarray:
    if array.ndim != 4:
        raise ValueError(f"Expected 4D detector output, got {array.shape}")
    if array.shape[1] == channels:
        return array
    if array.shape[-1] == channels:
        return np.transpose(array, (0, 3, 1, 2))
    raise ValueError(f"Cannot find {channels} channels in output {array.shape}")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    left = np.maximum(box[0], boxes[:, 0])
    top = np.maximum(box[1], boxes[:, 1])
    right = np.minimum(box[2], boxes[:, 2])
    bottom = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    area_a = np.maximum(0, box[2] - box[0]) * np.maximum(0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-8)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        order = remaining[_iou(boxes[current], boxes[remaining]) <= threshold]
    return keep


class FaceDetector:
    def __init__(self, session, config: DetectorConfig) -> None:
        self.session = session
        self.config = config
        self.input = session.inputs[0]
        self.height, self.width = _input_hw(self.input.shape, (480, 640))
        self.last_max_score = 0.0

    def detect(self, frame_rgb: np.ndarray) -> list[Face]:
        if self.config.model_id == "yolov5_face":
            return self._detect_yolov5_face(frame_rgb)
        return self._detect_face_det_lite(frame_rgb)

    def _detect_yolov5_face(self, frame_rgb: np.ndarray) -> list[Face]:
        original_h, original_w = frame_rgb.shape[:2]
        scale = min(self.width / original_w, self.height / original_h)
        resized_w = max(1, min(self.width, round(original_w * scale)))
        resized_h = max(1, min(self.height, round(original_h * scale)))
        pad_left = (self.width - resized_w) // 2
        pad_top = (self.height - resized_h) // 2
        letterboxed = np.full((self.height, self.width, 3), 114, dtype=np.uint8)
        letterboxed[
            pad_top : pad_top + resized_h,
            pad_left : pad_left + resized_w,
        ] = resize_image(frame_rgb, resized_w, resized_h)
        tensor = _for_layout(
            letterboxed.astype(np.float32) / 255.0,
            self.input.shape,
        )
        raw = self.session.run({self.input.name: tensor})

        if len(raw) == 1:
            predictions = np.asarray(raw[0], dtype=np.float32).reshape(-1, 16)
            boxes_xywh = predictions[:, :4]
            scores = predictions[:, 4] * predictions[:, 15]
            landmarks = predictions[:, 5:15]
        else:
            boxes_output = next((value for value in raw if value.shape[-1] == 4), None)
            scores_output = next((value for value in raw if value.shape[-1] == 1), None)
            landmarks_output = next((value for value in raw if value.shape[-1] == 10), None)
            if boxes_output is None or scores_output is None or landmarks_output is None:
                raise ValueError(
                    f"Unexpected YOLOv5-Face outputs: {[value.shape for value in raw]}"
                )
            boxes_xywh = np.asarray(boxes_output, dtype=np.float32).reshape(-1, 4)
            scores = np.asarray(scores_output, dtype=np.float32).reshape(-1)
            landmarks = np.asarray(landmarks_output, dtype=np.float32).reshape(-1, 10)

        self.last_max_score = float(np.max(scores)) if scores.size else 0.0
        selected = scores >= self.config.score_threshold
        if not np.any(selected):
            return []
        boxes_xywh = boxes_xywh[selected]
        scores = scores[selected]
        landmarks = landmarks[selected]
        boxes = np.column_stack(
            (
                boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2,
                boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2,
                boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2,
                boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2,
            )
        )
        keep = nms(boxes, scores, self.config.nms_iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        landmarks = landmarks[keep]

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_h - 1)
        landmarks[:, 0::2] = (landmarks[:, 0::2] - pad_left) / scale
        landmarks[:, 1::2] = (landmarks[:, 1::2] - pad_top) / scale
        landmarks[:, 0::2] = np.clip(landmarks[:, 0::2], 0, original_w - 1)
        landmarks[:, 1::2] = np.clip(landmarks[:, 1::2], 0, original_h - 1)

        return [
            Face(
                tuple(int(round(value)) for value in box),
                float(score),
                tuple(
                    (int(round(points[index])), int(round(points[index + 1])))
                    for index in range(0, 10, 2)
                ),
            )
            for box, score, points in zip(boxes, scores, landmarks)
        ]

    def _detect_face_det_lite(self, frame_rgb: np.ndarray) -> list[Face]:
        original_h, original_w = frame_rgb.shape[:2]
        resized = resize_image(frame_rgb, self.width, self.height).astype(np.float32)
        gray = (0.299 * resized[..., 0] + 0.587 * resized[..., 1] + 0.114 * resized[..., 2]) / 255.0
        tensor = _for_layout(gray[..., None], self.input.shape)
        raw = self.session.run({self.input.name: tensor})
        by_name = {meta.name.lower(): value for meta, value in zip(self.session.outputs, raw)}
        heatmap = next((v for k, v in by_name.items() if "heat" in k), raw[0])
        bbox = next((v for k, v in by_name.items() if "bbox" in k or "box" in k), raw[1])
        heatmap = _nchw(heatmap, 1)
        bbox = _nchw(bbox, 4)

        scores_map = _sigmoid(heatmap[0, 0])
        self.last_max_score = float(np.max(scores_map)) if scores_map.size else 0.0
        local_max = _local_maximum(scores_map)
        ys, xs = np.where(local_max & (scores_map >= self.config.score_threshold))
        if not len(xs):
            return []
        scores = scores_map[ys, xs]
        if len(scores) > 2000:
            top = np.argpartition(scores, -2000)[-2000:]
            xs, ys, scores = xs[top], ys[top], scores[top]

        # Index the spatial dimensions only after selecting the NCHW batch.
        # Combining ``:`` with the advanced ``ys/xs`` indices moves NumPy's
        # advanced-index axis to the front, producing (4, N) for some N values.
        distances = bbox[0][:, ys, xs].T
        stride_x = self.width / scores_map.shape[1]
        stride_y = self.height / scores_map.shape[0]
        boxes = np.column_stack(
            ((xs - distances[:, 0]) * stride_x, (ys - distances[:, 1]) * stride_y,
             (xs + distances[:, 2]) * stride_x, (ys + distances[:, 3]) * stride_y)
        )
        boxes[:, [0, 2]] *= original_w / self.width
        boxes[:, [1, 3]] *= original_h / self.height
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_h - 1)
        return [
            Face(tuple(int(x) for x in boxes[index]), float(scores[index]))
            for index in nms(boxes, scores, self.config.nms_iou_threshold)
        ]


class FaceEmbedder:
    def __init__(self, session, config) -> None:
        self.session = session
        self.config = config
        if len(session.inputs) not in (1, 2):
            raise ValueError("Embedder must have one input, or MobileFaceNet's img1/img2 inputs")
        self.height, self.width = _input_hw(
            session.inputs[0].shape,
            (config.input_height, config.input_width),
        )
        if config.output_name is None:
            self.output_index = 0
        else:
            output_names = [item.name for item in session.outputs]
            if config.output_name not in output_names:
                raise ValueError(
                    f"Embedder output {config.output_name!r} is unavailable; "
                    f"outputs={output_names}"
                )
            self.output_index = output_names.index(config.output_name)

    def embed(self, face_rgb: np.ndarray) -> np.ndarray:
        rgb = resize_image(face_rgb, self.width, self.height)
        ordered = rgb[..., ::-1] if self.config.channel_order == "bgr" else rgb
        normalized = ordered.astype(np.float32) / 255.0
        mean = np.asarray(self.config.image_mean, dtype=np.float32)
        std = np.asarray(self.config.image_std, dtype=np.float32)
        normalized = (normalized - mean) / std
        tensor = _for_layout(normalized, self.session.inputs[0].shape)
        feeds = {item.name: tensor for item in self.session.inputs}
        output = np.asarray(self.session.run(feeds)[self.output_index], dtype=np.float32)
        embedding = output.reshape(output.shape[0], -1)[0] if output.ndim > 1 else output
        norm = float(np.linalg.norm(embedding))
        if norm < 1e-12:
            raise RuntimeError("Embedding model returned a zero vector")
        return embedding / norm


def align_face(
    frame_rgb: np.ndarray,
    landmarks: tuple[tuple[int, int], ...],
    width: int = 112,
    height: int = 112,
) -> np.ndarray:
    """Align a face to the standard five-point ArcFace template."""
    if len(landmarks) < 5:
        raise ValueError("Five landmarks are required for ArcFace alignment")
    source = np.asarray(landmarks[:5], dtype=np.float64)
    template = np.asarray(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float64,
    )
    template[:, 0] *= width / 112.0
    template[:, 1] *= height / 112.0

    system = np.empty((10, 4), dtype=np.float64)
    target = template.reshape(-1)
    for index, (x, y) in enumerate(source):
        system[index * 2] = (x, -y, 1.0, 0.0)
        system[index * 2 + 1] = (y, x, 0.0, 1.0)
    a, b, tx, ty = np.linalg.lstsq(system, target, rcond=None)[0]
    forward = np.asarray(
        [[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    inverse = np.linalg.inv(forward)

    dest_y, dest_x = np.indices((height, width), dtype=np.float64)
    homogeneous = np.stack(
        (dest_x.reshape(-1), dest_y.reshape(-1), np.ones(width * height)), axis=0
    )
    mapped = inverse @ homogeneous
    source_x = mapped[0].reshape(height, width)
    source_y = mapped[1].reshape(height, width)

    frame_h, frame_w = frame_rgb.shape[:2]
    valid = (
        (source_x >= -1e-6)
        & (source_x <= frame_w - 1 + 1e-6)
        & (source_y >= -1e-6)
        & (source_y <= frame_h - 1 + 1e-6)
    )
    source_x = np.clip(source_x, 0, frame_w - 1)
    source_y = np.clip(source_y, 0, frame_h - 1)
    x0 = np.clip(np.floor(source_x).astype(np.int32), 0, frame_w - 1)
    y0 = np.clip(np.floor(source_y).astype(np.int32), 0, frame_h - 1)
    x1 = np.minimum(x0 + 1, frame_w - 1)
    y1 = np.minimum(y0 + 1, frame_h - 1)
    wx = (source_x - x0)[..., None]
    wy = (source_y - y0)[..., None]
    top = frame_rgb[y0, x0] * (1.0 - wx) + frame_rgb[y0, x1] * wx
    bottom = frame_rgb[y1, x0] * (1.0 - wx) + frame_rgb[y1, x1] * wx
    aligned = np.clip(
        np.rint(top * (1.0 - wy) + bottom * wy), 0, 255
    ).astype(np.uint8)
    aligned[~valid] = 0
    return aligned


def prepare_face(frame: np.ndarray, face: Face, embedder_config) -> np.ndarray:
    if embedder_config.align_landmarks and face.landmarks is not None:
        return align_face(
            frame,
            face.landmarks,
            embedder_config.input_width,
            embedder_config.input_height,
        )
    return crop_face(frame, face)


def crop_face(frame: np.ndarray, face: Face, padding: float = 0.05) -> np.ndarray:
    x1, y1, x2, y2 = face.xyxy
    width, height = x2 - x1, y2 - y1
    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, int(x1 - width * padding))
    y1 = max(0, int(y1 - height * padding))
    x2 = min(frame_w, int(x2 + width * padding))
    y2 = min(frame_h, int(y2 + height * padding))
    return frame[y1:y2, x1:x2]
