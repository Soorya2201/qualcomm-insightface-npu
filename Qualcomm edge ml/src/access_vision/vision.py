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


def _optional_nchw(array: np.ndarray | None, channels: int) -> np.ndarray | None:
    if array is None:
        return None
    return _nchw(array, channels)


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


def _resize_pad_rgb(
    frame_rgb: np.ndarray, width: int, height: int, pad_value: int = 0
) -> tuple[np.ndarray, float, int, int]:
    original_h, original_w = frame_rgb.shape[:2]
    scale = min(width / original_w, height / original_h)
    resized_w = max(1, min(width, round(original_w * scale)))
    resized_h = max(1, min(height, round(original_h * scale)))
    pad_left = (width - resized_w) // 2
    pad_top = (height - resized_h) // 2
    canvas = np.full((height, width, 3), pad_value, dtype=np.uint8)
    canvas[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resize_image(
        frame_rgb, resized_w, resized_h
    )
    return canvas, scale, pad_left, pad_top


def _mediapipe_face_anchors() -> np.ndarray:
    """Generate BlazeFace back-model anchors used by Qualcomm MediaPipe-Face."""
    anchors: list[tuple[float, float, float, float]] = []
    for stride, anchors_per_cell in ((16, 2), (32, 6)):
        grid = 256 // stride
        for y in range(grid):
            for x in range(grid):
                cx = (x + 0.5) / grid
                cy = (y + 0.5) / grid
                for _ in range(anchors_per_cell):
                    anchors.append((cx, cy, 1.0, 1.0))
    return np.asarray(anchors, dtype=np.float32).reshape(-1, 2, 2)


MEDIAPIPE_FACE_ANCHORS = _mediapipe_face_anchors()


def _retinaface_priors(width: int, height: int) -> np.ndarray:
    anchors: list[tuple[float, float, float, float]] = []
    for step, min_sizes in zip((8, 16, 32), ((16, 32), (64, 128), (256, 512))):
        feature_h = int(np.ceil(height / step))
        feature_w = int(np.ceil(width / step))
        for y in range(feature_h):
            for x in range(feature_w):
                for min_size in min_sizes:
                    anchors.append(
                        (
                            (x + 0.5) * step / width,
                            (y + 0.5) * step / height,
                            min_size / width,
                            min_size / height,
                        )
                    )
    return np.asarray(anchors, dtype=np.float32)


def _sigmoid_scalar(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64), -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def _mesh_arcface_landmarks(mesh: np.ndarray, x1: float, y1: float) -> tuple[tuple[int, int], ...]:
    points = mesh[:, :2].copy()
    points[:, 0] += x1
    points[:, 1] += y1
    left_eye = points[[33, 133]].mean(axis=0)
    right_eye = points[[263, 362]].mean(axis=0)
    nose = points[1]
    left_mouth = points[61]
    right_mouth = points[291]
    return tuple(
        (int(round(x)), int(round(y)))
        for x, y in (left_eye, right_eye, nose, left_mouth, right_mouth)
    )


class FaceDetector:
    def __init__(self, session, config: DetectorConfig, landmark_session=None) -> None:
        self.session = session
        self.landmark_session = landmark_session
        self.config = config
        self.input = session.inputs[0]
        self.height, self.width = _input_hw(self.input.shape, (480, 640))
        if self.config.model_id == "retinaface_yakhyo":
            self.height, self.width = _input_hw(self.input.shape, (640, 640))
        self.last_max_score = 0.0

    def detect(self, frame_rgb: np.ndarray) -> list[Face]:
        if self.config.model_id == "yolov5_face":
            return self._detect_yolov5_face(frame_rgb)
        if self.config.model_id == "mediapipe_face":
            return self._detect_mediapipe_face(frame_rgb)
        if self.config.model_id == "yolox":
            return self._detect_yolox(frame_rgb)
        if self.config.model_id == "retinaface_yakhyo":
            return self._detect_retinaface_yakhyo(frame_rgb)
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
        landmark = next((v for k, v in by_name.items() if "landmark" in k), None)
        heatmap = _nchw(heatmap, 1)
        bbox = _nchw(bbox, 4)
        landmark = _optional_nchw(landmark, 10)

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
        landmarks = None
        if landmark is not None:
            points = landmark[0][:, ys, xs].T
            centers_x = xs.astype(np.float32)[:, None] * stride_x
            centers_y = ys.astype(np.float32)[:, None] * stride_y
            landmarks = np.empty_like(points)
            landmarks[:, 0::2] = centers_x + points[:, 0::2] * stride_x
            landmarks[:, 1::2] = centers_y + points[:, 1::2] * stride_y
        boxes[:, [0, 2]] *= original_w / self.width
        boxes[:, [1, 3]] *= original_h / self.height
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_h - 1)
        if landmarks is not None:
            # Qualcomm face_det_lite emits points as left-mouth, left-eye,
            # right-eye, nose, right-mouth. ArcFace alignment expects
            # left-eye, right-eye, nose, left-mouth, right-mouth.
            landmarks = landmarks[:, [2, 3, 4, 5, 6, 7, 0, 1, 8, 9]]
            landmarks[:, 0::2] *= original_w / self.width
            landmarks[:, 1::2] *= original_h / self.height
            landmarks[:, 0::2] = np.clip(landmarks[:, 0::2], 0, original_w - 1)
            landmarks[:, 1::2] = np.clip(landmarks[:, 1::2], 0, original_h - 1)
        keep = nms(boxes, scores, self.config.nms_iou_threshold)
        return [
            Face(
                tuple(int(round(x)) for x in boxes[index]),
                float(scores[index]),
                (
                    tuple(
                        (int(round(points[offset])), int(round(points[offset + 1])))
                        for offset in range(0, 10, 2)
                    )
                    if landmarks is not None
                    else None
                ),
            )
            for index, points in (
                (index, landmarks[index] if landmarks is not None else None)
                for index in keep
            )
        ]

    def _detect_mediapipe_face(self, frame_rgb: np.ndarray) -> list[Face]:
        original_h, original_w = frame_rgb.shape[:2]
        image, scale, pad_left, pad_top = _resize_pad_rgb(
            frame_rgb, self.width, self.height, pad_value=0
        )
        tensor = _for_layout(image.astype(np.float32) / 255.0, self.input.shape)
        raw = self.session.run({self.input.name: tensor})
        by_name = {meta.name: value for meta, value in zip(self.session.outputs, raw)}
        coords = np.concatenate(
            (
                np.asarray(by_name.get("box_coords_1", raw[0]), dtype=np.float32),
                np.asarray(by_name.get("box_coords_2", raw[1]), dtype=np.float32),
            ),
            axis=1,
        )
        scores = np.concatenate(
            (
                np.asarray(by_name.get("box_scores_1", raw[2]), dtype=np.float32),
                np.asarray(by_name.get("box_scores_2", raw[3]), dtype=np.float32),
            ),
            axis=1,
        ).reshape(-1)
        scores = _sigmoid_scalar(scores)
        coords = coords.reshape(-1, 8, 2)
        anchors = MEDIAPIPE_FACE_ANCHORS
        offset = anchors[:, 0:1, :] * np.asarray([self.width, self.height], dtype=np.float32)
        decoded = coords * anchors[:, 1:2, :] + offset * (
            np.arange(coords.shape[1])[:, None] != 1
        )
        flat = decoded.reshape(decoded.shape[0], -1)
        boxes = np.column_stack(
            (
                flat[:, 0] - flat[:, 2] / 2,
                flat[:, 1] - flat[:, 3] / 2,
                flat[:, 0] + flat[:, 2] / 2,
                flat[:, 1] + flat[:, 3] / 2,
            )
        )
        keypoints = flat[:, 4:].reshape(-1, 6, 2)
        self.last_max_score = float(np.max(scores)) if scores.size else 0.0
        selected = scores >= self.config.score_threshold
        if not np.any(selected):
            return []
        boxes = boxes[selected]
        keypoints = keypoints[selected]
        scores = scores[selected]
        keep = nms(boxes, scores, self.config.nms_iou_threshold)[:4]
        faces: list[Face] = []
        for index in keep:
            box = boxes[index].copy()
            points = keypoints[index].copy()
            box[[0, 2]] = (box[[0, 2]] - pad_left) / scale
            box[[1, 3]] = (box[[1, 3]] - pad_top) / scale
            points[:, 0] = (points[:, 0] - pad_left) / scale
            points[:, 1] = (points[:, 1] - pad_top) / scale
            box[[0, 2]] = np.clip(box[[0, 2]], 0, original_w - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, original_h - 1)
            points[:, 0] = np.clip(points[:, 0], 0, original_w - 1)
            points[:, 1] = np.clip(points[:, 1], 0, original_h - 1)
            x1, y1, x2, y2 = box
            landmarks = (
                (points[0] + points[1]) / 2,
                (points[2] + points[3]) / 2,
                points[4],
                points[5],
                points[5],
            )
            if self.landmark_session is not None and x2 > x1 and y2 > y1:
                refined = self._mediapipe_mesh_landmarks(frame_rgb, box)
                if refined is not None:
                    landmarks = refined
            faces.append(
                Face(
                    tuple(int(round(value)) for value in box),
                    float(scores[index]),
                    tuple((int(round(x)), int(round(y))) for x, y in landmarks),
                )
            )
        return faces

    def _mediapipe_mesh_landmarks(
        self, frame_rgb: np.ndarray, box: np.ndarray
    ) -> tuple[tuple[int, int], ...] | None:
        if self.landmark_session is None:
            return None
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        side = max(w, h) * 1.35
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        crop_x1 = int(max(0, round(cx - side / 2)))
        crop_y1 = int(max(0, round(cy - side / 2)))
        crop_x2 = int(min(frame_rgb.shape[1], round(cx + side / 2)))
        crop_y2 = int(min(frame_rgb.shape[0], round(cy + side / 2)))
        crop = frame_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            return None
        landmark_input = self.landmark_session.inputs[0]
        lh, lw = _input_hw(landmark_input.shape, (192, 192))
        resized = resize_image(crop, lw, lh).astype(np.float32) / 255.0
        raw = self.landmark_session.run(
            {landmark_input.name: _for_layout(resized, landmark_input.shape)}
        )
        by_name = {meta.name: value for meta, value in zip(self.landmark_session.outputs, raw)}
        score = float(np.asarray(by_name.get("scores", raw[0]), dtype=np.float32).reshape(-1)[0])
        if score < 0.5:
            return None
        mesh = np.asarray(by_name.get("landmarks", raw[1]), dtype=np.float32).reshape(468, 3)
        mesh[:, 0] *= (crop_x2 - crop_x1)
        mesh[:, 1] *= (crop_y2 - crop_y1)
        return _mesh_arcface_landmarks(mesh, crop_x1, crop_y1)

    def _detect_yolox(self, frame_rgb: np.ndarray) -> list[Face]:
        original_h, original_w = frame_rgb.shape[:2]
        image, scale, pad_left, pad_top = _resize_pad_rgb(
            frame_rgb, self.width, self.height, pad_value=0
        )
        tensor = _for_layout(image.astype(np.float32) / 255.0, self.input.shape)
        raw = self.session.run({self.input.name: tensor})
        by_name = {meta.name: value for meta, value in zip(self.session.outputs, raw)}
        boxes = np.asarray(by_name.get("boxes", raw[0]), dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(by_name.get("scores", raw[1]), dtype=np.float32).reshape(-1)
        classes = np.asarray(by_name.get("class_idx", raw[2])).reshape(-1)
        person = classes == 0
        self.last_max_score = float(np.max(scores[person])) if np.any(person) else 0.0
        selected = person & (scores >= self.config.score_threshold)
        if not np.any(selected):
            return []
        boxes = boxes[selected]
        scores = scores[selected]
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_h - 1)
        return [
            Face(tuple(int(round(value)) for value in boxes[index]), float(scores[index]))
            for index in nms(boxes, scores, self.config.nms_iou_threshold)
        ]

    def _detect_retinaface_yakhyo(self, frame_rgb: np.ndarray) -> list[Face]:
        original_h, original_w = frame_rgb.shape[:2]
        image_rgb, scale, pad_left, pad_top = _resize_pad_rgb(
            frame_rgb, self.width, self.height, pad_value=0
        )
        image_bgr = image_rgb[..., ::-1].astype(np.float32)
        image_bgr -= np.asarray((104.0, 117.0, 123.0), dtype=np.float32)
        tensor = _for_layout(image_bgr, self.input.shape)
        raw = self.session.run({self.input.name: tensor})
        outputs = [np.asarray(value, dtype=np.float32) for value in raw]
        loc_output = next((value for value in outputs if value.shape[-1] == 4), None)
        conf_output = next((value for value in outputs if value.shape[-1] == 2), None)
        landmark_output = next((value for value in outputs if value.shape[-1] == 10), None)
        if loc_output is None or conf_output is None or landmark_output is None:
            raise ValueError(f"Unexpected RetinaFace outputs: {[value.shape for value in outputs]}")
        loc = loc_output.reshape(-1, 4)
        conf = conf_output.reshape(-1, 2)
        landmarks = landmark_output.reshape(-1, 10)
        scores = conf[:, 1]
        priors = _retinaface_priors(self.width, self.height)
        if loc.shape[0] != priors.shape[0]:
            raise ValueError(
                f"RetinaFace priors/output mismatch: priors={priors.shape}, loc={loc.shape}"
            )
        boxes = np.empty_like(loc)
        boxes[:, :2] = priors[:, :2] + loc[:, :2] * 0.1 * priors[:, 2:]
        boxes[:, 2:] = priors[:, 2:] * np.exp(loc[:, 2:] * 0.2)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        boxes *= np.asarray([self.width, self.height, self.width, self.height], dtype=np.float32)

        points = priors[:, None, :2] + landmarks.reshape(-1, 5, 2) * 0.1 * priors[:, None, 2:]
        points *= np.asarray([self.width, self.height], dtype=np.float32)
        points = points.reshape(-1, 10)

        self.last_max_score = float(np.max(scores)) if scores.size else 0.0
        selected = scores >= self.config.score_threshold
        if not np.any(selected):
            return []
        boxes = boxes[selected]
        points = points[selected]
        scores = scores[selected]
        if len(scores) > 1000:
            top = np.argpartition(scores, -1000)[-1000:]
            boxes, points, scores = boxes[top], points[top], scores[top]
        order = np.argsort(scores)[::-1]
        boxes, points, scores = boxes[order], points[order], scores[order]
        keep = nms(boxes, scores, self.config.nms_iou_threshold)[:10]
        boxes = boxes[keep]
        points = points[keep]
        scores = scores[keep]

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
        points[:, 0::2] = (points[:, 0::2] - pad_left) / scale
        points[:, 1::2] = (points[:, 1::2] - pad_top) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_h - 1)
        points[:, 0::2] = np.clip(points[:, 0::2], 0, original_w - 1)
        points[:, 1::2] = np.clip(points[:, 1::2], 0, original_h - 1)

        return [
            Face(
                tuple(int(round(value)) for value in box),
                float(score),
                tuple(
                    (int(round(face_points[index])), int(round(face_points[index + 1])))
                    for index in range(0, 10, 2)
                ),
            )
            for box, score, face_points in zip(boxes, scores, points)
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
