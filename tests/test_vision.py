from pathlib import Path

import numpy as np

from access_vision.config import DetectorConfig, EmbedderConfig
from access_vision.vision import FaceDetector, FaceEmbedder, align_face


class _Tensor:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _DetectorSession:
    inputs = [_Tensor("input", [1, 1, 16, 24])]
    outputs = [
        _Tensor("heatmap", [1, 1, 2, 3]),
        _Tensor("bbox", [1, 4, 2, 3]),
    ]

    def run(self, _feeds):
        heatmap = np.full((1, 1, 2, 3), -10.0, dtype=np.float32)
        heatmap[0, 0, 0, 0] = 10.0
        heatmap[0, 0, 1, 2] = 10.0
        bbox = np.ones((1, 4, 2, 3), dtype=np.float32)
        return [heatmap, bbox]


def test_detector_gathers_multiple_bounding_boxes_as_n_by_four():
    detector = FaceDetector(
        _DetectorSession(),
        DetectorConfig(path=Path("unused.onnx"), score_threshold=0.5),
    )
    faces = detector.detect(np.zeros((16, 24, 3), dtype=np.uint8))
    assert len(faces) == 2
    assert all(len(face.xyxy) == 4 for face in faces)


class _YoloDetectorSession:
    inputs = [_Tensor("input", [1, 3, 640, 640])]
    outputs = [_Tensor("output", [1, 2, 16])]

    def run(self, _feeds):
        output = np.zeros((1, 2, 16), dtype=np.float32)
        output[0, 0, :4] = [320, 320, 320, 160]
        output[0, 0, 4] = 0.9
        output[0, 0, 5:15] = [240, 280, 400, 280, 320, 320, 270, 360, 370, 360]
        output[0, 0, 15] = 0.9
        return [output]


def test_yolov5_face_letterbox_and_coordinate_restoration():
    detector = FaceDetector(
        _YoloDetectorSession(),
        DetectorConfig(
            path=Path("unused.onnx"),
            model_id="yolov5_face",
            score_threshold=0.5,
            nms_iou_threshold=0.5,
        ),
    )
    faces = detector.detect(np.zeros((200, 400, 3), dtype=np.uint8))
    assert len(faces) == 1
    assert faces[0].xyxy == (100, 50, 300, 150)
    assert len(faces[0].landmarks or ()) == 5


class _DinoEmbedderSession:
    inputs = [_Tensor("pixel_values", [1, 3, "height", "width"])]
    outputs = [
        _Tensor("last_hidden_state", [1, 201, 3]),
        _Tensor("pooler_output", [1, 3]),
    ]

    def __init__(self):
        self.tensor = None

    def run(self, feeds):
        self.tensor = feeds["pixel_values"]
        return [
            np.zeros((1, 201, 3), dtype=np.float32),
            np.array([[3, 0, 4]], dtype=np.float32),
        ]


def test_dinov3_embedder_selects_pooler_and_normalizes_input():
    session = _DinoEmbedderSession()
    config = EmbedderConfig(
        path=Path("unused.onnx"),
        model_id="dinov3_vits16",
        embedding_dimension=3,
        input_width=224,
        input_height=224,
        output_name="pooler_output",
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )
    embedding = FaceEmbedder(session, config).embed(
        np.full((16, 16, 3), 255, dtype=np.uint8)
    )

    assert session.tensor.shape == (1, 3, 224, 224)
    assert np.allclose(session.tensor, 1.0)
    assert np.allclose(embedding, [0.6, 0.0, 0.8])


class _InsightFaceSession:
    inputs = [_Tensor("input_1", [1, 3, 112, 112])]
    outputs = [_Tensor("output_0", [1, 3])]

    def __init__(self):
        self.tensor = None

    def run(self, feeds):
        self.tensor = feeds["input_1"]
        return [np.array([[1, 0, 0]], dtype=np.float32)]


def test_insightface_embedder_uses_bgr_and_minus_one_to_one_input():
    session = _InsightFaceSession()
    config = EmbedderConfig(
        path=Path("unused.onnx"),
        model_id="insightface_w600k_r50",
        embedding_dimension=3,
        output_name="output_0",
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        channel_order="bgr",
        align_landmarks=True,
    )
    FaceEmbedder(session, config).embed(
        np.full((112, 112, 3), [255, 128, 0], dtype=np.uint8)
    )

    assert np.allclose(
        session.tensor[0, :, 0, 0], [-1.0, 1 / 255, 1.0], atol=1e-6
    )


def test_arcface_alignment_preserves_the_reference_geometry():
    y, x = np.indices((112, 112))
    frame = np.stack((x, y, np.zeros_like(x)), axis=-1).astype(np.uint8)
    landmarks = (
        (38.2946, 51.6963),
        (73.5318, 51.5014),
        (56.0252, 71.7366),
        (41.5493, 92.3655),
        (70.7299, 92.2041),
    )
    aligned = align_face(frame, landmarks)

    assert aligned.shape == (112, 112, 3)
    assert np.mean(np.abs(aligned.astype(np.int16) - frame.astype(np.int16))) < 0.1
