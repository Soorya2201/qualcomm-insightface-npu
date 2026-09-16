import json

import numpy as np
import pytest

from access_vision.matching import AllowList
from access_vision.runtime import active_strictly_qnn
from access_vision.vision import nms


def test_allow_list_cosine_match():
    allow_list = AllowList(["alice", "bob"], np.array([[1, 0], [0, 1]]), 0.8)
    assert allow_list.match(np.array([0.99, 0.01])).person_id == "alice"
    assert not allow_list.match(np.array([0.7, 0.7])).allowed


def test_allow_list_rejects_query_with_wrong_dimension():
    allow_list = AllowList(["alice"], np.array([[1, 0]]), 0.8)
    with pytest.raises(ValueError, match="Expected a 2-value query embedding"):
        allow_list.match(np.array([1, 0, 0]))


def test_allow_list_loads_multiple_templates_and_legacy_vector(tmp_path):
    path = tmp_path / "embeddings.json"
    path.write_text(
        json.dumps({"alice": [[1, 0], [0.8, 0.2]], "bob": [0, 1]}),
        encoding="utf-8",
    )
    allow_list = AllowList.load(path, threshold=0.8, expected_dimension=2)

    assert allow_list.names == ["alice", "alice", "bob"]
    assert allow_list.embeddings.shape == (3, 2)
    assert allow_list.match(np.array([0.75, 0.25])).person_id == "alice"


def test_nms_keeps_non_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]], dtype=float)
    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5) == [0, 2]


def test_strict_npu_provider_check():
    assert active_strictly_qnn(["QNNExecutionProvider"], "QNNExecutionProvider")
    assert active_strictly_qnn(
        ["QNNExecutionProvider", "CPUExecutionProvider"], "QNNExecutionProvider"
    )
    assert not active_strictly_qnn(["CPUExecutionProvider"], "QNNExecutionProvider")
