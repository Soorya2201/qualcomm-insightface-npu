"""vlm.py: PNG encoder correctness, crop math, response parsing, the GenieX
HTTP client's request/error shapes, VlmDescriber's async/trigger/cooldown/
failure behavior, and FrameProcessor's per-person wiring into it.

What these tests do NOT cover, because it needs Windows ARM64 + GenieX + a
Snapdragon NPU to check: whether `geniex serve` actually resolves MODEL_ID to
a bundle fetched by scripts/ensure_vlm_model.py. See vlm.py's module
docstring and that script's own docstring for the verification steps once
this runs on the real device.
"""
from __future__ import annotations

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import numpy as np
import pytest

from access_vision.matching import AllowList
from access_vision.pipeline import FrameProcessor
from access_vision.vision import Face
from access_vision.vlm import (
    GenieXVlmClient,
    GenieXVlmError,
    PersonAttributes,
    VlmDescriber,
    encode_png,
    expand_crop,
    image_data_url,
    parse_attributes,
)


# --------------------------------------------------------------------------
# PNG encoder
# --------------------------------------------------------------------------

def test_encode_png_round_trips_through_pillow():
    """Byte-for-byte correctness check against a real decoder, not just
    'did not crash'. Pillow is a test-only dependency -- vlm.py itself never
    imports it, matching the rest of this package's no-Pillow convention."""
    PIL = pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(0)
    for shape in [(1, 1), (5, 7), (64, 48), (112, 112), (3, 300)]:
        image = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
        decoded = np.array(PIL.open(io.BytesIO(encode_png(image))).convert("RGB"))
        assert decoded.shape == image.shape
        assert np.array_equal(decoded, image), f"round-trip mismatch at {shape}"


@pytest.mark.parametrize("bad", [
    np.zeros((5, 5), dtype=np.uint8),       # missing channel dim
    np.zeros((0, 5, 3), dtype=np.uint8),    # zero height
    np.zeros((5, 5, 4), dtype=np.uint8),    # RGBA, not RGB
])
def test_encode_png_rejects_bad_shapes(bad):
    with pytest.raises(ValueError):
        encode_png(bad)


def test_image_data_url_is_a_valid_data_uri():
    url = image_data_url(np.zeros((4, 4, 3), dtype=np.uint8))
    assert url.startswith("data:image/png;base64,")


# --------------------------------------------------------------------------
# expand_crop
# --------------------------------------------------------------------------

def test_expand_crop_is_wider_than_the_tight_box():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    crop = expand_crop(frame, (100, 50, 150, 100), expand=1.0)  # 50x50 box, +100% each side
    assert crop.shape[0] > 50 and crop.shape[1] > 50
    assert crop.shape[0] <= 200 and crop.shape[1] <= 300


def test_expand_crop_clamps_at_frame_edges_instead_of_raising():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    crop = expand_crop(frame, (0, 0, 10, 10), expand=5.0)  # would go far negative unclamped
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert crop.shape[0] <= 200 and crop.shape[1] <= 300


def test_expand_crop_rejects_negative_expand():
    with pytest.raises(ValueError):
        expand_crop(np.zeros((10, 10, 3), dtype=np.uint8), (0, 0, 5, 5), expand=-1)


# --------------------------------------------------------------------------
# Response parsing -- tolerant of a small model not following the format
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,color,glasses,hat", [
    ("OUTFIT_COLOR: blue\nGLASSES: yes\nHAT: no", "blue", True, False),
    ("OUTFIT_COLOR: red and white\nGLASSES: no\nHAT: yes", "red and white", False, True),
    ("outfit_color: Black\nglasses: Yes\nhat: No", "black", True, False),  # case-insensitive
    ("Sure! Here is my answer:\nOUTFIT_COLOR: green\nGLASSES: unknown\nHAT: unknown",
     "green", None, None),  # preamble text before the fields
    ("OUTFIT_COLOR: unknown\nGLASSES: no\nHAT: no", None, False, False),
    ("GLASSES: yes.\nHAT: no!\nOUTFIT_COLOR: navy blue", "navy blue", True, False),  # reordered, punctuated
    ("I cannot determine this from the image.", None, None, None),  # total non-compliance
    ("", None, None, None),
    ("OUTFIT_COLOR: \"purple\"\nGLASSES: 'yes'\nHAT: 'no'", "purple", True, False),  # quoted
    ("OUTFIT_COLOR: gray\nGLASSES: not wearing glasses\nHAT: none visible", "gray", False, False),
])
def test_parse_attributes(text, color, glasses, hat):
    result = parse_attributes(text)
    assert result.outfit_color == color
    assert result.glasses is glasses
    assert result.hat is hat


# --------------------------------------------------------------------------
# GenieXVlmClient against a stub HTTP server standing in for GenieX
# --------------------------------------------------------------------------

class _StubGenieX(BaseHTTPRequestHandler):
    reply_content = "OUTFIT_COLOR: blue\nGLASSES: yes\nHAT: no"

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.server.last_request = json.loads(self.rfile.read(length))  # type: ignore[attr-defined]
        body = json.dumps({"choices": [{"message": {"content": self.reply_content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def stub_geniex():
    server = HTTPServer(("127.0.0.1", 0), _StubGenieX)
    server.last_request = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(2.0)


def test_client_sends_the_documented_genie_x_request_shape(stub_geniex):
    client = GenieXVlmClient(base_url=f"http://127.0.0.1:{stub_geniex.server_port}/v1",
                              model="Qwen2.5-VL-7B-Instruct")
    crop = np.full((20, 20, 3), 128, dtype=np.uint8)
    attributes = client.describe(crop)

    assert attributes == PersonAttributes("blue", True, False, _StubGenieX.reply_content)
    body = stub_geniex.last_request
    assert body["model"] == "Qwen2.5-VL-7B-Instruct"
    content = body["messages"][0]["content"]
    assert [part["type"] for part in content] == ["text", "image_url"]
    assert content[1]["image_url"]["url"] == image_data_url(crop)


def test_client_wraps_connection_failure(monkeypatch):
    client = GenieXVlmClient(base_url="http://127.0.0.1:1/v1", timeout=1.0)
    with pytest.raises(GenieXVlmError, match="could not reach GenieX"):
        client.describe(np.zeros((5, 5, 3), dtype=np.uint8))


def test_client_wraps_unexpected_response_shape():
    class _Malformed(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = b'{"unexpected": "shape"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Malformed)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = GenieXVlmClient(base_url=f"http://127.0.0.1:{server.server_port}/v1")
        with pytest.raises(GenieXVlmError, match="unexpected GenieX response shape"):
            client.describe(np.zeros((5, 5, 3), dtype=np.uint8))
    finally:
        server.shutdown()
        thread.join(2.0)


# --------------------------------------------------------------------------
# VlmDescriber: async, event-triggered, cooldown, failure isolation
# --------------------------------------------------------------------------

_CROP = np.zeros((10, 10, 3), dtype=np.uint8)


def test_only_the_trigger_status_calls_the_client_and_never_blocks():
    calls = []
    results = []
    client = lambda c: calls.append(1) or PersonAttributes("blue", True, False, "raw")
    describer = VlmDescriber(client, lambda pid, a: results.append((pid, a)),
                              trigger_on="unauthorized", cooldown_seconds=0.0)
    started = time.monotonic()
    describer.maybe_describe("allowed", "alice", _CROP)
    describer.maybe_describe("no_face", None, _CROP)
    describer.maybe_describe("unauthorized", "unknown", _CROP)
    elapsed = time.monotonic() - started
    time.sleep(0.3)
    describer.close()

    assert elapsed < 0.05, "maybe_describe must return immediately, never block on the network"
    assert len(calls) == 1, f"only the triggering status should call the client, got {len(calls)}"
    assert results == [("unknown", PersonAttributes("blue", True, False, "raw"))]


def test_cooldown_spaces_calls_at_the_configured_interval():
    """Verifies actual call spacing, not just a total count -- a total count
    alone cannot distinguish correct throttling from a burst-then-pause bug."""
    timestamps = []
    client = lambda c: timestamps.append(time.monotonic()) or PersonAttributes(None, None, None, "")
    describer = VlmDescriber(client, lambda *a: None, cooldown_seconds=0.15, max_queue=100)
    for _ in range(20):
        describer.maybe_describe("unauthorized", "x", _CROP)
        time.sleep(0.01)
    time.sleep(0.5)
    describer.close()

    assert len(timestamps) >= 3
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    assert min(gaps) >= 0.14, f"cooldown violated: gaps={gaps}"


def test_failing_client_does_not_raise_and_does_not_wedge_the_worker():
    def dying(_crop):
        raise RuntimeError("GenieX not running")

    describer = VlmDescriber(dying, lambda *a: None, cooldown_seconds=0.0)
    for index in range(3):
        describer.maybe_describe("unauthorized", f"p{index}", _CROP)  # must never raise
        time.sleep(0.05)
    time.sleep(0.2)
    assert describer._thread.is_alive()
    describer.close()


def test_bad_on_result_callback_does_not_kill_the_worker():
    client = lambda c: PersonAttributes("blue", True, False, "raw")

    def bad_callback(pid, attrs):
        raise ValueError("boom")

    describer = VlmDescriber(client, bad_callback, cooldown_seconds=0.0)
    describer.maybe_describe("unauthorized", "a", _CROP)
    time.sleep(0.1)
    describer.maybe_describe("unauthorized", "b", _CROP)  # worker must still pick this up
    time.sleep(0.1)
    assert describer._thread.is_alive()
    describer.close()


def test_queue_is_bounded_when_the_client_is_stuck():
    release = threading.Event()

    def blocked(_crop):
        release.wait(3.0)
        return PersonAttributes(None, None, None, "")

    describer = VlmDescriber(blocked, lambda *a: None, cooldown_seconds=0.0, max_queue=3)
    for index in range(10):
        describer.maybe_describe("unauthorized", f"p{index}", _CROP)
    assert len(describer._queue) <= 3
    release.set()
    time.sleep(0.3)
    describer.close()


# --------------------------------------------------------------------------
# FrameProcessor integration: per-person trigger through the real process()
# --------------------------------------------------------------------------

def _make_config(crop_expand=0.5):
    return SimpleNamespace(
        processing=SimpleNamespace(
            min_face_size=0, decision_mode="any_allowed",
            alert_cooldown_seconds=999.0, log_interval_seconds=999.0,
        ),
        embedder=SimpleNamespace(align_landmarks=False),
        vlm=SimpleNamespace(crop_expand=crop_expand),
    )


class _StubDetector:
    def __init__(self, faces):
        self._faces = faces

    def detect(self, frame_rgb):
        return self._faces


class _AlternatingEmbedder:
    """Returns a vector matching the allow-list on the first call, and an
    orthogonal (non-matching) one on every call after -- so the first face
    in a frame comes back authorized and the rest unauthorized, without
    depending on any real embedding model."""

    def __init__(self):
        self.calls = 0

    def embed(self, _crop):
        self.calls += 1
        return (
            np.array([1.0, 0.0], dtype=np.float32)
            if self.calls == 1
            else np.array([0.0, 1.0], dtype=np.float32)
        )


def test_process_calls_describer_once_per_person_with_the_right_status():
    calls = []

    class _RecordingDescriber:
        def maybe_describe(self, status, person_id, crop):
            calls.append((status, person_id, crop.shape))

    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    faces = [Face((10, 10, 40, 40), 0.9), Face((100, 10, 130, 40), 0.9)]
    allow_list = AllowList(["alice"], np.array([[1.0, 0.0]], dtype=np.float32), threshold=0.5)

    processor = FrameProcessor(
        _make_config(), _StubDetector(faces), _AlternatingEmbedder(), allow_list,
        describer=_RecordingDescriber(),
    )
    result = processor.process("cam1", frame)

    assert [f["allowed"] for f in result["faces"]] == [True, False]
    assert len(calls) == 2, f"expected one call per detected person, got {len(calls)}"
    assert calls[0][:2] == ("authorized", "alice")
    assert calls[1][:2] == ("unauthorized", None)
    # crop_expand=0.5 on a 30x30 box must produce something larger, proving
    # the VLM path gets a wider crop than the tight embedding box, not a copy
    # of it.
    assert calls[0][2][0] > 30 and calls[0][2][1] > 30


def test_process_with_no_describer_is_unaffected():
    """The default (describer=None) must cost nothing and change nothing --
    every pre-existing test in this suite already relies on this."""
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    faces = [Face((10, 10, 40, 40), 0.9)]
    allow_list = AllowList(["alice"], np.array([[1.0, 0.0]], dtype=np.float32), threshold=0.5)
    processor = FrameProcessor(_make_config(), _StubDetector(faces), _AlternatingEmbedder(), allow_list)
    result = processor.process("cam1", frame)  # must not raise with describer=None
    assert result["faces"][0]["allowed"] is True
