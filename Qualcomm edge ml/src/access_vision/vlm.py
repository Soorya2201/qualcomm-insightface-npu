"""Optional VLM attribute description via GenieX, running a second model on the
Snapdragon NPU alongside the face-recognition pipeline.

Why this exists
----------------
Face recognition answers "who". This answers three fixed questions about
someone the allow-list did not recognize, from one wider crop of the same
frame the detector already found them in:

    1. what color is their outfit?
    2. are they wearing glasses?
    3. are they wearing a hat?

Why GenieX, and which VLM
--------------------------
GenieX is Qualcomm's runtime for *generative* models -- an LLM or VLM, served
locally with an OpenAI-compatible API, typically on http://127.0.0.1:18181/v1.
It is not the runtime for the detector or embedder CNNs in this project; those
are ahead-of-time-compiled QNN context binaries loaded directly through
ONNX Runtime. A VLM needs tokenization, autoregressive decoding, and (for
image input) a vision encoder feeding that decoder -- GenieX's job, not QNN
Execution Provider's. See the root README's "On-device AI" section.

This module targets Qwen2.5-VL-7B-Instruct. One precision correction, made
before writing any download code: Qualcomm's own published AI Hub bundle for
this model is W4A16 (4-bit weights, 16-bit activations) -- there is no W8A8
/ INT8 variant in their catalog for it. W4A16 is not a fallback; it is
smaller and generally faster on the Hexagon NPU than an 8-bit scheme, so it
already serves "runs best on the NPU" better than INT8 would have. MODEL_ID
and scripts/ensure_vlm_model.py both request w4a16 for this reason -- not
because int8 was unavailable to ask for, but because it is not what Qualcomm
ships for this model, and shipping code should request a precision that
actually exists.

A 7B model is real weight on the NPU alongside the detector and embedder
CNNs already running there. possible_optimisations.txt already says not to
run a VLM on every frame, for exactly this contention reason; this module
follows that -- see VlmDescriber below, triggered only on an unauthorized
event, never per frame.

Why this is async and event-triggered, never per-frame
--------------------------------------------------------
A VLM call is a multi-second generation, not a few-millisecond CNN forward
pass. Calling it from the frame loop would stall recognition. VlmDescriber
runs it on a worker thread, triggered only on a state matching `trigger_on`
(unauthorized by default) and rate-limited by `cooldown_seconds`, mirroring
AlertPipeline's own alert cooldown in pipeline.py.

What is verified here vs. what needs the real device
-------------------------------------------------------
The PNG encoder, the crop, the HTTP request shape, and the response parser
are all tested against a stub HTTP server standing in for GenieX -- no NPU
or GenieX install is needed to prove those pieces work. What is NOT verified
here, because it needs Windows ARM64 + GenieX + a Snapdragon NPU to check:
whether `geniex serve` actually resolves the identifier this module sends as
`model` to a bundle fetched by scripts/ensure_vlm_model.py. That script and
this module's MODEL_ID constant must agree with whatever `geniex pull` (or
`qai-hub-models fetch --runtime geniex_qairt`) actually places into GenieX's
own model store on that machine -- confirm this once, on device, per the
verification steps in scripts/ensure_vlm_model.py's own docstring.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import struct
import threading
import time
import zlib
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

LOGGER = logging.getLogger(__name__)

# The AI Hub / GenieX catalog identifier for the model this module targets.
# scripts/ensure_vlm_model.py fetches this exact string; config.py defaults
# to it. Keep the two in sync if this ever changes.
MODEL_ID = "Qwen2.5-VL-7B-Instruct"
# W4A16 is the only quantized precision Qualcomm publishes for this model on
# AI Hub -- see the module docstring. Not a stand-in for int8; it is what
# actually exists, and it beats int8 on NPU size/speed for this model anyway.
MODEL_PRECISION = "w4a16"

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload))


def encode_png(image_rgb: np.ndarray) -> bytes:
    """Minimal, dependency-free RGB -> PNG encoder (no filtering, zlib level 6).

    Written so this module needs nothing beyond numpy and the stdlib -- no
    Pillow, matching the rest of this package (see the root README: "There is
    no OpenCV, Pillow, FFmpeg, or cloud service in the pipeline"). Correctness
    matters more than compression ratio here: one face crop, sent once, after
    an unauthorized event, not a video stream.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"encode_png expects an HxWx3 RGB array, got shape {image_rgb.shape}")
    height, width = image_rgb.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("encode_png got an empty image")
    image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)

    # Filter type 0 (None) per scanline, as PNG requires one filter-type byte
    # per row even when the filter is "no-op".
    filter_bytes = np.zeros((height, 1), dtype=np.uint8)
    raw = np.concatenate([filter_bytes, image_rgb.reshape(height, width * 3)], axis=1).tobytes()

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # bit depth 8, color type 2 = RGB
    idat = zlib.compress(raw, level=6)
    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def image_data_url(image_rgb: np.ndarray) -> str:
    """PNG-encode and wrap as a data: URL, per GenieX's documented image_url formats."""
    return "data:image/png;base64," + base64.b64encode(encode_png(image_rgb)).decode("ascii")


def expand_crop(frame_rgb: np.ndarray, xyxy: tuple[int, int, int, int], expand: float) -> np.ndarray:
    """A wider crop than the tight face box, to capture outfit/headwear.

    `expand` is a fraction of the face box's own width/height added on every
    side, so a face box that is mostly a head gets enough margin below and
    beside it to show shoulders and the top of the head. Reuses the same
    frame-bounds clamping vision.crop_face already relies on.
    """
    if expand < 0:
        raise ValueError("expand must be >= 0")
    x1, y1, x2, y2 = xyxy
    width, height = x2 - x1, y2 - y1
    frame_h, frame_w = frame_rgb.shape[:2]
    x1 = max(0, int(x1 - width * expand))
    y1 = max(0, int(y1 - height * expand))
    x2 = min(frame_w, int(x2 + width * expand))
    y2 = min(frame_h, int(y2 + height * expand))
    return frame_rgb[y1:y2, x1:x2]


PROMPT = """Look at this photo of one person. Answer all three questions, each on its own line, in exactly this format and nothing else:

OUTFIT_COLOR: <the single most prominent color of their clothing, one or two words>
GLASSES: <yes or no>
HAT: <yes or no>

If you cannot tell for a question, answer "unknown" for OUTFIT_COLOR or "unknown" for GLASSES/HAT."""

_YES_WORDS = {"yes", "y", "true", "wearing", "has", "present"}
_NO_WORDS = {"no", "n", "false", "not wearing", "none", "absent"}


@dataclass(frozen=True)
class PersonAttributes:
    outfit_color: str | None
    glasses: bool | None  # None means the model's answer could not be parsed as yes/no
    hat: bool | None
    raw: str  # the model's full response, kept for logging/debugging


def _parse_yes_no(value: str) -> bool | None:
    value = value.strip().lower().strip(".!\"'")
    if value in _YES_WORDS or value.startswith("yes"):
        return True
    if value in _NO_WORDS or value.startswith("no"):
        return False
    return None


def parse_attributes(text: str) -> PersonAttributes:
    """Tolerant of a small VLM not following the format exactly: matches each
    field by keyword anywhere in the response rather than requiring the exact
    three-line layout, and never raises on unexpected output."""
    color_match = re.search(r"OUTFIT_COLOR\s*:\s*(.+)", text, re.IGNORECASE)
    glasses_match = re.search(r"GLASSES\s*:\s*(\S+)", text, re.IGNORECASE)
    hat_match = re.search(r"HAT\s*:\s*(\S+)", text, re.IGNORECASE)

    color = None
    if color_match:
        color = color_match.group(1).strip().strip(".!\"'").lower()
        if color in ("unknown", ""):
            color = None

    return PersonAttributes(
        outfit_color=color,
        glasses=_parse_yes_no(glasses_match.group(1)) if glasses_match else None,
        hat=_parse_yes_no(hat_match.group(1)) if hat_match else None,
        raw=text,
    )


class GenieXVlmError(RuntimeError):
    """The GenieX server could not be reached, or returned an error."""


class GenieXVlmClient:
    """OpenAI-compatible chat-completions call to a local GenieX server.

    Uses urllib from the stdlib rather than the `openai` package, matching
    the rest of access_vision (board.py, server.py): one fewer dependency for
    a single-endpoint call whose request shape is simple and documented.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:18181/v1",
        model: str = MODEL_ID,
        max_tokens: int = 64,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

    def describe(self, image_rgb: np.ndarray, prompt: str = PROMPT) -> PersonAttributes:
        import urllib.error
        import urllib.request

        body = json.dumps({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_rgb)}},
                ],
            }],
            "max_tokens": self.max_tokens,
        }).encode("utf-8")

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # No proxy: this is always localhost.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.timeout) as response:
                answer = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise GenieXVlmError(f"GenieX returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise GenieXVlmError(
                f"could not reach GenieX at {self.base_url} ({exc}). "
                "Is `geniex serve` running?"
            ) from exc

        try:
            text = answer["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GenieXVlmError(f"unexpected GenieX response shape: {answer!r}") from exc
        return parse_attributes(text)


class VlmDescriber:
    """Async, event-triggered bridge from the pipeline to a GenieX VLM call.

    Mirrors board.py's BoardNotifier: update()-style call returns immediately,
    a worker thread owns the (slow) network call, and a failure never raises
    into the recognition pipeline. Unlike the board notifier this is
    fire-and-forget per event rather than a continuous state -- there is no
    single "current attributes", just a stream of (person_id, attributes)
    results delivered to `on_result`.
    """

    def __init__(
        self,
        client: Callable[[np.ndarray], PersonAttributes],
        on_result: Callable[[str | None, PersonAttributes], None],
        trigger_on: str = "unauthorized",
        cooldown_seconds: float = 10.0,
        max_queue: int = 4,
    ) -> None:
        self._client = client
        self._on_result = on_result
        self._trigger_on = trigger_on
        self._cooldown = max(0.0, float(cooldown_seconds))
        self._max_queue = int(max_queue)

        self._cond = threading.Condition()
        self._queue: list[tuple[str | None, np.ndarray]] = []
        self._stop = threading.Event()
        self._last_call = 0.0
        self._dropped = 0
        self._last_error_logged = 0.0

        self._thread = threading.Thread(target=self._run, name="vlm-describer", daemon=True)
        self._thread.start()

    def maybe_describe(self, status: str, person_id: str | None, crop_rgb: np.ndarray) -> None:
        """Call once per detected person; only queues work when `status` matches
        `trigger_on` and returns immediately either way."""
        if status != self._trigger_on:
            return
        with self._cond:
            if len(self._queue) >= self._max_queue:
                self._queue.pop(0)
                self._dropped += 1
                return
            self._queue.append((person_id, crop_rgb))
            self._cond.notify()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._cond:
                while not self._queue and not self._stop.is_set():
                    self._cond.wait(timeout=1.0)
                if self._stop.is_set():
                    return
                since = time.monotonic() - self._last_call
                if since < self._cooldown:
                    remaining = self._cooldown - since
                    self._cond.wait(timeout=remaining)
                    continue
                person_id, crop_rgb = self._queue.pop(0)

            self._last_call = time.monotonic()
            try:
                attributes = self._client(crop_rgb)
            except Exception as exc:  # noqa: BLE001 - a VLM failure must not stop recognition
                now = time.monotonic()
                if now - self._last_error_logged >= 60.0:
                    LOGGER.warning("VLM describe failed for person=%s: %s", person_id, exc)
                    self._last_error_logged = now
                continue
            LOGGER.info(
                "VLM described person=%s outfit_color=%s glasses=%s hat=%s",
                person_id, attributes.outfit_color, attributes.glasses, attributes.hat,
            )
            try:
                self._on_result(person_id, attributes)
            except Exception:  # noqa: BLE001 - a bad callback must not kill the worker
                LOGGER.exception("VLM on_result callback raised for person=%s", person_id)

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=timeout)
