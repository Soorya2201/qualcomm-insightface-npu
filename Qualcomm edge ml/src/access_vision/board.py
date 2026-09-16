"""Push authorization state to the Arduino Uno Q status light.

Why this does not hook AlertPipeline.sink
-----------------------------------------
`sink.emit()` fires only when a frame is UNAUTHORIZED, and only after the alert
cooldown. It is an alert channel. The board is a state display: it needs green
too, and it needs to clear when people leave. Driving it from the sink would
produce a light that can only ever turn red. So this consumes the return value
of `process()` instead -- every frame, every status.

Why the send happens on a worker thread
---------------------------------------
Reaching the board spawns an ssh process: TCP + key exchange + remote python
startup, on the order of 300ms-1s. Frames arrive every `frame_interval_ms`
(300ms by default) and inference itself is ~2ms. A synchronous send would stall
the capture loop and queue up backlog. The worker holds a single slot: while a
send is in flight, newer states overwrite the pending one, so the board always
converges on the latest truth instead of replaying a stale queue.

Failure policy
--------------
The light is an indicator, not the security decision. If the board is
unreachable the recognition pipeline keeps running and logging; the outage is
logged once, not once per frame. The board's own firmware is fail-closed
(unknown input shows red), which is the correct direction for the display.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)

# Board shows 8 LEDs; sending more is harmless but pointless.
MAX_PEOPLE = 8


def to_board_payload(result: dict[str, Any]) -> dict[str, Any]:
    """access_vision `process()` result -> the Uno Q contract.

    The board reads `people[].status` and sorts LEDs by `box[0]`. Our bbox is
    xyxy; the contract is [x, y, w, h].
    """
    people = []
    for face in result.get("faces", []):
        x1, y1, x2, y2 = face["bbox"]
        person_id = face.get("person_id") or "unknown"
        # send_to_board.send() rejects payloads containing a single quote,
        # because it embeds the JSON in a remote shell heredoc.
        person_id = str(person_id).replace("'", "")
        people.append({
            "id": person_id,
            "status": "authorized" if face.get("allowed") else "unauthorized",
            "box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            "confidence": float(face.get("cosine_similarity", 0.0)),
        })
    people.sort(key=lambda p: p["box"][0])
    return {"people": people[:MAX_PEOPLE]}


def state_key(payload: dict[str, Any]) -> tuple:
    """What counts as a change worth spending an ssh round-trip on.

    Identity and verdict only -- not the box, or every pixel of movement would
    look like a new state and we would send continuously.
    """
    return tuple((p["id"], p["status"]) for p in payload["people"])


class BoardNotifier:
    """Non-blocking bridge from the pipeline to the status light."""

    def __init__(
        self,
        sender: Callable[[dict], str],
        heartbeat_seconds: float = 30.0,
        min_interval_seconds: float = 0.5,
    ) -> None:
        self._sender = sender
        self._heartbeat = heartbeat_seconds
        self._min_interval = min_interval_seconds

        self._pending: dict | None = None
        self._pending_key: tuple | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()

        self._last_key: tuple | None = None
        self._last_sent = 0.0
        self._failing = False

        self._thread = threading.Thread(target=self._run, name="board-notifier", daemon=True)
        self._thread.start()

    def update(self, result: dict[str, Any]) -> None:
        """Call with every `process()` result. Returns immediately."""
        if result.get("status") == "no_face":
            payload = {"people": []}
        else:
            payload = to_board_payload(result)
        key = state_key(payload)

        now = time.monotonic()
        changed = key != self._last_key
        stale = now - self._last_sent >= self._heartbeat
        if not changed and not stale:
            return

        with self._lock:
            # Single slot: a newer state replaces an unsent older one.
            self._pending = payload
            self._pending_key = key
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            with self._lock:
                payload, key = self._pending, self._pending_key
                self._pending = self._pending_key = None
            if payload is None:
                continue

            # The slot may still hold a state that finished sending while it sat
            # here (update() compares against _last_key, which only advances once
            # a send completes). Drop it unless the heartbeat is actually due.
            if key == self._last_key and time.monotonic() - self._last_sent < self._heartbeat:
                continue

            # Rate limit even genuine changes; the light cannot usefully show
            # more than a couple of transitions per second.
            since = time.monotonic() - self._last_sent
            if since < self._min_interval:
                time.sleep(self._min_interval - since)

            try:
                summary = self._sender(payload)
            except Exception as exc:  # noqa: BLE001 - a dead light must not stop recognition
                if not self._failing:
                    LOGGER.warning("Board unreachable, continuing without it: %s", exc)
                    self._failing = True
                continue

            if self._failing:
                LOGGER.info("Board reachable again")
                self._failing = False
            self._last_key = key
            self._last_sent = time.monotonic()
            LOGGER.debug("Board updated people=%d -> %s", len(payload["people"]), summary)

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=timeout)


def build_notifier(scripts_dir: str | None = None, **kwargs) -> BoardNotifier | None:
    """Wire up send_to_board.send, or return None if the board tooling is absent."""
    import sys
    if scripts_dir and scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        from send_to_board import send  # type: ignore
    except ImportError:
        LOGGER.info("send_to_board not importable; board output disabled")
        return None
    return BoardNotifier(sender=send, **kwargs)
