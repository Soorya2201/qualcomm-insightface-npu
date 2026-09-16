"""Push authorization state to the Arduino Uno Q status light.

Why this does not hook AlertPipeline.sink
-----------------------------------------
`sink.emit()` fires only when a frame is UNAUTHORIZED, and only after the alert
cooldown. It is an alert channel. The board is a state display: it needs green
too, and it needs to clear when people leave. Driving it from the sink would
produce a light that can only ever turn red. So this consumes the return value
of `process()` instead -- every frame, every status.

Two kinds of send
-----------------
* A CHANGE is a frame whose people/verdicts differ from the last state queued.
  Changes go into a FIFO queue and are delivered in order; a failed change is
  put back at the head and retried, never dropped. This is what carries the
  red light and the beep, so losing one is the failure that matters.
* A HEARTBEAT re-sends the current state at most every `min_interval_seconds`,
  and only while frames are still arriving. It proves the relay is flowing.
  A failed heartbeat is not retried: the next frame produces a fresh one.

Changes always go before heartbeats.

Why there is a send budget (ntfy only)
--------------------------------------
Public ntfy.sh enforces two limits per client IP, both answered with HTTP 429:

* a request rate: a burst of 60, then one request every 5 seconds. Measured
  at one publish per second: 81 succeeded, then only one in four did.
* a DAILY MESSAGE QUOTA of 250 (documented; hit during testing, error code
  42908 "daily message quota reached"). When it is exhausted the relay refuses
  every publish -- changes included -- until the quota resets.

`_SendBudget` mirrors the request-rate limit so rate-limited requests are not
sent. Heartbeats may only spend tokens above `budget_reserve`, which keeps
tokens in hand so a change goes out immediately. A 429 that still slips through
(another client on the same IP, or a server-side count that differs from ours)
drains the local budget to resynchronize.

The daily quota cannot be budgeted around at a one-second heartbeat: 250
messages last roughly 17 minutes at the rate limit. When the relay reports the
quota exhausted, the notifier logs it once at ERROR, stops publishing for
`quota_backoff_seconds` instead of retrying into a wall, and keeps pending
changes queued. The durable fixes are a transport without a quota (direct HTTP
to the board) or a paid ntfy tier.

Why the send happens on a worker thread
---------------------------------------
A send is a network round-trip (hundreds of milliseconds to seconds). Frames
arrive every `frame_interval_ms` and inference is a few milliseconds.
`update()` only records state and returns; the worker owns the network.

Failure policy
--------------
The light is an indicator, not the security decision. An unreachable board
never raises into the recognition pipeline. The outage is logged at WARNING
when it starts and at most once a minute while it continues.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)

# Board shows 8 LEDs; sending more is harmless but pointless.
MAX_PEOPLE = 8

# How often a continuing problem (outage, queue overflow, budget throttling)
# is re-logged. Once at onset, then at most this often.
_RELOG_SECONDS = 60.0


class RateLimitedError(RuntimeError):
    """The relay refused a send because this client exceeded its request rate."""


class QuotaExhaustedError(RateLimitedError):
    """The relay refused a send because this client's daily message quota is used up."""


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
    """Identity and verdict only -- not the box, or every pixel of movement
    would count as a change."""
    return tuple((p["id"], p["status"]) for p in payload["people"])


def _describe_payload(payload: dict[str, Any]) -> str:
    """Human-readable summary of what is being sent, for the INFO-level log line."""
    people = payload.get("people", [])
    if not people:
        return "no people (lights cleared)"
    authorized = sum(1 for p in people if p.get("status") == "authorized")
    denied = len(people) - authorized
    ids = ", ".join(f"{p.get('id', '?')}:{p.get('status', '?')}" for p in people)
    return f"{len(people)} people ({authorized} authorized, {denied} denied) [{ids}]"


class _SendBudget:
    """Client-side token bucket mirroring the relay's per-client request limit."""

    def __init__(self, burst: int, refill_seconds: float) -> None:
        if burst < 1:
            raise ValueError("budget burst must be at least 1")
        if refill_seconds <= 0:
            raise ValueError("budget refill_seconds must be positive")
        self.burst = float(burst)
        self.refill_seconds = float(refill_seconds)
        self._tokens = float(burst)
        self._stamp = time.monotonic()

    def _refill(self, now: float) -> None:
        if now > self._stamp:
            self._tokens = min(self.burst, self._tokens + (now - self._stamp) / self.refill_seconds)
        self._stamp = now

    def available(self, now: float) -> float:
        self._refill(now)
        return self._tokens

    def take(self, now: float) -> bool:
        self._refill(now)
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def seconds_until(self, level: float, now: float) -> float:
        self._refill(now)
        return max(0.0, (level - self._tokens) * self.refill_seconds)

    def drain(self, now: float) -> None:
        self._refill(now)
        self._tokens = 0.0


class BoardNotifier:
    """Non-blocking bridge from the pipeline to the status light.

    `heartbeat_seconds` is accepted for compatibility and ignored: the
    heartbeat cadence is `min_interval_seconds`, which is also the minimum gap
    between any two sends.
    """

    def __init__(
        self,
        sender: Callable[[dict], str],
        heartbeat_seconds: float = 30.0,
        min_interval_seconds: float = 0.5,
        budget_burst: int | None = None,
        budget_refill_seconds: float = 5.0,
        budget_reserve: int = 0,
        max_queue: int = 100,
        quota_backoff_seconds: float = 600.0,
    ) -> None:
        del heartbeat_seconds  # see class docstring
        if max_queue < 1:
            raise ValueError("max_queue must be at least 1")
        self._sender = sender
        self._min_interval = max(0.0, float(min_interval_seconds))
        self._budget = (
            _SendBudget(budget_burst, budget_refill_seconds) if budget_burst is not None else None
        )
        if self._budget is not None and not 0 <= budget_reserve < budget_burst:
            raise ValueError("budget_reserve must be >= 0 and smaller than budget_burst")
        self._reserve = int(budget_reserve)
        self._max_queue = int(max_queue)
        self._quota_backoff = max(0.0, float(quota_backoff_seconds))
        self._paused_until = 0.0

        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._queue: collections.deque[tuple[dict, tuple]] = collections.deque()
        self._latest_key: tuple | None = None  # last state queued as a change
        self._current: tuple[dict, tuple] | None = None  # latest frame's state
        self._fresh = False  # a frame arrived since the last send of the current state

        self._last_attempt = 0.0
        self._failing = False
        self._last_failure_logged = 0.0
        self._dropped = 0
        self._last_drop_logged = -_RELOG_SECONDS
        self._last_throttle_logged = -_RELOG_SECONDS

        self._thread = threading.Thread(target=self._run, name="board-notifier", daemon=True)
        self._thread.start()

    @property
    def queued(self) -> int:
        with self._cond:
            return len(self._queue)

    def update(self, result: dict[str, Any]) -> None:
        """Call with every `process()` result. Returns immediately."""
        if result.get("status") == "no_face":
            payload = {"people": []}
        else:
            payload = to_board_payload(result)
        key = state_key(payload)
        dropped_now = 0
        with self._cond:
            self._current = (payload, key)
            self._fresh = True
            if key != self._latest_key:
                if len(self._queue) >= self._max_queue:
                    # Unbounded, a queue this far behind would replay minutes-old
                    # states. Keep the newest; the oldest pending change goes.
                    self._queue.popleft()
                    self._dropped += 1
                    dropped_now = self._dropped
                self._queue.append((payload, key))
                self._latest_key = key
            self._cond.notify()
        if dropped_now:
            now = time.monotonic()
            if now - self._last_drop_logged >= _RELOG_SECONDS:
                LOGGER.warning(
                    "Board queue full (%d pending); dropped the oldest pending change "
                    "(%d dropped so far). The board is falling behind the camera.",
                    self._max_queue, dropped_now,
                )
                self._last_drop_logged = now

    def _next_send_locked(self) -> tuple[tuple[dict, tuple] | None, bool, float]:
        """Pick what to send now: (item, is_heartbeat, seconds_to_wait)."""
        now = time.monotonic()
        if now < self._paused_until:
            return None, False, self._paused_until - now
        gap = self._min_interval - (now - self._last_attempt)
        if gap > 0:
            return None, False, gap

        if self._queue:
            if self._budget is None or self._budget.take(now):
                item = self._queue.popleft()
                if self._current is not None and self._current[1] == item[1]:
                    self._fresh = False
                return item, False, 0.0
            return None, False, self._budget.seconds_until(1.0, now)

        if self._fresh and self._current is not None:
            if self._budget is None:
                self._fresh = False
                return self._current, True, 0.0
            if self._budget.available(now) >= self._reserve + 1 and self._budget.take(now):
                self._fresh = False
                return self._current, True, 0.0
            if now - self._last_throttle_logged >= _RELOG_SECONDS:
                LOGGER.info(
                    "Board heartbeat slowed to one per %.0fs to stay under the relay rate "
                    "limit (%d tokens held back so changes still go out immediately)",
                    self._budget.refill_seconds, self._reserve,
                )
                self._last_throttle_logged = now
            return None, False, self._budget.seconds_until(self._reserve + 1, now)

        return None, False, 1.0

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._cond:
                item, is_heartbeat, wait = self._next_send_locked()
                if item is None:
                    # update() and close() notify, so a new change or a stop
                    # request cuts this wait short.
                    self._cond.wait(timeout=min(max(wait, 0.005), 1.0))
                    continue
                self._last_attempt = time.monotonic()

            payload, _key = item
            try:
                summary = self._sender(payload)
            except Exception as exc:  # noqa: BLE001 - a dead light must not stop recognition
                now = time.monotonic()
                with self._cond:
                    if not is_heartbeat:
                        self._queue.appendleft(item)  # retried, not dropped
                    if self._budget is not None and isinstance(exc, RateLimitedError):
                        self._budget.drain(now)
                    if isinstance(exc, QuotaExhaustedError):
                        self._paused_until = now + self._quota_backoff
                        queued = len(self._queue)
                if isinstance(exc, QuotaExhaustedError):
                    # Not transient: every publish will be refused until the
                    # quota resets. Retrying every few seconds only hides it.
                    LOGGER.error(
                        "Board relay DAILY QUOTA EXHAUSTED: %s. The board will receive "
                        "nothing -- not even changes -- until the quota resets. Pausing "
                        "publishes for %.0fs with %d change(s) queued. Fix: board.transport "
                        "= \"http\" (no quota) or a paid ntfy tier.",
                        exc, self._quota_backoff, queued,
                    )
                    self._failing = True
                    self._last_failure_logged = now
                    continue
                if not self._failing:
                    LOGGER.warning(
                        "Board send FAILED (%s): %s -- continuing without it",
                        _describe_payload(payload), exc,
                    )
                    self._failing = True
                    self._last_failure_logged = now
                elif now - self._last_failure_logged >= _RELOG_SECONDS:
                    LOGGER.warning("Board still unreachable (%s): %s", _describe_payload(payload), exc)
                    self._last_failure_logged = now
                continue

            if self._failing:
                LOGGER.info("Board reachable again")
                self._failing = False
            with self._cond:
                queued = len(self._queue)
                tokens = self._budget.available(time.monotonic()) if self._budget else None
            # INFO, not DEBUG: this is the line that proves a verdict this device
            # computed actually reached the board.
            LOGGER.info(
                "Board updated (%s): %s -> %s [queued=%d%s]",
                "heartbeat" if is_heartbeat else "change",
                _describe_payload(payload),
                summary,
                queued,
                f", budget={tokens:.0f}" if tokens is not None else "",
            )

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=timeout)


class HttpBoardSender:
    """POST verdicts straight to the Uno Q's own listener.

    The board runs verdict_server.py, joins Wi-Fi itself, and drives its own
    lights. Nothing sits in between -- no ssh session, no laptop holding a USB
    cable. One short-lived HTTP request per state change.
    """

    def __init__(self, url: str, token: str = "", timeout: float = 4.0) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def __call__(self, payload: dict) -> str:
        import json as _json
        import urllib.error
        import urllib.request

        body = _json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}/verdict",
            data=body,
            headers={"Content-Type": "application/json", **(
                {"X-Verdict-Token": self.token} if self.token else {}
            )},
            method="POST",
        )
        try:
            # No proxy: the board is a LAN/mDNS address and machine-wide proxies
            # routinely cannot route private addresses.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=self.timeout) as response:
                answer = _json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise RuntimeError("board rejected the token (check VERDICT_TOKEN)") from exc
            raise RuntimeError(f"board returned HTTP {exc.code}") from exc
        return answer.get("summary", "ok")

    def health(self) -> dict:
        import json as _json
        import urllib.request

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"{self.url}/health", timeout=self.timeout) as response:
            return _json.loads(response.read().decode("utf-8"))


def build_http_notifier(url: str, token: str = "", **kwargs) -> BoardNotifier:
    """Notifier that talks to the board's own listener over HTTP."""
    return BoardNotifier(sender=HttpBoardSender(url, token), **kwargs)


class NtfyBoardSender:
    """Publish verdicts to a public ntfy.sh topic instead of reaching the board directly.

    Sidesteps client isolation and NAT entirely: this laptop and the board each
    make an OUTBOUND https connection to ntfy.sh and never try to reach each
    other. This is the fallback for a venue network (hotel/motel/conference)
    that blocks device-to-device traffic but allows internet access.

    The topic name is the only access control ntfy.sh's free tier offers.
    Anyone who knows it can read or write it, so use a long random topic, not
    a guessable one, for anything beyond a demo.
    """

    def __init__(self, topic: str, base_url: str = "https://ntfy.sh", timeout: float = 5.0) -> None:
        self.url = f"{base_url.rstrip('/')}/{topic}"
        self.timeout = timeout

    def __call__(self, payload: dict) -> str:
        import json as _json
        import urllib.error
        import urllib.request

        body = _json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.timeout) as response:
                response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                detail = ""
                try:
                    body = _json.loads(exc.read().decode("utf-8") or "{}")
                    detail = str(body.get("error", ""))
                    code = int(body.get("code", 0))
                except (ValueError, OSError, TypeError):
                    code = 0
                message = f"ntfy publish failed: HTTP 429 {detail}".rstrip()
                # 42908 = "limit reached: daily message quota reached" on ntfy.sh.
                if code == 42908 or "daily message quota" in detail:
                    raise QuotaExhaustedError(message) from exc
                raise RateLimitedError(message) from exc
            raise RuntimeError(f"ntfy publish failed: {exc}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"ntfy publish failed: {exc}") from exc
        return f"published to {self.url}"


def build_ntfy_notifier(topic: str, base_url: str = "https://ntfy.sh", **kwargs) -> BoardNotifier:
    """Notifier that relays through ntfy.sh instead of reaching the board directly."""
    return BoardNotifier(sender=NtfyBoardSender(topic, base_url=base_url), **kwargs)


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
