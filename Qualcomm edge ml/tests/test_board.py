"""Board bridge: format contract, coalescing, and failure isolation."""
from __future__ import annotations

import time

import threading

import pytest

from access_vision.board import (
    BoardNotifier,
    QuotaExhaustedError,
    RateLimitedError,
    state_key,
    to_board_payload,
)


def _result(entries):
    return {
        "status": "allowed",
        "camera": "camera-1",
        "faces": [
            {
                "bbox": list(bbox),
                "allowed": allowed,
                "person_id": person_id,
                "cosine_similarity": 0.9,
            }
            for person_id, allowed, bbox in entries
        ],
    }


def test_payload_matches_board_contract():
    payload = to_board_payload(
        _result([("unknown", False, (180, 130, 275, 340)), ("alice", True, (40, 120, 130, 320))])
    )
    people = payload["people"]
    # Sorted by box x so the LEDs line up with the frame.
    assert [p["id"] for p in people] == ["alice", "unknown"]
    assert [p["status"] for p in people] == ["authorized", "unauthorized"]
    # Board contract is [x, y, w, h]; our Face.xyxy is corners.
    assert people[0]["box"] == [40, 120, 90, 200]


def test_unmatched_face_is_unauthorized_not_dropped():
    payload = to_board_payload(_result([(None, False, (10, 10, 60, 60))]))
    assert payload["people"] == [
        {"id": "unknown", "status": "unauthorized", "box": [10, 10, 50, 50], "confidence": 0.9}
    ]


def test_apostrophe_stripped_for_remote_heredoc():
    # send_to_board.send() raises ValueError on a single quote in the payload.
    payload = to_board_payload(_result([("O'Brien", True, (0, 0, 10, 10))]))
    assert payload["people"][0]["id"] == "OBrien"


def test_only_eight_leds_are_addressed():
    payload = to_board_payload(_result([(f"p{i}", True, (i * 10, 0, i * 10 + 5, 5)) for i in range(12)]))
    assert len(payload["people"]) == 8


def test_no_face_clears_the_board():
    sent = []
    notifier = BoardNotifier(lambda p: sent.append(state_key(p)), heartbeat_seconds=999, min_interval_seconds=0.0)
    notifier.update(_result([("alice", True, (0, 0, 10, 10))]))
    time.sleep(0.15)
    notifier.update({"status": "no_face", "camera": "camera-1", "faces": []})
    time.sleep(0.15)
    notifier.close()
    assert sent[-1] == (), "an empty frame must push an empty people list"


def test_repeated_state_is_resent_at_the_board_interval():
    sent = []
    notifier = BoardNotifier(lambda p: sent.append(state_key(p)), heartbeat_seconds=999, min_interval_seconds=0.05)
    started = time.monotonic()
    while time.monotonic() - started < 0.22:
        notifier.update(_result([("alice", True, (0, 0, 10, 10))]))
        time.sleep(0.01)
    time.sleep(0.15)
    notifier.close()
    assert len(sent) >= 3, f"same live state should be resent at the board interval, got {len(sent)}"
    assert all(item == (("alice", "authorized"),) for item in sent)


def test_slow_board_never_blocks_and_converges():
    sent = []

    def slow(payload):
        time.sleep(0.25)
        sent.append(state_key(payload))

    notifier = BoardNotifier(slow, heartbeat_seconds=999, min_interval_seconds=0.0)
    started = time.monotonic()
    for index in range(20):
        notifier.update(_result([("alice" if index < 10 else "bob", True, (0, 0, 10, 10))]))
        time.sleep(0.01)
    elapsed = time.monotonic() - started
    time.sleep(0.9)
    notifier.close()

    assert elapsed < 0.6, "update() must not wait on the sender"
    assert sent[-1] == (("bob", "authorized"),), "board must end on the latest state"
    assert len(sent) <= 3, f"stale states must be dropped, not queued (got {len(sent)})"


def test_dead_board_does_not_raise_into_the_pipeline():
    def dead(payload):
        raise RuntimeError("board not found")

    notifier = BoardNotifier(dead, heartbeat_seconds=999, min_interval_seconds=0.0)
    for index in range(5):
        notifier.update(_result([(f"p{index}", True, (0, 0, 10, 10))]))  # must not raise
        time.sleep(0.02)
    time.sleep(0.2)
    notifier.close()


def test_successful_send_logs_at_info_not_debug(caplog):
    """A working send must be visible at the app's normal log level.

    Buried at DEBUG, the one line proving a verdict actually reached the
    board would never appear during normal operation -- only failures would,
    which makes a healthy system look silent rather than confirmed-working.
    """
    import logging

    notifier = BoardNotifier(lambda p: "ok", heartbeat_seconds=999, min_interval_seconds=0.0)
    with caplog.at_level(logging.INFO, logger="access_vision.board"):
        notifier.update(_result([("alice", True, (0, 0, 10, 10))]))
        time.sleep(0.15)
    notifier.close()

    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("Board updated" in r.message for r in info_records), (
        "a successful send must log at INFO; found: " + repr([r.message for r in caplog.records])
    )


def test_long_outage_stays_visible_past_the_first_warning():
    """The onset warning must not be the only sign of trouble for a long outage."""
    from access_vision import board as board_module

    logged = []

    class _FakeLogger:
        def warning(self, msg, *args):
            logged.append(msg % args if args else msg)

        def info(self, *a, **k):
            pass

    original_logger = board_module.LOGGER
    board_module.LOGGER = _FakeLogger()
    try:
        notifier = BoardNotifier(
            lambda p: (_ for _ in ()).throw(RuntimeError("down")),
            heartbeat_seconds=999,
            min_interval_seconds=0.0,
        )
        clock = {"t": 0.0}
        original_monotonic = board_module.time.monotonic
        board_module.time.monotonic = lambda: clock["t"]
        try:
            notifier.update(_result([("alice", True, (0, 0, 10, 10))]))
            notifier._run_once_for_test = None  # no-op; loop thread drives this
            time.sleep(0.05)  # let the worker take one pass with the fake clock at t=0

            # Advance the fake clock past the 60s re-log threshold and nudge the
            # worker with a fresh (but same-key) update so it takes another pass.
            clock["t"] = 61.0
            notifier.update(_result([("alice", True, (0, 0, 10, 10))]))
            time.sleep(0.05)
        finally:
            board_module.time.monotonic = original_monotonic
        notifier.close()
    finally:
        board_module.LOGGER = original_logger

    onset = [m for m in logged if "FAILED" in m]
    still_down = [m for m in logged if "still unreachable" in m]
    assert onset, f"expected an onset warning; got {logged}"
    assert still_down, f"a 60s+ outage must re-log, not go silent after the first warning; got {logged}"


def _ids(key):
    return [person_id for person_id, _status in key]


def test_every_change_is_delivered_in_order_while_the_board_is_slow():
    """The queue holds every change: nothing between A and D is skipped."""
    sent = []

    def slow(payload):
        time.sleep(0.1)
        sent.append(state_key(payload))

    notifier = BoardNotifier(slow, min_interval_seconds=0.0)
    for person in ("a", "b", "c", "d"):
        notifier.update(_result([(person, person != "c", (0, 0, 10, 10))]))
        time.sleep(0.01)
    time.sleep(0.8)
    notifier.close()

    assert [_ids(key) for key in sent] == [["a"], ["b"], ["c"], ["d"]]
    assert sent[2] == (("c", "unauthorized"),), "the red state must not be skipped"


def test_failed_change_is_retried_not_dropped():
    calls = {"n": 0}
    sent = []

    def flaky(payload):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("timed out")
        sent.append(state_key(payload))

    notifier = BoardNotifier(flaky, min_interval_seconds=0.0)
    notifier.update(_result([("unknown", False, (0, 0, 10, 10))]))
    time.sleep(0.3)
    notifier.close()

    assert calls["n"] == 3, f"expected two failures then a success, got {calls['n']} attempts"
    assert sent == [(("unknown", "unauthorized"),)]


def test_heartbeats_stop_when_frames_stop():
    """A heartbeat proves the pipeline is running; no frames means no heartbeat."""
    sent = []
    notifier = BoardNotifier(lambda p: sent.append(state_key(p)), min_interval_seconds=0.02)
    started = time.monotonic()
    while time.monotonic() - started < 0.15:
        notifier.update(_result([("alice", True, (0, 0, 10, 10))]))
        time.sleep(0.01)
    time.sleep(0.1)
    count_after_frames_stop = len(sent)
    time.sleep(0.3)
    notifier.close()

    assert count_after_frames_stop >= 3
    assert len(sent) == count_after_frames_stop


def test_budget_caps_sends_at_the_relay_limit():
    sent = []
    notifier = BoardNotifier(
        lambda p: sent.append(state_key(p)),
        min_interval_seconds=0.0,
        budget_burst=3,
        budget_refill_seconds=0.2,
        budget_reserve=0,
    )
    started = time.monotonic()
    while time.monotonic() - started < 1.0:
        notifier.update(_result([("alice", True, (0, 0, 10, 10))]))
        time.sleep(0.01)
    notifier.close()

    # Burst of 3 plus one refill per 0.2s over 1.0s = 8; +1 for timing slack.
    # Without the budget this loop would send roughly 100 times.
    assert 3 <= len(sent) <= 9, f"budget exceeded: {len(sent)} sends"


def test_reserved_tokens_let_a_change_through_immediately():
    sent = []
    notifier = BoardNotifier(
        lambda p: sent.append((time.monotonic(), state_key(p))),
        min_interval_seconds=0.0,
        budget_burst=4,
        budget_refill_seconds=60.0,  # no refill during the test
        budget_reserve=2,
    )
    started = time.monotonic()
    while time.monotonic() - started < 0.2:
        notifier.update(_result([("alice", True, (0, 0, 10, 10))]))
        time.sleep(0.01)
    heartbeats_before_change = len(sent)

    changed_at = time.monotonic()
    notifier.update(_result([("unknown", False, (0, 0, 10, 10))]))
    time.sleep(0.15)
    notifier.close()

    # 1 change + 1 heartbeat, then heartbeats stop to protect the 2 reserved tokens.
    assert heartbeats_before_change == 2, f"heartbeats spent into the reserve: {sent}"
    delivered_at, key = sent[-1]
    assert key == (("unknown", "unauthorized"),)
    assert delivered_at - changed_at < 0.1, "a change must not wait for a refill"


def test_rate_limited_reply_drains_the_budget_and_keeps_the_change():
    calls = {"n": 0}

    def limited(payload):
        calls["n"] += 1
        raise RateLimitedError("HTTP Error 429: Too Many Requests")

    notifier = BoardNotifier(
        limited,
        min_interval_seconds=0.0,
        budget_burst=10,
        budget_refill_seconds=60.0,
        budget_reserve=0,
    )
    notifier.update(_result([("unknown", False, (0, 0, 10, 10))]))
    time.sleep(0.3)
    queued = notifier.queued
    notifier.close()

    assert calls["n"] == 1, "after a 429 the budget must drain, not keep hammering the relay"
    assert queued == 1, "the refused change must stay queued for retry"


def test_queue_is_bounded_and_keeps_the_newest_changes():
    release = threading.Event()
    sent = []

    def blocked(payload):
        release.wait(2.0)
        sent.append(state_key(payload))

    notifier = BoardNotifier(blocked, min_interval_seconds=0.0, max_queue=3)
    for index in range(10):
        notifier.update(_result([(f"p{index}", True, (0, 0, 10, 10))]))
        time.sleep(0.01)
    release.set()
    time.sleep(0.3)
    notifier.close()

    assert [_ids(key)[0] for key in sent] == ["p0", "p7", "p8", "p9"]


def test_reserve_must_be_below_burst():
    with pytest.raises(ValueError):
        BoardNotifier(lambda p: "ok", budget_burst=5, budget_reserve=5)


def test_ntfy_sender_maps_http_429_to_rate_limited():
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from access_vision.board import NtfyBoardSender

    class _Limited(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(429)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Limited)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    try:
        sender = NtfyBoardSender("topic", base_url=f"http://127.0.0.1:{server.server_port}")
        with pytest.raises(RateLimitedError) as caught:
            sender({"people": []})
        assert not isinstance(caught.value, QuotaExhaustedError), "a plain 429 is a rate limit, not the daily quota"
    finally:
        thread.join(2.0)
        server.server_close()


def test_daily_quota_is_logged_as_error_pauses_publishing_and_keeps_changes(caplog):
    import logging

    calls = {"n": 0}

    def exhausted(payload):
        calls["n"] += 1
        raise QuotaExhaustedError("ntfy publish failed: HTTP 429 limit reached: daily message quota reached")

    notifier = BoardNotifier(exhausted, min_interval_seconds=0.0, quota_backoff_seconds=60.0)
    with caplog.at_level(logging.INFO, logger="access_vision.board"):
        notifier.update(_result([("unknown", False, (0, 0, 10, 10))]))
        time.sleep(0.3)
        notifier.update(_result([("alice", True, (0, 0, 10, 10))]))  # another change arrives
        time.sleep(0.2)
    queued = notifier.queued
    notifier.close()

    assert calls["n"] == 1, "publishing must pause, not retry into an exhausted quota"
    assert queued == 2, "changes must stay queued while paused"
    errors = [r.message for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and "DAILY QUOTA EXHAUSTED" in errors[0], errors


def test_ntfy_sender_maps_daily_quota_body_to_quota_error():
    import json
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from access_vision.board import NtfyBoardSender

    body = json.dumps({
        "code": 42908,
        "http": 429,
        "error": "limit reached: daily message quota reached; increase your limits with a paid plan",
    }).encode()

    class _Quota(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Quota)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    try:
        sender = NtfyBoardSender("topic", base_url=f"http://127.0.0.1:{server.server_port}")
        with pytest.raises(QuotaExhaustedError, match="daily message quota"):
            sender({"people": []})
    finally:
        thread.join(2.0)
        server.server_close()
