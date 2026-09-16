"""Board bridge: format contract, coalescing, and failure isolation."""
from __future__ import annotations

import time

from access_vision.board import BoardNotifier, state_key, to_board_payload


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


def test_repeated_state_is_not_resent():
    sent = []
    notifier = BoardNotifier(lambda p: sent.append(state_key(p)), heartbeat_seconds=999, min_interval_seconds=0.0)
    for _ in range(10):
        notifier.update(_result([("alice", True, (0, 0, 10, 10))]))
        time.sleep(0.02)
    time.sleep(0.2)
    notifier.close()
    assert len(sent) == 1, f"one state should cost one send, got {len(sent)}"


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
