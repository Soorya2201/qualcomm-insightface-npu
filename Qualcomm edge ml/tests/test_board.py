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
