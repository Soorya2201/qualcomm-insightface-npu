"""Verdict relay via ntfy.sh -- for a venue Wi-Fi that blocks device-to-device
traffic (client isolation) but allows normal internet access.

Runs ON THE BOARD. The board makes ONE outbound https connection to ntfy.sh
and keeps it open; it never accepts an inbound connection from anything, so
NAT and client isolation are both irrelevant -- this side never gets talked
TO, it only reads a stream it opened itself.

    Snapdragon --https POST--> ntfy.sh <--https GET (streaming)-- Uno Q --RPC--> MCU

Standard library only (no pip on the board). Reuses check_auth.verdict/to_mask
and rpc_base so behaviour matches verdict_server.py and check_auth.py exactly.

Usage:
    NTFY_TOPIC=your-long-random-topic python3 ntfy_poller.py
"""
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_auth import PIXELS, to_mask, verdict  # noqa: E402

BASE_URL = os.environ.get("NTFY_BASE_URL", "https://ntfy.sh")
TOPIC = os.environ.get("NTFY_TOPIC", "")
STALE_SECONDS = float(os.environ.get("VERDICT_STALE_SECONDS", "20"))
# Reconnect backoff: the stream WILL drop (Wi-Fi hiccups, ntfy.sh restarts).
RECONNECT_SECONDS = 3.0


def show(people):
    from rpc_base import ArduinoBridge

    bridge = ArduinoBridge()
    try:
        bridge.call("set_people", min(len(people), PIXELS), to_mask(people))
    finally:
        bridge.close()


def apply_payload(raw_message):
    """Same file-based reuse of check_auth's logic as verdict_server.py."""
    handle, path = tempfile.mkstemp(prefix="verdict-", suffix=".json")
    try:
        with os.fdopen(handle, "wb") as f:
            f.write(raw_message if isinstance(raw_message, bytes) else raw_message.encode("utf-8"))
        people, denied, summary = verdict(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    show(people)
    return people, denied, summary


def stream():
    """Yield each ntfy message body as it arrives. Blocks between messages."""
    url = f"{BASE_URL.rstrip('/')}/{TOPIC}/json"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    # A long-lived chunked response; ntfy.sh sends periodic keepalive lines
    # ({"event":"keepalive"}) that we can just ignore.
    with opener.open(url, timeout=90) as response:
        for line in response:
            line = line.strip()
            if not line:
                continue
            yield line


# Shared with the watchdog thread. A list, not a float, so the watchdog sees
# updates the main loop makes without needing a lock for this single field.
_last_verdict = [0.0]
_cleared = [True]


def _watchdog():
    """Runs on its own timer, independent of message arrival.

    A check made only inside the message-handling loop -- as the first
    version of this function did -- only ever runs right after a message
    shows up, so it can never notice that messages have STOPPED arriving.
    Detecting silence requires a clock that ticks whether or not anything
    happens on the stream, which is exactly what a stale sender must trigger.
    """
    while True:
        time.sleep(1.0)
        if STALE_SECONDS <= 0 or not _last_verdict[0] or _cleared[0]:
            continue
        if time.monotonic() - _last_verdict[0] > STALE_SECONDS:
            try:
                show([])
                _cleared[0] = True
                print("stale: no verdict for %.0fs, lights cleared" % STALE_SECONDS, file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                print(f"watchdog could not clear lights: {exc}", file=sys.stderr)


def main():
    if not TOPIC:
        sys.exit("NTFY_TOPIC is not set. Use a long random topic, not a guessable one.")
    print(f"subscribing to {BASE_URL}/{TOPIC} ...", file=sys.stderr)
    threading.Thread(target=_watchdog, name="stale-watchdog", daemon=True).start()

    while True:
        try:
            for line in stream():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("event") != "message":
                    continue  # keepalive / open events carry no payload

                _last_verdict[0] = time.monotonic()
                _cleared[0] = False
                try:
                    people, denied, summary = apply_payload(event.get("message", ""))
                    print(f"{'AUTHORIZED' if people and not denied else 'UNAUTHORIZED'} - {summary}",
                          file=sys.stderr)
                except Exception as exc:  # noqa: BLE001 - one bad message must not kill the poller
                    print(f"could not apply verdict: {exc}", file=sys.stderr)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"stream dropped ({exc}); reconnecting in {RECONNECT_SECONDS}s", file=sys.stderr)
            time.sleep(RECONNECT_SECONDS)


if __name__ == "__main__":
    main()
