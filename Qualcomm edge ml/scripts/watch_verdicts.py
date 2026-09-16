"""Live terminal view of every verdict flowing through the ntfy.sh relay.

Run this in its own terminal, on ANY machine with internet -- this laptop,
the Snapdragon, a phone's Termux, doesn't matter. It is a second, independent
subscriber to the same public topic uno-q-listener/ntfy_poller.py listens on:
ntfy.sh broadcasts to every subscriber, so this needs no access to the board
at all, and keeps working even if the board is off or unreachable. It only
watches -- it never drives the lights.

The printed AUTHORIZED/UNAUTHORIZED decision is computed by importing
verdict() from the uno-q-board submodule's check_auth.py, so what you see
here is guaranteed to match what the board itself decides, not a
reimplementation that could quietly drift from it.

Usage:
    NTFY_TOPIC=your-topic python3 scripts/watch_verdicts.py
"""
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "uno-q-board", "MPU"))
try:
    from check_auth import verdict  # noqa: E402
except ImportError:
    sys.exit(
        "Could not import check_auth from the uno-q-board submodule.\n"
        "Run: git submodule update --init --recursive"
    )

BASE_URL = os.environ.get("NTFY_BASE_URL", "https://ntfy.sh")
TOPIC = os.environ.get("NTFY_TOPIC", "")
RECONNECT_SECONDS = 3.0

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def describe(raw_message: str) -> str:
    """Reuse the board's own decision logic on this one message."""
    handle, path = tempfile.mkstemp(prefix="watch-", suffix=".json")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            f.write(raw_message)
        people, denied, summary = verdict(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    if not people:
        return f"{DIM}EMPTY       {summary}{RESET}"
    if denied:
        return f"{RED}UNAUTHORIZED{RESET} {summary}"
    return f"{GREEN}AUTHORIZED  {RESET}{summary}"


def stream():
    url = f"{BASE_URL.rstrip('/')}/{TOPIC}/json"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=90) as response:
        for line in response:
            line = line.strip()
            if line:
                yield line


def main():
    if not TOPIC:
        sys.exit("NTFY_TOPIC is not set. Use the same topic the board is subscribed to.")

    url = f"{BASE_URL.rstrip('/')}/{TOPIC}"
    print(f"watching {url}  (Ctrl+C to stop)", file=sys.stderr)
    print(f"{DIM}waiting for the first message...{RESET}", file=sys.stderr)

    while True:
        try:
            for line in stream():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("event") != "message":
                    continue  # keepalive / open events -- not a real verdict

                ts = time.strftime("%H:%M:%S", time.localtime(event.get("time", time.time())))
                message = event.get("message", "")
                try:
                    desc = describe(message)
                except Exception as exc:  # noqa: BLE001 - one bad line must not kill the watcher
                    desc = f"could not parse ({exc}): {message!r}"
                print(f"[{ts}] {desc}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"{DIM}stream dropped ({exc}); reconnecting in {RECONNECT_SECONDS:.0f}s...{RESET}",
                  file=sys.stderr)
            time.sleep(RECONNECT_SECONDS)
        except KeyboardInterrupt:
            print(file=sys.stderr)
            return


if __name__ == "__main__":
    main()
