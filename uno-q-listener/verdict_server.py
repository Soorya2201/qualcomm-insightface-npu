"""Standalone verdict listener for the Arduino Uno Q Linux side (MPU).

Runs ON THE BOARD. The board joins Wi-Fi, listens, and drives its own lights.
No laptop holds a cable, opens an ssh session, or sits in the path:

    Snapdragon (face recognition) --HTTP POST--> Uno Q --RPC--> MCU --> Pixels + buzzer

Standard library only: the Uno Q has no pip. The authorization logic is not
reimplemented here -- it imports check_auth.verdict/to_mask and rpc_base, so the
board behaves identically whether driven by this server, check_auth.py on the
command line, or mcp_server.py.

Install (see README.md):
    /home/arduino/rpc/verdict_server.py
    systemctl --user enable --now verdict-light

Endpoints:
    POST /verdict   {"people":[{"status":"authorized","box":[x,y,w,h]}, ...]}
    GET  /health    liveness + seconds since the last verdict
"""
import json
import os
import socket
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_auth import PIXELS, to_mask, verdict  # noqa: E402

PORT = int(os.environ.get("VERDICT_PORT", "8770"))
# Shared secret. Anyone who can reach the port can drive the door light, so
# this is not optional on a venue network. Set the same value on the sender.
TOKEN = os.environ.get("VERDICT_TOKEN", "")
# If no verdict arrives for this long, blank the lights. A stale green light
# after the sender dies is the one failure this display must never show.
STALE_SECONDS = float(os.environ.get("VERDICT_STALE_SECONDS", "20"))
MAX_BODY = 64 * 1024

# rpc_base opens one AF_UNIX socket per call and is not concurrency-safe;
# ThreadingHTTPServer can deliver overlapping requests, so serialize.
_bridge_lock = threading.Lock()
_last_verdict = [0.0]
_last_summary = ["never"]


def show(people):
    """Drive the MCU exactly the way check_auth.py and mcp_server.py do."""
    from rpc_base import ArduinoBridge

    with _bridge_lock:
        bridge = ArduinoBridge()
        try:
            bridge.call("set_people", min(len(people), PIXELS), to_mask(people))
        finally:
            bridge.close()


def apply_payload(raw):
    """Write the payload to a temp file and reuse check_auth's file-based logic.

    Going through a file looks indirect, but it keeps exactly one implementation
    of "what counts as authorized" on this board, including its fail-closed
    handling of malformed input.
    """
    handle, path = tempfile.mkstemp(prefix="verdict-", suffix=".json")
    try:
        with os.fdopen(handle, "wb") as f:
            f.write(raw)
        people, denied, summary = verdict(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    show(people)
    _last_verdict[0] = time.monotonic()
    _last_summary[0] = summary
    return {
        "ok": True,
        "people": len(people),
        "denied": denied,
        "summary": summary,
        "authorized": bool(people) and denied == 0,
    }


def _ids(raw):
    """' [alice:authorized, unknown:unauthorized]' for the log line, or ''."""
    try:
        people = json.loads(raw.decode("utf-8")).get("people") or []
        parts = ["%s:%s" % (p.get("id", "?"), p.get("status", "?")) for p in people if isinstance(p, dict)]
    except (ValueError, AttributeError, UnicodeDecodeError):
        return ""
    return " [%s]" % ", ".join(parts) if parts else ""


def _watchdog():
    """Blank the lights when the sender goes quiet."""
    cleared = True
    while True:
        time.sleep(1.0)
        if STALE_SECONDS <= 0:
            continue
        age = time.monotonic() - _last_verdict[0]
        if _last_verdict[0] and age > STALE_SECONDS and not cleared:
            try:
                show([])  # empty -> every LED off
                _last_summary[0] = "stale: sender silent, lights cleared"
                print("STALE - no verdict for %.0fs, lights cleared" % STALE_SECONDS, file=sys.stderr)
                cleared = True
            except Exception as exc:  # noqa: BLE001
                print("watchdog could not clear lights: %s" % exc, file=sys.stderr)
        elif age <= STALE_SECONDS:
            cleared = False


class Handler(BaseHTTPRequestHandler):
    server_version = "UnoQVerdict/1.0"

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        if not TOKEN:
            return True
        return self.headers.get("X-Verdict-Token", "") == TOKEN

    def do_GET(self):
        if self.path.split("?")[0] != "/health":
            self._json({"error": "not_found"}, 404)
            return
        age = time.monotonic() - _last_verdict[0] if _last_verdict[0] else None
        self._json({
            "ok": True,
            "hostname": socket.gethostname(),
            "seconds_since_verdict": round(age, 1) if age is not None else None,
            "last": _last_summary[0],
            "stale_seconds": STALE_SECONDS,
            "auth_required": bool(TOKEN),
        })

    def do_POST(self):
        if self.path.split("?")[0] != "/verdict":
            self._json({"error": "not_found"}, 404)
            return
        if not self._authorized():
            self._json({"error": "unauthorized"}, 401)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json({"error": "bad_length"}, 400)
            return
        if length <= 0 or length > MAX_BODY:
            self._json({"error": "bad_length"}, 400)
            return

        raw = self.rfile.read(length)
        try:
            result = apply_payload(raw)
        except Exception as exc:  # noqa: BLE001
            # Never 500 silently: the caller needs to know the light is wrong.
            print("verdict failed: %s" % exc, file=sys.stderr)
            self._json({"ok": False, "error": str(exc)}, 500)
            return
        # The decision itself, in the same form ntfy_poller.py logs, so the
        # journal says what the lights now show -- not just "POST 200".
        if not result["people"]:
            tag = "EMPTY"
        elif result["authorized"]:
            tag = "AUTHORIZED"
        else:
            tag = "UNAUTHORIZED"
        print("%s - %s%s (from %s)" % (tag, result["summary"], _ids(raw), self.address_string()),
              file=sys.stderr)
        self._json(result)

    def log_request(self, code="-", size="-"):
        # A successful verdict already produced the decision line above, and
        # health checks are routine; logging them too would bury the verdicts.
        # Refusals and errors (401, 400, 404, 500) are always logged.
        path = self.path.split("?")[0]
        if str(code) == "200" and path in ("/verdict", "/health"):
            return
        super().log_request(code, size)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main():
    threading.Thread(target=_watchdog, name="stale-watchdog", daemon=True).start()
    # 0.0.0.0 on purpose: the whole point is that another machine reaches this
    # board directly. TOKEN is what protects it, not the bind address.
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("verdict server on 0.0.0.0:%d host=%s token=%s stale=%.0fs"
          % (PORT, socket.gethostname(), "yes" if TOKEN else "NO", STALE_SECONDS),
          file=sys.stderr)
    if not TOKEN:
        print("WARNING: VERDICT_TOKEN is unset; anyone on this network can drive the light",
              file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
