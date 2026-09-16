# Uno Q verdict listener

Makes the Arduino Uno Q **self-sufficient**. The board joins Wi-Fi, listens on
its own port, and drives its own Pixels and buzzer. No laptop holds a USB cable,
no ssh session, nothing in between:

```
Snapdragon X Elite                          Arduino Uno Q
  phone frames -> NPU face recognition        verdict_server.py (Linux/MPU)
                        |                              |
                        +---- HTTP POST /verdict ----->+
                                                       | RPC set_people
                                                       v
                                              auth_status.ino (MCU)
                                                       |
                                              Modulino Pixels + buzzer
```

Standard library only -- the Uno Q has no pip. The authorization rules are not
reimplemented: this imports `check_auth.verdict` / `to_mask` and `rpc_base`, so
the board behaves identically whether driven by this server, `check_auth.py`, or
`mcp_server.py`.

## Why a listener instead of ssh

Earlier the board was reached by `ssh`/`adb` from whichever machine held it.
That makes the board a passive peripheral of a host. A listener inverts it: the
board is a network device that anything on the LAN can address, which is what
"self-sufficient" requires. It also removes the 300ms-1s ssh handshake per
update -- an HTTP POST on a LAN is single-digit milliseconds.

## Install (once, from the machine with the USB cable)

The `--user` systemd bus needs a runtime directory that does not exist until
the user has logged in at least once *or* lingering is enabled -- without it,
`systemctl --user` fails with `Failed to connect to user scope bus`, and doing
this step out of order is the most common reason the service silently never
starts. Run `enable-linger` **first**, before touching `systemctl --user` at
all -- it needs no root/sudo despite the name:

```bash
adb shell 'loginctl enable-linger arduino'
adb push verdict_server.py /home/arduino/rpc/
adb shell 'mkdir -p ~/.config/systemd/user'
adb push verdict-light.service /home/arduino/.config/systemd/user/
adb shell 'export XDG_RUNTIME_DIR=/run/user/1000; systemctl --user daemon-reload && systemctl --user enable --now verdict-light'
```

Confirmed on real hardware: `systemctl --user` run through a plain
non-interactive `adb shell` command has no `XDG_RUNTIME_DIR` /
`DBUS_SESSION_BUS_ADDRESS` of its own, so it must be exported in the same
command, every time -- an interactive `adb shell` session behaves the same
way unless you export it once at the top of that session too.

Verify it survived: `systemctl --user status verdict-light` should show
`Main PID` and `active (running)`, and stay that way after you close the adb
session -- if the process (`ps aux | grep verdict`) disappears once the
session closes, systemd was never actually managing it and this step needs
redoing.

`MPU/check_auth.py`, `MPU/rpc_base.py` and the `msgpack` folder must already be
in `/home/arduino/rpc/` (see the board project's setup step 7), and the MCU
firmware flashed. After this, unplug from the host -- any USB charger will do.

Set a real token before exposing it:

```bash
adb shell "sed -i 's/CHANGE_ME/your-shared-secret/' ~/.config/systemd/user/verdict-light.service"
adb shell 'systemctl --user daemon-reload && systemctl --user restart verdict-light'
```

## Verify

```bash
curl http://SCL-UNOQ05.local:8770/health
```

```json
{"ok": true, "hostname": "SCL-UNOQ05", "seconds_since_verdict": null,
 "last": "never", "stale_seconds": 20.0, "auth_required": true}
```

Drive it by hand:

```bash
curl -X POST http://SCL-UNOQ05.local:8770/verdict \
  -H 'Content-Type: application/json' -H 'X-Verdict-Token: your-shared-secret' \
  -d '{"people":[{"id":"alice","status":"authorized","box":[40,120,90,200]},
                 {"id":"unknown","status":"unauthorized","box":[180,130,95,210]}]}'
```

Expect LED 0 green, LED 1 red, one beep.

## ntfy relay variant

Use `verdict-light-ntfy.service` instead when the venue network has client
isolation -- see [ntfy_poller.py](ntfy_poller.py) and the section below.
Same install pattern, same linger requirement.

## Endpoints

| | |
|---|---|
| `POST /verdict` | `{"people":[{"status":"authorized","box":[x,y,w,h]}, ...]}`; `401` on a bad token, `400` over 64 KB |
| `GET /health` | liveness, seconds since last verdict, last summary |

## Behaviour that matters

**Fail-closed.** Malformed JSON, a missing `status`, `"unknown"` -- all resolve
to a refusal (red + beep). Verified: a body of `{not json` produces
`set_people(1, 0)`, one red LED. It never defaults to green.

**Stale watchdog.** If no verdict arrives for `VERDICT_STALE_SECONDS`, the
lights blank. A stale green light after the sender crashes is the one thing an
access display must never show. `/health` then reports
`"stale: sender silent, lights cleared"`.

**Serialized MCU access.** `rpc_base` opens one AF_UNIX socket per call and is
not concurrency-safe, so overlapping HTTP requests are serialized behind a lock.

**The token is the only protection.** The server binds `0.0.0.0` by design --
another machine must reach it. Anyone who can reach the port can drive the
light, so do not leave `VERDICT_TOKEN` empty on a shared network. The server
warns loudly at startup if it is unset.

## ntfy relay (for venue Wi-Fi that blocks device-to-device traffic)

`ntfy_poller.py` is the fallback for a network with **client isolation** --
common on hotel/motel/conference guest Wi-Fi, where the router actively
refuses to route traffic between two guest devices even though each device
individually has internet access. `verdict_server.py` needs the laptop to
reach the board directly, which client isolation blocks by design; this file
routes through [ntfy.sh](https://ntfy.sh), a free public pub/sub relay, so
neither side ever talks to the other directly:

```
Snapdragon --https POST--> ntfy.sh <--https GET (streaming)-- Uno Q --RPC--> MCU
```

Both connections are OUTBOUND. The board never accepts an inbound connection
from anything, so NAT and client isolation are both irrelevant -- this side
only ever reads a stream it opened itself.

**This does not fix a captive portal.** The board still needs real outbound
internet to reach ntfy.sh at all; a splash-page login that has not been
completed blocks this exactly as it blocks anything else.

**Verified against the live public service** (not a mock): a message sent
from a real `NtfyBoardSender.__call__()` reached a real `ntfy_poller.py`
subscribed to the same topic over the actual internet, and drove the correct
`set_people` bitmask. Also verified: the stale watchdog runs on its own timer
thread, independent of message arrival -- an earlier version checked staleness
only inside the message loop, which can never detect that messages have
*stopped* arriving, since that code only runs when one arrives.

### Install

```bash
adb push ntfy_poller.py /home/arduino/rpc/
adb shell "sed -i 's/verdict_server.py/ntfy_poller.py/' ~/.config/systemd/user/verdict-light.service"
adb shell "echo 'Environment=NTFY_TOPIC=your-long-random-topic' >> ~/.config/systemd/user/verdict-light.service"
adb shell 'systemctl --user daemon-reload && systemctl --user restart verdict-light'
```

Pick a long random topic, not a guessable one -- ntfy.sh's free tier has no
access control beyond the topic name being hard to guess.

### Send a test verdict from anywhere with internet

```bash
curl -d '{"people":[{"id":"alice","status":"authorized","box":[40,120,90,200]}]}' \
  https://ntfy.sh/your-long-random-topic
```
