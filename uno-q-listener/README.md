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

```bash
adb push verdict_server.py /home/arduino/rpc/
adb shell 'mkdir -p ~/.config/systemd/user'
adb push verdict-light.service /home/arduino/.config/systemd/user/
adb shell 'systemctl --user daemon-reload && systemctl --user enable --now verdict-light'
adb shell 'loginctl enable-linger arduino'
```

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
