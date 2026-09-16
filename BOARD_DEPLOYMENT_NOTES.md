# Board Deployment Notes

These notes capture the live Arduino/Uno Q status-light fix that is intentionally
local-only in `config.toml`.

## Problem Found

The app was detecting unauthorized users, but the board did not turn red or
beep because the active local config had board output disabled:

```toml
[board]
enabled = false
```

With that setting, the application never creates a `BoardNotifier`, so no
verdict JSON reaches the Arduino.

Direct HTTP to the board was also not usable from this laptop during testing:

```text
http://SCL-UNOQ05.local:8770/health
```

failed because `SCL-UNOQ05.local` could not be resolved. That points to mDNS,
network isolation, or hostname visibility, not face recognition.

## Live Fix

Because direct `.local` discovery was unavailable, the active local config was
switched to the ntfy relay that the board had previously been run with:

```toml
[board]
enabled = true
transport = "ntfy"
ntfy_topic = "uno-q-live-ace934e73888e3a6"
ntfy_base_url = "https://ntfy.sh"
url = "http://SCL-UNOQ05.local:8770"
scripts_dir = "uno-q-board/scripts"
heartbeat_seconds = 30.0
min_interval_seconds = 0.5
```

`config.toml` is gitignored on purpose, so copy these settings into the active
runtime config on the machine that runs the app.

## Verification Performed

A manual unauthorized verdict was published to:

```text
https://ntfy.sh/uno-q-live-ace934e73888e3a6
```

ntfy accepted it with HTTP `200 OK`.

If the Arduino still does not turn red or beep after this, the board is likely
listening to a different ntfy topic or its listener process is not running.

## Expected Flow

```text
Access Vision detects face
  -> result status is unauthorized
  -> BoardNotifier converts faces to people[]
  -> NtfyBoardSender publishes JSON to ntfy topic
  -> Arduino listener receives topic message
  -> Uno Q shows red and beeps
```

## Payload Shape

The board expects:

```json
{
  "people": [
    {
      "id": "unknown",
      "status": "unauthorized",
      "box": [10, 10, 80, 80],
      "confidence": 0.0
    }
  ]
}
```

An empty frame sends:

```json
{"people":[]}
```

which should clear the lights.
