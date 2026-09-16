# Arduino Relay Troubleshooting

This note captures the live failure seen during the YOLOv5 Access Vision demo.

## Current observed state

The vision pipeline is alive:

- The server is running at `http://127.0.0.1:8765/`.
- The active detector is `yolov5_face`.
- The active embedder is `insightface_w600k_r50`.
- Logs show unauthorized faces being produced.
- Logs show frame throughput and detector latency.

Example live evidence:

```text
Pipeline mode=live detector=yolov5_face embedder=insightface_w600k_r50
NPU session ready model=yolov5n_face.onnx providers=['QNNExecutionProvider', 'CPUExecutionProvider']
NPU session ready model=w600k_r50_qnn.onnx providers=['QNNExecutionProvider', 'CPUExecutionProvider']
{"camera":"camera-1","tag":"unauthorized", ...}
```

So the face model is not the failing part.

## Where the red beep path is supposed to go

The full path is:

```text
Browser frame
  -> localhost Access Vision server
  -> YOLOv5 face detector
  -> InsightFace embedder
  -> allow-list matching
  -> BoardNotifier
  -> ntfy.sh topic
  -> Uno Q ntfy poller
  -> Uno Q RPC
  -> MCU Pixels and buzzer
```

The Arduino only blinks/beeps after it receives a board payload like this:

```json
{"people":[{"id":"unknown","status":"unauthorized","box":[180,130,95,210]}]}
```

If the payload dies before the Uno Q poller receives it, the model can still
detect an unauthorized face while the board stays silent.

## Exact failure observed

The current failure is at the laptop-to-ntfy publish step:

```text
Board still unreachable (no people (lights cleared)): ntfy publish failed: <urlopen error timed out>
```

Earlier runs also showed:

```text
ntfy publish failed: HTTP Error 429: Too Many Requests
```

That means Access Vision reached the board output code, but the public ntfy
relay did not reliably accept the publish. In that state the chain dies here:

```text
BoardNotifier -> NtfyBoardSender -> https://ntfy.sh/<topic>
                                      ^
                                      failure: timeout or 429 rate limit
```

The downstream Arduino poller has nothing new to read, so it cannot set red or
beep.

## Why this can happen

- The configured board transport is `ntfy`, so every status update depends on
  outbound HTTPS to `https://ntfy.sh`.
- Public ntfy topics can be rate-limited. A one-second repeated state heartbeat
  can trigger `429 Too Many Requests`.
- A timeout means the laptop could not complete the HTTPS publish to ntfy in
  time. This can be Wi-Fi, proxy, DNS, ntfy service delay, or venue-network
  instability.
- If the Uno Q is not subscribed to the same `NTFY_TOPIC`, messages can publish
  successfully and still never reach the board.
- If the Uno Q has no internet, is stuck behind captive portal, or its
  `verdict-light-ntfy` service is down, it will not receive the messages.

## Quick checks

Check that only one Access Vision server owns the live port:

```powershell
netstat -ano | Select-String '127\.0\.0\.1:8765\s+0\.0\.0\.0:0\s+LISTENING'
```

Confirm the page is using YOLOv5:

```powershell
$html = Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8765/
($html.Content | Select-String '"detector": "[^"]+"' -AllMatches).Matches.Value
```

Check whether the laptop can publish one red verdict to the ntfy topic:

```powershell
Invoke-WebRequest -UseBasicParsing -Method Post `
  -Uri https://ntfy.sh/uno-q-live-ace934e73888e3a6 `
  -Body '{"people":[{"id":"unknown","status":"unauthorized","box":[40,120,90,200]}]}' `
  -ContentType 'application/json' `
  -TimeoutSec 10
```

If this times out or returns `429`, the Arduino cannot react because the relay
message is not being accepted.

Check the board-side ntfy service over adb:

```bash
adb shell 'export XDG_RUNTIME_DIR=/run/user/1000; systemctl --user status verdict-light'
adb shell 'export XDG_RUNTIME_DIR=/run/user/1000; journalctl --user -u verdict-light -n 80 --no-pager'
```

Check that the board and laptop are using the same topic:

```powershell
$env:NTFY_TOPIC
```

```bash
adb shell 'export XDG_RUNTIME_DIR=/run/user/1000; systemctl --user show verdict-light -p Environment'
```

## Practical fixes

1. Reduce ntfy pressure.

   In `config.insightface.toml`, increase:

   ```toml
   [board]
   min_interval_seconds = 5.0
   ```

   One second is useful for proving liveness, but it can be too aggressive for
   a public relay during continuous frame processing.

2. Prefer direct board HTTP when the laptop and Uno Q are on the same LAN.

   ```toml
   [board]
   transport = "http"
   url = "http://SCL-UNOQ05.local:8770"
   ```

   This bypasses ntfy entirely. It will not work on networks with client
   isolation, but when it works it is faster and avoids public relay limits.

3. Keep ntfy only for isolated venue networks.

   Use ntfy when the laptop cannot directly reach the board. Make sure the Uno
   Q has working outbound internet and the same long random `NTFY_TOPIC`.

## What is not failing

- YOLOv5 detector startup is working.
- InsightFace QNN embedder startup is working.
- Unauthorized event creation is working.
- Board update code is being reached.

The current failure is the relay transport after board update creation, before
the Uno Q receives the message.
