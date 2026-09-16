# Snapdragon Access Vision

A local face allow-list prototype for Snapdragon X Elite. A browser captures
webcam frames or decodes enrollment photos and sends raw RGBA pixels to a
localhost-only Python service. Face detection and embedding inference execute
through ONNX Runtime's QNN HTP/NPU backend. NumPy handles preprocessing and
cosine matching.

There is no OpenCV, Pillow, FFmpeg, or cloud service in the pipeline.

## Models

The current adapters support:

- [YOLOv5-Face](https://aihub.qualcomm.com/compute/models/yolov5_face)
  (`yolov5_face`), 640x640 RGB input, aspect-ratio-preserving letterboxing,
  face boxes, and five facial landmarks. The current `config.toml` selects it.
- [Lightweight-Face-Detection](https://aihub.qualcomm.com/compute/models/face_det_lite)
  (`face_det_lite`), retained as a legacy 480x640 grayscale adapter.
- [MobileFaceNet](https://aihub.qualcomm.com/compute/models/mobile_facenet)
  (`mobile_facenet`), 112x112 RGB input and 128-value output.
- [CavaFace](https://aihub.qualcomm.com/compute/models/cavaface)
  (`cavaface`), 112x112 RGB input and 512-value output. The preserved
  `config.cavaface.toml` selects this model.
- InsightFace `w600k_r50`/ArcFace (`insightface_w600k_r50`), using the
  Snapdragon X Elite W8A16 QNN context from
  [qualcomm-insightface-npu](https://github.com/Soorya2201/qualcomm-insightface-npu).
  The default `config.toml` selects this model with YOLO five-point alignment,
  112x112 BGR input, `(x - 127.5) / 127.5`, and a normalized 512-value output.
- [DINOv3 ViT-S/16](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
  (`dinov3_vits16`), 224x224 ImageNet-normalized RGB input and a normalized
  384-value `pooler_output`. The experimental `config.dinov3.toml` selects it
  with a separate embedding database.

Download and extract the QNN-compatible ONNX Runtime assets into `models/`:

```powershell
qai-hub-models fetch mobile_facenet -r onnx -p w8a16 -o models
```

Qualcomm does not directly distribute YOLOv5-Face exports because of its GPL
license. This project currently uses the GPL `yolov5n_face.onnx` export from
[yakhyo/yolov5-face-onnx-inference](https://github.com/yakhyo/yolov5-face-onnx-inference),
stored at `models/yolov5n-face-onnx-float/yolov5n_face.onnx`. The adapter also
accepts Qualcomm Workbench exports with separate `boxes`, `scores`, and
`landmarks` outputs.

The checked-in `config.toml` points to the directories produced by these
commands. Keep each asset's `metadata.json` beside its ONNX and external `.data`
file; the application uses it to quantize inputs and dequantize outputs. QNN/HTP
is required as the primary execution provider. ONNX Runtime may retain small
input/output quantization wrapper nodes on CPU; the neural-network partition
runs through QNN on the NPU.

Other Snapdragon X Elite models worth trying include MediaPipe Face Detection,
CavaFace, HRNetFace, and YOLOv5. They require model-specific
preprocessing/output adapters and are not drop-in replacements. Re-enroll all
identities after changing an embedding model or its preprocessing.

The included InsightFace-derived weights are restricted by the upstream model
terms to non-commercial research use. The repository's pipeline code is MIT,
but that does not relicense the model weights. Obtain appropriate permission or
replace the weights before commercial deployment.

### DINOv3 ViT-S experiment

The experimental pipeline is YOLOv5-Face detection, largest-face crop, DINOv3
ViT-S pooler embedding, and cosine matching. It does not overwrite the CavaFace
database used by `config.toml`.

Authenticate Workbench and compile the downloaded ONNX for Snapdragon X Elite:

```powershell
.\.venv-workbench314\Scripts\qai-hub.exe configure --api_token YOUR_TOKEN
.\.venv-workbench314\Scripts\python.exe scripts\compile_dinov3_vits16_workbench.py
```

Until the compiled model is downloaded, `config.dinov3.toml` uses the source
ONNX, which has been validated through the local QNN execution-provider path.
After compilation, point its embedder `path` to the downloaded Workbench model.

Enroll and run the experiment with:

```powershell
python -m access_vision.enroll --config config.dinov3.toml
python -m access_vision.cli --config config.dinov3.toml
```

## Install

The prebuilt `onnxruntime-qnn` package currently requires native Windows ARM64
Python 3.11.x:

```powershell
Set-Location C:\Users\qc_de\Qualcomm_Hackathon
py install 3.11-arm64
py -V:3.11-arm64 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -c "import platform,sys; print(sys.version); print(platform.machine())"
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -e .
```

The verification command must report Python `3.11.x` and `ARM64`.

## Enroll allowed people

Arrange 3-4 clear, single-person photos per identity:

```text
data/
  allowed_people/
    alice/
      front.jpg
      left.jpg
      right.jpg
      different-light.jpg
```

Start enrollment:

```powershell
python -m access_vision.enroll --config config.toml
```

The browser opens `http://127.0.0.1:8765/`. Choose the `allowed_people` folder
with the folder picker and click **Generate embeddings**. The browser decodes
the images and sends raw pixels only to localhost. Python detects the largest
face in each image, generates and normalizes one NPU embedding per accepted
photo, then writes all templates to
`data/allowed_embeddings_insightface_w600k_r50.json`. Live
matching uses the highest cosine similarity across every stored template.

The previous CavaFace database remains available. To enroll or run it, replace
`config.toml` in the commands with `config.cavaface.toml`.

An existing database is left unchanged if no identity has the configured
minimum number of accepted images. Stop the enrollment server with `Ctrl+C`
after the browser reports that the database was saved.

Enrollment is a one-time operation for each embedder and preprocessing
configuration. Normal live startup only loads the configured JSON database; it
does not regenerate embeddings. Separate configuration files keep the
InsightFace, CavaFace, and DINO databases independent and reusable.

## Run the webcam

```powershell
python -m access_vision.cli --config config.toml
```

Open `http://127.0.0.1:8765/` if it does not open automatically. Choose either
the local `camera-1 (webcam)` or configured `camera-2 (mjpeg)`, then click
**Start monitoring**. Grant camera permission when using the webcam. The
remote MJPEG stream is proxied through this localhost service, allowing the
browser to decode frames without OpenCV or cross-origin canvas access. The
surveillance dashboard draws a green box with the identity and cosine score
for an allowed face, or a red `UNAUTHORIZED` box for an unknown face. It also
shows the recording state, camera/time overlay, processed FPS, inference
latency, current detections, rolling security events, and raw JSON. The browser uses
`navigator.mediaDevices.getUserMedia()` and posts frames to the local service.
No camera stream leaves the laptop.

If an MJPEG endpoint is offline or unreachable, the page reports a camera
connection error and waits to be restarted instead of attempting to draw an
empty image to the canvas.

Unauthorized detections are printed as newline-delimited JSON:

```json
{"camera":"camera-1","tag":"unauthorized","timestamp":"2026-09-15T18:22:31.120000+00:00","faces":[{"bbox":[101,52,238,241],"allowed":false,"person_id":null,"cosine_similarity":0.4123}]}
```

No event is produced for an empty frame. In `any_allowed` mode, an event is
produced when faces are present but none match. Use `all_faces_allowed` to flag
a frame whenever any detected face is unknown. Alerts are rate-limited by the
configured cooldown.

At `INFO` level, startup logs report the active detector, embedder, model paths,
QNN providers, model I/O, model load time, database path, identity count,
template count, and embedding dimension. Every five seconds, a performance log
reports processed FPS plus average total, detector, and embedding latency for
each camera. Use `--log-level DEBUG` to log every processed frame.

When `[board]` is enabled, every send to the board logs at `INFO`, e.g.:

    INFO Board updated: 2 people (1 authorized, 1 denied) [alice:authorized, unknown:unauthorized] -> published to https://ntfy.sh/your-topic

This is deliberate: a working send is the one line that proves a verdict this
device computed actually reached the board, so it must be visible at the
default level, not only failures. An unreachable board logs once at `WARNING`
when it first fails, then again at most once a minute while the outage
continues, so a long outage stays visible rather than disappearing after the
first line; `INFO Board reachable again` marks recovery.

A dated summary of the models, pipeline, completed work, performance, and
commands is maintained in [PROJECT_LOG.txt](PROJECT_LOG.txt). The remaining
performance ideas are recorded in
[possible_optimisations.txt](possible_optimisations.txt).
If the face model is producing unauthorized detections but the Uno Q does not
blink red or beep, use
[ARDUINO_RELAY_TROUBLESHOOTING.md](ARDUINO_RELAY_TROUBLESHOOTING.md) to locate
where the board notification path is failing.

## Configuration

[config.toml](config.toml) contains:

- Camera IDs and sources. Use `source = "webcam"` for the laptop camera, or
  `source = "mjpeg"` plus an HTTP(S) `url` for an MJPEG endpoint.
- Local host, port, frame size, and capture interval.
- Model IDs, paths, channel order, landmark alignment, normalization, and
  output embedding dimensions. Re-enroll after switching models.
- Allow-list path and cosine threshold.
- Enrollment minimum image count.
- Strict QNN/HTP runtime settings.

The application registers the installed `onnxruntime-qnn` plugin, discovers its
NPU device, and attaches that device explicitly to both model sessions. The
bundled HTP backend is selected by the NPU device; `runtime.performance_mode`
controls each inference run. Persistent context generation is disabled by
default so repeated launches do not collide with an existing `_ctx.onnx` file.

Keep `web.host` set to `127.0.0.1`. The application rejects non-local binds
because it receives raw camera pixels.

The InsightFace configuration starts at a `0.36` cosine threshold. Calibrate it
using genuine and impostor pairs from your actual cameras, lighting, angles,
and distances. Face embeddings are biometric data; protect the JSON file and
establish appropriate consent and deletion rules.

## Verification

```powershell
python -c "import access_vision; print(access_vision.__version__)"
python -m pip install -e ".[dev]"
python -m pytest -q
```

Expected project version: `0.2.0`.

## Status light and buzzer (Arduino Uno Q)

Frames come in from the phone over MJPEG and are decoded in the browser; only a
small JSON verdict leaves this laptop, and only to the Uno Q, which shows one
LED per detected person and beeps once per refusal.

```
phone (MJPEG) ──► browser ──raw pixels──► localhost service ──► YOLOv5-Face ──► InsightFace (NPU)
                                                                                      │
                                                                             verdict JSON
                                                                                      ▼
                                                              BoardNotifier ──ssh──► Uno Q ──► Pixels + buzzer
```

The board tooling is vendored as a submodule, so clone with:

```powershell
git clone --recurse-submodules https://github.com/Soorya2201/qualcomm-insightface-npu
```

Enable it in the config:

```toml
[board]
enabled = true
scripts_dir = "../uno-q-board/scripts"
heartbeat_seconds = 30.0
min_interval_seconds = 0.5
```

### Why it is driven from `process()` and not from the alert sink

`AlertPipeline.sink.emit()` fires only when a frame is UNAUTHORIZED, and only
past `alert_cooldown_seconds`. It is an alert channel. The board is a state
display: it needs green too, and it needs to clear when people leave. A board
wired to the sink could only ever turn red. `BoardNotifier.update()` therefore
consumes the return value of `process()` on every frame.

### Why the send is asynchronous

A send is a network round-trip — hundreds of milliseconds over the relay, up
to a second over `ssh` — while inference takes a few milliseconds.
`update()` only records state and returns; a worker thread owns the network.

It sends two kinds of message:

- **Changes** — someone appears, leaves, or flips authorized/unauthorized.
  Changes are queued and delivered **in order**, ahead of heartbeats, and a
  failed change is put back and retried rather than dropped. This is what
  carries the red light and the beep. The queue is bounded (`max_queue`); if the
  board falls that far behind, the oldest pending change is dropped with a
  warning rather than replaying minutes-old states.
- **Heartbeats** — the current state, re-sent at most every
  `min_interval_seconds` while frames keep arriving, as live proof the relay is
  flowing. When frames stop, heartbeats stop.

With `transport = "ntfy"` a send budget mirrors ntfy.sh's request-rate limit and
the relay's daily quota is detected explicitly; see
[ARDUINO_RELAY_TROUBLESHOOTING.md](ARDUINO_RELAY_TROUBLESHOOTING.md).

For the legacy `ssh` transport, cut the per-call handshake by reusing one
connection — add to `~/.ssh/config`:

```
Host SCL-UNOQ05.local
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```

### Failure policy

The light is an indicator, not the security decision. If the board is
unreachable the pipeline keeps running; the outage is logged at WARNING when it
starts and at most once a minute while it continues, and an exhausted relay
quota is logged once at ERROR. The board's firmware is fail-closed on its own side: unreadable input,
bad JSON, a missing `status` and `"unknown"` all show red.

### Before it can work

Both are owned by whoever holds the board and cannot be fixed from this laptop:

1. Your SSH **public** key installed on the board (`ssh-keygen -t ed25519`, send
   the `.pub` line).
2. Board and laptop on the same Wi-Fi, client isolation off, no VPN.

Verify first:

```powershell
cd ..\uno-q-board
$env:BOARD_TARGET="net"; python scripts/check_link.py; Remove-Item Env:\BOARD_TARGET
```

Expect `CONNECTED (over ssh (SCL-UNOQ05.local))`.

## Preflight

`config.toml` is gitignored, so the committed `config.insightface.toml` is the
reference wiring for the compiled NPU model. Every value under
`[models.embedder]` is part of the model's contract — a wrong one does not
raise, it produces plausible embeddings that match the wrong people. Check it:

```powershell
python scripts/preflight_insightface.py --config config.insightface.toml
```

It also warns when the database filename does not name the embedder: CavaFace
embeddings are **also** 512-d, so `AllowList`'s dimension guard cannot detect a
stale database from a different model.

## Watching live traffic in a terminal

```powershell
$env:NTFY_TOPIC="your-topic"
python scripts\watch_verdicts.py
```

Prints every verdict as it flows through the relay, color-coded, e.g.:

```
[14:32:07] AUTHORIZED   1 detected: 1 authorized, 0 denied
[14:32:11] UNAUTHORIZED 1 detected: 0 authorized, 1 denied
[14:32:15] EMPTY        no people detected
```

This is a second, independent subscriber to the same public ntfy.sh topic
`uno-q-listener/ntfy_poller.py` listens on -- it needs no access to the board
at all and keeps working even if the board is off, since ntfy.sh broadcasts
to every subscriber. It only watches; it never drives the lights.

The printed decision is computed by importing `verdict()` from the
`uno-q-board` submodule's `check_auth.py`, so what you see here is guaranteed
to match the board's own decision, not a separate implementation that could
drift from it. Requires the submodule (`git submodule update --init`).

Verified against the live relay: sent authorized, unauthorized, and empty
test messages and confirmed each printed correctly, color-coded, in real time.
