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

A dated summary of the models, pipeline, completed work, performance, and
commands is maintained in [PROJECT_LOG.txt](PROJECT_LOG.txt). The remaining
performance ideas are recorded in
[possible_optimisations.txt](possible_optimisations.txt).

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
