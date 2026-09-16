# InsightFace → Qualcomm Snapdragon X NPU

## Two format dead-ends, up front

**GGUF** and **Genie/GenieX** are both the wrong target for this model, for the
same underlying reason: they are *text-generation* stacks, and InsightFace is a
convolutional vision model.

| | what it is | runs InsightFace? |
|---|---|---|
| GGUF | llama.cpp weight container | No — no ggml graph for ResNet50/SCRFD, and no ONNX→GGUF converter exists |
| Genie / GenieX | QAIRT text-to-text runtime (`genie-t2t-run`) | No — bundle format requires a tokenizer + KV-cache decode loop; supported families are Llama 3.x, Qwen 2.5/3, Phi-4-mini, Falcon3 and VLMs only |
| **QNN context binary** | **Hexagon NPU executable** | **Yes — this is the target** |

A Genie bundle is `context binaries + tokenizer.json + genie_config.json`. A face
embedding model has no tokenizer, no autoregressive decode, and no KV cache —
there is nothing for Genie to drive. Putting `w600k_r50` in a Genie bundle is not
a conversion problem, it is a category mismatch.

The pipeline in this repo *is* the AI Hub Workbench flow (`qai-hub` package,
hosted devices, compile → profile → inference jobs). Workbench is the right tool;
Genie is simply the wrong runtime downstream of it.

## Why not GGUF (detail)

GGUF is llama.cpp's weight-container format. It is not a Qualcomm format and
there is no ONNX→GGUF converter, because:

- llama.cpp's converters map **named transformer architectures** (llama, qwen,
  gemma, …) onto ggml graphs. InsightFace models are CNNs — ArcFace/ResNet50 for
  recognition, SCRFD for detection. No ggml graph exists for them, so a GGUF file
  would hold tensors nothing can execute.
- GGUF buys nothing on Snapdragon anyway. llama.cpp's Hexagon NPU support is
  partial and LLM-shaped; a GGUF CNN would land on the CPU, which is the slowest
  of the three compute units on the chip.

The format that reaches the Hexagon NPU is a **QNN context binary**, and the
input to that toolchain is the ONNX file you already have.

    w600k_r50.onnx ──▶ static-shape ONNX ──▶ INT8/INT16 quantized ──▶ QNN context binary (.bin)
                                                                  └─▶ precompiled QNN ONNX (.onnx)

## Layout

    models/onnx/        stock InsightFace buffalo_l models (downloaded)
    models/prepared/    static-shape ONNX + calibration npz + golden outputs
    models/compiled/    QNN artifacts (produced by step 3)
    scripts/01..04      the pipeline
    calib/              put ~100-500 real aligned 112x112 face crops here

## Status

Done locally:

- venv on Python 3.11 with onnx / onnxruntime / onnxsim / qai-hub
- buffalo_l downloaded and unpacked (5 models)
- `w600k_r50` pinned to `1x3x112x112`, simplified, checker-clean, runs in ORT
  → input `input.1`, output `683` = `[1, 512]` embedding
- `det_10g` pinned to `1x3x640x640`, simplified, checker-clean, runs in ORT
  → 9 outputs (score/bbox/kps at strides 8/16/32)
- calibration npz + fp32 golden outputs for both

Blocked on you:

- **AI Hub API token** — it is account-bound, so steps 3 and 4 cannot run yet.
- **Real calibration faces** — current npz files are synthetic noise. They make
  the pipeline run; they do not give trustworthy quantized accuracy.

## Steps

### 1. Get a token (free)

Sign in at https://app.aihub.qualcomm.com/account/ , copy the API token, then:

    .venv/bin/qai-hub configure --api_token <YOUR_TOKEN>
    .venv/bin/python -c "import qai_hub; print([d.name for d in qai_hub.get_devices()])"

### 2. Drop in real calibration faces

Put aligned face crops in `calib/`, then regenerate:

    .venv/bin/python scripts/02_make_calib.py --images ./calib \
        --shape 1,3,112,112 --n 256 --input-name "input.1" \
        --out models/prepared/calib_w600k.npz

Alignment must match inference: InsightFace's 5-point similarity transform to
112x112, BGR, `(x - 127.5) / 127.5`. Mismatched preprocessing here is the most
common cause of a quantized face model that "works" but loses accuracy.

### 3. Quantize + compile

    .venv/bin/python scripts/03_compile_aihub.py \
        --model models/prepared/w600k_r50_static.onnx \
        --calib models/prepared/calib_w600k.npz \
        --device "Snapdragon X Elite CRD" \
        --precision w8a16 \
        --runtime qnn_context_binary \
        --profile

`w8a16` (INT8 weights, INT16 activations) is the right default for ArcFace —
cosine distance between embeddings is sensitive to activation clipping, and
full `w8a8` tends to cost recognition accuracy. Try `w8a8` only if you measure it.

Same command for the detector with `det_10g_static.onnx` / `calib_det10g.npz`.

### 4. Verify numerically before trusting it

    .venv/bin/python scripts/04_verify.py \
        --target models/compiled/w600k_r50_static \
        --golden models/prepared/w600k_r50_static_golden.npz \
        --input-name "input.1"

Runs the compiled model on real silicon and compares against the fp32 ONNX
output. For an embedding model, cosine similarity is the metric that matters;
below ~0.99 you have a calibration or precision problem, not a rounding one.

### 5. Deploy on the device

Two options on a Snapdragon X laptop (Windows on ARM):

**a. ONNX Runtime + QNN EP** — easiest. Ship `--runtime precompiled_qnn_onnx`
output and load it normally:

    import onnxruntime as ort
    sess = ort.InferenceSession(
        "w600k_r50_qnn.onnx",
        providers=["QNNExecutionProvider"],
        provider_options=[{"backend_path": "QnnHtp.dll"}],
    )

**b. QNN SDK directly** — load the `.bin` context binary via
`qnn-net-run` or the C API against `libQnnHtp`. Lower overhead, more integration
work. Get the SDK from Qualcomm's developer site.

### 6. Run it (on the device)

`scripts/05_run_on_device.py` is the device-side runner — copy it and the
compiled model to the Snapdragon machine.

    pip install onnxruntime-qnn numpy opencv-python     # native arm64 Python
    python 05_run_on_device.py --model w600k_r50_qnn.onnx --bench 100
    python 05_run_on_device.py --model w600k_r50_qnn.onnx --compare a.jpg b.jpg
    python 05_run_on_device.py --model w600k_r50_qnn.onnx --images faces/

It prints the active execution providers on startup and warns loudly if QNN EP
failed to load and it silently fell back to CPU — the most common and most
easily missed deployment failure. Verified working end-to-end on CPU locally
(35.9 ms/inference); expect roughly an order of magnitude better on the NPU.

## Caveats worth knowing up front

- A context binary is **SoC-specific**. One compiled for X Elite will not load on
  X Plus or an 8-series phone part; compile per target.
- Anything the NPU cannot run silently falls back to CPU. The `--profile` output
  prints the NPU/total layer split — check it, a low ratio explains bad latency.
- **arm64 Python matters.** `onnxruntime-qnn` will not load under x64 Python
  emulated on Windows-on-ARM. If QNN EP silently refuses to appear, check this first.
- SCRFD's detector head includes NMS-adjacent post-processing that often does not
  quantize or partition well. If `det_10g` profiles badly, cut the graph before
  the post-processing and do that part on CPU.

## Compiled artifact (included in this repo)

    models/compiled/w600k_r50_qnn_x_elite/
    ├── w600k_r50_qnn.onnx    333 B   EPContext wrapper — load THIS in ONNX Runtime
    └── model.bin              42 MB  compiled Hexagon context binary

Both files must stay in the same directory and `model.bin` must keep its name —
the wrapper references it by relative path.

| | |
|---|---|
| Source | InsightFace `buffalo_l` / `w600k_r50` (ArcFace ResNet50) |
| Target | Snapdragon X Elite (Hexagon NPU) |
| Precision | INT8 weights / INT16 activations |
| Input | `input.1`, `1x3x112x112`, BGR, `(x - 127.5) / 127.5` |
| Output | 512-d embedding (L2-normalize before cosine compare) |
| Inference | 1.71 ms (~585 FPS) |
| NPU coverage | 188 / 188 layers, no CPU fallback |
| Peak memory | 45 MB |
| Accuracy vs fp32 | cosine 0.998493 |
| Size | 166 MB → 42 MB |

AI Hub jobs: [quantize jp2r9886g](https://workbench.aihub.qualcomm.com/jobs/jp2r9886g/) ·
[compile jpyojee05](https://workbench.aihub.qualcomm.com/jobs/jpyojee05/) ·
[profile j568z6jng](https://workbench.aihub.qualcomm.com/jobs/j568z6jng/) ·
[inference j5wl3ovzp](https://workbench.aihub.qualcomm.com/jobs/j5wl3ovzp/)

### Accuracy caveat — read before production use

The included binary was calibrated on **synthetic noise**, not real faces. The
0.9985 cosine confirms the quantization is numerically sound on one random
input; it does **not** establish face-verification accuracy. Before relying on
this, put ~200 aligned crops in `calib/`, rerun steps 2-3, and re-measure on
real face pairs.

## License

Pipeline code here is MIT. The InsightFace models it derives from are released
by deepinsight for **non-commercial research use** — that license governs the
compiled artifact too. See https://github.com/deepinsight/insightface .
