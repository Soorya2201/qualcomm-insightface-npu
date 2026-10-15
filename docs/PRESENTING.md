# Presenting this work

Slide outline and live-demo script for the Snapdragon Edge Access Control project. The main [README](../README.md) is the technical write-up.


### One sentence

> We compiled a production face-recognition model to run entirely on the
> Snapdragon X Elite NPU — 4× smaller, 1.71 ms per face, zero CPU fallback — and
> used it to drive a self-sufficient Arduino access indicator, with no camera
> frame ever leaving the device.

### Three messages to land

1. **Real on-device AI performance, proven on hardware.** 188/188 layers on the NPU,
   1.71 ms, 42 MB — from Qualcomm AI Hub jobs, not estimates.
2. **The right Snapdragon runtime for each job.** QNN for vision models, GenieX for
   generative AI — and a clear reason why each model runs where it does.
3. **End to end, in the physical world.** Camera → NPU → decision → microcontroller → light,
   engineered to survive a real network.

### Suggested slides

| # | Slide | Show |
|---|---|---|
| 1 | Title & one-liner | The sentence above; photo of the lit board |
| 2 | The problem | Access control usually streams faces to a server. Biometric data leaves the building |
| 3 | Our approach | Architecture diagram: phone → laptop NPU → verdict JSON → Arduino |
| 4 | Results at a glance | The results table: 1.71 ms, 188/188, 4× smaller, cosine 0.9985 |
| 5 | Quantization pipeline | Pipeline diagram; W8A16 reasoning; AI Hub job IDs |
| 6 | Right runtime per workload | QNN vs. GenieX table; why not GGUF or a Genie bundle for a CNN |
| 7 | GenieX in the system | Qwen3-4B on-device agent controlling the board through MCP; incident-description roadmap |
| 8 | Engineering rigor | Model evaluation, including DINOv3 rejected on evidence; bottleneck located (detector is ~97% of time) |
| 9 | Real-world deployment | Client isolation → captive portal → relay limits, each diagnosed from evidence |
| 10 | Reliability | Queue, retries, budget, fail-closed, stale watchdog, live logs, 60 tests |
| 11 | Live demo | Authorized face → green; unknown face → red + beep; `watch_verdicts.py` on screen |
| 12 | What's next | Optimized detector (Qualcomm publishes 4.65 ms on QNN), real-face calibration, GenieX incident reports |

### Demo script

1. Start the board (power only), then the laptop app, and open `watch_verdicts.py` beside the dashboard.
2. An enrolled person steps in → dashboard green box → terminal `AUTHORIZED` → board LED green.
3. An unknown person steps in → red box → `UNAUTHORIZED` → board LED red, buzzer beeps.
4. Both in frame → two LEDs, left to right, matching where they stand.
5. Everyone leaves → lights clear.
6. Point at the laptop log line `Board updated ...` as live proof of delivery.

**Before a long demo:** use direct HTTP or a fresh relay quota. At one heartbeat
per second, public ntfy.sh's 250-message daily quota runs out in about 17 minutes.

### Questions to expect

| Question | Answer |
|---|---|
| How accurate is it? | The quantized model matches the float model's output (cosine 0.9985). End-to-end verification accuracy on real faces is our next measurement, and we've said so rather than guess |
| Why not run it in the cloud? | Privacy, latency, cost, and offline operation — the whole decision happens on the laptop |
| Why is the detector not quantized yet? | We measured first. The embedder was the model we could fully control; timings now show the detector is the bottleneck, and the pipeline to fix it is ready |
| Where is GenieX? | Driving the hardware through an on-device LLM agent, and describing unauthorized faces' outfit/glasses/hat (Qwen2.5-VL-7B, W4A16). Face recognition itself runs on QNN because GenieX serves generative models, not CNNs |
| What if the network drops? | Recognition continues; board updates are queued and retried; the board clears stale lights after 25 s rather than showing a wrong green |
