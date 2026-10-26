---
name: benchmark-methodology
description: Review or change VisionBench timing, precision, throughput, drop accounting, aggregation, energy, or capacity measurements.
---

Keep warmup and model downloads outside measurement. Preserve existing FPS and latency definitions unless an explicit, versioned change is required.

- HF adapters own preprocessing, synchronized inference, and postprocessing timings. Record device transfers and dtype conversion under preprocessing.
- Distinguish explicit FP16 weights/inputs from CUDA autocast with FP32 weights. Preserve integer masks and reject unsupported precision/device/backend combinations.
- E2E currently starts after decode; it is not camera-to-detection latency.
- Sequential throughput includes the pipeline. FPS/source FPS is an estimate, not validated camera capacity.
- Keep realtime buffers bounded and frame loss visible. Capacity requires independent decoders and per-stream SLA checks.
- Keep raw repetitions. Label medians of per-run percentiles; do not present them as pooled percentiles.
- Compare detection summaries on the same frames before attributing changes to precision. These summaries are not accuracy metrics.
- Label GPU power and energy as GPU-only; unavailable telemetry remains null.
- Do not run competing GPU workloads during measurement. Mocked tests establish control flow, not hardware performance or detection equivalence.
