# Benchmark roadmap

Objective: reproducible benchmarking and capacity planning for vision workloads on edge hardware. Rented GPUs are a compute laboratory; final appliance validation covers decode, CPU limits, thermals, and full-system power.

P1.1 is implemented with CPU/mocked regression coverage. GPU detection consistency and performance validation remain pending. See [the precision decision](../decisions/0002-precision-results.md).

## Phase 1: measurement foundation

- P1.1: validated FP32, explicit FP16, and CUDA AMP-FP16; integer-safe tensor transfer; explicit PyTorch backend and effective configuration metadata.
- P1.2: independent repeated runs, preserved raw results, collision-safe persistence, and separate aggregate statistics.
- P1.3: compatible-run comparisons, baseline speedups, source/checkpoint identity, environment metadata, and detection-drift checks on equivalent frames.

Keep warmup outside measurement. Preserve current FPS and post-decode E2E definitions; label estimated stream capacity and GPU-only energy explicitly. Read historical results without inventing missing metadata.

## Phase 1.5: reproducible GPU laboratory

- LAB.1: canonical suite parsing, strict source metadata/checksum validation, matrix expansion, deterministic IDs, seeded/interleaved order, and GPU-free dry-run.
- LAB.2: manifest with planned/successful/failed/interrupted experiments, atomic progress/results, environment-aware resume, repetitions, and aggregates.
- LAB.3: dependency lockfile, CUDA Docker image, immutable build identity, external data/results/model-cache mounts, and environment fingerprint.
- LAB.4: allowlisted artifact bundle, Markdown/CSV reports, compatible cross-host comparison, and usage documentation.
- LAB.5: optional provider and session rental-cost metadata; separate rental economics from appliance economics.

Acceptance: mount the same verified clip, run one suite command in the same image on two NVIDIA environments, and obtain auditable bundles with comparable configurations. Do not claim this until both executions are validated.

The canonical `104803.mp4` clip and checksum must be supplied or verified before publishing a runnable canonical suite. Pin model revisions and dependencies; record driver and hardware differences. Resume must not combine incompatible environments.

Before cross-GPU resource comparisons, map telemetry to the physical compute GPU. Current NVML sampling uses physical index 0, which may differ from an indexed or visibility-remapped CUDA device.

## Phase 2: deployment capacity

- CAP.1: independent decoders per stream, bounded latest-frame scheduling, per-stream drop/latency metrics, and host totals.
- CAP.2: capacity search with explicit drop, latency, resource, and optional temperature/headroom criteria.
- CAP.3: batching with bounded wait and queue/throughput/latency/energy measurements.

Only validated multicamera SLA results establish cameras per appliance.

## Phase 3: additional runtimes

Refine adapter capabilities for ONNX Runtime, TensorRT, and OpenVINO in separate changes. Keep model family distinct from runtime. Avoid speculative runtime implementations in Phase 1.

## Deferred

Provider provisioning, billing APIs, automatic termination, distributed orchestration, Kubernetes, and labelled accuracy evaluation remain separate work. Preserve source credential redaction and keep weights and clips out of artifacts and Git.
