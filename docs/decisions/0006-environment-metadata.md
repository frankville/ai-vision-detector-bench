# Environment metadata

P1.3 raw schema v6 captures a fresh `host` snapshot after model loading and before source opening, warmup, and measurement. Existing OS, architecture, CPU, memory, Python, and PyTorch fields remain. `host.packages` records an explicit allowlist of installed distribution versions; missing or unreadable metadata is null. No package paths, environment variables, hostnames, or command errors are exported.

`host.torch.device` and GPU properties follow the selected visible CUDA device; bare `cuda` uses PyTorch's current device. CPU/MPS runs do not attribute an available CUDA GPU to their inference. `host.torch.cuda_version` is PyTorch's CUDA **build** version, not a probe of the installed toolkit. `cudnn_version` is the integer returned by PyTorch, or null. `host.nvidia_driver` comes from a bounded driver-only query and is null when unavailable or ambiguous.

`host.torch.settings` records thread counts, cuDNN benchmark/determinism, and cuDNN/matmul TF32 flags when available. Optional probe failures remain null. The snapshot is descriptive; it does not replace LAB.3's pinned environment and build identity, or establish reproducibility by itself. NVML telemetry-to-device mapping remains separate work.

Readers preserve missing historical metadata as unknown. Aggregation accepts homogeneous raw v3–v6 groups and rejects differing host metadata. Timing and measurement definitions are unchanged.
