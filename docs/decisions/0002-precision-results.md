# Precision and execution metadata

P1.1 adds validated `fp32`, `fp16`, and CUDA-only `amp-fp16` with an explicit `pytorch` backend. Unsupported combinations fail before downloads or background session creation. Alternative runtimes remain deferred.

Schema v2 keeps requested settings in `config` and validated effective settings in `model`. Existing v1 results remain readable; their precision is unverified because CPU FP16 requests previously ran as FP32.

Explicit FP16 converts floating pixel inputs on CPU before device transfer. Both operations remain in preprocessing timing. AMP keeps FP32 model weights and uses autocast only around forward inference. Integer and boolean inputs retain their dtypes.

Warmup, completion-interval FPS, post-decode E2E latency, drop accounting, and energy definitions are unchanged. CPU/mocked regression tests verify control flow; CUDA/MPS detection consistency and performance require separate hardware runs.

Acceptance checks cover adapter/config/CLI/API validation, dtype and autocast boundaries, effective metadata, and v1 report compatibility. Repetitions, aggregate records, and suite execution are subsequent cases.
