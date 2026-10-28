# Repeated measurements

P1.2 adds `sweep --repeat N`. Each attempt uses the existing single-run engine with independent model/source setup and normal warmup. Models are interleaved by repetition and never run concurrently.

Raw schema v3 appends run ID, group ID, repetition, requested repeat count, record type, and status. Existing positional arguments and v1/v2 readers remain supported. Empty means fewer than two completed frames or no positive completion interval; FPS definitions are unchanged.

Setup and measurement share guarded cleanup. A failed attempt releases acquired resources before continuation. Interrupted results are saved, and interruption stops the sweep. Attempt callbacks distinguish failures from unattempted slots; persistence failures are not treated as model failures.

Aggregate schema v1 uses `record_type=aggregate`. It references raw run IDs and paths relative to the results root, validates explicit group membership and compatible configuration/effective metadata, and records outcome counts and exclusions. It does not infer groups from similar filenames or combine historical runs.

Unknown input shape on an interrupted zero-frame warmup is excluded from shape comparison; measured records and known shapes must still match. Groups with no raw result have no aggregate artifact; failures remain terminal output until suite manifests are implemented.

FPS summaries use per-run observations and sample standard deviation, null below two observations. Latency summaries are medians of per-run percentiles, not pooled frame percentiles. GPU utilization, power, and memory use medians of per-run means. GPU energy/frame uses medians of per-run values. Every metric records availability; missing values remain null and numeric zero is retained.

Result files publish atomically without overwriting existing files. Raw reports and dashboard queries exclude aggregate records by default. `report --kind aggregate` selects separate summary tables.

Acceptance covers known statistics, incompatible/duplicate membership, failed and interrupted slots, cleanup failures, atomic writes, same-second/concurrent saves, CLI behavior, and mixed-directory historical reads. Hardware performance, cross-host compatibility, suite manifests, and resume remain later validation or implementation work.
