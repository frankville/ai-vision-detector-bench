# Checkpoint identity

P1.3 adds optional `revision` to RunConfig and the session API, and `--revision` to CLI run/sweep. A sweep applies the requested revision to each model repository. Revisions are supported only for HF models.

Before warmup and measurement, resolve the requested Hub branch/tag/commit (default `main`) once. Load both processor and model from that immutable commit. Resolution/load failures do not retry another revision.

Raw schema v4 records the requested `config.revision` and `model.revision`, plus effective `model.checkpoint_commit`. Local checkpoint directories have a null commit and reject explicit Hub revisions. Missing historical identity remains unknown. Reports still read older records; aggregation accepts homogeneous v3 or v4 groups and rejects differing checkpoint metadata.

This identifies repository snapshots, not local file content, source clips, dependencies, or detection equivalence. E2E checks explicit offline revision loading and persisted identity; GPU validation remains separate.
