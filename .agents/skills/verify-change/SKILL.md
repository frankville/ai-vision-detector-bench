---
name: verify-change
description: Verify VisionBench code changes and report test evidence without conflating mocked checks with GPU validation.
---

Use the project environment with Python 3.10 or newer. Run `python -m pytest` and `ruff check src tests` for Python changes; scope extra checks to affected behavior.

- Add tests for externally visible behavior and failure cases, not implementation wording.
- Unit tests must not download weights, require camera credentials, or require a GPU. Use small real CPU tensors where useful and mock CUDA boundaries explicitly.
- Exercise CLI/API parsing, unsupported precision requests, schema compatibility, and persistence for affected changes.
- Record the reviewed revision or diff and exact commands with outcomes.
- State skipped or unavailable GPU/container checks. Hardware validation must compare equivalent inputs and run exclusively on its measurement device.
- For workflow-only edits, validate TOML syntax, skill manifests, paths, and role consistency; do not run model inference.
