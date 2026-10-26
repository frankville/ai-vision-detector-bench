# Development workflow

Use Codex-native role definitions and repo-local skills, with the main agent coordinating bounded work and independent review. Keep the workflow optional and separate from the benchmark experiment harness.

Measurement and schema changes receive methodology review. Code changes receive independent defect review and deterministic checks. Accepted requirements live in `docs/development/roadmap.md`; transient agent logs are not versioned.

Start with the existing detector/source/runner/report boundaries. Add suite orchestration around single measured runs rather than moving experiment state into adapters.
