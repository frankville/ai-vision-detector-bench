---
name: result-schema
description: Change VisionBench result persistence, benchmark identity, aggregation, resume, or compatible-run comparison.
---

Inspect RunConfig, RunResult, report readers, and dashboard consumers before changing results. Preserve positional dataclass arguments by appending optional fields.

- Version changed result contracts and continue reading existing v1 results where practical.
- Distinguish requested settings from effective device/backend/precision. Reject silent fallback.
- Keep run, aggregate, and suite-manifest records distinguishable.
- Separate workload/configuration identity, repetition identity, and execution session. Resume must also validate environment compatibility and successful completion.
- Include immutable source/checkpoint identity and preprocessing settings in comparable workload definitions. Exclude the intentionally varied dimension from a comparison key.
- Preserve raw results; avoid filename collisions and partial writes.
- Export allowlisted metadata only. Redact source credentials and avoid hostnames, account identifiers, tokens, and absolute personal paths.
- Represent missing historical fields as unknown unless their meaning can be established from the schema.
