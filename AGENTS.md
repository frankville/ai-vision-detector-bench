# Development workflow

- Keep public comments and documentation concise.
- Preserve CLI compatibility and measurement definitions. Document intentional changes.
- Keep the development agent workflow separate from the benchmark engine and suites.
- For substantial changes, the main agent coordinates bounded implementation and independent review. Delegate methodology review for timing, precision, aggregation, and capacity changes; delegate code review before completion. Handle small changes directly.
- Give each delegated task an objective, requirement references, edit scope, acceptance criteria, and expected evidence. Assign one writer per file; use separate worktrees for overlapping work.
- Agents may use the models and effort configured in `.codex/agents/`. Report unavailable settings rather than silently substituting them.
- Read the relevant repo skill before changing benchmark methodology, result schemas, or verification procedures.
- Review findings must identify the reviewed revision or diff, severity, evidence, and resolution. After two unresolved review rounds, the coordinator resolves or records the remaining decision.
- Run measured GPU workloads exclusively. Never compete with another benchmark or GPU test on the same device.
- Validate changes with `python -m pytest` and `ruff check src tests`. Report unavailable checks and hardware validation separately from passing checks.
- Keep credentials, agent transcripts, scratch artifacts, clips, and model weights out of Git. Keep accepted requirements and design decisions under `docs/`.
