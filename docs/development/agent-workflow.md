# Agent workflow

The main agent coordinates development. Custom roles live in `.codex/agents/`; reusable procedures live in `.agents/skills/`. This tooling is optional and is not part of benchmark execution.

For substantial changes:

1. Define requirement references, scope, acceptance criteria, and checks.
2. Plan the smallest coherent change; request design or methodology review when relevant.
3. Assign implementation with explicit file ownership.
4. Review the actual patch independently and run relevant checks.
5. Resolve findings and report evidence and remaining limitations.

Use only the roles a task needs. The coordinator normally plans small changes directly. Parallelize independent analysis; serialize shared-file edits and GPU measurements. Use separate worktrees when implementation scopes overlap.

Review output identifies severity, evidence, affected requirement, and resolution. Findings must name the reviewed revision or diff. After two unresolved rounds, the coordinator decides or raises a concrete remaining product question.

Role TOML files select model and effort; unavailable settings must be reported. Runtime permissions still apply: an instruction to avoid edits is not a read-only sandbox. Skills provide procedures, not isolated permissions or exclusive tool access.

Keep accepted requirements and design decisions in Git. Keep transcripts, local overrides, and scratch artifacts outside tracked files. Contributors can use the same checks without agents.
