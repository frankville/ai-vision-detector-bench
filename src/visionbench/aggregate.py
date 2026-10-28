"""Summaries of explicit repetition groups, separate from frame measurements."""

from __future__ import annotations

import copy
import math
import statistics
from pathlib import Path
from typing import Any


def _get(data: dict, *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _summary(values: list[float], *, full: bool = False) -> dict:
    result = {"count": len(values), "median": statistics.median(values) if values else None}
    if full:
        result.update(
            mean=statistics.fmean(values) if values else None,
            min=min(values) if values else None,
            max=max(values) if values else None,
            stddev=statistics.stdev(values) if len(values) > 1 else None,
        )
    return result


def _signature(run: dict, representative: dict) -> tuple:
    source = run.get("source", {})
    model = dict(run.get("model", {}))
    if (
        run.get("status") in ("interrupted", "empty")
        and _get(run, "throughput", "frames_processed") == 0
        and model.get("input_shape") is None
    ):
        reference_model = representative.get("model", {})
        if "input_shape" in reference_model:
            model["input_shape"] = reference_model["input_shape"]
        else:
            model.pop("input_shape", None)
    return (
        run.get("profile"),
        run.get("config"),
        run.get("host"),
        model,
        {key: source.get(key) for key in ("label", "mode", "width", "height", "nominal_fps")},
    )


def _reference(run: dict, root: Path | str | None) -> dict:
    path = run.get("_path")
    if path is not None:
        path = Path(path)
        if root is not None:
            path = path.resolve().relative_to(Path(root).resolve())
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("run references must be relative to the results directory")
        path = path.as_posix()
    return {
        "run_id": run["run_id"],
        "repetition": run["repetition"],
        "status": run["status"],
        "path": path,
    }


def aggregate_runs(
    runs: list[dict],
    *,
    attempts: list[dict],
    root: Path | str | None = None,
) -> dict:
    """Aggregate one sweep group using its explicit attempt outcomes."""
    if not runs:
        raise ValueError("at least one raw result is required")
    first = runs[0]
    group_id = first.get("group_id")
    requested = first.get("repeat_count")
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("an explicit repetition group is required")
    if not isinstance(requested, int) or isinstance(requested, bool) or requested < 1:
        raise ValueError("repeat_count must be a positive integer")

    representative = next(
        (
            run
            for run in runs
            if run.get("status") == "completed"
            and (_get(run, "throughput", "frames_processed") or 0) >= 2
        ),
        next((run for run in runs if _get(run, "model", "input_shape") is not None), first),
    )
    signature = _signature(representative, representative)
    members: dict[int, dict] = {}
    run_ids: set[str] = set()
    for run in runs:
        if run.get("record_type") != "run" or run.get("schema_version") != 3:
            raise ValueError("only schema v3 raw runs can be aggregated")
        if run.get("group_id") != group_id or run.get("repeat_count") != requested:
            raise ValueError("runs must belong to one repetition group")
        if _signature(run, representative) != signature:
            raise ValueError("incompatible configuration or effective run metadata")
        repetition = run.get("repetition")
        run_id = run.get("run_id")
        if (
            not isinstance(repetition, int)
            or isinstance(repetition, bool)
            or not 1 <= repetition <= requested
        ):
            raise ValueError("invalid repetition number")
        if not isinstance(run_id, str) or not run_id or run_id in run_ids or repetition in members:
            raise ValueError("duplicate or missing run identity")
        if run.get("status") not in ("completed", "empty", "interrupted"):
            raise ValueError("invalid raw run status")
        members[repetition] = run
        run_ids.add(run_id)

    outcomes: dict[int, dict] = {}
    for attempt in attempts:
        if attempt.get("group_id") != group_id or attempt.get("repeat_count") != requested:
            raise ValueError("attempts must belong to the same repetition group")
        if attempt.get("model") != first["config"]["model"]:
            raise ValueError("attempt model does not match raw results")
        repetition = attempt.get("repetition")
        if (
            not isinstance(repetition, int)
            or isinstance(repetition, bool)
            or not 1 <= repetition <= requested
            or repetition in outcomes
        ):
            raise ValueError("duplicate or invalid attempt repetition")
        status = attempt.get("status")
        if status not in ("completed", "empty", "interrupted", "failed"):
            raise ValueError("invalid attempt status")
        member = members.get(repetition)
        if member is None:
            if status not in ("failed", "interrupted") or attempt.get("run_id") is not None:
                raise ValueError("attempt refers to a missing raw result")
        elif attempt.get("run_id") != member["run_id"] or status != member["status"]:
            raise ValueError("attempt does not match its raw result")
        outcomes[repetition] = {
            "repetition": repetition,
            "status": status,
            "run_id": attempt.get("run_id"),
            "error_type": attempt.get("error_type"),
        }
    if not set(members) <= set(outcomes):
        raise ValueError("each raw run must have an attempt outcome")
    for repetition in range(1, requested + 1):
        outcomes.setdefault(
            repetition,
            {"repetition": repetition, "status": "unattempted", "run_id": None, "error_type": None},
        )

    eligible = []
    excluded = []
    for repetition, run in sorted(members.items()):
        fps = _get(run, "throughput", "processed_fps")
        frames = _get(run, "throughput", "frames_processed")
        if (
            run["status"] == "completed"
            and _number(frames)
            and frames >= 2
            and _number(fps)
            and fps >= 0
        ):
            eligible.append(run)
        else:
            excluded.append(
                {
                    "run_id": run["run_id"],
                    "repetition": repetition,
                    "reason": run["status"]
                    if run["status"] != "completed"
                    else "no measurable completion interval",
                }
            )

    def samples(*path: str) -> list[float]:
        return [
            float(value) for run in eligible if _number(value := _get(run, *path)) and value >= 0
        ]

    metrics = {"fps": _summary(samples("throughput", "processed_fps"), full=True)}
    for name, path in {
        "inference_p50_ms": ("latency", "inference_ms", "p50"),
        "inference_p95_ms": ("latency", "inference_ms", "p95"),
        "e2e_p95_ms": ("latency", "end_to_end_ms", "p95"),
        "gpu_util_pct": ("resources", "counters", "gpu_util_pct", "mean"),
        "gpu_power_w": ("resources", "counters", "gpu_power_w", "mean"),
        "gpu_mem_mb": ("resources", "counters", "gpu_mem_used_mb", "mean"),
        "energy_per_frame_j": ("resources", "energy_per_frame_j"),
    }.items():
        metrics[name] = _summary(samples(*path))

    counts = {"requested": requested, "eligible": len(eligible)}
    for status in ("completed", "empty", "interrupted", "failed", "unattempted"):
        counts[status] = sum(outcome["status"] == status for outcome in outcomes.values())
    return {
        "schema_version": 1,
        "record_type": "aggregate",
        "group_id": group_id,
        "started_at": min(run["started_at"] for run in runs),
        "profile": representative["profile"],
        "config": copy.deepcopy(representative["config"]),
        "host": copy.deepcopy(representative.get("host", {})),
        "model": copy.deepcopy(representative["model"]),
        "source": copy.deepcopy(representative["source"]),
        "counts": counts,
        "metrics": metrics,
        "runs": [_reference(run, root) for _, run in sorted(members.items())],
        "attempts": [outcomes[key] for key in sorted(outcomes)],
        "excluded_runs": excluded,
        "definitions": {
            "fps": "per-run completion-interval FPS; sample standard deviation",
            "latency": "medians of per-run percentiles; E2E begins after decode",
            "gpu_counters": "medians of per-run means; count reports available runs",
            "energy_per_frame_j": "median of per-run GPU-only joules per processed frame",
        },
    }
