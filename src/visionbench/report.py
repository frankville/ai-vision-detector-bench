"""Persisting results and rendering comparison tables."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visionbench.models import REGISTRY
from visionbench.runner import RunResult
from visionbench.storage import write_json

DEFAULT_DIR = Path("benchmarks")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "run"


def save(result: RunResult, root: Path | str = DEFAULT_DIR) -> Path:
    """Write one result as JSON under benchmarks/<profile>/."""
    root = Path(root)
    directory = root / _slug(result.profile)
    stamp = re.sub(r"[^0-9]", "", result.started_at)[:14]
    stem = f"{_slug(result.config['model'])}__{_slug(result.source['label'])}__{stamp}__{_slug(result.run_id)}"
    return write_json(directory, stem, result.to_dict())


def save_aggregate(result: dict, root: Path | str = DEFAULT_DIR) -> Path:
    if result.get("record_type") != "aggregate":
        raise ValueError("expected an aggregate record")
    directory = Path(root) / _slug(result["profile"]) / "aggregates"
    stem = f"{_slug(result['config']['model'])}__{_slug(result['group_id'])}"
    return write_json(directory, stem, result)


def _valid_aggregate(data: dict) -> bool:
    if data.get("schema_version") != 1 or data.get("record_type") != "aggregate":
        return False
    if not all(isinstance(data.get(key), dict) for key in ("config", "model", "source", "counts", "metrics")):
        return False
    if not isinstance(data["config"].get("model"), str) or not isinstance(data.get("group_id"), str):
        return False
    if not isinstance(data.get("profile"), str):
        return False
    if not all(isinstance(data["model"].get(key, "unknown"), str) for key in ("backend", "precision")):
        return False
    if not isinstance(data["source"].get("mode", "?"), str):
        return False
    counts = data["counts"]
    keys = ("requested", "eligible", "completed", "empty", "interrupted", "failed", "unattempted")
    if not all(type(counts.get(key)) is int and counts[key] >= 0 for key in keys):
        return False
    if not counts["requested"] or counts["eligible"] > counts["completed"]:
        return False
    if sum(counts[key] for key in keys[2:]) != counts["requested"]:
        return False
    if "fps" not in data["metrics"]:
        return False
    for metric in data["metrics"].values():
        if not isinstance(metric, dict) or type(metric.get("count")) is not int or not 0 <= metric["count"] <= counts["eligible"]:
            return False
        median = metric.get("median")
        if median is not None and (type(median) not in (int, float) or not math.isfinite(median)):
            return False
    return True


def load_all(root: Path | str = DEFAULT_DIR, *, kind: str = "run") -> list[dict]:
    """Read raw runs or aggregates, newest first. Legacy records are raw runs."""
    if kind not in ("run", "aggregate"):
        raise ValueError("kind must be 'run' or 'aggregate'")
    root = Path(root)
    if not root.exists():
        return []
    results: list[dict] = []
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or "schema_version" not in data:
            continue
        if data.get("record_type", "run") != kind:
            continue
        required = ("config", "model", "throughput", "latency") if kind == "run" else ("config", "model", "counts", "metrics")
        if not all(isinstance(data.get(key), dict) for key in required):
            continue
        if not isinstance(data.get("started_at", ""), str):
            continue
        if kind == "aggregate" and not _valid_aggregate(data):
            continue
        data["_path"] = str(path)
        results.append(data)
    results.sort(key=lambda d: d.get("started_at", ""), reverse=True)
    return results


def _get(data: dict, *path: str, default: Any = None) -> Any:
    node: Any = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


#: Tolerance on the real-time factor. The fps estimator divides by (n-1)
#: intervals and normal scheduler jitter costs a fraction of a frame, so
#: demanding exactly 1.0 would label a run that plainly kept up as failing.
KEEPS_UP_FACTOR = 0.97


def verdict(realtime_factor: float | None) -> str:
    if realtime_factor is None:
        return "n/a"
    if realtime_factor >= KEEPS_UP_FACTOR:
        return "keeps up"
    return f"{realtime_factor:.2f}x"


def _fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


@dataclass(frozen=True)
class Row:
    model: str
    license: str
    profile: str
    device: str
    precision: str
    mode: str
    processed_fps: float | None
    realtime_factor: float | None
    capacity_streams: float | None
    infer_p50: float | None
    infer_p95: float | None
    e2e_p95: float | None
    drop_pct: float | None
    rss_mb: float | None
    gpu_mem_mb: float | None
    energy_j: float | None
    detections_per_frame: float | None
    backend: str = "unknown"
    status: str = "unknown"

    @property
    def verdict(self) -> str:
        return verdict(self.realtime_factor)


def to_row(data: dict) -> Row:
    model_key = _get(data, "config", "model", default="?")
    spec = REGISTRY.get(model_key)
    drop = _get(data, "throughput", "drop_ratio")
    precision = _get(data, "model", "precision", default="unknown")
    backend = _get(data, "model", "backend", default="unknown")
    if data.get("schema_version", 1) == 1:
        precision = f"{precision} (unverified)" if precision != "unknown" else precision
        backend = "unknown"
    return Row(
        model=model_key,
        license=spec.license if spec else _get(data, "model", "license", default="?"),
        profile=data.get("profile", "?"),
        device=_get(data, "model", "device", default="?"),
        precision=precision,
        mode=_get(data, "source", "mode", default="?"),
        processed_fps=_get(data, "throughput", "processed_fps"),
        realtime_factor=_get(data, "throughput", "realtime_factor"),
        capacity_streams=_get(data, "throughput", "capacity_streams"),
        infer_p50=_get(data, "latency", "inference_ms", "p50"),
        infer_p95=_get(data, "latency", "inference_ms", "p95"),
        e2e_p95=_get(data, "latency", "end_to_end_ms", "p95"),
        drop_pct=round(drop * 100, 2) if isinstance(drop, (int, float)) else None,
        rss_mb=_get(data, "resources", "counters", "process_rss_mb", "mean"),
        gpu_mem_mb=_get(data, "resources", "counters", "gpu_mem_used_mb", "mean"),
        energy_j=_get(data, "resources", "energy_per_frame_j"),
        detections_per_frame=_get(data, "detections", "per_frame", "mean"),
        backend=backend,
        status=data.get("status", "unknown"),
    )


COLUMNS = [
    ("Model", lambda r: r.model),
    ("Licence", lambda r: r.license),
    ("Host", lambda r: r.profile),
    ("Device", lambda r: r.device),
    ("Backend", lambda r: r.backend),
    ("Precision", lambda r: r.precision),
    ("Mode", lambda r: r.mode),
    ("Status", lambda r: r.status),
    ("FPS", lambda r: _fmt(r.processed_fps)),
    ("Real time", lambda r: r.verdict),
    ("Est. streams", lambda r: _fmt(r.capacity_streams, 2)),
    ("Infer p50", lambda r: _fmt(r.infer_p50, 1, " ms")),
    ("Infer p95", lambda r: _fmt(r.infer_p95, 1, " ms")),
    ("E2E p95", lambda r: _fmt(r.e2e_p95, 1, " ms")),
    ("Drop", lambda r: _fmt(r.drop_pct, 1, " %")),
    ("RSS", lambda r: _fmt(r.rss_mb, 0, " MB")),
    ("GPU mem", lambda r: _fmt(r.gpu_mem_mb, 0, " MB")),
    ("Det/frame", lambda r: _fmt(r.detections_per_frame, 2)),
]

#: Columns shown in compact terminal reports.
COMPACT = {"Model", "Backend", "Precision", "Mode", "FPS", "Infer p50", "Infer p95", "Drop"}


def _sorted_rows(results: list[dict]) -> list[Row]:
    if any(result.get("record_type", "run") != "run" for result in results):
        raise ValueError("use aggregate tables for aggregate records")
    return sorted(
        (to_row(r) for r in results), key=lambda r: (r.profile, -(r.processed_fps or 0))
    )


def markdown_table(results: list[dict]) -> str:
    """Comparison table ready to paste into a README."""
    rows = _sorted_rows(results)
    if not rows:
        return "_No results yet._"
    header = "| " + " | ".join(name for name, _ in COLUMNS) + " |"
    divider = "|" + "|".join("---" for _ in COLUMNS) + "|"
    body = ["| " + " | ".join(render(row) for _, render in COLUMNS) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def rich_table(results: list[dict], full: bool = False):
    """The same comparison as a rich table for terminal output."""
    from rich.table import Table

    columns = COLUMNS if full else [c for c in COLUMNS if c[0] in COMPACT]
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    for name, _ in columns:
        table.add_column(name, no_wrap=True)
    for row in _sorted_rows(results):
        style = "" if row.license.startswith("Apache") else "yellow"
        table.add_row(*[render(row) for _, render in columns], style=style)
    return table


def context_line(results: list[dict]) -> str:
    """Host, device and licence context, which a compact table leaves out."""
    rows = _sorted_rows(results)
    if not rows:
        return ""
    hosts = sorted({f"{r.profile} ({r.device}/{r.backend}/{r.precision})" for r in rows})
    restricted = sorted({r.model for r in rows if not r.license.startswith("Apache")})
    parts = [f"host: {', '.join(hosts)}"]
    if restricted:
        parts.append(f"licence-restricted: {', '.join(restricted)}")
    return " | ".join(parts)


def summarise(result: RunResult) -> str:
    """One-paragraph plain-language read of a single run."""
    tp = result.throughput
    lat = result.latency.get("inference_ms", {})
    e2e = result.latency.get("end_to_end_ms", {})
    parts = [
        f"{result.config['model']} on {result.profile} ({result.model.get('device')}, "
        f"{result.model.get('backend', 'unknown')}, {result.model.get('precision', 'unknown')})",
        f"processed {tp['frames_processed']} frames in {result.elapsed_s:.1f}s "
        f"at {tp['processed_fps']:.1f} fps",
    ]
    if tp.get("stream_fps_reference"):
        parts.append(f"against a {tp['stream_fps_reference']:.1f} fps source")
    if tp.get("realtime_factor") is not None:
        keeps_up = tp["realtime_factor"] >= KEEPS_UP_FACTOR
        parts.append(
            f"{'keeps up' if keeps_up else 'falls behind'} (factor {tp['realtime_factor']:.2f})"
        )
    if tp.get("frames_dropped"):
        parts.append(f"dropping {tp['frames_dropped']} frames ({tp['drop_ratio'] * 100:.1f}%)")
    if lat.get("p50") is not None:
        parts.append(f"inference p50 {lat['p50']:.1f} ms, p95 {lat['p95']:.1f} ms")
    if e2e.get("p95") is not None:
        parts.append(f"capture-to-detection p95 {e2e['p95']:.1f} ms")
    return "; ".join(parts) + "."


def _aggregate_value(data: dict, metric: str, suffix: str = "") -> str:
    summary = data["metrics"].get(metric, {})
    value = _fmt(summary.get("median"), 2, suffix)
    count = summary.get("count", 0)
    eligible = data["counts"]["eligible"]
    return f"{value} ({count}/{eligible})" if count < eligible else value


AGGREGATE_COLUMNS = [
    ("Model", lambda d: d["config"]["model"]),
    ("Host", lambda d: d.get("profile", "?")),
    ("Backend", lambda d: d["model"].get("backend", "unknown")),
    ("Precision", lambda d: d["model"].get("precision", "unknown")),
    ("Mode", lambda d: d.get("source", {}).get("mode", "?")),
    ("Runs", lambda d: f"{d['counts']['eligible']}/{d['counts']['requested']}"),
    ("FPS median", lambda d: _aggregate_value(d, "fps")),
    ("FPS mean", lambda d: _fmt(_get(d, "metrics", "fps", "mean"), 2)),
    ("FPS min", lambda d: _fmt(_get(d, "metrics", "fps", "min"), 2)),
    ("FPS max", lambda d: _fmt(_get(d, "metrics", "fps", "max"), 2)),
    ("FPS sample SD", lambda d: _fmt(_get(d, "metrics", "fps", "stddev"), 2)),
    ("Infer p50 median", lambda d: _aggregate_value(d, "inference_p50_ms", " ms")),
    ("Infer p95 median", lambda d: _aggregate_value(d, "inference_p95_ms", " ms")),
    ("E2E p95 median", lambda d: _aggregate_value(d, "e2e_p95_ms", " ms")),
    ("GPU util median", lambda d: _aggregate_value(d, "gpu_util_pct", "%")),
    ("GPU mem median", lambda d: _aggregate_value(d, "gpu_mem_mb", " MB")),
    ("GPU W median", lambda d: _aggregate_value(d, "gpu_power_w", " W")),
    ("GPU J/frame median", lambda d: _aggregate_value(d, "energy_per_frame_j", " J")),
]
AGGREGATE_COMPACT = {"Model", "Backend", "Precision", "Runs", "FPS median", "FPS sample SD", "Infer p50 median"}


def _aggregate_rows(results: list[dict]) -> list[dict]:
    if any(not _valid_aggregate(result) for result in results):
        raise ValueError("expected aggregate records")
    return sorted(results, key=lambda d: (d.get("profile", ""), d["config"]["model"], d["group_id"]))


def aggregate_markdown_table(results: list[dict]) -> str:
    rows = _aggregate_rows(results)
    if not rows:
        return "_No aggregates yet._"
    return "\n".join([
        "| " + " | ".join(name for name, _ in AGGREGATE_COLUMNS) + " |",
        "|" + "|".join("---" for _ in AGGREGATE_COLUMNS) + "|",
        *["| " + " | ".join(render(row) for _, render in AGGREGATE_COLUMNS) + " |" for row in rows],
    ])


def aggregate_rich_table(results: list[dict], full: bool = False):
    from rich.table import Table

    columns = AGGREGATE_COLUMNS if full else [column for column in AGGREGATE_COLUMNS if column[0] in AGGREGATE_COMPACT]
    table = Table(title="Aggregates (eligible/requested runs)", box=None, pad_edge=False)
    for name, _ in columns:
        table.add_column(name, no_wrap=True)
    for row in _aggregate_rows(results):
        table.add_row(*[render(row) for _, render in columns])
    return table
