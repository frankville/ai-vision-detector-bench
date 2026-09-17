"""Persisting results and rendering comparison tables."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visionbench.models import REGISTRY
from visionbench.runner import RunResult

DEFAULT_DIR = Path("benchmarks")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "run"


def save(result: RunResult, root: Path | str = DEFAULT_DIR) -> Path:
    """Write one result as JSON under benchmarks/<profile>/."""
    root = Path(root)
    directory = root / _slug(result.profile)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = re.sub(r"[^0-9]", "", result.started_at)[:14]
    name = f"{_slug(result.config['model'])}__{_slug(result.source['label'])}__{stamp}.json"
    path = directory / name
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n")
    return path


def load_all(root: Path | str = DEFAULT_DIR) -> list[dict]:
    """Read every result JSON under a directory tree, newest first."""
    root = Path(root)
    if not root.exists():
        return []
    results: list[dict] = []
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "schema_version" not in data:
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

    @property
    def verdict(self) -> str:
        return verdict(self.realtime_factor)


def to_row(data: dict) -> Row:
    model_key = _get(data, "config", "model", default="?")
    spec = REGISTRY.get(model_key)
    drop = _get(data, "throughput", "drop_ratio")
    return Row(
        model=model_key,
        license=spec.license if spec else _get(data, "model", "license", default="?"),
        profile=data.get("profile", "?"),
        device=_get(data, "model", "device", default="?"),
        precision=_get(data, "model", "precision", default="fp32"),
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
    )


COLUMNS = [
    ("Model", lambda r: r.model),
    ("Licence", lambda r: r.license),
    ("Host", lambda r: r.profile),
    ("Device", lambda r: f"{r.device}/{r.precision}"),
    ("Mode", lambda r: r.mode),
    ("FPS", lambda r: _fmt(r.processed_fps)),
    ("Real time", lambda r: r.verdict),
    ("Streams", lambda r: _fmt(r.capacity_streams, 2)),
    ("Infer p50", lambda r: _fmt(r.infer_p50, 1, " ms")),
    ("Infer p95", lambda r: _fmt(r.infer_p95, 1, " ms")),
    ("E2E p95", lambda r: _fmt(r.e2e_p95, 1, " ms")),
    ("Drop", lambda r: _fmt(r.drop_pct, 1, " %")),
    ("RSS", lambda r: _fmt(r.rss_mb, 0, " MB")),
    ("GPU mem", lambda r: _fmt(r.gpu_mem_mb, 0, " MB")),
    ("Det/frame", lambda r: _fmt(r.detections_per_frame, 2)),
]

#: Terminal default. Fourteen columns do not fit an 80-column terminal, and a
#: table that wraps every cell is harder to read than one that omits what is
#: usually constant across a sweep.
COMPACT = {"Model", "Mode", "FPS", "Real time", "Infer p50", "Infer p95", "Drop", "Det/frame"}


def _sorted_rows(results: list[dict]) -> list[Row]:
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
    hosts = sorted({f"{r.profile} ({r.device}/{r.precision})" for r in rows})
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
        f"{result.model.get('precision', 'fp32')})",
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
