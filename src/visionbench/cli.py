"""Command line entry point."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from visionbench import report as report_mod
from visionbench.models import DEFAULT_SWEEP, SPECS, permissive_keys
from visionbench.runner import RunConfig, run, sweep

app = typer.Typer(
    add_completion=False,
    help="Benchmark and compare real-time object detection models on live video.",
)
console = Console()


def _split(values: list[str] | None) -> list[str]:
    """Accept both repeated flags and comma-separated lists."""
    out: list[str] = []
    for value in values or []:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


@app.command("models")
def list_models(
    permissive_only: bool = typer.Option(
        False, "--permissive-only", help="Hide models whose licence blocks commercial use."
    ),
) -> None:
    """List the model catalogue with licence terms."""
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    for column in ("Key", "Model", "Family", "Licence", "Commercial", "Weights", "Extra"):
        table.add_column(column, overflow="fold")
    for spec in SPECS:
        if permissive_only and not spec.permissive:
            continue
        table.add_row(
            spec.key,
            spec.display_name,
            spec.family,
            spec.license,
            spec.commercial,
            spec.weights,
            spec.extra or "-",
            style="" if spec.permissive else "yellow",
        )
    console.print(table)
    console.print(
        "\n[dim]Rows in yellow carry licence obligations that make them unsuitable as a "
        "shipping default. See docs/licensing.md.[/dim]"
    )
    notes = [(s.key, s.notes) for s in SPECS if s.notes]
    if notes:
        console.print("\n[bold]Notes[/bold]")
        for key, note in notes:
            console.print(f"  [cyan]{key}[/cyan]: {note}")


@app.command()
def doctor() -> None:
    """Report what this machine can run and which telemetry is available."""
    from visionbench.hostinfo import default_profile, host_info
    from visionbench.metrics.system import detect_gpu_backend
    from visionbench.models.base import resolve_device

    info = host_info()
    device = resolve_device("auto")
    backend, _ = detect_gpu_backend(device)

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("k", style="bold cyan")
    table.add_column("v", overflow="fold")
    table.add_row("profile", default_profile())
    table.add_row("host", f"{info['hostname']} ({info['os']}, {info['arch']})")
    table.add_row("cpu", f"{info['cpu_model']}")
    table.add_row(
        "cores", f"{info['cpu_cores_physical']} physical / {info['cpu_cores_logical']} logical"
    )
    table.add_row("ram", f"{info['ram_total_gb']} GB")
    table.add_row("python", info["python"])
    torch_info = info["torch"]
    if torch_info.get("available"):
        table.add_row("torch", torch_info["version"])
        table.add_row("device", device)
        if torch_info.get("gpu_name"):
            table.add_row("gpu", f"{torch_info['gpu_name']} ({torch_info.get('gpu_total_memory_mb')} MB)")
    else:
        table.add_row("torch", "[red]not installed[/red]")
    table.add_row("gpu telemetry", backend.kind)
    console.print(table)
    if backend.note:
        console.print(f"\n[yellow]{backend.note}[/yellow]")


@app.command("run")
def run_cmd(
    source: str = typer.Option(..., "-s", "--source", help="File path, RTSP URL, or MJPEG URL."),
    model: str = typer.Option("dfine-nano", "-m", "--model", help="Registry key."),
    duration: float = typer.Option(30.0, "-d", "--duration", help="Seconds to run."),
    frames: int | None = typer.Option(None, "--frames", help="Stop after N frames instead."),
    device: str = typer.Option("auto", "--device", help="auto | cpu | cuda | mps"),
    confidence: float = typer.Option(0.3, "--conf", help="Detection score threshold."),
    classes: list[str] | None = typer.Option(
        None, "-c", "--class", help="Keep only these class names, e.g. -c person."
    ),
    mode: str = typer.Option("realtime", "--mode", help="realtime | sequential"),
    imgsz: int | None = typer.Option(None, "--imgsz", help="Override model input size."),
    precision: str = typer.Option("fp32", "--precision", help="fp32 | fp16"),
    profile: str | None = typer.Option(None, "--profile", help="Label for this machine."),
    pace_fps: float | None = typer.Option(
        None, "--pace-fps", help="Replay a file at this rate to imitate a camera."
    ),
    save: bool = typer.Option(True, "--save/--no-save"),
    out: Path = typer.Option(report_mod.DEFAULT_DIR, "--out", help="Results directory."),
) -> None:
    """Benchmark one model against one source."""
    config = RunConfig(
        source=source,
        model=model,
        device=device,
        duration_s=duration,
        max_frames=frames,
        confidence=confidence,
        classes=_split(classes) or None,
        mode=mode,
        imgsz=imgsz,
        precision=precision,
        profile=profile,
        pace_fps=pace_fps,
    )
    console.print(f"[dim]loading {model}...[/dim]")
    result = run(config)
    console.print()
    console.print(report_mod.summarise(result))
    for warning in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    if save:
        path = report_mod.save(result, out)
        console.print(f"[dim]saved {path}[/dim]")


@app.command("sweep")
def sweep_cmd(
    source: str = typer.Option(..., "-s", "--source"),
    models: list[str] | None = typer.Option(None, "-m", "--model"),
    all_permissive: bool = typer.Option(
        False, "--all-permissive", help="Every model with a permissive licence."
    ),
    duration: float = typer.Option(30.0, "-d", "--duration"),
    device: str = typer.Option("auto", "--device"),
    confidence: float = typer.Option(0.3, "--conf"),
    classes: list[str] | None = typer.Option(None, "-c", "--class"),
    mode: str = typer.Option("realtime", "--mode"),
    imgsz: int | None = typer.Option(None, "--imgsz"),
    precision: str = typer.Option("fp32", "--precision"),
    profile: str | None = typer.Option(None, "--profile"),
    pace_fps: float | None = typer.Option(None, "--pace-fps"),
    out: Path = typer.Option(report_mod.DEFAULT_DIR, "--out"),
) -> None:
    """Run several models against the same source, one after another."""
    keys = _split(models)
    if all_permissive:
        keys = permissive_keys()
    if not keys:
        keys = DEFAULT_SWEEP
    console.print(f"[bold]sweeping[/bold] {', '.join(keys)}")

    config = RunConfig(
        source=source,
        model=keys[0],
        device=device,
        duration_s=duration,
        confidence=confidence,
        classes=_split(classes) or None,
        mode=mode,
        imgsz=imgsz,
        precision=precision,
        profile=profile,
        pace_fps=pace_fps,
    )

    saved: list[dict] = []
    failed: list[tuple[str, str]] = []

    def _on_result(result) -> None:
        console.print(report_mod.summarise(result))
        for warning in result.warnings:
            console.print(f"[yellow]warning:[/yellow] {warning}")
        path = report_mod.save(result, out)
        console.print(f"[dim]saved {path}[/dim]\n")
        saved.append(result.to_dict())

    def _on_error(key: str, exc: Exception) -> None:
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        console.print(f"[red]{key} failed:[/red] {first_line}\n")
        failed.append((key, first_line))

    sweep(config, keys, on_result=_on_result, on_error=_on_error)

    if saved:
        console.print(report_mod.rich_table(saved))
        console.print(f"[dim]{report_mod.context_line(saved)}[/dim]")
    if failed:
        console.print(f"\n[red]{len(failed)} model(s) failed:[/red] {', '.join(k for k, _ in failed)}")
    if not saved:
        raise typer.Exit(code=1)


@app.command("report")
def report_cmd(
    directory: Path = typer.Argument(report_mod.DEFAULT_DIR, help="Results directory."),
    markdown: bool = typer.Option(False, "--markdown", help="Emit a README-ready table."),
    full: bool = typer.Option(False, "--full", help="Show every column, not the compact set."),
    limit: int | None = typer.Option(None, "--limit", help="Only the N most recent runs."),
) -> None:
    """Render saved results as a comparison table."""
    results = report_mod.load_all(directory)
    if limit:
        results = results[:limit]
    if not results:
        console.print(f"[yellow]no results under {directory}[/yellow]")
        raise typer.Exit(code=1)
    if markdown:
        print(report_mod.markdown_table(results))
    else:
        console.print(report_mod.rich_table(results, full=full))
        console.print(f"[dim]{report_mod.context_line(results)}[/dim]")
        console.print(f"[dim]{len(results)} run(s) from {directory}[/dim]")


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the web dashboard."""
    import uvicorn

    uvicorn.run("visionbench.web.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
