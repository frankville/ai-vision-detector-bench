from dataclasses import replace

import pytest
from typer.testing import CliRunner

from visionbench import cli, report


def fake_runner(result, *, fail=None, interrupt=None, status=None):
    calls = []

    def execute(config):
        calls.append(config.model)
        attempt = len(calls)
        if attempt == fail:
            raise RuntimeError("simulated failure")
        if attempt == interrupt:
            raise KeyboardInterrupt
        current = replace(result, schema_version=3)
        current.config = config.redacted()
        current.source = {**result.source, "mode": config.mode}
        current.status = status or "completed"
        return current

    return execute, calls


def test_sweep_repeats_save_raw_before_aggregates(monkeypatch, tmp_path, result):
    execute, calls = fake_runner(result)
    monkeypatch.setattr("visionbench.runner.run", execute)
    response = CliRunner().invoke(
        cli.app,
        [
            "sweep",
            "-s",
            "clip.mp4",
            "--device",
            "cpu",
            "-m",
            "dfine-nano,rtdetrv2-r18",
            "--repeat",
            "3",
            "--out",
            str(tmp_path),
        ],
    )
    assert response.exit_code == 0, response.output
    assert calls == ["dfine-nano", "rtdetrv2-r18"] * 3
    raw = report.load_all(tmp_path)
    aggregates = report.load_all(tmp_path, kind="aggregate")
    assert len(raw) == 6
    assert len(aggregates) == 2
    assert all(aggregate["counts"]["eligible"] == 3 for aggregate in aggregates)
    assert len({run["run_id"] for run in raw}) == 6
    assert "eligible/requested" in response.output
    for aggregate in aggregates:
        assert all((tmp_path / reference["path"]).is_file() for reference in aggregate["runs"])


def test_partial_failures_preserve_runs_and_fail_exit(monkeypatch, tmp_path, result):
    execute, calls = fake_runner(result, fail=2)
    monkeypatch.setattr("visionbench.runner.run", execute)
    response = CliRunner().invoke(
        cli.app,
        [
            "sweep",
            "-s",
            "clip.mp4",
            "--device",
            "cpu",
            "-m",
            "dfine-nano",
            "--repeat",
            "3",
            "--out",
            str(tmp_path),
        ],
    )
    assert response.exit_code == 1, response.output
    assert len(calls) == 3
    assert len(report.load_all(tmp_path)) == 2
    aggregate = report.load_all(tmp_path, kind="aggregate")[0]
    assert aggregate["counts"]["eligible"] == 2
    assert aggregate["counts"]["failed"] == 1
    assert aggregate["counts"]["unattempted"] == 0
    assert "2/3 eligible" in response.output


def test_setup_interrupt_exports_completed_subset(monkeypatch, tmp_path, result):
    execute, calls = fake_runner(result, interrupt=2)
    monkeypatch.setattr("visionbench.runner.run", execute)
    response = CliRunner().invoke(
        cli.app,
        [
            "sweep",
            "-s",
            "clip.mp4",
            "--device",
            "cpu",
            "-m",
            "dfine-nano",
            "--repeat",
            "3",
            "--out",
            str(tmp_path),
        ],
    )
    assert response.exit_code == 130, response.output
    assert len(calls) == 2
    aggregate = report.load_all(tmp_path, kind="aggregate")[0]
    assert aggregate["counts"]["eligible"] == 1
    assert aggregate["counts"]["interrupted"] == 1
    assert aggregate["counts"]["unattempted"] == 1


@pytest.mark.parametrize("status,code", [("interrupted", 130), ("empty", 1)])
def test_ineligible_results_are_saved_but_not_averaged(monkeypatch, tmp_path, result, status, code):
    execute, calls = fake_runner(result, status=status)
    monkeypatch.setattr("visionbench.runner.run", execute)
    response = CliRunner().invoke(
        cli.app,
        [
            "sweep",
            "-s",
            "clip.mp4",
            "--device",
            "cpu",
            "-m",
            "dfine-nano",
            "--repeat",
            "3",
            "--out",
            str(tmp_path),
        ],
    )
    assert response.exit_code == code, response.output
    aggregate = report.load_all(tmp_path, kind="aggregate")[0]
    assert aggregate["counts"]["eligible"] == 0
    assert aggregate["metrics"]["fps"]["median"] is None
    assert len(calls) == (1 if status == "interrupted" else 3)


def test_default_sweep_saves_only_raw(monkeypatch, tmp_path, result):
    execute, calls = fake_runner(result)
    monkeypatch.setattr("visionbench.runner.run", execute)
    response = CliRunner().invoke(
        cli.app,
        ["sweep", "-s", "clip.mp4", "--device", "cpu", "-m", "dfine-nano", "--out", str(tmp_path)],
    )
    assert response.exit_code == 0, response.output
    assert calls == ["dfine-nano"]
    assert len(report.load_all(tmp_path)) == 1
    assert not report.load_all(tmp_path, kind="aggregate")


@pytest.mark.parametrize(
    "flags", [["--repeat", "0"], ["--repeat", "-2"], ["-m", "dfine-nano,dfine-nano"]]
)
def test_invalid_repetition_request_does_not_start(monkeypatch, tmp_path, flags):
    def no_run(*args, **kwargs):
        pytest.fail("started benchmark")

    monkeypatch.setattr(cli, "sweep", no_run)
    response = CliRunner().invoke(
        cli.app, ["sweep", "-s", "clip.mp4", "--device", "cpu", "--out", str(tmp_path), *flags]
    )
    assert response.exit_code == 2, response.output


def test_report_selects_aggregate_records(monkeypatch, tmp_path, result):
    execute, _ = fake_runner(result)
    monkeypatch.setattr("visionbench.runner.run", execute)
    CliRunner().invoke(
        cli.app,
        [
            "sweep",
            "-s",
            "clip.mp4",
            "--device",
            "cpu",
            "-m",
            "dfine-nano",
            "--repeat",
            "2",
            "--out",
            str(tmp_path),
        ],
    )
    response = CliRunner().invoke(
        cli.app, ["report", str(tmp_path), "--kind", "aggregate", "--markdown"]
    )
    assert response.exit_code == 0, response.output
    assert "FPS median" in response.output
    assert "2/2" in response.output
    raw = CliRunner().invoke(cli.app, ["report", str(tmp_path), "--markdown"])
    assert raw.exit_code == 0
    assert "FPS median" not in raw.output


def test_bad_report_kind_fails():
    response = CliRunner().invoke(cli.app, ["report", "--kind", "manifest"])
    assert response.exit_code == 2
