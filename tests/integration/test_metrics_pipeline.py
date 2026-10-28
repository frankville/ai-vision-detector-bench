"""Deterministic timing samples composed into saved runs and aggregate reports."""

import pytest

from visionbench import report
from visionbench.aggregate import aggregate_runs
from visionbench.metrics import timers
from visionbench.metrics.system import energy_per_frame_j
from visionbench.metrics.timers import RateMeter, Series, StageTimers
from visionbench.models.base import stage
from visionbench.runner import RunResult

pytestmark = pytest.mark.integration


def measured_result(monkeypatch, repetition, durations, completions, power):
    clock = [0.0]
    stages = StageTimers()
    rate = RateMeter(window_s=0.6)
    with monkeypatch.context() as patch:
        patch.setattr(timers.time, "perf_counter", lambda: clock[0])
        for duration, completion in zip(durations, completions, strict=True):
            clock[0] = completion - duration / 1000
            with stage() as elapsed:
                clock[0] = completion
            stages.add("inference_ms", elapsed[0])
            rate.mark()
    return RunResult(
        schema_version=3, profile="deterministic-cpu", started_at="2026-10-27T12:00:00+0000",
        elapsed_s=rate.elapsed_s, config={"model": "metrics-pipeline"}, host={},
        model={"device": "cpu", "backend": "pytorch", "precision": "fp32"},
        source={"label": "synthetic-timings", "mode": "sequential"},
        throughput={"frames_processed": rate.total, "processed_fps": rate.mean_fps},
        latency=stages.summary(),
        resources={"energy_per_frame_j": energy_per_frame_j(power, rate.mean_fps)},
        detections={}, group_id="timing-group", repetition=repetition, repeat_count=2,
    ), rate


def test_stage_clocks_rates_and_energy_survive_aggregate_report(monkeypatch, tmp_path):
    power = Series("gpu_power_w")
    power.add(30)
    power.add(60)
    first, first_rate = measured_result(
        monkeypatch, 1, [1, 2, 4, 9], [0, 0.25, 0.5, 1], power,
    )
    second, second_rate = measured_result(
        monkeypatch, 2, [10, 20, 40, 90], [0, 0.5, 1, 2], None,
    )
    assert first.latency["inference_ms"] == {
        "count": 4, "mean": 4.0, "p50": 3.0, "p95": 8.25, "p99": 8.85, "max": 9.0,
    }
    assert first_rate.mean_fps == 3.0
    assert first_rate.instant_fps == 2.0
    assert first_rate.elapsed_s == 1.0
    assert second_rate.mean_fps == 1.5
    assert second_rate.instant_fps == 0.0
    assert first.resources["energy_per_frame_j"] == 15.0
    assert second.resources["energy_per_frame_j"] is None

    attempts = []
    for result in (first, second):
        report.save(result, tmp_path)
        attempts.append({
            "group_id": result.group_id, "repeat_count": 2, "model": "metrics-pipeline",
            "repetition": result.repetition, "run_id": result.run_id, "status": "completed",
        })
    raw = report.load_all(tmp_path)
    assert {report.to_row(run).infer_p95 for run in raw} == {8.25, 82.5}
    aggregate = aggregate_runs(raw, attempts=attempts, root=tmp_path)
    assert aggregate["metrics"]["fps"] == {
        "count": 2, "median": 2.25, "mean": 2.25,
        "min": 1.5, "max": 3.0, "stddev": pytest.approx(1.0606601718),
    }
    assert aggregate["metrics"]["inference_p95_ms"] == {"count": 2, "median": 45.375}
    assert aggregate["metrics"]["energy_per_frame_j"] == {"count": 1, "median": 15.0}
    report.save_aggregate(aggregate, tmp_path)
    loaded, = report.load_all(tmp_path, kind="aggregate")
    assert loaded["metrics"] == aggregate["metrics"]
    table = report.aggregate_markdown_table([loaded])
    assert "45.38 ms" in table
    assert "15.00 J (1/2)" in table
