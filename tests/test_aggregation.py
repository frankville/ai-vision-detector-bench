import copy
import json
import math
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from visionbench import report
from visionbench.aggregate import aggregate_runs
from visionbench.web import app as web


@pytest.fixture
def group(result):
    runs = []
    for repetition, fps in enumerate((10.0, 20.0, 60.0), 1):
        run = replace(
            result,
            schema_version=3,
            run_id=f"run{repetition}",
            group_id="group1",
            repetition=repetition,
            repeat_count=3,
        ).to_dict()
        run["throughput"]["processed_fps"] = fps
        run["latency"] = {
            "inference_ms": {"p50": repetition * 10.0, "p95": repetition * 30.0},
            "end_to_end_ms": {"p95": repetition * 50.0},
        }
        run["resources"] = {
            "counters": {
                "gpu_power_w": {"mean": (0.0, 100.0, None)[repetition - 1]},
                "gpu_mem_used_mb": {"mean": 200.0 * repetition},
                "gpu_util_pct": {"mean": 10.0 * repetition},
            },
            "energy_per_frame_j": 0.5 * repetition,
        }
        runs.append(run)
    return runs


def outcomes(runs):
    return [
        {key: run[key] for key in ("group_id", "repetition", "repeat_count", "run_id", "status")}
        | {"model": run["config"]["model"]}
        for run in runs
    ]


def test_known_statistics_and_optional_telemetry(group):
    original = copy.deepcopy(group)
    aggregate = aggregate_runs(group, attempts=outcomes(group))
    metrics = aggregate["metrics"]
    assert metrics["fps"] == {
        "count": 3,
        "median": 20.0,
        "mean": 30.0,
        "min": 10.0,
        "max": 60.0,
        "stddev": pytest.approx(math.sqrt(700)),
    }
    assert metrics["inference_p50_ms"] == {"count": 3, "median": 20.0}
    assert metrics["inference_p95_ms"] == {"count": 3, "median": 60.0}
    assert metrics["e2e_p95_ms"] == {"count": 3, "median": 100.0}
    assert metrics["gpu_power_w"] == {"count": 2, "median": 50.0}
    assert metrics["gpu_mem_mb"] == {"count": 3, "median": 400.0}
    assert metrics["gpu_util_pct"] == {"count": 3, "median": 20.0}
    assert metrics["energy_per_frame_j"] == {"count": 3, "median": 1.0}
    assert aggregate["counts"] == {
        "requested": 3,
        "eligible": 3,
        "completed": 3,
        "empty": 0,
        "interrupted": 0,
        "failed": 0,
        "unattempted": 0,
    }
    assert group == original


def test_partial_group_distinguishes_failure_and_unattempted(group):
    runs = group[:1]
    attempts = outcomes(runs) + [
        {
            "group_id": "group1",
            "repeat_count": 3,
            "repetition": 2,
            "model": "dfine-nano",
            "status": "failed",
            "run_id": None,
            "error_type": "RuntimeError",
        }
    ]
    aggregate = aggregate_runs(runs, attempts=attempts)
    assert aggregate["counts"]["eligible"] == 1
    assert aggregate["counts"]["failed"] == 1
    assert aggregate["counts"]["unattempted"] == 1
    assert aggregate["metrics"]["fps"]["stddev"] is None
    assert aggregate["attempts"][1]["error_type"] == "RuntimeError"
    assert aggregate["attempts"][2]["status"] == "unattempted"


@pytest.mark.parametrize("status", ["interrupted", "empty"])
def test_incomplete_runs_preserved_but_excluded(group, status):
    group[1]["status"] = status
    aggregate = aggregate_runs(group, attempts=outcomes(group))
    assert aggregate["counts"]["eligible"] == 2
    assert aggregate["counts"][status] == 1
    assert aggregate["metrics"]["fps"]["median"] == 35.0
    assert len(aggregate["runs"]) == 3
    assert aggregate["excluded_runs"][0]["run_id"] == "run2"


def test_zero_eligible_group_has_null_stats(group):
    for run in group:
        run["status"] = "empty"
        run["throughput"]["frames_processed"] = 1
    aggregate = aggregate_runs(group, attempts=outcomes(group))
    assert aggregate["counts"]["eligible"] == 0
    assert all(
        metric["count"] == 0 and metric["median"] is None
        for metric in aggregate["metrics"].values()
    )


def test_warmup_interrupt_preserves_completed_subset(group):
    for run in group:
        run["model"]["input_shape"] = [1, 3, 640, 640]
    group[1]["status"] = "interrupted"
    group[1]["throughput"]["frames_processed"] = 0
    group[1]["model"]["input_shape"] = None
    ordered = [group[1], group[0]]
    aggregate = aggregate_runs(ordered, attempts=outcomes(ordered))
    assert aggregate["counts"]["eligible"] == 1
    assert aggregate["counts"]["interrupted"] == 1
    assert aggregate["counts"]["unattempted"] == 1
    assert aggregate["model"]["input_shape"] == [1, 3, 640, 640]
    assert aggregate["runs"][1]["status"] == "interrupted"


def test_measured_interrupted_shape_mismatch_rejected(group):
    group[0]["model"]["input_shape"] = [1, 3, 640, 640]
    group[1]["model"]["input_shape"] = None
    group[1]["status"] = "interrupted"
    group[1]["throughput"]["frames_processed"] = 1
    with pytest.raises(ValueError, match="incompatible"):
        aggregate_runs(group[:2], attempts=outcomes(group[:2]))


def test_warmup_interrupt_still_checks_precision(group):
    group[1]["status"] = "interrupted"
    group[1]["throughput"]["frames_processed"] = 0
    group[1]["model"]["precision"] = "fp16"
    with pytest.raises(ValueError, match="incompatible"):
        aggregate_runs(group[:2], attempts=outcomes(group[:2]))


def test_missing_invalid_telemetry_is_not_zero(group):
    for run in group:
        run["resources"] = {"counters": {"gpu_power_w": {"mean": float("nan")}}}
    aggregate = aggregate_runs(group, attempts=outcomes(group))
    assert aggregate["metrics"]["gpu_power_w"] == {"count": 0, "median": None}
    assert aggregate["metrics"]["energy_per_frame_j"] == {"count": 0, "median": None}


def test_rounded_elapsed_does_not_discard_valid_interval(group):
    group[0]["elapsed_s"] = 0.0
    group[0]["throughput"]["processed_fps"] = 0.0
    aggregate = aggregate_runs(group, attempts=outcomes(group))
    assert aggregate["counts"]["eligible"] == 3


@pytest.mark.parametrize(
    "key,value",
    [
        ("group_id", "other"),
        ("run_id", "run1"),
        ("repetition", 1),
        ("repetition", 5),
        ("schema_version", 2),
        ("record_type", "aggregate"),
        ("repeat_count", 4),
        ("status", "failed"),
    ],
)
def test_rejects_invalid_or_duplicate_members(group, key, value):
    group[1][key] = value
    with pytest.raises(ValueError):
        aggregate_runs(group, attempts=outcomes(group))


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("config", "precision", "fp16"),
        ("config", "confidence", 0.7),
        ("config", "note", "live preview"),
        ("model", "backend", "tensorrt"),
        ("model", "input_shape", [1, 3, 8, 8]),
        ("source", "width", 1920),
        ("host", "cpu_model", "different"),
    ],
)
def test_rejects_incompatible_members(group, section, key, value):
    group[1][section][key] = value
    with pytest.raises(ValueError, match="incompatible"):
        aggregate_runs(group, attempts=outcomes(group))


def test_rejects_missing_and_duplicate_attempts(group):
    with pytest.raises(ValueError, match="attempt"):
        aggregate_runs(group, attempts=outcomes(group)[:2])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_runs(group, attempts=outcomes(group) + outcomes(group)[:1])


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        aggregate_runs([], attempts=[])


def test_round_trip_and_mixed_directory_filtering(group, tmp_path, monkeypatch):
    from visionbench.runner import RunResult

    for run in group:
        path = report.save(RunResult(**run), tmp_path)
        run["_path"] = str(path)
    aggregate = aggregate_runs(group, attempts=outcomes(group), root=tmp_path)
    aggregate_path = report.save_aggregate(aggregate, tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 1, "config": {}}))
    (tmp_path / "array.json").write_text("[]")
    (tmp_path / "malformed-aggregate.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "aggregate",
                "config": {},
                "model": {},
                "counts": {},
                "metrics": {},
            }
        )
    )
    for section, key, value in [
        (None, "profile", None),
        ("model", "backend", {}),
        ("model", "precision", []),
        ("source", "mode", None),
    ]:
        malformed = copy.deepcopy(aggregate)
        target = malformed if section is None else malformed[section]
        target[key] = value
        (tmp_path / f"invalid-{key}.json").write_text(json.dumps(malformed))
    raw = report.load_all(tmp_path)
    summaries = report.load_all(tmp_path, kind="aggregate")
    assert len(raw) == 3
    assert len(summaries) == 1
    assert summaries[0]["_path"] == str(aggregate_path)
    for reference in summaries[0]["runs"]:
        assert (tmp_path / reference["path"]).is_file()
        assert not reference["path"].startswith("/")
    assert "50.00 W (2/3)" in report.aggregate_markdown_table(summaries)
    assert "3/3" in report.aggregate_markdown_table(summaries)
    from typer.testing import CliRunner

    from visionbench.cli import app

    response = CliRunner().invoke(
        app, ["report", str(tmp_path), "--kind", "aggregate", "--markdown"]
    )
    assert response.exit_code == 0, response.output
    assert "3/3" in response.output
    with pytest.raises(ValueError):
        report.markdown_table(summaries)
    monkeypatch.setattr(web.report_mod, "load_all", lambda: raw)
    assert TestClient(web.app).get("/api/results").json()["count"] == 3


def test_rejects_external_result_reference(group, tmp_path):
    group[0]["_path"] = str(tmp_path.parent / "external.json")
    with pytest.raises(ValueError):
        aggregate_runs(group, attempts=outcomes(group), root=tmp_path)


def test_raw_save_same_second_preserves_results(result, tmp_path):
    first = report.save(result, tmp_path)
    second = report.save(result, tmp_path)
    assert first != second
    assert json.loads(first.read_text()) == json.loads(second.read_text())
