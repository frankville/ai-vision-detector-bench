"""Opt-in subprocess checks of the installed CLI with real CPU inference."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.e2e


def test_installed_cli_run_sweep_and_reports(tmp_path):
    executable = Path(sys.executable).parent / (
        "visionbench.exe" if os.name == "nt" else "visionbench"
    )
    assert executable.is_file(), "Install VisionBench before running E2E tests"
    artifact_root = Path(os.environ.get("VISIONBENCH_E2E_ARTIFACTS", tmp_path / "artifacts")).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="cli-", dir=artifact_root))
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NO_COLOR": "1",
        "COLUMNS": "240",
        "PYTHONSAFEPATH": "1",
    }
    env.pop("PYTHONPATH", None)

    def command(label, args):
        process = subprocess.run(
            args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=240
        )
        (artifacts / f"{label}.stdout.log").write_text(process.stdout)
        (artifacts / f"{label}.stderr.log").write_text(process.stderr)
        assert process.returncode == 0, f"{label} failed:\n{process.stdout}\n{process.stderr}"
        return process.stdout

    installed = json.loads(
        command(
            "installation",
            [
                sys.executable,
                "-c",
                (
                    "import importlib.resources, json, visionbench; "
                    "print(json.dumps({'module': visionbench.__file__, 'dashboard': "
                    "importlib.resources.files('visionbench.web').joinpath('static/index.html').is_file()}))"
                ),
            ],
        )
    )
    assert installed["dashboard"], "The installed package must include the dashboard"
    if os.environ.get("VISIONBENCH_EXPECT_WHEEL") == "1":
        checkout = Path(__file__).resolve().parents[2]
        assert not Path(installed["module"]).resolve().is_relative_to(checkout)

    clip = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened(), "MJPG encoding is required for this smoke test"
    try:
        for index in range(12):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            cv2.rectangle(frame, (index, 8), (index + 16, 32), (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()

    common = [
        "-s",
        str(clip),
        "-m",
        "dfine-nano",
        "--device",
        "cpu",
        "--precision",
        "fp32",
        "--backend",
        "pytorch",
        "--mode",
        "sequential",
        "--imgsz",
        "320",
        "--profile",
        "cpu-e2e",
    ]
    revision = os.environ.get("E2E_MODEL_REVISION", "066438d3d8f0da137a37b38fdf3368fd4afceced")
    common.extend(["--revision", revision])
    single_dir = artifacts / "single"
    command("run", [str(executable), "run", *common, "--frames", "3", "--out", str(single_dir)])
    single_paths = list(single_dir.rglob("*.json"))
    assert len(single_paths) == 1
    single = json.loads(single_paths[0].read_text())
    assert single["throughput"]["frames_processed"] == 3
    assert single["latency"]["inference_ms"]["count"] == 3

    sweep_dir = artifacts / "repeated"
    command(
        "sweep",
        [str(executable), "sweep", *common, "--repeat", "2", "-d", "8", "--out", str(sweep_dir)],
    )
    records = {path: json.loads(path.read_text()) for path in sweep_dir.rglob("*.json")}
    raw = {path: data for path, data in records.items() if data["record_type"] == "run"}
    (aggregate,) = [data for data in records.values() if data["record_type"] == "aggregate"]
    assert len(raw) == 2
    assert len({data["run_id"] for data in raw.values()}) == 2
    assert {data["repetition"] for data in raw.values()} == {1, 2}
    assert aggregate["counts"]["eligible"] == aggregate["counts"]["requested"] == 2
    assert aggregate["metrics"]["fps"]["count"] == 2
    for reference in aggregate["runs"]:
        assert records[sweep_dir / reference["path"]]["run_id"] == reference["run_id"]

    for data in [single, *raw.values()]:
        assert data["schema_version"] == 4
        assert data["config"]["revision"] == revision
        assert data["model"]["revision"] == revision
        assert data["model"]["checkpoint_commit"] == revision
        assert data["status"] == "completed"
        assert data["model"]["device"] == "cpu"
        assert data["model"]["weights"] == "ustc-community/dfine-nano-coco"
        assert data["model"]["backend"] == "pytorch"
        assert data["model"]["precision"] == "fp32"
        assert data["model"]["input_shape"] == [1, 3, 320, 320]
        assert data["throughput"]["frames_dropped"] == 0
        assert data["throughput"]["frames_processed"] >= 2
        for stage in data["latency"].values():
            assert stage["count"] == data["throughput"]["frames_processed"]
            assert all(
                math.isfinite(stage[key]) and stage[key] >= 0 for key in ("p50", "p95", "p99")
            )

    raw_report = command("raw-report", [str(executable), "report", str(sweep_dir), "--markdown"])
    aggregate_report = command(
        "aggregate-report",
        [str(executable), "report", str(sweep_dir), "--kind", "aggregate", "--markdown"],
    )
    assert "dfine-nano" in raw_report
    assert "FPS median" not in raw_report
    assert "FPS median" in aggregate_report and "2/2" in aggregate_report
