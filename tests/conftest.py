from __future__ import annotations

import pytest

from visionbench.runner import RunConfig, RunResult


@pytest.fixture
def result() -> RunResult:
    config = RunConfig("clip.mp4", "dfine-nano", device="cpu")
    return RunResult(
        schema_version=2,
        profile="test-cpu",
        started_at="2026-10-25T12:00:00+0000",
        elapsed_s=1.0,
        config=config.redacted(),
        host={},
        model={"device": "cpu", "backend": "pytorch", "precision": "fp32"},
        source={"label": "clip.mp4", "mode": "sequential"},
        throughput={
            "frames_processed": 10,
            "processed_fps": 10.0,
            "realtime_factor": 1.0,
            "capacity_streams": 1.0,
            "frames_dropped": 0,
            "drop_ratio": 0.0,
        },
        latency={"inference_ms": {"p50": 1.0, "p95": 2.0}},
        resources={},
        detections={},
    )
