from __future__ import annotations

import socket

import pytest

from visionbench.runner import RunConfig, RunResult


def pytest_addoption(parser):
    parser.addoption("--run-e2e", action="store_true", help="Run opt-in CPU CLI/model tests.")


def pytest_collection_modifyitems(config, items):
    for item in items:
        layers = [name for name in ("unit", "integration", "e2e") if item.get_closest_marker(name)]
        if len(layers) > 1:
            raise pytest.UsageError(f"{item.nodeid} must have exactly one test layer")
        if not layers:
            item.add_marker(pytest.mark.unit)
        if item.get_closest_marker("e2e") and not config.getoption("--run-e2e"):
            item.add_marker(pytest.mark.skip(reason="use --run-e2e with a staged HF model cache"))


@pytest.fixture(autouse=True)
def offline_fast_tests(request, monkeypatch):
    if request.node.get_closest_marker("e2e"):
        return
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError("network access is disabled for unit/integration tests")
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError("network access is disabled for unit/integration tests")
        return original_connect_ex(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


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
