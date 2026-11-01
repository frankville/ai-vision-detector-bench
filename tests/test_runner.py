import time

import numpy as np
import pytest

from visionbench import runner
from visionbench.models.base import Detector
from visionbench.types import Detections, Frame, InferTimings, SourceStats

pytestmark = pytest.mark.integration


def test_measured_run_records_effective_settings_and_excludes_warmup(monkeypatch):
    events = []
    built_settings = {}
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    class TestDetector(Detector):
        def load(self):
            events.append("load")

        def infer(self, frame):
            events.append("infer")
            return Detections.empty(), InferTimings(1.0, 2.0, 3.0)

        def unload(self):
            events.append("unload")

    def build(key, **settings):
        built_settings.update(settings)
        return TestDetector(**settings)

    class Source:
        seq = 0

        def read(self, timeout):
            frame = Frame(image, self.seq, time.perf_counter())
            self.seq += 1
            return frame

        def reset_counters(self):
            events.append("reset")

        def close(self):
            events.append("close")

        def stats(self):
            return SourceStats(2, 2, 0, 10.0, 10.0, 2, 2)

    class Sampler:
        series = {}

        def __init__(self, device):
            assert device == "cpu"

        def start(self):
            events.append("measurement_start")

        def stop(self):
            events.append("measurement_stop")

        def summary(self):
            return {}

    monkeypatch.setattr(runner, "build_model", build)
    monkeypatch.setattr(runner, "open_source", lambda *a, **kw: Source())
    monkeypatch.setattr(runner, "SystemSampler", Sampler)
    monkeypatch.setattr(runner, "host_info", lambda: {})
    config = runner.RunConfig(
        "clip.mp4", "dfine-nano", device="cpu", max_frames=2,
        warmup_frames=3, profile="test-cpu", revision="release",
    )
    result = runner.run(config)

    assert built_settings["backend"] == "pytorch"
    assert built_settings["precision"] == "fp32"
    assert built_settings["revision"] == result.config["revision"] == "release"
    assert result.schema_version == 4
    assert result.config["device"] == "cpu"
    assert result.config["backend"] == result.model["backend"] == "pytorch"
    assert result.config["precision"] == result.model["precision"] == "fp32"
    assert result.throughput["frames_processed"] == 2
    assert result.latency["inference_ms"]["count"] == 2
    assert result.resources["energy_per_frame_j"] is None
    assert events == [
        "load", "infer", "infer", "infer", "reset", "measurement_start",
        "infer", "infer", "measurement_stop", "close", "unload",
    ]


def test_api_result_rows_include_backend(monkeypatch, result):
    from fastapi.testclient import TestClient

    from visionbench.web import app as web

    monkeypatch.setattr(web.report_mod, "load_all", lambda: [result.to_dict()])
    response = TestClient(web.app).get("/api/results")
    assert response.status_code == 200
    assert response.json()["rows"][0]["backend"] == "pytorch"
    assert response.json()["rows"][0]["precision"] == "fp32"
