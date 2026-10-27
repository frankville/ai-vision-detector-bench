import time
from dataclasses import replace

import numpy as np
import pytest

from visionbench import runner
from visionbench.sources import base
from visionbench.types import Detections, Frame, InferTimings, SourceStats


def config():
    return runner.RunConfig("clip.mp4", "a", device="cpu", max_frames=2, profile="cpu")


def test_interleaved_order_independent_configs_and_identity(monkeypatch, result):
    seen = []
    cfg = config()
    cfg.classes = ["person"]

    def run(attempt):
        seen.append((attempt.model, attempt.classes[:], attempt))
        attempt.classes.append("changed")
        return result

    monkeypatch.setattr(runner, "run", run)
    saved, events = [], []
    rows = runner.sweep(cfg, ["a", "b"], saved.append, repeat=3, on_attempt=events.append)
    assert [key for key, _, _ in seen] == ["a", "b"] * 3
    assert all(classes == ["person"] for _, classes, _ in seen)
    assert len({id(attempt) for _, _, attempt in seen}) == 6
    assert cfg.classes == ["person"]
    assert len({row.run_id for row in rows}) == 6
    assert len({row.group_id for row in rows[::2]}) == 1
    assert rows[0].group_id != rows[1].group_id
    assert [row.repetition for row in rows] == [1, 1, 2, 2, 3, 3]
    assert all(row.repeat_count == 3 for row in rows)
    assert rows == saved
    assert [event["run_id"] for event in events] == [row.run_id for row in rows]
    assert result.group_id is None
    assert result.repetition == result.repeat_count == 1
    assert result.record_type == "run"


@pytest.mark.parametrize("repeat", [True, False, 0, -1, 1.5, "2", None])
def test_invalid_repeat_before_attempt(monkeypatch, repeat):
    monkeypatch.setattr(runner, "run", lambda _: pytest.fail("attempted invalid repeat"))
    with pytest.raises(ValueError, match="repeat"):
        runner.sweep(config(), ["a"], repeat=repeat)


@pytest.mark.parametrize("models", [[], ["a", "a"], [""], [" "]])
def test_invalid_models_before_attempt(models):
    with pytest.raises(ValueError, match="models"):
        runner.sweep(config(), models)


def test_invalid_mutated_config_before_attempt(monkeypatch):
    cfg = config()
    cfg.precision = "invalid"
    monkeypatch.setattr(runner, "run", lambda _: pytest.fail("attempted invalid config"))
    with pytest.raises(ValueError):
        runner.sweep(cfg, ["a"], on_error=lambda *_: None)


def test_failures_continue_and_attempt_events_are_safe(monkeypatch, result):
    called, errors, events = [], [], []

    def run(cfg):
        called.append(cfg.model)
        if len(called) == 1:
            raise RuntimeError("secret source credential")
        return result

    monkeypatch.setattr(runner, "run", run)
    rows = runner.sweep(config(), ["a", "b"], on_error=lambda *e: errors.append(e),
                        repeat=2, on_attempt=events.append)
    assert called == ["a", "b", "a", "b"]
    assert len(rows) == 3
    assert errors[0][0] == "a (repeat 1/2)"
    assert events[0]["status"] == "failed"
    assert events[0]["error_type"] == "RuntimeError"
    assert events[0]["run_id"] is None
    assert "secret" not in str(events)
    assert events[0]["group_id"] == events[2]["group_id"]


def test_default_repeat_and_error_propagation(monkeypatch, result):
    monkeypatch.setattr(runner, "run", lambda _: result)
    rows = runner.sweep(config(), ["a"])
    assert len(rows) == 1 and rows[0].repeat_count == 1
    monkeypatch.setattr(runner, "run", lambda _: (_ for _ in ()).throw(ValueError("bad")))
    errors = []
    runner.sweep(config(), ["a"], on_error=lambda *args: errors.append(args))
    assert errors[0][0] == "a"
    with pytest.raises(ValueError):
        runner.sweep(config(), ["a"])


def test_interruption_saved_before_attempt_event_then_stops(monkeypatch, result):
    monkeypatch.setattr(runner, "run", lambda _: replace(result, status="interrupted"))
    calls = []
    rows = runner.sweep(config(), ["a", "b"], lambda row: calls.append("saved"),
                        repeat=3, on_attempt=lambda event: calls.append(event["status"]))
    assert len(rows) == 1
    assert calls == ["saved", "interrupted"]


def test_load_interrupt_propagates_with_attempt_event(monkeypatch):
    monkeypatch.setattr(runner, "run", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
    events = []
    with pytest.raises(KeyboardInterrupt):
        runner.sweep(config(), ["a"], repeat=2, on_attempt=events.append)
    assert len(events) == 1 and events[0]["status"] == "interrupted"


def test_callback_failure_is_not_reported_as_model_failure(monkeypatch, result):
    monkeypatch.setattr(runner, "run", lambda _: result)
    errors, events = [], []
    with pytest.raises(OSError):
        runner.sweep(config(), ["a"],
                     on_result=lambda _: (_ for _ in ()).throw(OSError("disk full")),
                     on_error=lambda *args: errors.append(args), on_attempt=events.append)
    assert errors == events == []


@pytest.fixture
def fake_runtime(monkeypatch):
    events = []
    fail = {}
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    def event(name):
        events.append(name)
        if name in fail:
            raise fail[name]

    class Detector:
        device = "cpu"

        def load(self):
            event("load")

        def unload(self):
            event("unload")

        def info(self):
            event("info")
            return {}

        def warmup(self, image, rounds):
            event("warmup")

        def infer(self, image):
            event("infer")
            return Detections.empty(), InferTimings(1, 2, 3)

    class Source:
        def read(self, timeout):
            event("read")
            return Frame(image, 0, time.perf_counter())

        def reset_counters(self):
            event("reset")

        def close(self):
            event("close")

        def stats(self):
            return SourceStats(2, 2, 0, 10, 10, 2, 2)

    class Sampler:
        series = {}

        def __init__(self, device):
            event("sampler_init")

        def start(self):
            event("start")

        def stop(self):
            event("stop")

        def summary(self):
            return {}

    def open_source(*args, **kwargs):
        event("open")
        return Source()

    monkeypatch.setattr(runner, "build_model", lambda *a, **kw: Detector())
    monkeypatch.setattr(runner, "open_source", open_source)
    monkeypatch.setattr(runner, "SystemSampler", Sampler)
    monkeypatch.setattr(runner, "host_info", lambda: {})
    return events, fail


@pytest.mark.integration
def test_warmup_interrupt_exports_completed_subset(fake_runtime, monkeypatch, tmp_path):
    from visionbench import report
    from visionbench.aggregate import aggregate_runs

    loads, saved, attempts = [], [], []

    class ShapeDetector:
        device = "cpu"
        shape = None

        def load(self):
            loads.append(self)

        def unload(self):
            pass

        def info(self):
            return {"device": "cpu", "backend": "pytorch", "precision": "fp32", "input_shape": self.shape}

        def warmup(self, image, rounds):
            if len(loads) == 2:
                raise KeyboardInterrupt
            self.shape = [1, 3, 2, 2]

        def infer(self, image):
            return Detections.empty(), InferTimings(1, 2, 3)

    def save_result(result):
        path = report.save(result, tmp_path)
        saved.append(result.to_dict() | {"_path": str(path)})

    monkeypatch.setattr(runner, "build_model", lambda *args, **kwargs: ShapeDetector())
    runner.sweep(config(), ["a"], on_result=save_result, repeat=3, on_attempt=attempts.append)
    aggregate = aggregate_runs(saved, attempts=attempts, root=tmp_path)
    report.save_aggregate(aggregate, tmp_path)
    assert len(loads) == 2
    assert aggregate["counts"]["eligible"] == 1
    assert aggregate["counts"]["interrupted"] == 1
    assert aggregate["counts"]["unattempted"] == 1
    assert aggregate["model"]["input_shape"] == [1, 3, 2, 2]
    assert all((tmp_path / reference["path"]).is_file() for reference in aggregate["runs"])
    assert report.load_all(tmp_path, kind="aggregate")[0]["counts"]["eligible"] == 1


@pytest.mark.parametrize("failure,cleanup", [
    ("load", ["unload"]), ("open", ["unload"]),
    ("sampler_init", ["close", "unload"]),
    ("info", ["stop", "close", "unload"]),
    ("warmup", ["stop", "close", "unload"]),
    ("start", ["stop", "close", "unload"]),
    ("infer", ["stop", "close", "unload"]),
])
def test_failed_run_cleans_up_acquired_resources(fake_runtime, failure, cleanup):
    events, fail = fake_runtime
    fail[failure] = RuntimeError("injected")
    with pytest.raises(RuntimeError):
        runner.run(config())
    assert events[-len(cleanup):] == cleanup


def test_each_repeat_reloads_warms_and_cleans_up(fake_runtime):
    events, _ = fake_runtime
    rows = runner.sweep(config(), ["a", "b"], repeat=2)
    assert len(rows) == 4
    assert events.count("load") == events.count("warmup") == events.count("unload") == 4
    for start in [i for i, event in enumerate(events) if event == "start"]:
        assert events[start - 2:start] == ["info", "reset"]
    assert all(row.schema_version == 3 for row in rows)


@pytest.mark.parametrize("count", [0, 1])
def test_short_runs_are_empty_without_changing_fps(fake_runtime, count):
    cfg = replace(config(), max_frames=count)
    row = runner.run(cfg)
    assert row.status == "empty"
    assert row.throughput["frames_processed"] == count
    assert row.throughput["processed_fps"] == 0


def test_zero_completion_interval_is_empty(fake_runtime, monkeypatch):
    monkeypatch.setattr(runner.time, "perf_counter", lambda: 100.0)
    row = runner.run(config())
    assert row.status == "empty"
    assert row.throughput["frames_processed"] == 2
    assert row.throughput["processed_fps"] == 0


def test_should_stop_and_inference_interrupt_mark_partial_run(fake_runtime):
    events, fail = fake_runtime
    row = runner.run(config(), should_stop=lambda: True)
    assert row.status == "interrupted"
    fail["infer"] = KeyboardInterrupt()
    row = runner.run(config())
    assert row.status == "interrupted"
    assert events[-3:] == ["stop", "close", "unload"]
    fail.clear()
    fail["load"] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        runner.run(config())
    assert events[-1] == "unload"


def test_cleanup_continues_when_sampler_stop_fails(fake_runtime):
    events, fail = fake_runtime
    fail["stop"] = RuntimeError("stop failed")
    with pytest.raises(RuntimeError):
        runner.run(config())
    assert events[-3:] == ["stop", "close", "unload"]


def test_open_source_closes_partial_open(monkeypatch):
    events = []

    def open(self):
        events.append("open")
        raise RuntimeError("partial open")

    monkeypatch.setattr(base.SequentialSource, "open", open)
    monkeypatch.setattr(base.SequentialSource, "close", lambda self: events.append("close"))
    with pytest.raises(RuntimeError):
        base.open_source("file://clip.mp4", mode="sequential")
    assert events == ["open", "close"]


def test_failed_repetition_releases_resources_before_fresh_warmup(fake_runtime):
    events, fail = fake_runtime
    fail["warmup"] = RuntimeError("warmup failed")

    def continue_after_failure(key, exc):
        assert events[-3:] == ["stop", "close", "unload"]
        fail.clear()

    rows = runner.sweep(config(), ["a"], repeat=2, on_error=continue_after_failure)
    assert len(rows) == 1 and rows[0].repetition == 2
    assert events.count("warmup") == events.count("unload") == 2
    assert events.index("unload") < events.index("load", events.index("load") + 1)


def test_capture_released_when_build_fails(monkeypatch):
    events = []

    class Capture:
        def isOpened(self):
            return False

        def release(self):
            events.append("released")

    monkeypatch.setattr(base.cv2, "VideoCapture", lambda _: Capture())
    with pytest.raises(RuntimeError, match="could not open"):
        base.SequentialSource("clip.mp4").open()
    assert events == ["released"]
