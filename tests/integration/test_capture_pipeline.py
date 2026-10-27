"""Real local decoding through source, runner, persistence, and reporting."""

import threading

import cv2
import numpy as np
import pytest

from visionbench import report, runner
from visionbench.aggregate import aggregate_runs
from visionbench.models.base import Detector
from visionbench.sources.base import RealtimeSource, SequentialSource
from visionbench.types import Detections, InferTimings

pytestmark = pytest.mark.integration

LEVELS = [20 + 25 * index for index in range(8)]


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "generated.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (64, 48))
    assert writer.isOpened(), "OpenCV must support MJPG encoding for capture integration tests"
    try:
        for level in LEVELS:
            writer.write(np.full((48, 64, 3), level, dtype=np.uint8))
    finally:
        writer.release()
    return path


def assert_image(frame, index):
    assert frame.image.shape == (48, 64, 3)
    assert frame.image.dtype == np.uint8
    assert float(frame.image.mean()) == pytest.approx(LEVELS[index], abs=3)


def test_sequential_decoder_order_eof_and_counter_reset(clip):
    source = SequentialSource(str(clip))
    with source:
        first = source.read()
        assert first.seq == 0
        assert_image(first, 0)
        source.reset_counters()
        frames = list(iter(source.read, None))
        assert [frame.seq for frame in frames] == list(range(1, len(LEVELS)))
        for index, frame in enumerate(frames, 1):
            assert_image(frame, index)
        assert source.read() is None
        stats = source.stats()
        assert stats.frames_decoded == stats.frames_delivered == 7
        assert stats.frames_dropped == 0
        assert (stats.width, stats.height, stats.nominal_fps) == (64, 48, 20)
    assert source.cap is None


def test_realtime_decoder_keeps_latest_and_counts_observed_gaps(clip):
    second_decode = threading.Event()
    continue_decoding = threading.Event()

    class CoordinatedSource(RealtimeSource):
        def _note_decode(self):
            if self.frames_decoded == 1:
                second_decode.set()
                if not continue_decoding.wait(5):
                    raise TimeoutError("consumer did not release decoder")
            super()._note_decode()

    source = CoordinatedSource(str(clip), pace_fps=0)
    source.open()
    thread = source._thread
    try:
        assert second_decode.wait(5)
        first = source.read(timeout=1)
        assert first.seq == 0
        assert_image(first, 0)
        continue_decoding.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        newest = source.read(timeout=1)
        assert newest.seq == len(LEVELS) - 1
        assert_image(newest, newest.seq)
        assert source.read(timeout=1) is None
        stats = source.stats()
        assert stats.frames_delivered == 2
        assert stats.frames_decoded == len(LEVELS)
        assert stats.frames_dropped == newest.seq - first.seq - 1
    finally:
        continue_decoding.set()
        source.close()
    assert not thread.is_alive()
    assert source.cap is None


@pytest.fixture
def detectors(monkeypatch):
    instances = []

    class LocalDetector(Detector):
        license = "Apache-2.0"

        def load(self):
            self.loaded = True
            self.images = []

        def infer(self, image):
            assert self.loaded
            self.images.append(float(image.mean()))
            detections = Detections(
                np.array([[1, 2, 10, 12], [3, 4, 20, 24]], dtype=np.float32),
                np.array([0.9, 0.8], dtype=np.float32),
                np.array([0, 1], dtype=np.int32),
                {0: "person", 1: "car"},
            )
            # A distinct warmup timing makes accidental inclusion visible.
            return detections, InferTimings(1, 1000 if image.mean() < 30 else 2, 3)

        def unload(self):
            self.loaded = False

    def build(key, **settings):
        detector = LocalDetector(**settings)
        detector.key = key
        instances.append(detector)
        return detector

    monkeypatch.setattr(runner, "build_model", build)
    return instances


def config(clip):
    return runner.RunConfig(
        str(clip), "local-detector", device="cpu", mode="sequential", loop=False,
        duration_s=0, warmup_frames=3, profile="offline-capture", classes=["person"],
    )


def test_real_runner_excludes_warmup_and_roundtrips_report(clip, detectors, tmp_path):
    sequences = []
    result = runner.run(config(clip), on_frame=lambda frame, *_: sequences.append(frame.seq))
    assert sequences == list(range(1, len(LEVELS)))
    assert len(detectors) == 1
    assert detectors[0].images[:3] == pytest.approx([LEVELS[0]] * 3, abs=3)
    assert detectors[0].images[3:] == pytest.approx(LEVELS[1:], abs=3)
    assert not detectors[0].loaded
    assert result.status == "completed"
    assert result.throughput["frames_processed"] == result.throughput["frames_decoded"] == 7
    assert result.throughput["frames_dropped"] == 0
    assert result.latency["inference_ms"] == {
        "count": 7, "mean": 2.0, "p50": 2.0, "p95": 2.0, "p99": 2.0, "max": 2.0,
    }
    assert result.latency["end_to_end_ms"]["count"] == 7
    assert result.detections["class_totals"] == {"person": 7}
    assert result.detections["frames_with_detections"] == 7
    assert result.resources["energy_per_frame_j"] is None
    path = report.save(result, tmp_path / "results")
    loaded, = report.load_all(tmp_path / "results")
    assert loaded["_path"] == str(path)
    assert loaded["latency"] == result.latency
    row = report.to_row(loaded)
    assert row.infer_p95 == 2.0
    assert row.detections_per_frame == 1.0
    assert row.energy_j is None
    assert "local-detector" in report.markdown_table([loaded])


def test_repetitions_open_independent_decoders_and_aggregate(clip, detectors, tmp_path):
    attempts = []
    results = runner.sweep(
        config(clip), ["local-detector"], repeat=2, on_attempt=attempts.append,
        on_result=lambda result: report.save(result, tmp_path),
    )
    assert len(results) == len(detectors) == 2
    assert detectors[0] is not detectors[1]
    assert detectors[0].images == detectors[1].images
    assert all(not detector.loaded for detector in detectors)
    assert [result.throughput["frames_processed"] for result in results] == [7, 7]
    raw = report.load_all(tmp_path)
    assert len({run["run_id"] for run in raw}) == 2
    aggregate = aggregate_runs(raw, attempts=attempts, root=tmp_path)
    assert aggregate["counts"]["eligible"] == 2
    assert aggregate["metrics"]["inference_p95_ms"] == {"count": 2, "median": 2.0}
    assert aggregate["metrics"]["energy_per_frame_j"] == {"count": 0, "median": None}
    report.save_aggregate(aggregate, tmp_path)
    loaded, = report.load_all(tmp_path, kind="aggregate")
    assert loaded["metrics"] == aggregate["metrics"]
    assert "2/2" in report.aggregate_markdown_table([loaded])
    assert len(report.load_all(tmp_path)) == 2
