"""Boundary behavior of timing series, completion rates, and GPU-only energy."""

import pytest

from visionbench.metrics import timers
from visionbench.metrics.system import energy_per_frame_j
from visionbench.metrics.timers import RateMeter, Series

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("power_values,fps", [(None, 3), ([], 3), ([30], 0), ([30], -1)])
def test_unavailable_gpu_energy_stays_null(power_values, fps):
    power = None
    if power_values is not None:
        power = Series("gpu_power_w")
        for value in power_values:
            power.add(value)
    assert energy_per_frame_j(power, fps) is None


def test_series_empty_and_capped_samples_preserve_percentiles_and_recent_window():
    series = Series("latency", cap=3, window=2)
    assert series.summary() == {
        "count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None,
    }
    assert series.recent_mean == 0
    for value in (1, 3, 5, 100):
        series.add(value)
    assert len(series) == 3
    assert series.dropped_samples == 1
    assert series.summary() == {
        "count": 3, "mean": 3.0, "p50": 3.0, "p95": 4.8, "p99": 4.96, "max": 5.0,
    }
    assert series.recent_mean == 52.5
    assert series.count_above(3) == 1


def test_rate_empty_single_completion_and_window_eviction(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(timers.time, "perf_counter", lambda: clock[0])
    rate = RateMeter(window_s=1)
    assert (rate.mean_fps, rate.instant_fps, rate.elapsed_s) == (0, 0, 0)
    rate.mark()
    assert (rate.mean_fps, rate.instant_fps, rate.elapsed_s) == (0, 0, 0)
    clock[0] = 10.5
    rate.mark()
    clock[0] = 11
    rate.mark()
    assert (rate.mean_fps, rate.instant_fps, rate.elapsed_s) == (2, 2, 1)
    clock[0] = 12
    rate.mark()
    assert rate.mean_fps == 1.5
    assert rate.instant_fps == 1
    clock[0] = 14
    rate.mark()
    assert rate.total == 5
    assert rate.mean_fps == 1
    assert rate.instant_fps == 0
