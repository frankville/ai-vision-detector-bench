"""Timing series and rate meters.

Percentiles matter more than means here. A model with a good mean latency and a
bad p95 drops frames in bursts, which on a live stream looks like the system
missing exactly the moment you cared about.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np


class Series:
    """A named stream of samples with percentile summaries.

    Keeps every sample up to `cap` so percentiles are exact rather than
    estimated. A ten minute run at 30 fps is 18k samples, which is nothing.
    """

    def __init__(self, name: str, cap: int = 500_000, window: int = 120) -> None:
        self.name = name
        self.cap = cap
        self._values: list[float] = []
        self._recent: deque[float] = deque(maxlen=window)
        self.dropped_samples = 0

    def add(self, value: float) -> None:
        if len(self._values) < self.cap:
            self._values.append(float(value))
        else:
            self.dropped_samples += 1
        self._recent.append(float(value))

    def __len__(self) -> int:
        return len(self._values)

    @property
    def values(self) -> list[float]:
        return self._values

    @property
    def recent_mean(self) -> float:
        return float(np.mean(self._recent)) if self._recent else 0.0

    def count_above(self, threshold: float) -> int:
        return sum(1 for v in self._values if v > threshold)

    def summary(self) -> dict[str, float | int | None]:
        if not self._values:
            return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
        arr = np.asarray(self._values, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean": round(float(arr.mean()), 3),
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "p99": round(float(np.percentile(arr, 99)), 3),
            "max": round(float(arr.max()), 3),
        }


class StageTimers:
    """One `Series` per pipeline stage, created on first use."""

    def __init__(self) -> None:
        self.stages: dict[str, Series] = {}

    def add(self, stage: str, ms: float) -> None:
        if stage not in self.stages:
            self.stages[stage] = Series(stage)
        self.stages[stage].add(ms)

    def summary(self) -> dict[str, dict]:
        return {name: series.summary() for name, series in self.stages.items()}

    def recent(self) -> dict[str, float]:
        return {name: round(series.recent_mean, 2) for name, series in self.stages.items()}


class RateMeter:
    """Frames per second over a sliding wall-clock window, plus a run-long mean."""

    def __init__(self, window_s: float = 5.0) -> None:
        self.window_s = window_s
        self._marks: deque[float] = deque()
        self.total = 0
        self.started_at: float | None = None
        self.last_at: float | None = None

    def mark(self, n: int = 1) -> None:
        now = time.perf_counter()
        if self.started_at is None:
            self.started_at = now
        self.last_at = now
        self.total += n
        for _ in range(n):
            self._marks.append(now)
        cutoff = now - self.window_s
        while self._marks and self._marks[0] < cutoff:
            self._marks.popleft()

    @property
    def instant_fps(self) -> float:
        if len(self._marks) < 2:
            return 0.0
        span = self._marks[-1] - self._marks[0]
        return round((len(self._marks) - 1) / span, 3) if span > 0 else 0.0

    @property
    def mean_fps(self) -> float:
        if self.started_at is None or self.last_at is None or self.total < 2:
            return 0.0
        span = self.last_at - self.started_at
        return round((self.total - 1) / span, 3) if span > 0 else 0.0

    @property
    def elapsed_s(self) -> float:
        if self.started_at is None or self.last_at is None:
            return 0.0
        return round(self.last_at - self.started_at, 3)
