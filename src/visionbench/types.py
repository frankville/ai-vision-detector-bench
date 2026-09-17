"""Core data types shared by sources, detectors, and the runner.

Kept dependency-light on purpose: numpy only. Conversions to third-party
formats (supervision, COCO) live in the modules that need them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np


@dataclass(slots=True)
class Frame:
    """A decoded frame plus the bookkeeping the runner needs to detect drops.

    `seq` is the index in the *decoded* sequence, not the processed one. Gaps
    between consecutive frames handed to the consumer are dropped frames, which
    is the whole point of measuring on a live stream.
    """

    image: np.ndarray  # (H, W, 3) uint8, BGR as OpenCV delivers it
    seq: int
    captured_at: float  # time.perf_counter() at the moment the decoder produced it

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass(slots=True)
class Detections:
    """Detector output in absolute pixel coordinates.

    Deliberately mirrors the shape of supervision.Detections so converting is
    trivial, without taking the dependency in the core package.
    """

    xyxy: np.ndarray  # (N, 4) float32
    confidence: np.ndarray  # (N,) float32
    class_id: np.ndarray  # (N,) int32
    names: dict[int, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    @classmethod
    def empty(cls, names: dict[int, str] | None = None) -> Detections:
        return cls(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            confidence=np.zeros((0,), dtype=np.float32),
            class_id=np.zeros((0,), dtype=np.int32),
            names=names or {},
        )

    def _take(self, mask: np.ndarray) -> Detections:
        return Detections(
            xyxy=self.xyxy[mask],
            confidence=self.confidence[mask],
            class_id=self.class_id[mask],
            names=self.names,
        )

    def above(self, min_confidence: float) -> Detections:
        if len(self) == 0:
            return self
        return self._take(self.confidence >= min_confidence)

    def only(self, keep: set[str]) -> Detections:
        """Keep detections whose class name is in `keep` (case-insensitive)."""
        if len(self) == 0 or not keep:
            return self
        wanted = {k.lower() for k in keep}
        mask = np.array(
            [self.names.get(int(c), str(c)).lower() in wanted for c in self.class_id],
            dtype=bool,
        )
        return self._take(mask)

    def label(self, i: int) -> str:
        return self.names.get(int(self.class_id[i]), str(int(self.class_id[i])))

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for i in range(len(self)):
            name = self.label(i)
            counts[name] = counts.get(name, 0) + 1
        return counts


class InferTimings(NamedTuple):
    """Wall-clock split of one inference call, in milliseconds.

    Split matters: a model that looks slow may be losing time in preprocessing
    (resize, normalise, host-to-device copy) rather than in the forward pass,
    and that changes which hardware fixes it.
    """

    pre_ms: float
    infer_ms: float
    post_ms: float

    @property
    def total_ms(self) -> float:
        return self.pre_ms + self.infer_ms + self.post_ms


class SourceStats(NamedTuple):
    """Counters a source reports at the end of a run."""

    frames_decoded: int
    frames_delivered: int
    frames_dropped: int
    nominal_fps: float  # what the stream claims, 0.0 if unknown
    decode_fps: float  # what we actually pulled off the wire
    width: int
    height: int
