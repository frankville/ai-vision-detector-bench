"""Detector interface and device resolution.

Every model in this project is reached through `Detector`. Adding a model means
adding one subclass and one registry entry, never touching the runner.

Note that `infer` returns its own stage timings. Only the adapter knows where
preprocessing ends and the forward pass begins, and that boundary is where the
interesting differences between these models show up.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any

import numpy as np

from visionbench.types import Detections, InferTimings


def resolve_device(requested: str = "auto") -> str:
    """Pick a torch device, preferring the fastest available.

    Apple Silicon reports as `mps`. Some operators still fall back to CPU there,
    so an mps run is a real measurement of that machine but not a clean
    measurement of the GPU alone.
    """
    import torch

    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync_device(device: str) -> None:
    """Block until queued GPU work has finished.

    Without this, a timer around the forward pass measures how fast the call was
    *enqueued*, not how long the GPU took, and every model looks equally fast.
    """
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


@contextmanager
def stage(device: str | None = None):
    """Time a block in milliseconds, synchronising the device first if given."""
    start = time.perf_counter()
    result: list[float] = [0.0]
    try:
        yield result
    finally:
        if device:
            sync_device(device)
        result[0] = (time.perf_counter() - start) * 1000.0


class Detector(ABC):
    """One object detection model, loaded and ready to run on frames."""

    #: Registry key, set by the spec that built this instance.
    key: str = "unset"
    #: Human label for reports.
    display_name: str = "unset"
    #: SPDX-style identifier of the weights licence.
    license: str = "unknown"
    #: Where the weights come from.
    weights: str = "unset"

    def __init__(
        self,
        device: str = "auto",
        confidence: float = 0.3,
        imgsz: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.device = resolve_device(device)
        self.confidence = confidence
        self.imgsz = imgsz
        self.extra = kwargs
        self.loaded = False

    @abstractmethod
    def load(self) -> None:
        """Fetch weights and move the model to the device."""

    @abstractmethod
    def infer(self, frame_bgr: np.ndarray) -> tuple[Detections, InferTimings]:
        """Run one frame. Returns detections in absolute pixel coordinates."""

    def warmup(self, frame_bgr: np.ndarray, rounds: int = 3) -> None:
        """Run a few throwaway frames.

        The first call to a freshly loaded model pays for lazy kernel
        compilation and memory allocation, and on MPS that first frame can be
        an order of magnitude slower than steady state. Excluding it is the
        difference between a benchmark and a rumour.
        """
        for _ in range(max(0, rounds)):
            self.infer(frame_bgr)

    def parameters_m(self) -> float | None:
        """Parameter count in millions, if the backend exposes it."""
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "parameters"):
            return None
        try:
            total = sum(p.numel() for p in model.parameters())
        except Exception:
            return None
        return round(total / 1e6, 2)

    def info(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "license": self.license,
            "weights": self.weights,
            "device": self.device,
            "confidence": self.confidence,
            "imgsz": self.imgsz,
            "parameters_m": self.parameters_m(),
        }

    def unload(self) -> None:
        """Release weights and clear device caches between models in a sweep."""
        import gc

        import torch

        for attr in ("model", "processor", "_impl"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        self.loaded = False
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif self.device.startswith("mps"):
            torch.mps.empty_cache()
