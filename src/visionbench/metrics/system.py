"""Resource telemetry sampled in the background while a run is in flight.

Honesty about what is measurable matters more than filling every column. On
NVIDIA hardware NVML gives utilisation, memory, power and temperature. On Apple
Silicon, PyTorch reports how much GPU memory it allocated, but utilisation and
power need `powermetrics` with root, so those fields stay null rather than being
guessed at.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import psutil

from visionbench.metrics.timers import Series


@dataclass
class GPUBackend:
    """What GPU telemetry this host can actually provide."""

    kind: str = "none"  # "nvml" | "mps" | "none"
    name: str | None = None
    total_memory_mb: float | None = None
    utilisation_available: bool = False
    power_available: bool = False
    note: str = ""


class _NVMLProbe:
    def __init__(self) -> None:
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(self.handle)
        self.name = name.decode() if isinstance(name, bytes) else str(name)
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.total_mb = info.total / 1024**2

    def sample(self) -> dict[str, float | None]:
        nv = self._nvml
        out: dict[str, float | None] = {}
        try:
            util = nv.nvmlDeviceGetUtilizationRates(self.handle)
            out["gpu_util_pct"] = float(util.gpu)
            out["gpu_mem_util_pct"] = float(util.memory)
        except Exception:
            out["gpu_util_pct"] = None
        try:
            mem = nv.nvmlDeviceGetMemoryInfo(self.handle)
            out["gpu_mem_used_mb"] = mem.used / 1024**2
        except Exception:
            out["gpu_mem_used_mb"] = None
        try:
            out["gpu_power_w"] = nv.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except Exception:
            out["gpu_power_w"] = None
        try:
            out["gpu_temp_c"] = float(
                nv.nvmlDeviceGetTemperature(self.handle, nv.NVML_TEMPERATURE_GPU)
            )
        except Exception:
            out["gpu_temp_c"] = None
        return out

    def shutdown(self) -> None:
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass


class _MPSProbe:
    """Apple Silicon. Memory only, and only what PyTorch itself allocated."""

    def __init__(self) -> None:
        import torch

        self._torch = torch
        self.name = "Apple Silicon GPU (Metal)"
        self.total_mb = None

    def sample(self) -> dict[str, float | None]:
        torch = self._torch
        try:
            return {
                "gpu_mem_used_mb": torch.mps.current_allocated_memory() / 1024**2,
                "gpu_mem_driver_mb": torch.mps.driver_allocated_memory() / 1024**2,
                "gpu_util_pct": None,
                "gpu_power_w": None,
                "gpu_temp_c": None,
            }
        except Exception:
            return {}

    def shutdown(self) -> None:
        return None


def detect_gpu_backend(device: str) -> tuple[GPUBackend, object | None]:
    """Choose the telemetry probe that matches the compute device."""
    if device.startswith("cuda"):
        try:
            probe = _NVMLProbe()
        except Exception as exc:
            return (
                GPUBackend(
                    kind="none",
                    note=f"CUDA device in use but NVML unavailable ({exc}). "
                    "Install the 'cuda' extra for GPU telemetry.",
                ),
                None,
            )
        return (
            GPUBackend(
                kind="nvml",
                name=probe.name,
                total_memory_mb=round(probe.total_mb, 1),
                utilisation_available=True,
                power_available=True,
            ),
            probe,
        )
    if device.startswith("mps"):
        try:
            probe = _MPSProbe()
        except Exception:
            return GPUBackend(kind="none", note="MPS probe unavailable"), None
        return (
            GPUBackend(
                kind="mps",
                name=probe.name,
                utilisation_available=False,
                power_available=False,
                note="Apple Silicon reports PyTorch-allocated GPU memory only. "
                "Utilisation and power require powermetrics with root and are "
                "reported as null rather than estimated.",
            ),
            probe,
        )
    return GPUBackend(kind="none", note="CPU device; no GPU telemetry collected."), None


@dataclass
class SystemSampler:
    """Background sampler for CPU, memory and GPU counters."""

    device: str = "cpu"
    interval_s: float = 0.5

    series: dict[str, Series] = field(default_factory=dict)
    backend: GPUBackend = field(default_factory=GPUBackend)
    _probe: object | None = None
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _samples: int = 0

    def __post_init__(self) -> None:
        self.backend, self._probe = detect_gpu_backend(self.device)
        self._process = psutil.Process()
        self.cpu_count = psutil.cpu_count(logical=True) or 1

    def _record(self, key: str, value: float | None) -> None:
        if value is None:
            return
        if key not in self.series:
            self.series[key] = Series(key)
        self.series[key].add(value)

    def _tick(self) -> None:
        self._record("process_cpu_pct", self._process.cpu_percent(interval=None))
        self._record("system_cpu_pct", psutil.cpu_percent(interval=None))
        self._record("process_rss_mb", self._process.memory_info().rss / 1024**2)
        self._record("system_mem_used_pct", psutil.virtual_memory().percent)
        if self._probe is not None:
            for key, value in self._probe.sample().items():  # type: ignore[attr-defined]
                self._record(key, value)
        self._samples += 1

    def _loop(self) -> None:
        # Prime the CPU counters; the first psutil reading is always 0.
        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        while not self._stop.wait(self.interval_s):
            try:
                self._tick()
            except Exception:
                # Telemetry must never take a benchmark down with it.
                continue

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._probe is not None:
            self._probe.shutdown()  # type: ignore[attr-defined]
            self._probe = None

    def snapshot(self) -> dict[str, float]:
        """Latest-window view, for the live UI."""
        return {name: round(series.recent_mean, 2) for name, series in self.series.items()}

    def summary(self) -> dict:
        out: dict = {
            "samples": self._samples,
            "interval_s": self.interval_s,
            "logical_cores": self.cpu_count,
            "gpu_backend": {
                "kind": self.backend.kind,
                "name": self.backend.name,
                "total_memory_mb": self.backend.total_memory_mb,
                "utilisation_available": self.backend.utilisation_available,
                "power_available": self.backend.power_available,
                "note": self.backend.note,
            },
        }
        out["counters"] = {name: series.summary() for name, series in self.series.items()}
        return out


def energy_per_frame_j(power_series: Series | None, processed_fps: float) -> float | None:
    """Joules per processed frame, when power telemetry exists.

    This is the number that turns a latency comparison into a running-cost
    comparison across a fleet of sites.
    """
    if power_series is None or len(power_series) == 0 or processed_fps <= 0:
        return None
    mean_watts = power_series.summary()["mean"]
    if mean_watts is None:
        return None
    return round(float(mean_watts) / processed_fps, 3)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
