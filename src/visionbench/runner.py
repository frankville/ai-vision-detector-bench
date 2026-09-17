"""The benchmark loop.

One model, one source, for a fixed wall-clock duration or frame count. The same
loop backs the CLI and the live web view, so what you see in the browser is the
same measurement that lands in the result file, not a second implementation
that drifts away from it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from visionbench.hostinfo import default_profile, host_info
from visionbench.metrics.system import SystemSampler, energy_per_frame_j, now_iso
from visionbench.metrics.timers import RateMeter, Series, StageTimers
from visionbench.models import build as build_model
from visionbench.sources import describe, open_source
from visionbench.types import Detections, Frame, InferTimings

SCHEMA_VERSION = 1

#: Callback signature for live consumers: (frame, detections, timings) -> None
FrameHook = Callable[[Frame, Detections, InferTimings], None]


@dataclass
class RunConfig:
    source: str
    model: str
    device: str = "auto"
    duration_s: float = 60.0
    max_frames: int | None = None
    confidence: float = 0.3
    classes: list[str] | None = None
    mode: str = "realtime"
    imgsz: int | None = None
    precision: str = "fp32"
    profile: str | None = None
    warmup_frames: int = 5
    pace_fps: float | None = None
    loop: bool = True
    note: str = ""

    def redacted(self) -> dict[str, Any]:
        """Config safe to write to disk: the source URL loses its credentials."""
        data = asdict(self)
        data["source"] = describe(self.source)
        return data


@dataclass
class RunResult:
    schema_version: int
    profile: str
    started_at: str
    elapsed_s: float
    config: dict
    host: dict
    model: dict
    source: dict
    throughput: dict
    latency: dict
    resources: dict
    detections: dict
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def key(self) -> str:
        return f"{self.config['model']}@{self.profile}"


def _detection_stats(per_frame: Series, confidence: Series, class_totals: dict[str, int]) -> dict:
    return {
        "per_frame": per_frame.summary(),
        "confidence": confidence.summary(),
        "class_totals": dict(sorted(class_totals.items(), key=lambda kv: -kv[1])),
        "total": int(sum(class_totals.values())),
        "frames_with_detections": per_frame.count_above(0),
    }


def run(
    cfg: RunConfig,
    on_frame: FrameHook | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> RunResult:
    """Execute one benchmark run and return its result."""
    warnings: list[str] = []
    profile = cfg.profile or default_profile()

    detector = build_model(
        cfg.model,
        device=cfg.device,
        confidence=cfg.confidence,
        imgsz=cfg.imgsz,
        precision=cfg.precision,
    )
    detector.load()

    source = open_source(cfg.source, mode=cfg.mode, pace_fps=cfg.pace_fps, loop=cfg.loop)

    timers = StageTimers()
    processed = RateMeter()
    per_frame_detections = Series("detections_per_frame")
    mean_confidence = Series("mean_confidence")
    class_totals: dict[str, int] = {}
    sampler = SystemSampler(device=detector.device)

    wanted = set(cfg.classes) if cfg.classes else None
    started_wall = now_iso()
    # Captured before the finally block unloads the weights, since parameter
    # counts are read off the live model.
    model_info = detector.info()
    count = 0

    try:
        # Warm up before any clock starts. The first inference on a fresh model
        # pays for kernel compilation and allocator growth; on MPS that can be
        # ten times the steady-state cost.
        first = source.read(timeout=15.0)
        if first is None:
            raise RuntimeError(f"no frames from source: {describe(cfg.source)}")
        detector.warmup(first.image, rounds=cfg.warmup_frames)

        model_info = detector.info()  # refresh: input shape is known after warmup
        # Everything decoded while the model was warming up is water under the
        # bridge. Measurement starts here.
        source.reset_counters()

        sampler.start()
        loop_started = time.perf_counter()
        deadline = loop_started + cfg.duration_s if cfg.duration_s else float("inf")

        while True:
            if should_stop is not None and should_stop():
                break
            if time.perf_counter() >= deadline:
                break
            if cfg.max_frames is not None and count >= cfg.max_frames:
                break

            frame = source.read(timeout=10.0)
            if frame is None:
                break

            detections, timings = detector.infer(frame.image)
            if wanted:
                detections = detections.only(wanted)

            ready_at = time.perf_counter()
            timers.add("preprocess_ms", timings.pre_ms)
            timers.add("inference_ms", timings.infer_ms)
            timers.add("postprocess_ms", timings.post_ms)
            timers.add("model_total_ms", timings.total_ms)
            # Capture to detections-ready. Includes the time the frame spent
            # waiting for the consumer, which is what an alert pipeline feels.
            timers.add("end_to_end_ms", (ready_at - frame.captured_at) * 1000.0)

            processed.mark()
            count += 1

            per_frame_detections.add(len(detections))
            if len(detections):
                mean_confidence.add(float(np.mean(detections.confidence)))
                for name, n in detections.class_counts().items():
                    class_totals[name] = class_totals.get(name, 0) + n

            if on_frame is not None:
                on_frame(frame, detections, timings)

    except KeyboardInterrupt:
        warnings.append("run interrupted by user; results cover the frames completed so far")
    finally:
        sampler.stop()
        source.close()
        # Free the weights before the next model in a sweep loads its own,
        # otherwise two models briefly share VRAM and the second one's memory
        # figures are inflated by the first.
        detector.unload()

    stats = source.stats()
    elapsed = processed.elapsed_s
    processed_fps = processed.mean_fps

    # In realtime mode the decoder runs free, so its rate is the stream's true
    # rate. In sequential mode decode and inference are serialised, so the only
    # meaningful reference is what the file claims.
    stream_fps = stats.decode_fps if cfg.mode == "realtime" else stats.nominal_fps
    if not stream_fps:
        stream_fps = stats.nominal_fps

    drop_ratio = stats.frames_dropped / stats.frames_decoded if stats.frames_decoded else 0.0
    realtime_factor = processed_fps / stream_fps if stream_fps > 0 else None
    capacity = processed_fps / stream_fps if stream_fps > 0 else None

    if cfg.mode == "sequential":
        warnings.append(
            "sequential mode processes every frame, so drop counts are always zero; "
            "use --mode realtime to measure whether the host keeps up with a live stream"
        )
    if getattr(source, "reconnects", 0):
        warnings.append(f"source reconnected {source.reconnects} time(s) during the run")
    if count < 30:
        warnings.append(f"only {count} frames processed; percentiles are unreliable below ~30")

    power = sampler.series.get("gpu_power_w")
    resources = sampler.summary()
    resources["energy_per_frame_j"] = energy_per_frame_j(power, processed_fps)

    return RunResult(
        schema_version=SCHEMA_VERSION,
        profile=profile,
        started_at=started_wall,
        elapsed_s=round(elapsed, 3),
        config=cfg.redacted(),
        host=host_info(),
        model=model_info,
        source={
            "label": describe(cfg.source),
            "mode": cfg.mode,
            "width": stats.width,
            "height": stats.height,
            "nominal_fps": stats.nominal_fps,
            "reconnects": getattr(source, "reconnects", 0),
        },
        throughput={
            "frames_decoded": stats.frames_decoded,
            "frames_processed": count,
            "frames_dropped": stats.frames_dropped,
            "drop_ratio": round(drop_ratio, 4),
            "decode_fps": stats.decode_fps,
            "processed_fps": round(processed_fps, 3),
            "stream_fps_reference": round(stream_fps, 3) if stream_fps else None,
            # >= 1.0 means this host keeps up with one stream at its own rate.
            "realtime_factor": round(realtime_factor, 3) if realtime_factor else None,
            # How many cameras of this kind one host could serve, ignoring the
            # extra decode cost of each additional stream.
            "capacity_streams": round(capacity, 2) if capacity else None,
        },
        latency=timers.summary(),
        resources=resources,
        detections=_detection_stats(per_frame_detections, mean_confidence, class_totals),
        warnings=warnings,
    )


def sweep(
    cfg: RunConfig,
    models: list[str],
    on_result: Callable[[RunResult], None] | None = None,
    on_error: Callable[[str, Exception], None] | None = None,
) -> list[RunResult]:
    """Run the same source and settings across several models, in sequence.

    Models are run one at a time and unloaded between runs. Running two models
    concurrently on one device would have them compete for the same silicon and
    make both numbers wrong.

    A sweep is long-running and often unattended, so one model failing to load
    must not discard the results of the ones that already succeeded. Pass
    `on_error` to keep going; without it, the first failure propagates.
    """
    results: list[RunResult] = []
    for key in models:
        try:
            result = run(replace(cfg, model=key))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if on_error is None:
                raise
            on_error(key, exc)
            continue
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results
