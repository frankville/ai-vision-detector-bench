"""Live benchmark sessions behind the web dashboard.

A session runs the ordinary `runner.run` loop on a worker thread and publishes
two things: the newest annotated frame as JPEG, and the progress snapshot the
runner builds from its own meters.

One honest caveat, surfaced in the UI as well as here. Annotating and encoding
frames costs CPU inside the measured loop, so a run watched live reports
slightly lower throughput than the same run headless. Preview encoding is
throttled to keep that effect small, and authoritative numbers should come from
`visionbench run` or `visionbench sweep` with no browser attached.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from visionbench import draw
from visionbench.runner import RunConfig, RunResult, run
from visionbench.types import Detections, Frame, InferTimings


@dataclass
class LiveSession:
    config: RunConfig
    preview_fps: float = 12.0
    preview_quality: int = 72

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: str = "starting"  # starting | running | finished | failed | stopped
    error: str | None = None
    result: RunResult | None = None
    progress: dict[str, Any] = field(default_factory=dict)

    latest_jpeg: bytes | None = None
    frame_index: int = 0
    started_at: float = field(default_factory=time.time)

    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _last_preview: float = 0.0

    # -- runner hooks ------------------------------------------------------

    def _on_frame(self, frame: Frame, detections: Detections, timings: InferTimings) -> None:
        now = time.perf_counter()
        interval = 1.0 / self.preview_fps if self.preview_fps > 0 else 0.0
        if interval and now - self._last_preview < interval:
            return
        self._last_preview = now

        hud = [
            f"{self.config.model}  {self.config.device}",
            f"infer {timings.infer_ms:.0f} ms   objects {len(detections)}",
        ]
        fps = self.progress.get("processed_fps")
        if fps:
            hud.append(f"{fps:.1f} fps processed")
        annotated = draw.annotate(frame.image, detections, hud=hud)
        jpeg = draw.encode_jpeg(annotated, quality=self.preview_quality)
        if jpeg is not None:
            self.latest_jpeg = jpeg
            self.frame_index += 1

    def _on_progress(self, snapshot: dict) -> None:
        self.progress = snapshot

    # -- lifecycle ---------------------------------------------------------

    def _work(self) -> None:
        try:
            self.latest_jpeg = draw.placeholder(640, 360, f"loading {self.config.model}...")
            self.state = "running"
            # Recorded in the result file so a perturbed run is never mistaken
            # for a clean headless measurement later.
            if not self.config.note:
                self.config.note = "live preview attached; throughput is perturbed by annotation and JPEG encoding"
            self.result = run(
                self.config,
                on_frame=self._on_frame,
                should_stop=self._stop.is_set,
                on_progress=self._on_progress,
            )
            self.state = "stopped" if self._stop.is_set() else "finished"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = "failed"
            self.latest_jpeg = draw.placeholder(640, 360, "run failed")

    def start(self) -> LiveSession:
        self._thread = threading.Thread(target=self._work, name=f"session-{self.id}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 10.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        return self.state in {"starting", "running"}

    def status(self) -> dict:
        payload: dict[str, Any] = {
            "id": self.id,
            "state": self.state,
            "error": self.error,
            "model": self.config.model,
            "source": self.config.redacted()["source"],
            "device": self.config.device,
            "mode": self.config.mode,
            "duration_s": self.config.duration_s,
            "age_s": round(time.time() - self.started_at, 1),
            "progress": self.progress,
        }
        if self.result is not None:
            payload["result"] = {
                "throughput": self.result.throughput,
                "latency": self.result.latency,
                "detections": self.result.detections,
                "warnings": self.result.warnings,
                "profile": self.result.profile,
                "elapsed_s": self.result.elapsed_s,
            }
        return payload


class SessionStore:
    """In-memory registry of live sessions, with a cap so a long-lived server
    cannot accumulate finished runs forever."""

    def __init__(self, max_sessions: int = 12) -> None:
        self._sessions: dict[str, LiveSession] = {}
        self._lock = threading.Lock()
        self.max_sessions = max_sessions

    def create(self, config: RunConfig, **kwargs: Any) -> LiveSession:
        session = LiveSession(config=config, **kwargs)
        with self._lock:
            self._evict()
            self._sessions[session.id] = session
        return session.start()

    def _evict(self) -> None:
        if len(self._sessions) < self.max_sessions:
            return
        finished = sorted(
            (s for s in self._sessions.values() if not s.running),
            key=lambda s: s.started_at,
        )
        for session in finished[: max(1, len(self._sessions) - self.max_sessions + 1)]:
            self._sessions.pop(session.id, None)

    def get(self, session_id: str) -> LiveSession | None:
        return self._sessions.get(session_id)

    def all(self) -> list[LiveSession]:
        return sorted(self._sessions.values(), key=lambda s: -s.started_at)

    def stop_all(self) -> None:
        for session in self._sessions.values():
            session.stop()
