"""Frame sources.

Two modes, and the difference between them is the single most important
measurement decision in this project.

`sequential` reads every frame in order and never drops. Use it for
reproducible throughput numbers on a file: every model sees identical input.

`realtime` runs the decoder in its own thread and keeps only the newest frame.
The consumer always gets the latest one available, and the gap in sequence
numbers is the count of frames it was too slow to process. This is what a live
camera actually does to you. Reading frames inline with inference instead would
let FFmpeg's buffer absorb the backlog, turning a throughput problem into a
silently growing latency problem, which is exactly the failure you want the
benchmark to expose rather than hide.
"""

from __future__ import annotations

import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import cv2

from visionbench.types import Frame, SourceStats

# FFmpeg defaults to UDP for RTSP, which shreds frames on a congested link and
# makes throughput numbers meaningless. Force TCP before any capture is built.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")


def redact(url: str) -> str:
    """Strip credentials from a URL so it is safe to print, log, or commit."""
    try:
        parts = urlparse(url)
    except ValueError:
        return "<unparseable url>"
    if not parts.netloc or "@" not in parts.netloc:
        return url
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunparse(parts._replace(netloc=f"***:***@{host}"))


def describe(url: str) -> str:
    """Short human label for a source, with credentials removed."""
    if "://" in url:
        return redact(url)
    return Path(url).name


class FrameSource(ABC):
    """Common interface over a file, an RTSP stream, or an MJPEG endpoint."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.cap: cv2.VideoCapture | None = None
        self.width = 0
        self.height = 0
        self.nominal_fps = 0.0
        self.frames_decoded = 0
        self.frames_delivered = 0
        self.frames_dropped = 0
        self.reconnects = 0
        self._t_first_decode: float | None = None
        self._t_last_decode: float = 0.0

    # -- lifecycle ---------------------------------------------------------

    def _build_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            raise RuntimeError(f"could not open source: {describe(self.url)}")
        # Ask FFmpeg to hold one frame, not a queue of them. Honoured by some
        # backends only, which is why the realtime source drops explicitly too.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def open(self) -> None:
        self.cap = self._build_capture()
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        # Some RTSP servers report nonsense here.
        self.nominal_fps = fps if 0.0 < fps < 240.0 else 0.0

    @abstractmethod
    def read(self, timeout: float = 5.0) -> Frame | None:
        """Return the next frame to process, or None when the source is done."""

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # -- bookkeeping -------------------------------------------------------

    def _note_decode(self) -> None:
        now = time.perf_counter()
        if self._t_first_decode is None:
            self._t_first_decode = now
        self._t_last_decode = now
        self.frames_decoded += 1

    def reset_counters(self) -> None:
        """Zero the counters so warm-up frames do not pollute the measurement.

        A realtime source starts decoding the moment it is opened, which is
        before the model has finished warming up. Those frames are genuinely
        dropped, but they were dropped by a model that was still compiling
        kernels, so counting them would report a 28% drop rate for a run that
        actually kept up.
        """
        self.frames_decoded = 0
        self.frames_delivered = 0
        self.frames_dropped = 0
        self._t_first_decode = None
        self._t_last_decode = 0.0

    def stats(self) -> SourceStats:
        span = 0.0
        if self._t_first_decode is not None:
            span = self._t_last_decode - self._t_first_decode
        decode_fps = (self.frames_decoded - 1) / span if span > 0 else 0.0
        return SourceStats(
            frames_decoded=self.frames_decoded,
            frames_delivered=self.frames_delivered,
            frames_dropped=self.frames_dropped,
            nominal_fps=self.nominal_fps,
            decode_fps=round(decode_fps, 3),
            width=self.width,
            height=self.height,
        )

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class SequentialSource(FrameSource):
    """Every frame, in order, no drops. Reproducible throughput on a file."""

    def __init__(self, url: str, loop: bool = False) -> None:
        super().__init__(url)
        self.loop = loop
        self._seq = 0

    def read(self, timeout: float = 5.0) -> Frame | None:
        if self.cap is None:
            raise RuntimeError("source not open")
        ok, image = self.cap.read()
        if not ok:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, image = self.cap.read()
            if not ok:
                return None
        self._note_decode()
        self.frames_delivered += 1
        frame = Frame(image=image, seq=self._seq, captured_at=time.perf_counter())
        self._seq += 1
        return frame


class RealtimeSource(FrameSource):
    """Decode in a background thread, hand out only the newest frame.

    `pace_fps` makes a local file behave like a camera by throttling the reader
    to a fixed rate. Without it, the thread would race through the file and the
    drop count would say more about disk speed than about the model.
    """

    def __init__(
        self,
        url: str,
        pace_fps: float | None = None,
        loop: bool = False,
        max_reconnects: int = 5,
    ) -> None:
        super().__init__(url)
        self.pace_fps = pace_fps
        self.loop = loop
        self.max_reconnects = max_reconnects

        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._arrived = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ended = False
        self._error: BaseException | None = None
        self._seq = 0
        self._last_delivered_seq: int | None = None

    def open(self) -> None:
        super().open()
        if self.pace_fps is None and self.nominal_fps and not self._is_network():
            # A file replayed as "live" defaults to its own frame rate.
            self.pace_fps = self.nominal_fps
        self._stop.clear()
        self._thread = threading.Thread(target=self._pump, name="decoder", daemon=True)
        self._thread.start()

    def _is_network(self) -> bool:
        return "://" in self.url and not self.url.startswith("file://")

    def _pump(self) -> None:
        interval = 1.0 / self.pace_fps if self.pace_fps else 0.0
        next_at = time.perf_counter()
        try:
            while not self._stop.is_set():
                if self.cap is None:
                    break
                ok, image = self.cap.read()
                if not ok:
                    if self.loop and not self._is_network():
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    if self._is_network() and self.reconnects < self.max_reconnects:
                        self.reconnects += 1
                        time.sleep(min(2.0 * self.reconnects, 10.0))
                        try:
                            self.cap.release()
                            self.cap = self._build_capture()
                        except RuntimeError:
                            continue
                        continue
                    self._ended = True
                    self._arrived.set()
                    return

                self._note_decode()
                with self._lock:
                    self._latest = Frame(
                        image=image, seq=self._seq, captured_at=time.perf_counter()
                    )
                    self._seq += 1
                self._arrived.set()

                if interval:
                    next_at += interval
                    slack = next_at - time.perf_counter()
                    if slack > 0:
                        time.sleep(slack)
                    else:
                        # We are behind the pacing target; resync rather than
                        # accumulate an ever-growing debt.
                        next_at = time.perf_counter()
        except BaseException as exc:  # surfaced to the consumer on next read
            self._error = exc
            self._ended = True
            self._arrived.set()

    def read(self, timeout: float = 5.0) -> Frame | None:
        deadline = time.perf_counter() + timeout
        while True:
            if self._error is not None:
                raise RuntimeError(f"decoder failed: {self._error}") from self._error
            with self._lock:
                frame = self._latest
                unseen = frame is not None and frame.seq != self._last_delivered_seq
                if unseen:
                    self._latest = None
            if unseen and frame is not None:
                if self._last_delivered_seq is not None:
                    self.frames_dropped += max(0, frame.seq - self._last_delivered_seq - 1)
                self._last_delivered_seq = frame.seq
                self.frames_delivered += 1
                return frame
            if self._ended:
                return None
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return None
            self._arrived.clear()
            self._arrived.wait(min(remaining, 0.25))

    def reset_counters(self) -> None:
        with self._lock:
            super().reset_counters()
            # Discard whatever is buffered and forget the sequence position, so
            # the first measured frame is fresh and starts a new gap baseline.
            self._latest = None
            self._last_delivered_seq = None

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        super().close()


def open_source(
    url: str,
    mode: str = "realtime",
    pace_fps: float | None = None,
    loop: bool = False,
) -> FrameSource:
    """Build the right source for a URL and mode, and open it."""
    if mode not in {"realtime", "sequential"}:
        raise ValueError(f"mode must be 'realtime' or 'sequential', got {mode!r}")
    if "://" not in url and not Path(url).exists():
        raise FileNotFoundError(f"no such file: {url}")
    source: FrameSource = (
        RealtimeSource(url, pace_fps=pace_fps, loop=loop)
        if mode == "realtime"
        else SequentialSource(url, loop=loop)
    )
    source.open()
    return source
