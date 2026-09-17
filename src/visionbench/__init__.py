"""visionbench: benchmark and compare real-time object detection models on live video.

Quick start::

    from visionbench import RunConfig, run

    result = run(RunConfig(source="clips/corridor.mp4", model="dfine-nano", duration_s=30))
    print(result.throughput["processed_fps"])
"""

__version__ = "0.1.0"

from visionbench.runner import RunConfig, RunResult, run, sweep
from visionbench.types import Detections, Frame, InferTimings

__all__ = [
    "Detections",
    "Frame",
    "InferTimings",
    "RunConfig",
    "RunResult",
    "__version__",
    "run",
    "sweep",
]
