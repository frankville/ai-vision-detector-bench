"""Detector implementations and the model catalogue."""

from visionbench.models.base import Detector, resolve_device
from visionbench.models.registry import (
    DEFAULT_SWEEP,
    REGISTRY,
    SPECS,
    ModelSpec,
    build,
    get,
    permissive_keys,
)

__all__ = [
    "DEFAULT_SWEEP",
    "REGISTRY",
    "SPECS",
    "Detector",
    "ModelSpec",
    "build",
    "get",
    "permissive_keys",
    "resolve_device",
]
