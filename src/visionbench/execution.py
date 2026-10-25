"""Supported execution settings and their device constraints."""

PRECISIONS = ("fp32", "fp16", "amp-fp16")
BACKENDS = ("pytorch",)


def validate_execution(backend: str, precision: str, device: str = "auto") -> None:
    """Reject unsupported settings; defer automatic-device checks until resolution."""
    if backend not in BACKENDS:
        raise ValueError(f"unsupported backend {backend!r}; choose from {', '.join(BACKENDS)}")
    if precision not in PRECISIONS:
        raise ValueError(f"unsupported precision {precision!r}; choose from {', '.join(PRECISIONS)}")
    device_type = device.split(":", 1)[0]
    if device_type not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError(f"unsupported device {device!r}; choose auto, cpu, cuda[:index], or mps")
    if device == "auto":
        return
    if precision == "fp16" and device_type == "cpu":
        raise ValueError("fp16 precision requires a CUDA or MPS device; CPU fp16 is unsupported")
    if precision == "amp-fp16" and device_type != "cuda":
        raise ValueError("amp-fp16 precision requires a CUDA device")
