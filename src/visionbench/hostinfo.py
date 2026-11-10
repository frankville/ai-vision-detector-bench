"""Host fingerprint recorded with every run.

A benchmark number without the machine it came from is noise, and the whole
point of this tool is comparing the same model across very different boxes.
"""

from __future__ import annotations

import platform
import re
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import psutil

PACKAGES = (
    "visionbench", "torch", "torchvision", "transformers", "huggingface-hub",
    "numpy", "opencv-python", "opencv-python-headless", "pillow", "timm", "psutil",
)


def _package_versions() -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except (PackageNotFoundError, OSError, ValueError):
            packages[name] = None
    return packages


def _optional(probe):
    try:
        return probe()
    except Exception:
        return None


def _nvidia_driver() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
        )
        versions = set(result.stdout.split())
        if result.returncode == 0 and len(versions) == 1:
            value = versions.pop()
            if re.fullmatch(r"\d+(?:\.\d+)+", value):
                return value
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _cpu_model() -> str:
    system = platform.system()
    if system == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    elif system == "Linux":
        try:
            text = Path("/proc/cpuinfo").read_text()
            match = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
            if match:
                return match.group(1).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown"


def _torch_info(device: str | None = None) -> dict:
    try:
        import torch
    except ImportError:
        return {"available": False}

    info: dict = {
        "available": True,
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": None,
        "device": device,
        "gpu_name": None,
        "gpu_total_memory_mb": None,
        "gpu_capability": None,
        "cuda_available": _optional(torch.cuda.is_available),
        "mps_available": _optional(lambda: bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )),
        "settings": {
            "num_threads": _optional(torch.get_num_threads),
            "num_interop_threads": _optional(torch.get_num_interop_threads),
            "cudnn_benchmark": _optional(lambda: torch.backends.cudnn.benchmark),
            "cudnn_deterministic": _optional(lambda: torch.backends.cudnn.deterministic),
            "cudnn_allow_tf32": _optional(lambda: torch.backends.cudnn.allow_tf32),
            "matmul_allow_tf32": _optional(lambda: torch.backends.cuda.matmul.allow_tf32),
        },
    }
    if info["cuda_available"] and (device is None or device.startswith("cuda")):
        try:
            selected = torch.device(device or "cuda")
            index = selected.index if selected.index is not None else torch.cuda.current_device()
            info["device"] = f"cuda:{index}"
            info["gpu_name"] = torch.cuda.get_device_name(index)
            props = torch.cuda.get_device_properties(index)
            info["gpu_total_memory_mb"] = round(props.total_memory / 1024**2, 1)
            info["gpu_capability"] = f"{props.major}.{props.minor}"
            info["cudnn_version"] = torch.backends.cudnn.version()
        except Exception:
            pass
    return info


def host_info(device: str | None = None) -> dict:
    """Hardware and software fingerprint for a run.

    Deliberately excludes the machine's hostname. Result files are meant to be
    committed to a public repository, and hostnames routinely carry a person's
    name or an employer's, neither of which tells you anything about how fast
    the model ran. Use `--profile` to label a machine with a name you chose.
    """
    virtual = psutil.virtual_memory()
    torch_info = _torch_info(device)
    return {
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "arch": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(virtual.total / 1024**3, 2),
        "python": platform.python_version(),
        "torch": torch_info,
        "packages": _package_versions(),
        "nvidia_driver": _nvidia_driver() if torch_info.get("cuda_available") else None,
    }


def default_profile(*, host: dict | None = None) -> str:
    """A short slug naming this machine, used to group result files.

    Override with --profile when the auto-derived name is not meaningful, for
    example on a rented GPU instance where the hostname is a random string.
    """
    info = host if host is not None else host_info()
    cpu = info["cpu_model"]
    torch_info = info["torch"]
    if torch_info.get("gpu_name"):
        base = torch_info["gpu_name"]
    elif "Apple" in cpu:
        base = cpu
    else:
        base = f"{cpu}-cpu"
    slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
    slug = re.sub(r"^(nvidia|intel-r|intel|amd)-", "", slug)
    slug = slug.replace("-cpu-", "-").replace("core-tm-", "")
    return slug or "unknown-host"
