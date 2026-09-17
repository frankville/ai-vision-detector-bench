"""Host fingerprint recorded with every run.

A benchmark number without the machine it came from is noise, and the whole
point of this tool is comparing the same model across very different boxes.
"""

from __future__ import annotations

import platform
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import psutil


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


def _torch_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"available": False}

    info: dict = {
        "available": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ),
    }
    if info["cuda_available"]:
        try:
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_memory_mb"] = round(props.total_memory / 1024**2, 1)
            info["gpu_capability"] = f"{props.major}.{props.minor}"
        except Exception:
            pass
    return info


@lru_cache(maxsize=1)
def host_info() -> dict:
    virtual = psutil.virtual_memory()
    return {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "arch": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(virtual.total / 1024**3, 2),
        "python": platform.python_version(),
        "torch": _torch_info(),
    }


def default_profile() -> str:
    """A short slug naming this machine, used to group result files.

    Override with --profile when the auto-derived name is not meaningful, for
    example on a rented GPU instance where the hostname is a random string.
    """
    info = host_info()
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
