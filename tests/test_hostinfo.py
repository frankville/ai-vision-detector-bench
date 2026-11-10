import json
import subprocess
from types import SimpleNamespace

import pytest
import torch

from visionbench import hostinfo


def test_package_allowlist_and_missing_metadata(monkeypatch):
    queried = []

    def version(name):
        queried.append(name)
        if name == "timm":
            raise hostinfo.PackageNotFoundError(name)
        return "1.2.3"

    monkeypatch.setattr(hostinfo, "version", version)
    packages = hostinfo._package_versions()
    assert queried == list(hostinfo.PACKAGES)
    assert set(packages) == {
        "visionbench", "torch", "torchvision", "transformers", "huggingface-hub",
        "numpy", "opencv-python", "opencv-python-headless", "pillow", "timm", "psutil",
    }
    assert packages["timm"] is None and packages["torch"] == "1.2.3"


def test_host_snapshots_refresh_and_exclude_private_metadata(monkeypatch):
    monkeypatch.setenv("PRIVATE_TOKEN", "secret-token")
    monkeypatch.setattr(hostinfo, "version", lambda _: "1.0")
    monkeypatch.setattr(hostinfo, "_torch_info", lambda device: {"device": device})
    first = hostinfo.host_info("cpu")
    first["packages"]["torch"] = "mutated"
    monkeypatch.setattr(hostinfo, "version", lambda _: "2.0")
    second = hostinfo.host_info("cpu")
    assert second["packages"]["torch"] == "2.0"
    assert second["nvidia_driver"] is None
    assert set(second) == {
        "os", "platform", "arch", "cpu_model", "cpu_cores_physical",
        "cpu_cores_logical", "ram_total_gb", "python", "torch", "packages", "nvidia_driver",
    }
    assert "secret-token" not in json.dumps(second)


@pytest.mark.parametrize("device,expected", [("cuda:1", 1), ("cuda", 2), (None, 2)])
def test_torch_uses_visible_selected_or_current_gpu(monkeypatch, device, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"GPU-{index}")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(
        total_memory=(index + 1) * 1024**3, major=7, minor=5,
    ))
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 90000)
    info = hostinfo._torch_info(device)
    assert info["device"] == f"cuda:{expected}"
    assert info["gpu_name"] == f"GPU-{expected}"
    assert info["gpu_total_memory_mb"] == (expected + 1) * 1024
    assert info["gpu_capability"] == "7.5" and info["cudnn_version"] == 90000
    assert info["cuda_version"] == torch.version.cuda
    assert info["settings"]["num_threads"] == torch.get_num_threads()


def test_cpu_run_does_not_query_cuda_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: pytest.fail("GPU queried"))
    info = hostinfo._torch_info("cpu")
    assert info["device"] == "cpu" and info["gpu_name"] is None
    assert info["gpu_capability"] is None and info["gpu_total_memory_mb"] is None


def test_gpu_metadata_failure_remains_unknown(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: (_ for _ in ()).throw(RuntimeError("private-path")))
    info = hostinfo._torch_info("cuda:0")
    assert info["gpu_name"] is None
    assert "private-path" not in json.dumps(info)


def test_optional_package_and_cudnn_failures_are_nonfatal(monkeypatch):
    def fail(*a):
        raise OSError("private-path")

    monkeypatch.setattr(hostinfo, "version", fail)
    assert all(value is None for value in hostinfo._package_versions().values())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "T4")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(
        total_memory=1024**3, major=7, minor=5,
    ))
    monkeypatch.setattr(torch.backends.cudnn, "version", fail)
    info = hostinfo._torch_info("cuda:0")
    assert info["gpu_name"] == "T4" and info["cudnn_version"] is None
    assert "private-path" not in json.dumps(info)


@pytest.mark.parametrize("stdout,code,expected", [
    ("580.82.07\n580.82.07\n", 0, "580.82.07"), ("", 0, None),
    ("secret-path", 0, None), ("580.82.07", 1, None),
    ("580.82.07\n590.1", 0, None),
])
def test_driver_version_allowlist_and_bounded_query(monkeypatch, stdout, code, expected):
    def query(args, **options):
        assert args == ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        assert options["timeout"] == 3
        return SimpleNamespace(stdout=stdout, returncode=code)

    monkeypatch.setattr(subprocess, "run", query)
    assert hostinfo._nvidia_driver() == expected


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 3)])
def test_unavailable_driver_query_is_nonfatal(monkeypatch, error):
    def fail(*a, **kw):
        raise error

    monkeypatch.setattr(subprocess, "run", fail)
    assert hostinfo._nvidia_driver() is None
