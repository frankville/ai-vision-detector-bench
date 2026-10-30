from dataclasses import fields

import pytest

from visionbench.runner import RunConfig, run


@pytest.mark.parametrize("device,precision", [
    ("cpu", "fp32"), ("cuda", "fp16"), ("cuda:1", "amp-fp16"), ("mps", "fp16"),
])
def test_config_accepts_supported_settings(device, precision):
    config = RunConfig("clip.mp4", "dfine-nano", device=device, precision=precision)
    assert config.backend == "pytorch"
    assert config.redacted()["precision"] == precision


@pytest.mark.parametrize("settings", [
    {"precision": "int8"}, {"backend": "tensorrt"},
    {"device": "cpu", "precision": "fp16"},
    {"device": "mps", "precision": "amp-fp16"},
])
def test_config_rejects_unsupported_settings(settings):
    with pytest.raises(ValueError):
        RunConfig("clip.mp4", "dfine-nano", **settings)


def test_backend_appended_without_changing_positional_arguments():
    assert [item.name for item in fields(RunConfig)][-2:] == ["backend", "revision"]
    config = RunConfig("clip.mp4", "dfine-nano", "cpu", 12.0, 4)
    assert config.duration_s == 12.0
    assert config.max_frames == 4


@pytest.mark.parametrize("precision", ["fp16", "amp-fp16"])
def test_auto_device_validated_after_resolution(monkeypatch, precision):
    monkeypatch.setattr("visionbench.models.base.resolve_device", lambda _: "cpu")
    config = RunConfig("clip.mp4", "dfine-nano", precision=precision)
    with pytest.raises(ValueError):
        config.validate(resolve=True)


def test_run_rejects_mutated_config_before_model_build(monkeypatch):
    config = RunConfig("clip.mp4", "dfine-nano", device="cpu")
    config.precision = "typo"
    monkeypatch.setattr("visionbench.runner.build_model", lambda *a, **kw: pytest.fail("model built"))
    with pytest.raises(ValueError):
        run(config)


def test_config_preserves_source_redaction():
    config = RunConfig("rtsp://user:secret@camera/stream", "dfine-nano")
    assert "secret" not in str(config.redacted())
    assert config.redacted()["backend"] == "pytorch"


@pytest.mark.parametrize("revision", ["", " ", "../secret token", 1])
def test_invalid_revision_rejected(revision):
    with pytest.raises(ValueError, match="revision"):
        RunConfig("clip.mp4", "dfine-nano", revision=revision)


def test_revision_rejected_for_non_hf_model():
    with pytest.raises(ValueError, match="only for Hugging Face"):
        RunConfig("clip.mp4", "yolo11n", revision="main")


def test_registry_rejects_unsupported_revision_before_import(monkeypatch):
    from visionbench.models import registry

    monkeypatch.setattr(registry, "import_module", lambda *a: pytest.fail("backend imported"))
    with pytest.raises(ValueError, match="only for Hugging Face"):
        registry.build("yolo11n", revision="main")
