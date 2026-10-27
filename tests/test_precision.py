"""Precision control-flow checks using CPU tensors and mocked accelerator boundaries."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from visionbench.execution import BACKENDS, PRECISIONS, validate_execution
from visionbench.models.base import resolve_device, sync_device
from visionbench.models.hf_detection import HFDetectionModel


@pytest.mark.parametrize(
    ("precision", "device"),
    [
        ("fp32", "cpu"),
        ("fp32", "cuda:1"),
        ("fp16", "cuda:1"),
        ("fp16", "mps"),
        ("amp-fp16", "cuda:1"),
        *[(precision, "auto") for precision in PRECISIONS],
    ],
)
def test_supported_execution(precision, device):
    validate_execution("pytorch", precision, device)
    assert BACKENDS == ("pytorch",)


@pytest.mark.parametrize(
    ("backend", "precision", "device", "message"),
    [
        ("tensorrt", "fp32", "cuda", "unsupported backend"),
        ("pytorch", "int8", "cuda", "unsupported precision"),
        ("pytorch", "fp16", "cpu", "CPU fp16"),
        ("pytorch", "fp16", "cpu:0", "CPU fp16"),
        ("pytorch", "amp-fp16", "cpu", "requires a CUDA"),
        ("pytorch", "amp-fp16", "mps", "requires a CUDA"),
        ("pytorch", "fp32", "xpu", "unsupported device"),
    ],
)
def test_invalid_execution(backend, precision, device, message):
    with pytest.raises(ValueError, match=message):
        validate_execution(backend, precision, device)


@pytest.fixture
def fake_hf(monkeypatch):
    state = SimpleNamespace(events=[], active=False, inputs=None, inference_mode=False)
    state.encoded = {
        "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
        "pixel_mask": torch.ones((1, 2, 2), dtype=torch.int64),
        "metadata": torch.tensor([2.0], dtype=torch.float64),
        "flags": torch.tensor([True]),
    }

    class Model:
        config = SimpleNamespace(id2label={0: "object"})

        def __init__(self):
            self.weight = torch.ones(2, dtype=torch.float64)

        def eval(self):
            state.events.append("eval")
            return self

        def float(self):
            self.weight = self.weight.float()
            return self

        def half(self):
            self.weight = self.weight.half()
            return self

        def to(self, device):
            state.events.append(("weights", device, self.weight.dtype))
            return self

        def parameters(self):
            return [self.weight]

        def __call__(self, **inputs):
            state.inputs = inputs
            state.inference_mode = torch.is_inference_mode_enabled()
            state.events.append(("forward", state.active))
            return object()

    class Processor:
        size = {"height": 2, "width": 2}

        def __call__(self, *, images, return_tensors):
            assert images.shape == (2, 2, 3)
            assert return_tensors == "pt"
            state.events.append(("processor", state.active))
            return dict(state.encoded)

        def post_process_object_detection(self, outputs, *, threshold, target_sizes):
            state.events.append(("postprocess", state.active))
            assert target_sizes == [(2, 2)]
            return [{
                "boxes": torch.tensor([[0, 0, 1, 1]], dtype=torch.float16),
                "scores": torch.tensor([0.9], dtype=torch.float16),
                "labels": torch.tensor([0]),
            }]

    state.model = Model()
    state.processor = Processor()

    def load_processor(weights):
        state.events.append("download_processor")
        return state.processor

    def load_model(weights):
        state.events.append("download_model")
        return state.model

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoImageProcessor=SimpleNamespace(from_pretrained=load_processor),
        AutoModelForObjectDetection=SimpleNamespace(from_pretrained=load_model),
    ))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: state.events.append(("sync", device)))
    monkeypatch.setattr(torch.mps, "synchronize", lambda: state.events.append(("sync", "mps")))

    original_to = torch.Tensor.to

    def transfer(tensor, *args, **kwargs):
        device = args[0] if args else kwargs.get("device")
        if isinstance(device, str):
            state.events.append(("transfer", device, tensor.dtype, tensor.device.type))
            if device.startswith(("cuda", "mps")):
                return tensor
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", transfer)

    @contextmanager
    def autocast(*, device_type, dtype):
        assert device_type == "cuda"
        assert dtype == torch.float16
        assert not state.active
        state.active = True
        state.events.append("autocast_enter")
        try:
            yield
        finally:
            state.events.append("autocast_exit")
            state.active = False

    monkeypatch.setattr(torch, "autocast", autocast)
    return state


@pytest.mark.parametrize("precision", PRECISIONS)
def test_weights_inputs_and_autocast_scope(fake_hf, precision):
    detector = HFDetectionModel("test/checkpoint", "cuda:1", 0.3, None, precision)
    detector.load()
    detections, timings = detector.infer(np.zeros((2, 2, 3), dtype=np.uint8))

    dtype = torch.float16 if precision == "fp16" else torch.float32
    assert fake_hf.model.weight.dtype == dtype
    assert fake_hf.inputs["pixel_values"].dtype == dtype
    assert fake_hf.inputs["pixel_mask"].dtype == torch.int64
    assert fake_hf.inputs["metadata"].dtype == torch.float64
    assert fake_hf.inputs["flags"].dtype == torch.bool
    transfers = [event for event in fake_hf.events if isinstance(event, tuple) and event[0] == "transfer"]
    assert transfers == [
        ("transfer", "cuda:1", dtype, "cpu"),
        ("transfer", "cuda:1", torch.int64, "cpu"),
        ("transfer", "cuda:1", torch.float64, "cpu"),
        ("transfer", "cuda:1", torch.bool, "cpu"),
    ]
    assert ("processor", False) in fake_hf.events
    assert ("forward", precision == "amp-fp16") in fake_hf.events
    assert ("postprocess", False) in fake_hf.events
    assert fake_hf.inference_mode
    assert fake_hf.events.count(("sync", "cuda:1")) == 2
    if precision == "amp-fp16":
        assert fake_hf.events.index("autocast_exit") < fake_hf.events.index(("postprocess", False))
        forward = fake_hf.events.index(("forward", True))
        assert fake_hf.events[forward + 1:forward + 3] == ["autocast_exit", ("sync", "cuda:1")]
    else:
        assert "autocast_enter" not in fake_hf.events
    assert detector.info()["backend"] == "pytorch"
    assert detector.info()["precision"] == precision
    assert detector.info()["input_shape"] == [1, 3, 2, 2]
    assert detections.xyxy.dtype == np.float32
    assert detections.confidence.dtype == np.float32
    assert detections.class_id.dtype == np.int32
    assert all(value >= 0 for value in (timings.pre_ms, timings.infer_ms, timings.post_ms))


def test_cpu_fp32_uses_real_tensors(fake_hf):
    detector = HFDetectionModel("test/checkpoint", device="cpu")
    detector.load()
    detector.infer(np.zeros((2, 2, 3), dtype=np.uint8))
    assert fake_hf.inputs["pixel_values"].device.type == "cpu"
    assert fake_hf.inputs["pixel_values"].dtype == torch.float32
    assert fake_hf.model.weight.dtype == torch.float32
    assert ("forward", False) in fake_hf.events
    assert not any(isinstance(event, tuple) and event[0] == "sync" for event in fake_hf.events)


def test_mps_fp16_is_explicit_half(fake_hf):
    detector = HFDetectionModel("test/checkpoint", device="mps", precision="fp16")
    detector.load()
    detector.infer(np.zeros((2, 2, 3), dtype=np.uint8))
    assert fake_hf.model.weight.dtype == torch.float16
    assert ("transfer", "mps", torch.float16, "cpu") in fake_hf.events
    assert "autocast_enter" not in fake_hf.events


def test_integer_pixel_values_are_not_cast(fake_hf):
    fake_hf.encoded["pixel_values"] = torch.ones((1, 3, 2, 2), dtype=torch.int32)
    detector = HFDetectionModel("test/checkpoint", device="cuda", precision="fp16")
    detector.load()
    detector.infer(np.zeros((2, 2, 3), dtype=np.uint8))
    assert fake_hf.inputs["pixel_values"].dtype == torch.int32
    assert ("transfer", "cuda", torch.int32, "cpu") in fake_hf.events


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"device": "cpu", "precision": "fp16"}, "CPU fp16"),
        ({"device": "mps", "precision": "amp-fp16"}, "requires a CUDA"),
        ({"backend": "onnx"}, "unsupported backend"),
        ({"precision": "bf16"}, "unsupported precision"),
    ],
)
def test_invalid_model_rejected_before_download(fake_hf, kwargs, message):
    with pytest.raises(ValueError, match=message):
        HFDetectionModel("test/checkpoint", **kwargs).load()
    assert fake_hf.events == []


@pytest.mark.parametrize("precision", ["fp16", "amp-fp16"])
def test_auto_device_is_checked_after_resolution(fake_hf, monkeypatch, precision):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CPU fp16|requires a CUDA"):
        HFDetectionModel("test/checkpoint", precision=precision).load()
    assert fake_hf.events == []


def test_auto_mps_rejects_amp_before_download(fake_hf, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="requires a CUDA"):
        HFDetectionModel("test/checkpoint", precision="amp-fp16").load()
    assert fake_hf.events == []


def test_auto_device_records_resolved_execution(fake_hf, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    detector = HFDetectionModel("test/checkpoint", precision="fp16")
    detector.load()
    assert detector.info()["device"] == "mps"
    assert detector.info()["precision"] == "fp16"
    assert detector.info()["backend"] == "pytorch"


def test_load_revalidates_before_download(fake_hf):
    detector = HFDetectionModel("test/checkpoint", device="cpu")
    detector.precision = "amp-fp16"
    with pytest.raises(ValueError, match="requires a CUDA"):
        detector.load()
    assert fake_hf.events == []


def test_unavailable_cuda_rejected_before_download(fake_hf, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        HFDetectionModel("test/checkpoint", device="cuda:0").load()
    assert fake_hf.events == []


def test_unavailable_cuda_index_rejected_before_download(fake_hf):
    with pytest.raises(ValueError, match="index 2 is unavailable"):
        HFDetectionModel("test/checkpoint", device="cuda:2").load()
    assert fake_hf.events == []


def test_resolve_and_synchronize_indexed_cuda(fake_hf):
    assert resolve_device("cuda:1") == "cuda:1"
    sync_device("cuda:1")
    assert fake_hf.events == [("sync", "cuda:1")]


@pytest.mark.parametrize("device", ["cuda:invalid", "cuda:-1", "xpu", "mps:1"])
def test_resolve_invalid_device(fake_hf, device):
    with pytest.raises(ValueError):
        resolve_device(device)


@pytest.mark.parametrize("size,expected", [
    ({"height": 640, "width": 640}, {"height": 320, "width": 320}),
    ({"shortest_edge": 640, "longest_edge": 1333}, {"shortest_edge": 320, "longest_edge": 1333}),
])
@pytest.mark.parametrize("structured", [False, True])
def test_resize_override_supports_dictionary_protocol(fake_hf, size, expected, structured):
    class StructuredSize:
        def __iter__(self):
            yield from size.items()

    fake_hf.processor.size = StructuredSize() if structured else size
    detector = HFDetectionModel("test/checkpoint", device="cpu", imgsz=320)
    detector.load()
    assert detector.processor.size == expected
