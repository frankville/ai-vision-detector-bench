"""Exercise the real HF adapter with a tiny local, randomly initialized DETR."""

from __future__ import annotations

import numpy as np
import pytest

from visionbench.models.hf_detection import HFDetectionModel

pytestmark = pytest.mark.integration


def test_height_width_processor_honors_input_size():
    from transformers import RTDetrImageProcessor

    detector = HFDetectionModel("unused", device="cpu", imgsz=64)
    detector.processor = RTDetrImageProcessor(size={"height": 96, "width": 96})
    detector._apply_imgsz()
    encoded = detector.processor(images=np.zeros((48, 80, 3), dtype=np.uint8), return_tensors="pt")
    assert tuple(encoded["pixel_values"].shape) == (1, 3, 64, 64)


def test_hf_adapter_with_local_random_detr(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    import torch
    from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor, ResNetConfig

    previous_threads = torch.get_num_threads()
    detector = None
    try:
        torch.set_num_threads(1)
        checkpoint = tmp_path / "random-detr"
        names = {0: "first", 1: "second"}
        config = DetrConfig(
            use_timm_backbone=False,
            use_pretrained_backbone=False,
            backbone=None,
            backbone_config=ResNetConfig(
                depths=[1, 1, 1, 1],
                hidden_sizes=[8, 16, 32, 64],
                embedding_size=8,
                out_features=["stage4"],
            ),
            d_model=32,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=64,
            decoder_ffn_dim=64,
            num_queries=5,
            num_labels=2,
            id2label=names,
            label2id={name: index for index, name in names.items()},
        )
        DetrForObjectDetection(config).save_pretrained(checkpoint)
        DetrImageProcessor(size={"shortest_edge": 96, "longest_edge": 96}).save_pretrained(
            checkpoint
        )
        detector = HFDetectionModel(
            weights=str(checkpoint), device="cpu", precision="fp32", imgsz=64, confidence=0.0
        )
        detector.load()
        assert detector.loaded
        frame = np.full((96, 96, 3), 127, dtype=np.uint8)
        detector.warmup(frame, rounds=1)
        detections, timings = detector.infer(frame)

        # Random weights validate integration contracts, not detection accuracy.
        assert detections.xyxy.shape == (config.num_queries, 4)
        assert detections.confidence.shape == (config.num_queries,)
        assert detections.class_id.shape == (config.num_queries,)
        assert detections.xyxy.dtype == np.float32
        assert detections.confidence.dtype == np.float32
        assert detections.class_id.dtype == np.int32
        assert np.isfinite(detections.xyxy).all()
        assert np.isfinite(detections.confidence).all()
        assert (detections.xyxy[:, 2:] >= detections.xyxy[:, :2]).all()
        assert ((detections.confidence >= 0) & (detections.confidence <= 1)).all()
        assert set(detections.class_id).issubset(names)
        assert detections.names == names
        assert detector.input_shape == (1, 3, 64, 64)
        info = detector.info()
        assert info["device"] == "cpu"
        assert info["backend"] == "pytorch"
        assert info["precision"] == "fp32"
        assert info["weights"] == str(checkpoint)
        assert info["checkpoint_commit"] is None
        assert info["input_shape"] == [1, 3, 64, 64]
        assert all(np.isfinite(value) and value >= 0 for value in timings)
        detector.unload()
        assert not detector.loaded
        assert detector.model is None
        assert detector.processor is None
    finally:
        if detector is not None and detector.loaded:
            detector.unload()
        torch.set_num_threads(previous_threads)
