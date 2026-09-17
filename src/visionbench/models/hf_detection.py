"""Adapter for any Hugging Face `AutoModelForObjectDetection` checkpoint.

DETR, RT-DETRv2 and D-FINE all expose the same processor plus model plus
post-process contract, so one adapter covers three families. Adding a fourth
that follows the same contract costs one registry entry and no code.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from visionbench.models.base import Detector, stage
from visionbench.types import Detections, InferTimings


class HFDetectionModel(Detector):
    def __init__(
        self,
        weights: str,
        device: str = "auto",
        confidence: float = 0.3,
        imgsz: int | None = None,
        precision: str = "fp32",
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, confidence=confidence, imgsz=imgsz, **kwargs)
        self.weights = weights
        self.precision = precision
        self.model = None
        self.processor = None
        self.names: dict[int, str] = {}
        self.input_shape: tuple[int, ...] | None = None

    @property
    def _half(self) -> bool:
        return self.precision == "fp16" and not self.device.startswith("cpu")

    def _apply_imgsz(self) -> None:
        """Override the processor's resize target, honouring its own schema.

        DETR-style processors describe size as shortest/longest edge; RT-DETR
        and D-FINE use explicit height and width. Writing the wrong shape is
        silently ignored by the processor, so branch on what is already there.
        """
        if not self.imgsz or self.processor is None:
            return
        size = getattr(self.processor, "size", None)
        if not isinstance(size, dict):
            return
        target = int(self.imgsz)
        if "shortest_edge" in size:
            longest = size.get("longest_edge") or target
            self.processor.size = {
                "shortest_edge": target,
                "longest_edge": int(max(target, longest)),
            }
        elif "height" in size or "width" in size:
            self.processor.size = {"height": target, "width": target}

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.processor = AutoImageProcessor.from_pretrained(self.weights)
        self._apply_imgsz()

        model = AutoModelForObjectDetection.from_pretrained(self.weights)
        model.eval()
        model.to(self.device)
        if self._half:
            model.half()
        self.model = model

        id2label = getattr(model.config, "id2label", {}) or {}
        self.names = {int(k): str(v) for k, v in id2label.items()}
        self.loaded = True

    def infer(self, frame_bgr: np.ndarray) -> tuple[Detections, InferTimings]:
        import torch

        if self.model is None or self.processor is None:
            raise RuntimeError(f"{self.key}: load() not called")

        height, width = frame_bgr.shape[:2]

        with stage(self.device) as t_pre:
            # The processors accept numpy RGB directly; going through PIL adds a
            # copy per frame for nothing.
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            encoded = self.processor(images=rgb, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in encoded.items()}
            if self._half and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].half()
            pixel_values = inputs.get("pixel_values")
            if pixel_values is not None:
                self.input_shape = tuple(pixel_values.shape)

        with stage(self.device) as t_infer, torch.inference_mode():
            outputs = self.model(**inputs)

        with stage(None) as t_post:
            processed = self.processor.post_process_object_detection(
                outputs,
                threshold=self.confidence,
                target_sizes=[(height, width)],
            )[0]
            detections = Detections(
                xyxy=processed["boxes"].detach().float().cpu().numpy().astype(np.float32),
                confidence=processed["scores"].detach().float().cpu().numpy().astype(np.float32),
                class_id=processed["labels"].detach().cpu().numpy().astype(np.int32),
                names=self.names,
            )

        return detections, InferTimings(t_pre[0], t_infer[0], t_post[0])

    def info(self) -> dict[str, Any]:
        base = super().info()
        base["precision"] = self.precision
        base["input_shape"] = list(self.input_shape) if self.input_shape else None
        return base
