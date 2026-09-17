"""Frame annotation for the live view.

Drawing is deliberately cheap and is timed as its own stage, because in a
dashboard it competes with inference for the same CPU and can quietly become
the bottleneck on a small box.
"""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

from visionbench.types import Detections

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for(class_id: int) -> tuple[int, int, int]:
    """Stable, well-separated BGR colour per class id via the golden ratio."""
    hue = (class_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def annotate(
    frame_bgr: np.ndarray,
    detections: Detections,
    hud: list[str] | None = None,
    show_confidence: bool = True,
) -> np.ndarray:
    """Draw boxes, labels and an optional heads-up display. Returns a new image."""
    canvas = frame_bgr.copy()
    height, width = canvas.shape[:2]
    thickness = max(1, round(min(width, height) / 400))
    scale = max(0.4, min(width, height) / 1200)

    for i in range(len(detections)):
        x1, y1, x2, y2 = (int(v) for v in detections.xyxy[i])
        colour = color_for(int(detections.class_id[i]))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)

        label = detections.label(i)
        if show_confidence:
            label = f"{label} {detections.confidence[i]:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, _FONT, scale, thickness)
        # Keep the caption inside the frame when the box touches the top edge.
        top = y1 - text_h - baseline
        if top < 0:
            top = y1
        cv2.rectangle(
            canvas, (x1, top), (x1 + text_w + 4, top + text_h + baseline), colour, -1
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 2, top + text_h),
            _FONT,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    if hud:
        _draw_hud(canvas, hud, scale, thickness)
    return canvas


def _draw_hud(canvas: np.ndarray, lines: list[str], scale: float, thickness: int) -> None:
    pad = 8
    scale = scale * 1.1
    sizes = [cv2.getTextSize(line, _FONT, scale, thickness)[0] for line in lines]
    box_w = max(w for w, _ in sizes) + pad * 2
    line_h = max(h for _, h in sizes) + 6
    box_h = line_h * len(lines) + pad

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    y = pad + line_h - 6
    for line in lines:
        cv2.putText(canvas, line, (pad, y), _FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_h


def encode_jpeg(frame_bgr: np.ndarray, quality: int = 75) -> bytes | None:
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes() if ok else None


def placeholder(width: int, height: int, message: str) -> bytes | None:
    """A dark frame carrying a message, for before a stream has connected."""
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    scale = max(0.5, width / 900)
    (text_w, text_h), _ = cv2.getTextSize(message, _FONT, scale, 1)
    cv2.putText(
        canvas,
        message,
        ((width - text_w) // 2, (height + text_h) // 2),
        _FONT,
        scale,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return encode_jpeg(canvas)
