"""Model catalogue.

Licence is a first-class field here, not a footnote. The point of this catalogue
is to answer "which of these can I actually ship" at the same time as "which of
these is fast enough", because a model that wins on latency and loses on licence
has not won anything.

This module deliberately imports neither torch nor transformers, so listing the
catalogue stays instant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from visionbench.models.base import Detector

APACHE = "Apache-2.0"
AGPL = "AGPL-3.0"

#: Verdicts used in the `commercial` column.
PERMISSIVE = "yes"
COPYLEFT = "no, without a paid licence"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    family: str
    weights: str
    license: str
    license_url: str
    commercial: str
    factory: str  # "module.path:ClassName"
    kwargs: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    #: Models not installed by default need an extra; named here for the error message.
    extra: str | None = None

    @property
    def permissive(self) -> bool:
        return self.commercial == PERMISSIVE


_HF = "visionbench.models.hf_detection:HFDetectionModel"
_RFDETR = "visionbench.models.rfdetr_model:RFDETRModel"
_ULTRA = "visionbench.models.ultralytics_model:UltralyticsModel"

_APACHE_URL = "https://www.apache.org/licenses/LICENSE-2.0"
_AGPL_URL = "https://www.gnu.org/licenses/agpl-3.0.en.html"


def _spec(key: str, name: str, family: str, weights: str, factory: str, **kw: Any) -> ModelSpec:
    return ModelSpec(
        key=key,
        display_name=name,
        family=family,
        weights=weights,
        license=kw.pop("license", APACHE),
        license_url=kw.pop("license_url", _APACHE_URL),
        commercial=kw.pop("commercial", PERMISSIVE),
        factory=factory,
        kwargs=kw.pop("kwargs", {}),
        notes=kw.pop("notes", ""),
        extra=kw.pop("extra", None),
    )


SPECS: list[ModelSpec] = [
    # -- DETR, the original transformer detector. Slow, but the reference point.
    _spec(
        "detr-r50",
        "DETR ResNet-50",
        "DETR",
        "facebook/detr-resnet-50",
        _HF,
        notes="Original 2020 DETR. Included as the accuracy/latency baseline.",
    ),
    _spec(
        "detr-r101",
        "DETR ResNet-101",
        "DETR",
        "facebook/detr-resnet-101",
        _HF,
        notes="Heavier DETR backbone.",
    ),
    # -- RT-DETRv2, real-time transformer detector from Baidu.
    _spec(
        "rtdetrv2-r18",
        "RT-DETRv2 R18",
        "RT-DETR",
        "PekingU/rtdetr_v2_r18vd",
        _HF,
        notes="Smallest real-time DETR variant. Strong edge candidate.",
    ),
    _spec(
        "rtdetrv2-r50",
        "RT-DETRv2 R50",
        "RT-DETR",
        "PekingU/rtdetr_v2_r50vd",
        _HF,
    ),
    # -- D-FINE, refines box regression as distribution refinement.
    _spec(
        "dfine-nano",
        "D-FINE Nano",
        "D-FINE",
        "ustc-community/dfine-nano-coco",
        _HF,
        notes="Lightest permissive model in the catalogue.",
    ),
    _spec(
        "dfine-small",
        "D-FINE Small",
        "D-FINE",
        "ustc-community/dfine-small-coco",
        _HF,
    ),
    _spec(
        "dfine-medium",
        "D-FINE Medium",
        "D-FINE",
        "ustc-community/dfine-medium-coco",
        _HF,
    ),
    # -- RF-DETR from Roboflow.
    _spec(
        "rfdetr-nano",
        "RF-DETR Nano",
        "RF-DETR",
        "rfdetr:RFDETRNano",
        _RFDETR,
        extra="rfdetr",
        kwargs={"variant": "RFDETRNano"},
        notes="Apache-2.0. The RF-DETR Plus variants (XL/2XL) are under Roboflow "
        "PML 1.0 and are deliberately excluded from this catalogue.",
    ),
    _spec(
        "rfdetr-small",
        "RF-DETR Small",
        "RF-DETR",
        "rfdetr:RFDETRSmall",
        _RFDETR,
        extra="rfdetr",
        kwargs={"variant": "RFDETRSmall"},
    ),
    _spec(
        "rfdetr-base",
        "RF-DETR Base",
        "RF-DETR",
        "rfdetr:RFDETRBase",
        _RFDETR,
        extra="rfdetr",
        kwargs={"variant": "RFDETRBase"},
    ),
    # -- AGPL baseline. Never a default; see docs/licensing.md.
    _spec(
        "yolo11n",
        "YOLO11 Nano",
        "YOLO",
        "yolo11n.pt",
        _ULTRA,
        license=AGPL,
        license_url=_AGPL_URL,
        commercial=COPYLEFT,
        extra="agpl-baseline",
        notes="Reference only. AGPL-3.0 obliges you to publish the source of any "
        "networked service built on it, or buy an Ultralytics licence.",
    ),
    _spec(
        "yolo11s",
        "YOLO11 Small",
        "YOLO",
        "yolo11s.pt",
        _ULTRA,
        license=AGPL,
        license_url=_AGPL_URL,
        commercial=COPYLEFT,
        extra="agpl-baseline",
        notes="Reference only. See yolo11n.",
    ),
]

REGISTRY: dict[str, ModelSpec] = {s.key: s for s in SPECS}

#: Sensible sweep when you just want a first comparison.
DEFAULT_SWEEP = ["dfine-nano", "rtdetrv2-r18", "detr-r50"]


def get(key: str) -> ModelSpec:
    try:
        return REGISTRY[key]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown model {key!r}. Known models: {known}") from None


def permissive_keys() -> list[str]:
    return [s.key for s in SPECS if s.permissive]


def build(key: str, **overrides: Any) -> Detector:
    """Instantiate a detector by registry key.

    Imports the backend lazily so that a missing optional dependency only bites
    when that particular model is actually requested.
    """
    spec = get(key)
    module_path, _, class_name = spec.factory.partition(":")
    try:
        module = import_module(module_path)
    except ImportError as exc:
        hint = f"  pip install 'visionbench[{spec.extra}]'" if spec.extra else ""
        raise ImportError(
            f"model {spec.key!r} needs a backend that is not installed: {exc}\n{hint}"
        ) from exc
    cls = getattr(module, class_name)
    kwargs = {**spec.kwargs, **overrides}
    detector = cls(weights=spec.weights, **kwargs)
    detector.key = spec.key
    detector.display_name = spec.display_name
    detector.license = spec.license
    return detector
