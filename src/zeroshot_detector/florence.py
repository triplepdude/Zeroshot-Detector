"""Florence-2 object detection and parsing of its ``<OD>`` output.

Florence-2 emits detections as text: each label is followed by one or more
groups of four quantised coordinate tokens, e.g.::

    car<loc_34><loc_160><loc_597><loc_371>wheel<loc_96><loc_280><loc_198><loc_371><loc_454><loc_276><loc_553><loc_370></s>

The original study decoded this with ``skip_special_tokens=True``, which
removes the ``<loc_*>`` tokens and glues the labels together (``"carwheel"``).
:func:`parse_od` recovers the label list and boxes instead;
:func:`legacy_text` reproduces the glued string for comparison with the
archived data.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

_OD_GROUP = re.compile(r"([^<>]+?)((?:<loc_\d+>){4,})")
_LOC = re.compile(r"<loc_(\d+)>")
_SPECIAL = re.compile(r"<[^<>]*>")
NUM_BINS = 1000


@dataclass(frozen=True)
class Detection:
    label: str
    box: tuple[float, float, float, float] | None  #: (x1, y1, x2, y2) in pixels, or bins/1000 if size unknown


@dataclass
class FrameDetections:
    raw_text: str
    detections: list[Detection] = field(default_factory=list)

    @property
    def labels(self) -> list[str]:
        """Distinct object labels in order of first appearance."""
        return list(dict.fromkeys(d.label for d in self.detections))

    @property
    def legacy_text(self) -> str:
        return legacy_text(self.raw_text)


def _dequantise(bins: Sequence[int], image_size: tuple[int, int] | None) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bins
    if image_size is None:
        w = h = 1.0
    else:
        w, h = image_size
    return (
        (x1 + 0.5) / NUM_BINS * w,
        (y1 + 0.5) / NUM_BINS * h,
        (x2 + 0.5) / NUM_BINS * w,
        (y2 + 0.5) / NUM_BINS * h,
    )


def parse_od(raw_text: str, image_size: tuple[int, int] | None = None) -> FrameDetections:
    """Parse Florence-2 ``<OD>`` output (decoded *with* special tokens)."""
    text = raw_text.replace("</s>", "").replace("<s>", "").replace("<pad>", "")
    detections: list[Detection] = []
    for match in _OD_GROUP.finditer(text):
        label = match.group(1).strip()
        if not label:
            continue
        bins = [int(b) for b in _LOC.findall(match.group(2))]
        for i in range(0, len(bins) - len(bins) % 4, 4):
            detections.append(Detection(label=label, box=_dequantise(bins[i : i + 4], image_size)))
    return FrameDetections(raw_text=raw_text, detections=detections)


def legacy_text(raw_text: str) -> str:
    """The string the original pipeline stored: special tokens removed, labels glued."""
    return _SPECIAL.sub("", raw_text)


class FlorenceDetector:
    """Batched Florence-2 inference (``transformers`` remote code)."""

    def __init__(
        self,
        model: str = "microsoft/Florence-2-large",
        revision: str | None = None,
        task: str = "<OD>",
        num_beams: int = 3,
        max_new_tokens: int = 1024,
        dtype: str = "float32",
        device: str | None = None,
        backend: str = "auto",
    ) -> None:
        if backend not in ("auto", "remote", "native"):
            raise ValueError("backend must be auto, remote or native")
        #: ``remote``: Microsoft's ``trust_remote_code`` implementation (what the study used);
        #: ``native``: ``transformers.Florence2ForConditionalGeneration`` (transformers >= 4.56,
        #: converted ``florence-community/*`` checkpoints, no remote code).
        self.backend = (
            backend if backend != "auto" else ("native" if model.startswith("florence-community/") else "remote")
        )
        self.model_id = model
        self.revision = revision
        self.task = task
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.dtype_name = dtype
        self._device = device
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch

        self._torch = torch
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = getattr(torch, self.dtype_name)
        if self.backend == "native":
            from transformers import AutoProcessor, Florence2ForConditionalGeneration

            self._model = Florence2ForConditionalGeneration.from_pretrained(
                self.model_id, revision=self.revision, torch_dtype=dtype
            ).to(device)
            self._processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
        else:
            from transformers import AutoModelForCausalLM, AutoProcessor

            # The remote code predates SDPA detection and the cache classes of
            # recent transformers releases: eager attention + use_cache=False
            # (see detect()) work across versions.
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.revision,
                torch_dtype=dtype,
                trust_remote_code=True,
                attn_implementation="eager",
            ).to(device)
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, revision=self.revision, trust_remote_code=True
            )
        self._model.eval()
        self._device = device
        self._dtype = dtype

    @property
    def resolved_revision(self) -> str | None:
        return None if self._model is None else getattr(self._model.config, "_commit_hash", None)

    def detect(self, images: Sequence) -> list[FrameDetections]:
        """Run detection on PIL images; returns one :class:`FrameDetections` per image."""
        if not images:
            return []
        self._load()
        torch = self._torch
        inputs = self._processor(text=[self.task] * len(images), images=list(images), return_tensors="pt")
        with torch.inference_mode():
            generated = self._model.generate(
                input_ids=inputs["input_ids"].to(self._device),
                pixel_values=inputs["pixel_values"].to(self._device, self._dtype),
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
                use_cache=self.backend == "native",
            )
        raw = self._processor.batch_decode(generated, skip_special_tokens=False)
        return [parse_od(text, image.size) for text, image in zip(raw, images, strict=True)]
