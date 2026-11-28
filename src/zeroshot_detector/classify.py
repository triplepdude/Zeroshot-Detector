"""Frame-level decision rule: keyword override, then thresholded zero-shot.

The original rule (``Recursive Florence 2 + Bart + Confidence.py``) is::

    if any military keyword in text:            -> ground_vehicles
    elif top zero-shot score >= 0.8:            -> top zero-shot label
    else:                                       -> nonmilitary

:class:`FrameClassifier` implements that rule in two modes:

``legacy=True``
    bit-for-bit the original behaviour: object labels are concatenated without
    a separator (what ``batch_decode(..., skip_special_tokens=True)`` produced)
    and the keyword test is a raw substring test (:func:`legacy_keyword_match`).
``legacy=False``
    the corrected behaviour: object labels are joined with ``", "`` for the NLI
    model and the keyword test is :class:`~zeroshot_detector.rules.KeywordRule`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from .rules import KeywordRule, legacy_keyword_match
from .taxonomy import LABELS, LEGACY_MILITARY_KEYWORDS, NONMILITARY, validate_labels
from .zero_shot import ZeroShotScorer

#: Decision sources recorded for every frame.
SOURCE_KEYWORD = "keyword"
SOURCE_ZERO_SHOT = "zero_shot"
SOURCE_BELOW_THRESHOLD = "below_threshold"
SOURCE_EMPTY = "empty"


@dataclass(frozen=True)
class FrameDecision:
    """Outcome of classifying one frame description."""

    text: str
    category: str | None  #: ``None`` when the detector produced no text (frame not counted)
    source: str
    top_label: str | None = None
    top_score: float | None = None
    scores: dict[str, float] = field(default_factory=dict)
    matched_keyword: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def decide(scores: dict[str, float], threshold: float, fallback: str = NONMILITARY) -> tuple[str, str, str, float]:
    """Apply the confidence threshold to a ``{label: score}`` dict.

    Returns ``(category, source, top_label, top_score)``. Ties on the top score
    are broken by the order of ``scores`` (the label order), as ``max`` does.
    """
    top_label = max(scores, key=scores.get)
    top_score = scores[top_label]
    if top_score >= threshold:
        return top_label, SOURCE_ZERO_SHOT, top_label, top_score
    return fallback, SOURCE_BELOW_THRESHOLD, top_label, top_score


class FrameClassifier:
    """Classify Florence-2 object lists (or stored description strings)."""

    def __init__(
        self,
        scorer: ZeroShotScorer,
        labels: Sequence[str] = LABELS,
        threshold: float = 0.8,
        legacy: bool = True,
        keyword_category: str = "ground_vehicles",
        keywords: Sequence[str] = LEGACY_MILITARY_KEYWORDS,
        keyword_rule: KeywordRule | None = None,
        always_score: bool = False,
    ) -> None:
        self.scorer = scorer
        self.labels = validate_labels(labels)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if keyword_category not in self.labels:
            raise ValueError(f"keyword_category {keyword_category!r} not in labels")
        self.threshold = threshold
        self.legacy = legacy
        self.keyword_category = keyword_category
        self.keywords = tuple(keywords)
        self.keyword_rule = keyword_rule or KeywordRule({keyword_category: self.keywords})
        #: Also run the zero-shot model on keyword-matched frames (for analysis).
        self.always_score = always_score

    # -- text construction -------------------------------------------------
    def text_for(self, object_labels: Sequence[str]) -> str:
        """The string given to the NLI model for a list of detected objects."""
        if self.legacy:
            return "".join(object_labels)
        return ", ".join(object_labels)

    def _keyword(self, text: str, object_labels: Sequence[str] | None) -> str | None:
        if self.legacy:
            return legacy_keyword_match(text, self.keywords)
        objects = object_labels if object_labels is not None else [text]
        match = self.keyword_rule.match(objects)
        return match.keyword if match else None

    # -- classification ----------------------------------------------------
    def classify_objects(self, frames: Sequence[Sequence[str]]) -> list[FrameDecision]:
        """Classify frames given as lists of detected object labels."""
        texts = [self.text_for(objects) for objects in frames]
        return self._classify(texts, [list(objects) for objects in frames])

    def classify_texts(self, texts: Sequence[str]) -> list[FrameDecision]:
        """Classify stored description strings (e.g. to re-derive archived labels)."""
        return self._classify(list(texts), [None] * len(texts))

    def _classify(self, texts: list[str], objects: list[list[str] | None]) -> list[FrameDecision]:
        keywords = [self._keyword(t, o) if t else None for t, o in zip(texts, objects, strict=True)]
        to_score = [i for i, t in enumerate(texts) if t and (keywords[i] is None or self.always_score)]
        scored = self.scorer.score([texts[i] for i in to_score], self.labels) if to_score else []
        score_by_index = dict(zip(to_score, scored, strict=True))

        decisions: list[FrameDecision] = []
        for i, text in enumerate(texts):
            scores = score_by_index.get(i, {})
            top_label = max(scores, key=scores.get) if scores else None
            top_score = scores[top_label] if scores else None
            if not text:
                decisions.append(FrameDecision(text=text, category=None, source=SOURCE_EMPTY))
            elif keywords[i] is not None:
                decisions.append(
                    FrameDecision(
                        text=text,
                        category=self.keyword_category,
                        source=SOURCE_KEYWORD,
                        top_label=top_label,
                        top_score=top_score,
                        scores=scores,
                        matched_keyword=keywords[i],
                    )
                )
            else:
                category, source, top_label, top_score = decide(scores, self.threshold)
                decisions.append(
                    FrameDecision(
                        text=text,
                        category=category,
                        source=source,
                        top_label=top_label,
                        top_score=top_score,
                        scores=scores,
                    )
                )
        return decisions
