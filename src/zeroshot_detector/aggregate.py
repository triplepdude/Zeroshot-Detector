"""Video-level aggregation of frame decisions.

Rules operate on a count matrix (videos × labels) so that they can be applied
to tens of thousands of videos at once; :func:`aggregate_video` wraps them for
a single video.

``legacy``
    Exactly the original script: ``max(counts, key=counts.get)`` over all five
    labels (ties go to the label listed first, i.e. towards ``weapons``), then
    an override to ``nonmilitary`` when the number of *below-threshold* frames
    exceeds half of the **total decoded frames** of the video (not of the
    sampled frames). A video with no sampled frames therefore gets the first
    label, ``weapons``.
``plurality``
    Most frequent label among sampled frames; ties that involve
    ``nonmilitary`` go to ``nonmilitary``, other ties to the label order.
    Videos without frames are ``None`` (undetermined).
``military_fraction``
    A video is military when at least ``min_fraction`` of its sampled frames
    carry a military label; it then gets its most frequent military label.
    ``min_fraction=0`` is not allowed; use a small value (e.g. ``1e-9``) for
    "any military frame".
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .taxonomy import LABELS, NONMILITARY

RULES = ("legacy", "plurality", "military_fraction")


def count_matrix(frame_labels: Iterable[Sequence[str | None]], labels: Sequence[str] = LABELS) -> np.ndarray:
    """``(n_videos, n_labels)`` counts from per-video sequences of frame labels (``None`` ignored)."""
    index = {label: i for i, label in enumerate(labels)}
    rows = []
    for seq in frame_labels:
        row = np.zeros(len(labels), dtype=np.int64)
        for label in seq:
            if label is not None:
                row[index[label]] += 1
        rows.append(row)
    return np.vstack(rows) if rows else np.zeros((0, len(labels)), dtype=np.int64)


def legacy_labels(
    counts: np.ndarray,
    labels: Sequence[str] = LABELS,
    below_threshold_counts: np.ndarray | None = None,
    total_frames: np.ndarray | None = None,
) -> np.ndarray:
    """Original rule. ``argmax`` returns the first maximum, matching ``max(dict, key=...)``."""
    counts = np.asarray(counts)
    out = np.asarray(labels, dtype=object)[counts.argmax(axis=1)]
    if below_threshold_counts is not None and total_frames is not None:
        override = np.asarray(below_threshold_counts) > 0.5 * np.asarray(total_frames)
        out = np.where(override, NONMILITARY, out)
    return out


def plurality_labels(counts: np.ndarray, labels: Sequence[str] = LABELS) -> np.ndarray:
    counts = np.asarray(counts)
    labels = list(labels)
    nm = labels.index(NONMILITARY)
    best = counts.max(axis=1, keepdims=True)
    is_max = counts == best
    out = np.asarray(labels, dtype=object)[is_max.argmax(axis=1)]
    out = np.where(is_max[:, nm], NONMILITARY, out)
    out = np.where(counts.sum(axis=1) == 0, None, out)
    return out


def military_fraction_labels(counts: np.ndarray, min_fraction: float, labels: Sequence[str] = LABELS) -> np.ndarray:
    if not 0.0 < min_fraction <= 1.0:
        raise ValueError("min_fraction must be in (0, 1]")
    counts = np.asarray(counts)
    labels = list(labels)
    nm = labels.index(NONMILITARY)
    military_idx = [i for i in range(len(labels)) if i != nm]
    total = counts.sum(axis=1)
    military = counts[:, military_idx]
    frac = np.divide(military.sum(axis=1), total, out=np.zeros(len(counts)), where=total > 0)
    top_military = np.asarray([labels[i] for i in military_idx], dtype=object)[military.argmax(axis=1)]
    out = np.where(frac >= min_fraction, top_military, NONMILITARY)
    out = np.where(total == 0, None, out)
    return out


def apply_rule(counts: np.ndarray, rule: str, labels: Sequence[str] = LABELS, **kwargs) -> np.ndarray:
    if rule == "legacy":
        return legacy_labels(counts, labels, **kwargs)
    if rule == "plurality":
        return plurality_labels(counts, labels)
    if rule == "military_fraction":
        return military_fraction_labels(counts, labels=labels, **kwargs)
    raise ValueError(f"unknown aggregation rule {rule!r}; expected one of {RULES}")


@dataclass(frozen=True)
class VideoDecision:
    category: str | None
    counts: dict[str, int]
    n_frames: int
    rule: str
    tie: bool


def aggregate_video(
    frame_labels: Sequence[str | None],
    rule: str = "plurality",
    labels: Sequence[str] = LABELS,
    **kwargs,
) -> VideoDecision:
    counts = count_matrix([frame_labels], labels)
    category = apply_rule(counts, rule, labels, **kwargs)[0]
    row = counts[0]
    tie = bool(row.sum() > 0 and (row == row.max()).sum() > 1)
    return VideoDecision(
        category=category,
        counts={label: int(c) for label, c in zip(labels, row, strict=True)},
        n_frames=int(row.sum()),
        rule=rule,
        tie=tie,
    )
