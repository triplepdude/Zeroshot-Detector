"""Design-weighted evaluation of string classifiers against the text-level reference.

The sample key (one row per sampled frame) carries the stratum, ``N_h``,
``n_h`` and the design weight ``N_h / n_h``; system labels and reference
labels are attached through the frame's string. Estimates are Horvitz–Thompson
(weighted) with stratified-bootstrap intervals (``annotation.estimate``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..annotation import estimate
from ..evaluation.metrics import cohen_kappa, confusion_matrix, metrics_from_confusion
from ..taxonomy import LABELS, NONMILITARY
from .reference import READINGS


def attach(sample: pd.DataFrame, by_text: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Add one column per system, mapping each frame's string to the system's label."""
    out = sample.copy()
    for name, labels in by_text.items():
        out[name] = out["florence_text"].map(labels)
        missing = out[name].isna().sum()
        if missing:
            raise ValueError(f"system {name!r} has no label for {missing} sampled frames")
    return out


def gold_table(sample: pd.DataFrame, reference: pd.DataFrame, reading: str) -> pd.DataFrame:
    """``sample_id, gold`` for one reading (``reference``: item_id + reading columns)."""
    if reading not in READINGS:
        raise ValueError(f"unknown reading {reading!r}")
    lab = reference.set_index("item_id")[reading]
    return pd.DataFrame({"sample_id": sample["sample_id"], "gold": sample["item_id"].map(lab)})


def lenient_gold(sample: pd.DataFrame, reference: pd.DataFrame, system: str) -> pd.DataFrame:
    """Gold for *lenient* matching under the equipment reading.

    A system's military label counts as correct when that category is among the
    categories any annotator pass recorded; ``nonmilitary`` counts as correct
    when the reference primary is ``none``. Implemented by replacing the gold
    label with the system label where the system label is acceptable.
    """
    ref = reference.set_index("item_id")
    cats = sample["item_id"].map(ref["categories_any"])
    equipment = sample["item_id"].map(ref["equipment"])
    sys_lab = sample[system]
    ok = [
        (s == NONMILITARY and e == NONMILITARY) or (s != NONMILITARY and s in c)
        for s, e, c in zip(sys_lab, equipment, cats, strict=True)
    ]
    gold = np.where(ok, sys_lab, equipment)
    return pd.DataFrame({"sample_id": sample["sample_id"], "gold": gold})


def evaluate_systems(
    sample: pd.DataFrame,
    reference: pd.DataFrame,
    systems: Sequence[str],
    n_boot: int,
    seed: int,
) -> dict:
    """Per reading: ``annotation.estimate`` for every system (+ lenient matching)."""
    out = {}
    for reading in READINGS:
        out[reading] = estimate(
            sample, gold_table(sample, reference, reading), systems=systems, n_boot=n_boot, seed=seed
        )
    out["equipment_lenient"] = {
        system: estimate(sample, lenient_gold(sample, reference, system), systems=[system], n_boot=n_boot, seed=seed)[
            "systems"
        ][system]
        for system in systems
    }
    return out


def point_metrics(sample: pd.DataFrame, gold: pd.Series, predicted: pd.Series) -> dict:
    """Weighted accuracy / κ / macro-F1 without intervals (for sweeps)."""
    cm = confusion_matrix(gold, predicted, LABELS, weights=sample["weight"].to_numpy())
    m = metrics_from_confusion(cm, LABELS)
    return {
        "accuracy": m["accuracy"],
        "kappa": cohen_kappa(cm),
        "macro_f1": m["macro_f1"],
        "military_share": float(cm[:, [i for i, lab in enumerate(LABELS) if lab != NONMILITARY]].sum() / cm.sum()),
    }
