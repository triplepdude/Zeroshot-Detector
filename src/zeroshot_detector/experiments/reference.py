"""Text-level reference standard: sampling, annotation batches, adjudication, readings.

See ``docs/experiments/PLAN.md``. The sampled unit is a frame (stratified by
pipeline × LLM label); the labelled unit is the frame's Florence-2 string.
Every string is labelled in several independent annotator passes; the
adjudicated label is mapped back to all sampled frames carrying that string.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..annotation import fleiss_kappa, stratified_sample
from ..taxonomy import MILITARY_LABELS, NONMILITARY

PRIMARY_VALUES = (*MILITARY_LABELS, "none")
MILITARY_VALUES = ("yes", "ambiguous", "no")
READINGS = ("equipment", "lenient", "strict")


def item_id(text: str) -> str:
    """Stable, non-informative id of a string (first 10 hex digits of its SHA-1)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


# -- sampling ------------------------------------------------------------------
def draw_frame_sample(frames: pd.DataFrame, n_per_stratum: int, n_overrides: dict[str, int], seed: int) -> pd.DataFrame:
    """Stratified frame sample (pipeline × LLM label) with design weights; adds ``item_id``."""
    sample = stratified_sample(frames, n_per_stratum=n_per_stratum, seed=seed, n_overrides=n_overrides)
    sample["item_id"] = sample["florence_text"].map(item_id)
    return sample


def annotation_items(texts: Iterable[str], segmentation: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    """One row per distinct string: ``item_id, raw, objects`` (objects joined with ``"; "``)."""
    rows = []
    for text in dict.fromkeys(texts):
        parts = segmentation.get(text, (text,))
        rows.append({"item_id": item_id(text), "raw": text, "objects": "; ".join(parts)})
    items = pd.DataFrame(rows)
    if items["item_id"].duplicated().any():  # pragma: no cover - 40-bit collision
        raise ValueError("item id collision")
    return items


def make_batches(item_ids: Sequence[str], n_passes: int, batch_size: int, seed: int) -> list[dict]:
    """Independent random order per pass, cut into batches of ``batch_size``."""
    batches = []
    for p in range(1, n_passes + 1):
        rng = np.random.default_rng(seed + 1000 * p)
        order = list(np.asarray(item_ids, dtype=object)[rng.permutation(len(item_ids))])
        for b, start in enumerate(range(0, len(order), batch_size)):
            batches.append({"pass": p, "batch": b, "item_ids": order[start : start + batch_size]})
    return batches


# -- annotation results ----------------------------------------------------------
def normalise_answer(answer: dict) -> dict:
    """Validate and canonicalise one annotator answer."""
    primary = str(answer["primary"]).strip().lower()
    military = str(answer["military"]).strip().lower()
    cats = sorted({str(c).strip().lower() for c in answer.get("categories", [])})
    if primary not in PRIMARY_VALUES:
        raise ValueError(f"invalid primary {primary!r}")
    if military not in MILITARY_VALUES:
        raise ValueError(f"invalid military {military!r}")
    bad = set(cats) - set(MILITARY_LABELS)
    if bad:
        raise ValueError(f"invalid categories {sorted(bad)}")
    if primary != "none" and primary not in cats:
        cats = sorted({*cats, primary})
    confidence = int(answer.get("confidence", 0) or 0)
    return {
        "item_id": str(answer["item_id"]),
        "categories": cats,
        "primary": primary,
        "military": military,
        "confidence": confidence,
    }


def load_passes(path: str | Path) -> pd.DataFrame:
    """Read the JSONL of raw annotator answers (one object per item and pass)."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def _majority(values: Sequence[str]) -> tuple[str | None, int]:
    counts = Counter(values).most_common()
    if not counts:
        return None, 0
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None, counts[0][1]
    return counts[0][0], counts[0][1]


def adjudicate(passes: pd.DataFrame, adjudications: pd.DataFrame | None = None) -> pd.DataFrame:
    """Majority per item for ``primary`` and ``military``; ties resolved by ``adjudications``.

    ``passes``: columns ``item_id, pass, primary, military, categories``.
    ``adjudications``: optional ``item_id, primary, military`` used only where
    the majority is tied. Returns one row per item with the final labels and
    the vote shares.
    """
    adj = adjudications.set_index("item_id") if adjudications is not None and len(adjudications) else None
    rows = []
    for iid, g in passes.groupby("item_id"):
        prim, n_prim = _majority(list(g["primary"]))
        mil, n_mil = _majority(list(g["military"]))
        tie_prim, tie_mil = prim is None, mil is None
        if adj is not None and iid in adj.index:
            if tie_prim:
                prim = adj.loc[iid, "primary"]
            if tie_mil:
                mil = adj.loc[iid, "military"]
        cats = sorted({c for cs in g["categories"] for c in cs})
        rows.append(
            {
                "item_id": iid,
                "primary": prim,
                "military": mil,
                "primary_agreement": n_prim / len(g),
                "military_agreement": n_mil / len(g),
                "primary_tied": tie_prim,
                "military_tied": tie_mil,
                "categories_any": cats,
                "n_passes": len(g),
            }
        )
    return pd.DataFrame(rows)


def agreement(passes: pd.DataFrame) -> dict:
    """Fleiss' κ across passes for ``primary`` and ``military`` (items with all passes)."""
    out = {}
    n_passes = passes.groupby("item_id")["pass"].nunique()
    complete = n_passes[n_passes == n_passes.max()].index
    p = passes[passes["item_id"].isin(complete)]
    for field, values in (("primary", PRIMARY_VALUES), ("military", MILITARY_VALUES)):
        table = p.groupby("item_id")[field].apply(lambda s, v=values: [int((s == x).sum()) for x in v])
        arr = np.array(table.tolist())
        out[field] = {
            "fleiss_kappa": fleiss_kappa(arr),
            "items": len(arr),
            "unanimous_share": float((arr.max(axis=1) == arr.sum(axis=1)).mean()) if len(arr) else float("nan"),
        }
    return out


def readings(reference: pd.DataFrame) -> pd.DataFrame:
    """The three pre-registered reference readings per item."""
    prim = reference["primary"].fillna("none")
    mil = reference["military"].fillna("no")
    equipment = prim.where(prim != "none", NONMILITARY)
    lenient = equipment.where(mil.isin(["yes", "ambiguous"]), NONMILITARY)
    strict = equipment.where(mil == "yes", NONMILITARY)
    return pd.DataFrame({"item_id": reference["item_id"], "equipment": equipment, "lenient": lenient, "strict": strict})
