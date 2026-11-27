"""Text-level reference: ingest annotator passes, derive readings, adjudicate (PLAN v2 §7).

Each annotator answer gives, for one string, the equipment ``categories``, the
``primary`` category and a 0–3 military ``score`` per vehicle category. From
every single pass the four reading labels are derived; the reference label of a
reading is the **majority of the per-pass labels**. Ties are broken by a
further blind vote (configuration D) if available, otherwise the item is
*unresolved* (never imputed).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..annotation import fleiss_kappa
from ..taxonomy import NONMILITARY

VEHICLES = ("aircraft", "maritime", "ground_vehicles")
ORDER = ("weapons", "aircraft", "maritime", "ground_vehicles")
READINGS = ("equipment", "type", "lenient", "strict")
THRESHOLDS = {"equipment": 0, "type": 1, "lenient": 2, "strict": 3}
#: Configurations whose passes form the reference (A1 = single-item check, not a vote).
VOTING_CONFIGS = ("A", "B", "C", "D", "E", "TB")
UNRESOLVED = "unresolved"


def item_id(text: str) -> str:
    """Stable, non-informative id of a string (first 10 hex digits of its SHA-1)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


# -- per-pass readings ---------------------------------------------------------------
def pass_readings(categories: Sequence[str], primary: str, scores: Mapping[str, int]) -> dict[str, str]:
    """Reading labels implied by one pass.

    A category qualifies for a reading when it is ``weapons`` or a vehicle
    category whose score reaches the reading's threshold (equipment: any). The
    label is the pass's ``primary`` if it qualifies; otherwise the qualifying
    category ranked first by (weapons, highest score, label order).
    """
    out = {}
    cats = [c for c in ORDER if c in set(categories)]
    for reading, threshold in THRESHOLDS.items():
        q = [c for c in cats if c == "weapons" or int(scores.get(c, 0)) >= threshold]
        if not q:
            out[reading] = NONMILITARY
        elif primary in q:
            out[reading] = primary
        else:
            out[reading] = sorted(q, key=lambda c: (c != "weapons", -int(scores.get(c, 0)), ORDER.index(c)))[0]
    return out


# -- ingestion ---------------------------------------------------------------------
def load_workflow_results(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Flatten annotation-workflow outputs into one row per (item, configuration answer)."""
    rows = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        results = payload.get("result", payload)
        for batch in results:
            if not batch:
                continue
            for pos, a in enumerate(batch.get("answers", [])):
                scores = {k: int(v) for k, v in (a.get("scores") or {}).items() if k in VEHICLES}
                rows.append(
                    {
                        "item_id": a["item_id"],
                        "raw": a["raw"],
                        "config": batch["config"] if not batch["batch_id"].startswith("TB") else "TB",
                        "batch_id": batch["batch_id"],
                        "position": pos,
                        "fallback": bool(batch.get("fallback")),
                        "categories": sorted(set(a["categories"])),
                        "primary": a["primary"],
                        "scores": scores,
                        "depiction": a["depiction"],
                        "confidence": int(a["confidence"]),
                        **{f"pass_{r}": lab for r, lab in pass_readings(a["categories"], a["primary"], scores).items()},
                    }
                )
    return pd.DataFrame(rows)


# -- adjudication --------------------------------------------------------------------
def _majority(values: Sequence[str]) -> str:
    counts = Counter(values).most_common()
    if not counts:
        return UNRESOLVED
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return UNRESOLVED
    return counts[0][0]


def adjudicate(passes: pd.DataFrame) -> pd.DataFrame:
    """One row per item: reference label per reading, vote shares, median scores, ≥2-pass categories."""
    voting = passes[passes["config"].isin(VOTING_CONFIGS)]
    rows = []
    for iid, g in voting.groupby("item_id"):
        row = {"item_id": iid, "raw": g["raw"].iloc[0], "n_votes": len(g)}
        for r in READINGS:
            labels = list(g[f"pass_{r}"])
            row[r] = _majority(labels)
            row[f"{r}_agreement"] = Counter(labels).most_common(1)[0][1] / len(labels)
        cat_votes = Counter(c for cs in g["categories"] for c in cs)
        row["categories_2plus"] = sorted(c for c, n in cat_votes.items() if n >= 2)
        med = {}
        for v in VEHICLES:
            vals = [s[v] for s in g["scores"] if v in s]
            if vals:
                med[v] = float(np.median(vals))
        row["median_scores"] = med
        row["depiction"] = _majority(list(g["depiction"]))
        rows.append(row)
    return pd.DataFrame(rows)


def tied_items(reference: pd.DataFrame, readings: Sequence[str] = READINGS) -> list[str]:
    mask = np.zeros(len(reference), dtype=bool)
    for r in readings:
        mask |= (reference[r] == UNRESOLVED).to_numpy()
    return reference.loc[mask, "item_id"].tolist()


def resampler(passes: pd.DataFrame, reading: str, text_of: Mapping[str, str]):
    """Annotation-aware bootstrap: resample each item's votes with replacement and re-adjudicate."""
    voting = passes[passes["config"].isin(VOTING_CONFIGS)]
    votes = {iid: list(g[f"pass_{reading}"]) for iid, g in voting.groupby("item_id")}

    def draw(rng: np.random.Generator) -> dict[str, str]:
        out = {}
        for iid, v in votes.items():
            pick = [v[i] for i in rng.integers(0, len(v), size=len(v))]
            out[text_of[iid]] = _majority(pick)
        return out

    return draw


def agreement_stats(passes: pd.DataFrame, items: Iterable[str], configs: Sequence[str] = ("A", "B", "C")) -> dict:
    """Fleiss' κ across configurations for each reading (items with one answer per configuration)."""
    p = passes[passes["config"].isin(configs) & passes["item_id"].isin(set(items))]
    complete = p.groupby("item_id")["config"].nunique()
    complete = complete[complete == len(configs)].index
    p = p[p["item_id"].isin(complete)]
    out = {"items": len(complete)}
    cats = (*ORDER, NONMILITARY)
    for r in READINGS:
        table = p.groupby("item_id")[f"pass_{r}"].apply(lambda s: [int((s == c).sum()) for c in cats])
        arr = np.array(table.tolist()) if len(table) else np.zeros((0, len(cats)))
        out[r] = {
            "fleiss_kappa": fleiss_kappa(arr) if len(arr) else float("nan"),
            "unanimous_share": float((arr.max(axis=1) == len(configs)).mean()) if len(arr) else float("nan"),
        }
    return out
