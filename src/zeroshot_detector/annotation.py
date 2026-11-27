"""Gold-standard evaluation protocol: stratified sampling, blind annotation, weighted estimation.

The archived human labels cannot estimate accuracy (they exist only for one
kind of disagreement, were made from text, and annotators saw both system
labels). This module implements a design that can:

1. :func:`stratified_sample` draws a simple random sample of frames within
   each stratum (by default: pipeline label × LLM label), so rare cells such as
   ``pipeline=weapons, llm=nonmilitary`` are represented, and records the
   inclusion probability of every sampled frame.
2. The **annotation sheet** contains only ``sample_id, video, frame_number`` —
   annotators label the *image* without seeing either system's label (or, by
   default, the Florence-2 text). System labels are kept in a separate key file.
3. :func:`inter_annotator_agreement` reports pairwise Cohen's κ and Fleiss' κ;
   :func:`adjudicate` forms the gold label by majority vote.
4. :func:`estimate` computes design-weighted (Horvitz–Thompson) accuracy,
   per-class precision / recall / F1, Cohen's κ and the estimated true
   category prevalence for each system, with stratified-bootstrap intervals.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from .evaluation.metrics import cohen_kappa, confusion_matrix, metrics_from_confusion
from .taxonomy import LABELS

DEFAULT_STRATA = ("pipeline_label", "llm_label")
#: Extra answers annotators may give; excluded from the gold standard.
NON_LABELS = ("unsure", "unviewable")


def stratified_sample(
    frames: pd.DataFrame,
    n_per_stratum: int = 50,
    strata: Sequence[str] = DEFAULT_STRATA,
    seed: int = 0,
    exclude_duplicate_files: bool = True,
) -> pd.DataFrame:
    """Simple random sample of frames within each stratum.

    Returns the *key* table: one row per sampled frame with its stratum,
    stratum size ``N_h``, sample size ``n_h`` and design weight ``N_h / n_h``.
    With ``exclude_duplicate_files`` the population keeps one file per post.
    """
    pop = frames
    if exclude_duplicate_files and "post_id" in frames:
        first = frames.drop_duplicates("post_id")["video"]
        pop = frames[frames["video"].isin(first)]
    rng = np.random.default_rng(seed)
    parts = []
    for key, group in pop.groupby(list(strata), sort=True):
        n_h = min(n_per_stratum, len(group))
        idx = rng.choice(len(group), size=n_h, replace=False)
        chosen = group.iloc[np.sort(idx)].copy()
        chosen["stratum"] = "|".join(key if isinstance(key, tuple) else (key,))
        chosen["N_h"] = len(group)
        chosen["n_h"] = n_h
        chosen["weight"] = len(group) / n_h
        parts.append(chosen)
    sample = pd.concat(parts, ignore_index=True)
    # random (but seeded), non-informative ids: the id reveals nothing about the stratum
    ids: list[str] = []
    seen: set[str] = set()
    while len(ids) < len(sample):
        token = f"{int(rng.integers(0, 2**40)):010x}"
        if token not in seen:
            seen.add(token)
            ids.append(token)
    sample.insert(0, "sample_id", ids)
    sample["population_frames"] = len(pop)
    return sample


def annotation_sheet(sample: pd.DataFrame, seed: int = 0, include_text: bool = False) -> pd.DataFrame:
    """Blind sheet for annotators: shuffled, without strata or system labels."""
    cols = ["sample_id", "video", "frame_number"] + (["florence_text"] if include_text else [])
    return sample[cols].sample(frac=1.0, random_state=seed).reset_index(drop=True)


# -- agreement between annotators --------------------------------------------
def fleiss_kappa(table: np.ndarray) -> float:
    """Fleiss' κ from an ``(items, categories)`` count table with equal raters per item."""
    table = np.asarray(table, dtype=float)
    n_raters = table.sum(axis=1)
    if len(table) == 0 or not np.allclose(n_raters, n_raters[0]) or n_raters[0] < 2:
        return float("nan")
    m = n_raters[0]
    p_j = table.sum(axis=0) / table.sum()
    p_i = ((table**2).sum(axis=1) - m) / (m * (m - 1))
    p_bar, p_e = p_i.mean(), (p_j**2).sum()
    return float((p_bar - p_e) / (1 - p_e)) if p_e < 1 else float("nan")


def inter_annotator_agreement(annotations: pd.DataFrame, labels: Sequence[str] = LABELS) -> dict:
    """``annotations``: columns ``sample_id, annotator, label`` (non-labels are dropped)."""
    a = annotations[annotations["label"].isin(labels)]
    wide = a.pivot_table(index="sample_id", columns="annotator", values="label", aggfunc="first")
    pairwise = {}
    for r1, r2 in combinations(wide.columns, 2):
        both = wide[[r1, r2]].dropna()
        if len(both):
            cm = confusion_matrix(both[r1], both[r2], labels)
            pairwise[f"{r1}|{r2}"] = {
                "n": len(both),
                "agreement": float(np.trace(cm) / cm.sum()),
                "cohen_kappa": cohen_kappa(cm),
            }
    complete = wide.dropna()
    counts = np.array([[sum(row == label) for label in labels] for row in complete.to_numpy()])
    return {
        "annotators": list(wide.columns),
        "items_with_all_annotators": len(complete),
        "pairwise": pairwise,
        "fleiss_kappa": fleiss_kappa(counts) if len(complete) else float("nan"),
    }


def adjudicate(annotations: pd.DataFrame, labels: Sequence[str] = LABELS) -> pd.DataFrame:
    """Majority label per item; ties and items without a valid label are ``unresolved``."""
    rows = []
    for sid, group in annotations.groupby("sample_id"):
        votes = Counter(label for label in group["label"] if label in labels)
        if not votes:
            rows.append({"sample_id": sid, "gold": "unresolved", "n_votes": 0})
            continue
        (top, n), *rest = votes.most_common()
        tie = bool(rest) and rest[0][1] == n
        rows.append({"sample_id": sid, "gold": "unresolved" if tie else top, "n_votes": int(sum(votes.values()))})
    return pd.DataFrame(rows)


# -- design-weighted estimation -------------------------------------------------
def _weighted_metrics(df: pd.DataFrame, system: str, labels: Sequence[str]) -> dict:
    cm = confusion_matrix(df["gold"], df[system], labels, weights=df["weight"].to_numpy())
    m = metrics_from_confusion(cm, labels)
    prevalence = cm.sum(axis=1) / cm.sum()
    return {
        "metrics": m,
        "confusion_weighted": cm.tolist(),
        "prevalence": dict(zip(labels, map(float, prevalence), strict=True)),
    }


def estimate(
    sample: pd.DataFrame,
    gold: pd.DataFrame,
    systems: Sequence[str] = ("pipeline_label", "llm_label"),
    labels: Sequence[str] = LABELS,
    n_boot: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict:
    """Design-weighted accuracy etc. of each system against adjudicated gold labels.

    Items whose gold label is ``unresolved`` are dropped (reported as a count);
    if that removes many items the estimates may be biased.
    """
    df = sample.merge(gold[["sample_id", "gold"]], on="sample_id", how="inner")
    unresolved = int((df["gold"] == "unresolved").sum())
    df = df[df["gold"].isin(labels)].reset_index(drop=True)
    out: dict = {"n_annotated": len(df), "n_unresolved": unresolved, "systems": {}}
    rng = np.random.default_rng(seed)
    strata = [g.index.to_numpy() for _, g in df.groupby("stratum")]
    census = [len(idx) >= int(df.loc[idx[0], "N_h"]) for idx in strata]
    alpha = (1 - confidence) / 2
    for system in systems:
        point = _weighted_metrics(df, system, labels)
        reps = {"accuracy": [], "kappa": [], "macro_f1": []}
        per_class = {label: {"precision": [], "recall": []} for label in labels}
        prev = {label: [] for label in labels}
        for _ in range(n_boot):
            idx = np.concatenate(
                [
                    ix if full else rng.choice(ix, size=len(ix), replace=True)
                    for ix, full in zip(strata, census, strict=True)
                ]
            )
            r = _weighted_metrics(df.loc[idx], system, labels)
            for k in reps:
                reps[k].append(r["metrics"][k])
            for label in labels:
                per_class[label]["precision"].append(r["metrics"]["per_class"][label]["precision"])
                per_class[label]["recall"].append(r["metrics"]["per_class"][label]["recall"])
                prev[label].append(r["prevalence"][label])

        def interval(values: list[float]) -> list[float]:
            arr = np.asarray(values, dtype=float)
            arr = arr[~np.isnan(arr)]
            return [float(x) for x in np.quantile(arr, [alpha, 1 - alpha])] if len(arr) else [float("nan")] * 2

        out["systems"][system] = {
            **point,
            "ci": {k: interval(v) for k, v in reps.items()},
            "per_class_ci": {label: {k: interval(v) for k, v in d.items()} for label, d in per_class.items()},
            "prevalence_ci": {label: interval(v) for label, v in prev.items()},
        }
    out["method"] = "Horvitz-Thompson weights N_h/n_h; stratified bootstrap (census strata held fixed)"
    return out
