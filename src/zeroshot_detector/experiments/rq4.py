"""RQ4: alternative string classifiers evaluated against the reference (PLAN v2 §8).

* zero-shot grid: every configuration × keyword rule {off, on} × threshold;
* nested selection of the "best" configuration over 5 folds of strings;
* supervised bag-of-labels logistic regression: (a) trained on the reference
  strings with grouped cross-validation; (b) weakly supervised on the original
  LLM labels of all strings, cross-fitted.

All metrics are design-weighted over the sample records (``design.metrics``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..taxonomy import LABELS, NONMILITARY
from .design import _weighted_cm, metrics
from .grid import BagOfLabels, ZeroShotConfig, decide_labels, keyword_fires

THRESHOLDS: tuple[float | None, ...] = (None, *[round(x, 2) for x in np.arange(0.05, 0.951, 0.05)])
C_GRID = (0.01, 0.1, 1.0, 10.0)
N_FOLDS = 5


def fold_of(text: str, n_folds: int = N_FOLDS) -> int:
    return int(hashlib.sha1(text.encode()).hexdigest(), 16) % n_folds


def record_metrics(records: pd.DataFrame, reference: Mapping[str, str], labels: Mapping[str, str]) -> dict:
    texts = records["text"].to_numpy()
    gold = np.array([reference.get(t, "unresolved") for t in texts], dtype=object)
    pred = np.array([labels[t] for t in texts], dtype=object)
    return metrics(_weighted_cm(gold, pred, records["weight"].to_numpy(dtype=float)))


def config_labels(
    config: ZeroShotConfig,
    scores: pd.DataFrame,
    segmentation: Mapping[str, Sequence[str]],
    keyword: bool,
    threshold: float | None,
) -> dict[str, str]:
    texts = scores["text"].tolist()
    kw = keyword_fires(texts, segmentation, config.input) if keyword else None
    return decide_labels(scores, threshold, kw).to_dict()


def evaluate_grid(
    records: pd.DataFrame,
    reference: Mapping[str, str],
    scores: Mapping[ZeroShotConfig, pd.DataFrame],
    segmentation: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    rows = []
    for config, sc in scores.items():
        for keyword in (False, True):
            for th in THRESHOLDS:
                m = record_metrics(records, reference, config_labels(config, sc, segmentation, keyword, th))
                rows.append(
                    {
                        "config": config.name,
                        **config.__dict__,
                        "keyword": keyword,
                        "threshold": "argmax" if th is None else th,
                        **{
                            k: m.get(k)
                            for k in ("agreement", "kappa", "macro_f1", "binary_agreement", "military_share_system")
                        },
                    }
                )
    return pd.DataFrame(rows)


def nested_selection(
    records: pd.DataFrame,
    reference: Mapping[str, str],
    scores: Mapping[ZeroShotConfig, pd.DataFrame],
    segmentation: Mapping[str, Sequence[str]],
) -> dict:
    """Select (configuration, keyword, threshold) on 4 folds of strings, predict the 5th; pool."""
    folds = records["text"].map(fold_of).to_numpy()
    candidates = {
        (c.name, kw, th): config_labels(c, sc, segmentation, kw, th)
        for c, sc in scores.items()
        for kw in (False, True)
        for th in THRESHOLDS
    }
    pooled: dict[str, str] = {}
    chosen = []
    for k in range(N_FOLDS):
        train = records[folds != k]
        best = max(candidates, key=lambda key: record_metrics(train, reference, candidates[key]).get("agreement", -1))
        chosen.append(
            {"fold": k, "config": best[0], "keyword": best[1], "threshold": "argmax" if best[2] is None else best[2]}
        )
        for t in records.loc[folds == k, "text"]:
            pooled[t] = candidates[best][t]
    return {"chosen": chosen, "out_of_fold": record_metrics(records, reference, pooled), "labels": pooled}


def _fit_predict(x_train, y_train, x_test, c: float):
    from sklearn.linear_model import LogisticRegression

    if len(set(y_train)) < 2:
        return np.array([y_train[0]] * len(x_test))
    model = LogisticRegression(C=c, max_iter=5000)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def supervised_reference(
    records: pd.DataFrame, reference: Mapping[str, str], objects: Mapping[str, Sequence[str]]
) -> dict:
    """Grouped (by string) 5-fold CV with inner grouped CV for C; unweighted fit, weighted evaluation."""
    texts = sorted({t for t in records["text"] if reference.get(t, "unresolved") in LABELS})
    y = np.array([reference[t] for t in texts], dtype=object)
    feats = BagOfLabels().fit([objects[t] for t in texts])
    x = feats.transform([objects[t] for t in texts])
    folds = np.array([fold_of(t) for t in texts])
    pred = {}
    chosen_c = []
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        # inner grouped CV over the remaining folds
        inner_scores = {}
        for c in C_GRID:
            hits = []
            for j in set(folds[tr]) - {k}:
                itr, ite = tr & (folds != j), tr & (folds == j)
                p = _fit_predict(x[itr], y[itr], x[ite], c)
                hits.extend(p == y[ite])
            inner_scores[c] = float(np.mean(hits)) if hits else 0.0
        c_best = max(inner_scores, key=inner_scores.get)
        chosen_c.append(c_best)
        for t, p in zip(np.array(texts)[te], _fit_predict(x[tr], y[tr], x[te], c_best), strict=True):
            pred[t] = p
    labels = {t: pred.get(t, NONMILITARY) for t in records["text"]}
    return {
        "chosen_C": chosen_c,
        "out_of_fold": record_metrics(records, reference, labels),
        "labels": labels,
        "n_training_strings": len(texts),
    }


def weak_supervision(strings: pd.DataFrame, records: pd.DataFrame, reference: Mapping[str, str]) -> dict:
    """Cross-fitted (5 folds of all strings) logistic regression on the original LLM labels."""
    texts = strings["text"].tolist()
    objs = [o.split("; ") for o in strings["objects"]]
    y = strings["llm_label"].to_numpy(dtype=object)
    folds = np.array([fold_of(t) for t in texts])
    pred = np.empty(len(texts), dtype=object)
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        feats = BagOfLabels().fit([objs[i] for i in np.flatnonzero(tr)], min_count=2)
        xtr = feats.transform([objs[i] for i in np.flatnonzero(tr)])
        xte = feats.transform([objs[i] for i in np.flatnonzero(te)])
        pred[te] = _fit_predict(xtr, y[tr], xte, 1.0)
    labels = dict(zip(texts, pred, strict=True))
    return {"out_of_fold": record_metrics(records, reference, labels), "labels": labels}
