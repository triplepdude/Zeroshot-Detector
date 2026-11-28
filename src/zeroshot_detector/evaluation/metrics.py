"""Classification agreement metrics computed from (optionally weighted) confusion matrices.

All functions take a confusion matrix ``cm`` with rows = reference ("true")
labels and columns = compared ("predicted") labels. When neither labelling is
ground truth (e.g. pipeline vs. LLM), "precision"/"recall" should be read as
agreement conditional on one system's label, not as accuracy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def confusion_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence[str],
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """``(K, K)`` matrix; entries outside ``labels`` raise ``ValueError``."""
    index = {label: i for i, label in enumerate(labels)}
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred have different lengths")
    try:
        t = np.fromiter((index[y] for y in y_true), dtype=np.int64, count=len(y_true))
        p = np.fromiter((index[y] for y in y_pred), dtype=np.int64, count=len(y_pred))
    except KeyError as exc:  # pragma: no cover - message path
        raise ValueError(f"label {exc.args[0]!r} not in {list(labels)}") from None
    k = len(labels)
    w = None if weights is None else np.asarray(weights, dtype=float)
    flat = np.bincount(t * k + p, weights=w, minlength=k * k)
    return flat.reshape(k, k)


def cohen_kappa(cm: np.ndarray) -> float:
    """Cohen's kappa; ``nan`` when expected agreement is 1 (degenerate marginals)."""
    cm = np.asarray(cm, dtype=float)
    n = cm.sum()
    if n == 0:
        return float("nan")
    po = np.trace(cm) / n
    pe = float((cm.sum(axis=0) * cm.sum(axis=1)).sum() / n**2)
    if math.isclose(pe, 1.0):
        return float("nan")
    return float((po - pe) / (1.0 - pe))


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def metrics_from_confusion(cm: np.ndarray, labels: Sequence[str]) -> dict:
    """Accuracy, Cohen's kappa, per-class precision/recall/F1 and macro / weighted F1.

    Classes with zero support *and* zero predictions are excluded from the
    macro average (their F1 is undefined).
    """
    cm = np.asarray(cm, dtype=float)
    n = cm.sum()
    per_class = {}
    f1s, supports = [], []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        support = cm[i, :].sum()
        predicted = cm[:, i].sum()
        precision = _safe_div(tp, predicted)
        recall = _safe_div(tp, support)
        if math.isnan(precision) or math.isnan(recall) or precision + recall == 0:
            f1 = 0.0 if (support > 0 or predicted > 0) else float("nan")
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
            "predicted": float(predicted),
        }
        if not math.isnan(f1):
            f1s.append(f1)
            supports.append(support)
    macro_f1 = float(np.mean(f1s)) if f1s else float("nan")
    weighted_f1 = float(np.average(f1s, weights=supports)) if f1s and sum(supports) > 0 else float("nan")
    return {
        "n": float(n),
        "accuracy": _safe_div(np.trace(cm), n),
        "kappa": cohen_kappa(cm),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }


def wilson_interval(successes: float, n: float, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (suitable near 0 and 1)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    z = _z(confidence)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    low = 0.0 if successes <= 0 else max(0.0, centre - half)
    high = 1.0 if successes >= n else min(1.0, centre + half)
    return (low, high)


def _z(confidence: float) -> float:
    """Two-sided standard-normal quantile via the inverse error function (no SciPy needed)."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    return math.sqrt(2) * _erfinv(confidence)


def _erfinv(y: float) -> float:
    # Initial approximation (Giles 2010) refined by two Newton steps on math.erf.
    w = -math.log((1.0 - y) * (1.0 + y))
    if w < 5.0:
        w -= 2.5
        p = 2.81022636e-08
        for c in (
            3.43273939e-07,
            -3.5233877e-06,
            -4.39150654e-06,
            0.00021858087,
            -0.00125372503,
            -0.00417768164,
            0.246640727,
            1.50140941,
        ):
            p = c + p * w
    else:
        w = math.sqrt(w) - 3.0
        p = -0.000200214257
        for c in (
            0.000100950558,
            0.00134934322,
            -0.00367342844,
            0.00573950773,
            -0.0076224613,
            0.00943887047,
            1.00167406,
            2.83297682,
        ):
            p = c + p * w
    x = p * y
    for _ in range(2):
        x -= (math.erf(x) - y) / (2 / math.sqrt(math.pi) * math.exp(-x * x))
    return x
