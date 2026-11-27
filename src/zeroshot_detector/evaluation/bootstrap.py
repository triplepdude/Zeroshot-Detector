"""Cluster (block) bootstrap for statistics built from additive per-cluster tables.

Frames sampled from the same video are strongly correlated (consecutive frames
usually produce the same description), so treating 1.1 M frames as
independent gives confidence intervals that are far too narrow. Resampling
whole videos with replacement accounts for that correlation.

Any statistic that is a function of a *sum* of per-cluster tables (confusion
matrices, success/total counts, ...) can be bootstrapped cheaply: a bootstrap
replicate is ``w @ tables`` with ``w ~ Multinomial(n_clusters, uniform)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    low: float
    high: float
    std_error: float
    n_boot: int
    n_clusters: int
    confidence: float

    def to_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "ci_low": self.low,
            "ci_high": self.high,
            "std_error": self.std_error,
            "n_boot": self.n_boot,
            "n_clusters": self.n_clusters,
            "confidence": self.confidence,
            "method": "cluster percentile bootstrap",
        }


def cluster_confusion_tables(
    clusters: Sequence,
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence[str],
) -> np.ndarray:
    """Per-cluster flattened confusion matrices, shape ``(n_clusters, K*K)``."""
    index = {label: i for i, label in enumerate(labels)}
    k = len(labels)
    _, cluster_idx = np.unique(np.asarray(clusters, dtype=object).astype(str), return_inverse=True)
    t = np.fromiter((index[y] for y in y_true), dtype=np.int64)
    p = np.fromiter((index[y] for y in y_pred), dtype=np.int64)
    n_clusters = int(cluster_idx.max()) + 1 if len(cluster_idx) else 0
    flat = np.bincount(cluster_idx * k * k + t * k + p, minlength=n_clusters * k * k)
    return flat.reshape(n_clusters, k * k)


def bootstrap_additive(
    tables: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    chunk: int = 50,
) -> BootstrapResult:
    """Percentile cluster bootstrap of ``statistic(tables.sum(axis=0))``.

    Parameters
    ----------
    tables:
        ``(n_clusters, d)`` array of additive per-cluster sufficient statistics.
    statistic:
        Maps a ``d``-vector (the pooled table) to a scalar. Replicates where
        the statistic is ``nan`` are dropped from the interval.
    """
    return bootstrap_additive_many(tables, {"_": statistic}, n_boot, confidence, seed, chunk)["_"]


def bootstrap_additive_many(
    tables: np.ndarray,
    statistics: Mapping[str, Callable[[np.ndarray], float]],
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    chunk: int = 50,
) -> dict[str, BootstrapResult]:
    """Like :func:`bootstrap_additive` for several statistics sharing the same replicates."""
    tables = np.asarray(tables, dtype=float)
    n_clusters = tables.shape[0]
    pooled_all = tables.sum(axis=0)
    estimates = {name: float(fn(pooled_all)) for name, fn in statistics.items()}
    nan = float("nan")
    if n_clusters < 2 or n_boot < 2:
        return {name: BootstrapResult(est, nan, nan, nan, 0, n_clusters, confidence) for name, est in estimates.items()}
    rng = np.random.default_rng(seed)
    probs = np.full(n_clusters, 1.0 / n_clusters)
    reps: dict[str, list[float]] = {name: [] for name in statistics}
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        weights = rng.multinomial(n_clusters, probs, size=size).astype(float)
        pooled = weights @ tables
        for name, fn in statistics.items():
            reps[name].extend(float(fn(row)) for row in pooled)
    alpha = (1 - confidence) / 2
    out = {}
    for name, values in reps.items():
        arr = np.asarray(values)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 2:
            out[name] = BootstrapResult(estimates[name], nan, nan, nan, n_boot, n_clusters, confidence)
            continue
        low, high = np.quantile(arr, [alpha, 1 - alpha])
        out[name] = BootstrapResult(
            estimate=estimates[name],
            low=float(low),
            high=float(high),
            std_error=float(arr.std(ddof=1)),
            n_boot=n_boot,
            n_clusters=n_clusters,
            confidence=confidence,
        )
    return out


def proportion_tables(clusters: Sequence, successes: Sequence[bool]) -> np.ndarray:
    """Per-cluster ``[successes, trials]`` tables for :func:`bootstrap_additive`."""
    _, cluster_idx = np.unique(np.asarray(clusters, dtype=object).astype(str), return_inverse=True)
    s = np.asarray(successes, dtype=float)
    n_clusters = int(cluster_idx.max()) + 1 if len(cluster_idx) else 0
    k = np.bincount(cluster_idx, weights=s, minlength=n_clusters)
    n = np.bincount(cluster_idx, minlength=n_clusters).astype(float)
    return np.column_stack([k, n])


def ratio(table: np.ndarray) -> float:
    """Statistic for :func:`proportion_tables`: pooled successes / pooled trials."""
    return float(table[0] / table[1]) if table[1] else float("nan")
