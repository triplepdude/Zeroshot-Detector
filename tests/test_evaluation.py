import math

import numpy as np
import pytest
from conftest import has_module

from zeroshot_detector.annotation import fleiss_kappa
from zeroshot_detector.evaluation.bootstrap import (
    bootstrap_additive,
    bootstrap_additive_many,
    cluster_confusion_tables,
    proportion_tables,
    ratio,
)
from zeroshot_detector.evaluation.metrics import (
    _z,
    cohen_kappa,
    confusion_matrix,
    metrics_from_confusion,
    wilson_interval,
)

LABELS = ["a", "b", "c", "nonmilitary"]


@pytest.fixture
def labels_pair():
    rng = np.random.default_rng(3)
    t = rng.choice(LABELS, 400, p=[0.1, 0.2, 0.3, 0.4])
    p = np.where(rng.random(400) < 0.75, t, rng.choice(LABELS, 400))
    return t, p


@pytest.mark.skipif(not has_module("sklearn"), reason="scikit-learn not installed")
def test_metrics_match_sklearn(labels_pair):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_recall_fscore_support

    t, p = labels_pair
    m = metrics_from_confusion(confusion_matrix(t, p, LABELS), LABELS)
    assert m["accuracy"] == pytest.approx(accuracy_score(t, p))
    assert m["kappa"] == pytest.approx(cohen_kappa_score(t, p))
    assert m["macro_f1"] == pytest.approx(f1_score(t, p, average="macro"))
    assert m["weighted_f1"] == pytest.approx(f1_score(t, p, average="weighted"))
    prec, rec, _, _ = precision_recall_fscore_support(t, p, labels=LABELS)
    for i, label in enumerate(LABELS):
        assert m["per_class"][label]["precision"] == pytest.approx(prec[i])
        assert m["per_class"][label]["recall"] == pytest.approx(rec[i])


@pytest.mark.skipif(not has_module("sklearn"), reason="scikit-learn not installed")
def test_weighted_confusion_matches_sklearn(labels_pair):
    from sklearn.metrics import cohen_kappa_score

    t, p = labels_pair
    w = np.random.default_rng(0).uniform(0.5, 3, len(t))
    cm = confusion_matrix(t, p, LABELS, weights=w)
    assert cm.sum() == pytest.approx(w.sum())
    assert cohen_kappa(cm) == pytest.approx(cohen_kappa_score(t, p, sample_weight=w))


def test_kappa_edge_cases():
    assert math.isnan(cohen_kappa(np.zeros((2, 2))))
    assert math.isnan(cohen_kappa(np.array([[5, 0], [0, 0]])))  # one class only
    assert cohen_kappa(np.array([[5, 0], [0, 5]])) == pytest.approx(1.0)


def test_wilson_and_z():
    assert _z(0.95) == pytest.approx(1.959964, abs=1e-5)
    assert _z(0.99) == pytest.approx(2.575829, abs=1e-5)
    low, high = wilson_interval(0, 20)
    assert low == 0.0 and 0.1 < high < 0.2
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3) and high == pytest.approx(0.5962, abs=1e-3)


def test_fleiss_kappa_reference_example():
    # Fleiss (1971) / Wikipedia example: kappa = 0.210
    table = np.array(
        [
            [0, 0, 0, 0, 14],
            [0, 2, 6, 4, 2],
            [0, 0, 3, 5, 6],
            [0, 3, 9, 2, 0],
            [2, 2, 8, 1, 1],
            [7, 7, 0, 0, 0],
            [3, 2, 6, 3, 0],
            [2, 5, 3, 2, 2],
            [6, 5, 2, 1, 0],
            [0, 2, 2, 3, 7],
        ]
    )
    assert fleiss_kappa(table) == pytest.approx(0.210, abs=1e-3)


class TestBootstrap:
    def test_cluster_tables_sum_to_confusion(self, labels_pair):
        t, p = labels_pair
        clusters = np.arange(len(t)) // 7
        tables = cluster_confusion_tables(clusters, t, p, LABELS)
        assert tables.shape == (len(set(clusters)), 16)
        np.testing.assert_array_equal(tables.sum(axis=0).reshape(4, 4), confusion_matrix(t, p, LABELS))

    def test_deterministic_and_covers_estimate(self):
        rng = np.random.default_rng(1)
        clusters = rng.integers(0, 200, 2000)
        success = rng.random(2000) < 0.3
        tables = proportion_tables(clusters, success)
        r1 = bootstrap_additive(tables, ratio, n_boot=300, seed=7)
        r2 = bootstrap_additive(tables, ratio, n_boot=300, seed=7)
        assert r1 == r2
        assert r1.low < r1.estimate < r1.high
        assert r1.estimate == pytest.approx(success.mean())

    def test_clustering_widens_interval(self):
        # perfectly correlated clusters: 50 clusters of 40 identical outcomes
        rng = np.random.default_rng(2)
        cluster_outcome = rng.random(50) < 0.5
        clusters = np.repeat(np.arange(50), 40)
        success = np.repeat(cluster_outcome, 40)
        clustered = bootstrap_additive(proportion_tables(clusters, success), ratio, n_boot=500, seed=0)
        naive = bootstrap_additive(proportion_tables(np.arange(2000), success), ratio, n_boot=500, seed=0)
        assert (clustered.high - clustered.low) > 3 * (naive.high - naive.low)

    def test_many_shares_replicates(self):
        tables = proportion_tables(np.arange(100) % 10, np.arange(100) % 3 == 0)
        out = bootstrap_additive_many(tables, {"p": ratio, "q": lambda t: 1 - ratio(t)}, n_boot=200, seed=0)
        assert out["p"].low == pytest.approx(1 - out["q"].high)

    def test_single_cluster(self):
        r = bootstrap_additive(np.array([[3.0, 10.0]]), ratio)
        assert r.estimate == pytest.approx(0.3) and math.isnan(r.low)
