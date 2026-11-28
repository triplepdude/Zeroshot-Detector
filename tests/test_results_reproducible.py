"""The committed results/ must be reproducible from data/ (run in CI with ``-m slow``)."""

import json
import math

import pytest
from conftest import DATA, REPO

from zeroshot_detector.analysis.report import run_analysis

RESULTS = REPO / "results"
#: Bootstrap-derived fields may move slightly if a NumPy release changes its random streams.
BOOTSTRAP_KEYS = {"ci_low", "ci_high", "std_error"}


def compare(a, b, path="", key=""):
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys differ"
        for k in a:
            compare(a[k], b[k], f"{path}.{k}", k)
    elif isinstance(a, list):
        assert len(a) == len(b), f"{path}: lengths differ"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            compare(x, y, f"{path}[{i}]", key)
    elif isinstance(a, float) and isinstance(b, (int, float)):
        tol = 0.01 if key in BOOTSTRAP_KEYS else 1e-9
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=tol), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


@pytest.mark.slow
@pytest.mark.skipif(not (RESULTS / "summary.json").exists(), reason="no committed results")
def test_committed_results_are_reproducible(tmp_path):
    committed = json.loads((RESULTS / "summary.json").read_text())
    prov = committed["provenance"]
    run_analysis(DATA, tmp_path, RESULTS / "relabel", n_boot=prov["n_boot"], seed=prov["seed"], figures=False)
    fresh = json.loads((tmp_path / "summary.json").read_text())
    for section in committed:
        if section == "provenance":
            continue
        compare(committed[section], fresh[section], section)
