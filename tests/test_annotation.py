import numpy as np
import pandas as pd
import pytest

from zeroshot_detector.annotation import (
    adjudicate,
    annotation_sheet,
    estimate,
    inter_annotator_agreement,
    stratified_sample,
)
from zeroshot_detector.taxonomy import LABELS


def population(seed=0):
    """Large stratum where the system is accurate, small strata where it is not."""
    rng = np.random.default_rng(seed)
    rows = []
    sizes = {("nonmilitary", "nonmilitary"): 5000, ("weapons", "nonmilitary"): 300, ("weapons", "weapons"): 400}
    acc = {("nonmilitary", "nonmilitary"): 0.98, ("weapons", "nonmilitary"): 0.4, ("weapons", "weapons"): 0.9}
    i = 0
    for (pipe, llm), n in sizes.items():
        for _ in range(n):
            correct = rng.random() < acc[(pipe, llm)]
            gold = pipe if correct else ("nonmilitary" if pipe == "weapons" else "weapons")
            rows.append(
                {
                    "video": f"{i % 997}.mp4",
                    "post_id": str(i % 997),
                    "frame_number": 30 * i,
                    "pipeline_label": pipe,
                    "llm_label": llm,
                    "gold_true": gold,
                }
            )
            i += 1
    return pd.DataFrame(rows)


def test_sample_design():
    pop = population()
    s = stratified_sample(pop, n_per_stratum=50, seed=1, exclude_duplicate_files=False)
    assert s["sample_id"].is_unique and len(s) == 150
    for _, g in s.groupby("stratum"):
        assert g["weight"].iloc[0] == pytest.approx(g["N_h"].iloc[0] / g["n_h"].iloc[0])
    assert s["weight"].sum() == pytest.approx(len(pop))
    s2 = stratified_sample(pop, n_per_stratum=50, seed=1, exclude_duplicate_files=False)
    pd.testing.assert_frame_equal(s, s2)


def test_sheet_is_blind():
    s = stratified_sample(population(), n_per_stratum=10, seed=0, exclude_duplicate_files=False)
    sheet = annotation_sheet(s, seed=0)
    assert set(sheet.columns) == {"sample_id", "video", "frame_number"}
    assert "florence_text" not in sheet


def test_weighted_estimate_is_unbiased_where_naive_is_not():
    pop = population()
    true_acc = (pop["gold_true"] == pop["pipeline_label"]).mean()
    est, naive = [], []
    for seed in range(30):
        s = stratified_sample(pop, n_per_stratum=60, seed=seed, exclude_duplicate_files=False)
        gold = pd.DataFrame({"sample_id": s["sample_id"], "gold": s["gold_true"]})
        r = estimate(s, gold, systems=["pipeline_label"], n_boot=50, seed=seed)
        est.append(r["systems"]["pipeline_label"]["metrics"]["accuracy"])
        naive.append((s["gold_true"] == s["pipeline_label"]).mean())
    assert np.mean(est) == pytest.approx(true_acc, abs=0.01)
    assert abs(np.mean(naive) - true_acc) > 0.1


def test_estimate_reports_intervals_and_prevalence():
    pop = population()
    s = stratified_sample(pop, n_per_stratum=40, seed=0, exclude_duplicate_files=False)
    gold = pd.DataFrame({"sample_id": s["sample_id"], "gold": s["gold_true"]})
    r = estimate(s, gold, n_boot=100, seed=0)
    sys_ = r["systems"]["pipeline_label"]
    low, high = sys_["ci"]["accuracy"]
    assert low <= sys_["metrics"]["accuracy"] <= high
    assert sum(sys_["prevalence"].values()) == pytest.approx(1.0)


def test_agreement_and_adjudication():
    ans = pd.DataFrame(
        [
            ("s1", "ann1", "weapons"),
            ("s1", "ann2", "weapons"),
            ("s1", "ann3", "nonmilitary"),
            ("s2", "ann1", "aircraft"),
            ("s2", "ann2", "nonmilitary"),
            ("s3", "ann1", "unsure"),
            ("s3", "ann2", "unsure"),
            ("s4", "ann1", "maritime"),
            ("s4", "ann2", "maritime"),
            ("s4", "ann3", "maritime"),
        ],
        columns=["sample_id", "annotator", "label"],
    )
    gold = adjudicate(ans).set_index("sample_id")["gold"]
    assert (
        gold["s1"] == "weapons"
        and gold["s2"] == "unresolved"
        and gold["s3"] == "unresolved"
        and gold["s4"] == "maritime"
    )
    ia = inter_annotator_agreement(ans, LABELS)
    assert ia["pairwise"]["ann1|ann2"]["n"] == 3
    assert ia["items_with_all_annotators"] == 2
