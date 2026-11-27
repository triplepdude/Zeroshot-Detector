import numpy as np
import pandas as pd
import pytest

from zeroshot_detector.experiments.design import draw_string_design, estimate_design, metrics
from zeroshot_detector.experiments.labels import rule_reference
from zeroshot_detector.taxonomy import LABELS

# Worked examples of docs/experiments/CODEBOOK.md (strings absent from the archive)
CODEBOOK_EXAMPLES = [
    (["tank", "poster"], ["ground_vehicles"], "ground_vehicles", {"ground_vehicles": 3}, "screen_or_print"),
    (["truck", "vehicle"], ["ground_vehicles"], "ground_vehicles", {"ground_vehicles": 2}, "real_or_unclear"),
    (["truck", "luggage and bags"], ["ground_vehicles"], "ground_vehicles", {"ground_vehicles": 1}, "real_or_unclear"),
    (["helicopter", "swimming pool"], ["aircraft"], "aircraft", {"aircraft": 1}, "real_or_unclear"),
    (["van", "weapon"], ["weapons", "ground_vehicles"], "ground_vehicles", {"ground_vehicles": 3}, "real_or_unclear"),
    (["sword", "person"], ["weapons"], "weapons", {}, "real_or_unclear"),
    (
        ["helmet", "person", "rifle", "truck"],
        ["weapons", "ground_vehicles"],
        "weapons",
        {"ground_vehicles": 3},
        "real_or_unclear",
    ),
    (["kayak", "person"], ["maritime"], "maritime", {"maritime": 0}, "real_or_unclear"),
    (["boat", "toy"], [], "none", {}, "toy_or_model"),
    (["tank", "person", "flag"], ["ground_vehicles"], "ground_vehicles", {"ground_vehicles": 3}, "real_or_unclear"),
    (["person", "tank top", "woman"], [], "none", {}, "real_or_unclear"),
]


@pytest.mark.parametrize("objects,categories,primary,scores,depiction", CODEBOOK_EXAMPLES)
def test_rule_reference_matches_codebook_examples(objects, categories, primary, scores, depiction):
    r = rule_reference(objects)
    assert (r.categories, r.primary, r.scores, r.depiction) == (categories, primary, scores, depiction)


def test_readings_thresholds():
    heli = rule_reference(["helicopter"]).readings
    assert heli == {"equipment": "aircraft", "type": "aircraft", "lenient": "aircraft", "strict": "nonmilitary"}
    car = rule_reference(["car", "wheel"]).readings
    assert car["equipment"] == "ground_vehicles" and car["lenient"] == "nonmilitary"
    weapon = rule_reference(["person", "weapon"]).readings
    assert set(weapon.values()) == {"weapons"}  # weapons are never gated
    assert rule_reference(["vehicle"]).primary == "none"
    assert rule_reference(["No object detected."]).readings["equipment"] == "nonmilitary"


def test_parent_and_conditional_rules():
    assert rule_reference(["helicopter", "vehicle"]).categories == ["aircraft"]
    assert rule_reference(["rocket"]).categories == []
    assert rule_reference(["missile", "rocket"]).categories == ["weapons"]
    assert rule_reference(["kitchen knife", "person"]).categories == []
    assert rule_reference(["knife", "rifle"]).categories == ["weapons"]


def toy_population(seed=0):
    rng = np.random.default_rng(seed)
    texts = [f"s{i}" for i in range(300)]
    frames = np.maximum(1, (rng.pareto(1.2, 300) * 20).astype(int))
    pipe = rng.choice(["nonmilitary", "weapons", "aircraft"], 300, p=[0.8, 0.1, 0.1])
    llm = np.where(rng.random(300) < 0.85, pipe, "nonmilitary")
    strings = pd.DataFrame(
        {"text": texts, "frames": frames, "objects": ["x"] * 300, "pipeline_label": pipe, "llm_label": llm}
    )
    pop = strings.loc[strings.index.repeat(strings["frames"])].rename(columns={"text": "florence_text"})[
        ["florence_text", "pipeline_label", "llm_label"]
    ]
    truth = dict(zip(texts, np.where(rng.random(300) < 0.9, pipe, rng.choice(LABELS, 300)), strict=True))
    return strings, pop, truth


def test_design_weights_sum_to_population_and_estimate_is_unbiased():
    strings, pop, truth = toy_population()
    systems = {"pipeline": dict(zip(strings["text"], strings["pipeline_label"], strict=True))}
    w = strings["frames"].to_numpy()
    exact = np.average(
        [truth[t] == p for t, p in zip(strings["text"], strings["pipeline_label"], strict=True)], weights=w
    )
    estimates = []
    for seed in range(40):
        rec = draw_string_design(strings, pop, k_certainty=20, n_tail=10, n_tail_overrides={}, seed=seed)
        assert rec["weight"].sum() == pytest.approx(len(pop))
        r = estimate_design(rec, lambda _: truth, systems, n_boot=0, seed=0)
        estimates.append(r["systems"]["pipeline"]["agreement"]["estimate"])
    assert np.mean(estimates) == pytest.approx(exact, abs=0.01)


def test_contrast_and_bootstrap_interval():
    strings, pop, truth = toy_population(1)
    systems = {
        "pipeline": dict(zip(strings["text"], strings["pipeline_label"], strict=True)),
        "llm": dict(zip(strings["text"], strings["llm_label"], strict=True)),
    }
    rec = draw_string_design(strings, pop, k_certainty=30, n_tail=15, n_tail_overrides={}, seed=0)
    r = estimate_design(rec, lambda _: truth, systems, n_boot=200, seed=0, contrasts=[("llm", "pipeline")])
    a = r["systems"]["pipeline"]["agreement"]
    assert a["ci_low"] <= a["estimate"] <= a["ci_high"]
    c = r["contrasts"]["llm - pipeline: agreement"]
    assert c["estimate"] == pytest.approx(
        r["systems"]["llm"]["agreement"]["estimate"] - r["systems"]["pipeline"]["agreement"]["estimate"]
    )


def test_metrics_binary_and_prevalence():
    cm = np.zeros((5, 5))
    cm[4, 4] = 80
    cm[0, 0] = 10
    cm[0, 4] = 10
    m = metrics(cm)
    assert m["agreement"] == pytest.approx(0.9)
    assert m["prevalence_weapons"] == pytest.approx(0.2)
    assert m["recall_weapons"] == pytest.approx(0.5)
    assert m["military_share_system"] == pytest.approx(0.1)


# -- v2 adjudication -------------------------------------------------------------
from zeroshot_detector.experiments.reference import UNRESOLVED, adjudicate, pass_readings, resampler  # noqa: E402


def test_pass_readings_thresholds_and_primary():
    r = pass_readings(["aircraft"], "aircraft", {"aircraft": 2})
    assert r == {"equipment": "aircraft", "type": "aircraft", "lenient": "aircraft", "strict": "nonmilitary"}
    r = pass_readings(["weapons", "ground_vehicles"], "ground_vehicles", {"ground_vehicles": 1})
    # primary does not qualify under lenient -> weapons (always qualifies)
    assert r["equipment"] == "ground_vehicles" and r["lenient"] == "weapons"
    assert pass_readings([], "none", {})["strict"] == "nonmilitary"


def _passes(rows):
    out = []
    for iid, cfg, cats, prim, sc in rows:
        out.append(
            {
                "item_id": iid,
                "raw": iid,
                "config": cfg,
                "categories": cats,
                "primary": prim,
                "scores": sc,
                "depiction": "real_or_unclear",
                **{f"pass_{k}": v for k, v in pass_readings(cats, prim, sc).items()},
            }
        )
    return pd.DataFrame(out)


def test_adjudicate_majority_ties_and_categories():
    p = _passes(
        [
            ("x", "A", ["aircraft"], "aircraft", {"aircraft": 2}),
            ("x", "B", ["aircraft"], "aircraft", {"aircraft": 3}),
            ("x", "C", ["aircraft"], "aircraft", {"aircraft": 1}),
            ("y", "A", ["weapons"], "weapons", {}),
            ("y", "B", [], "none", {}),
            ("y", "A1", ["weapons"], "weapons", {}),  # not a vote
        ]
    )
    ref = adjudicate(p).set_index("item_id")
    assert ref.loc["x", "lenient"] == "aircraft" and ref.loc["x", "strict"] == "nonmilitary"
    assert ref.loc["x", "median_scores"] == {"aircraft": 2.0}
    assert ref.loc["y", "lenient"] == UNRESOLVED  # 1-1 tie, A1 ignored
    assert ref.loc["x", "categories_2plus"] == ["aircraft"]


def test_resampler_is_seeded():
    p = _passes([("x", c, ["aircraft"], "aircraft", {"aircraft": s}) for c, s in (("A", 2), ("B", 1), ("C", 3))])
    draw = resampler(p, "lenient", {"x": "x"})
    a = [draw(np.random.default_rng(i))["x"] for i in range(20)]
    b = [draw(np.random.default_rng(i))["x"] for i in range(20)]
    assert a == b and set(a) <= {"aircraft", "nonmilitary"}


# -- regressions from the adversarial code review --------------------------------
def test_annotation_bootstrap_keeps_unvoted_strings():
    strings, pop, truth = toy_population(2)
    systems = {"pipeline": dict(zip(strings["text"], strings["pipeline_label"], strict=True))}
    rec = draw_string_design(strings, pop, k_certainty=30, n_tail=15, n_tail_overrides={}, seed=0)
    voted = {t: v for t, v in truth.items() if t != rec["text"].iloc[0]}  # one fixed string never voted on

    total = estimate_design(rec, lambda _: truth, systems, 200, 0, resample_passes=lambda rng: voted)
    sampling = estimate_design(rec, lambda _: truth, systems, 200, 0)
    a, b = total["systems"]["pipeline"]["agreement"], sampling["systems"]["pipeline"]["agreement"]
    # unanimous votes: annotation-aware and sampling-only intervals coincide and contain the estimate
    assert (a["ci_low"], a["ci_high"]) == (b["ci_low"], b["ci_high"])
    assert a["ci_low"] <= a["estimate"] <= a["ci_high"]


def test_weapon_first_sensitivity_removes_only_the_armament_rule():
    from zeroshot_detector.experiments.report import weapon_first_passes

    p = _passes(
        [
            ("arm", "A", ["weapons", "ground_vehicles"], "ground_vehicles", {"ground_vehicles": 3}),  # armament
            ("tnk", "A", ["weapons", "ground_vehicles"], "ground_vehicles", {"ground_vehicles": 3}),  # tank + weapon
            ("per", "A", ["weapons", "ground_vehicles"], "weapons", {"ground_vehicles": 2}),  # person present
        ]
    )
    objects = {"arm": ["van", "weapon"], "tnk": ["tank", "weapon"], "per": ["person", "rifle", "truck"]}
    wf = weapon_first_passes(p, objects).set_index("item_id")
    assert wf.loc["arm", "pass_lenient"] == "weapons"
    assert wf.loc["tnk", "pass_lenient"] == "ground_vehicles"
    assert wf.loc["per", "pass_lenient"] == "weapons"
    assert rule_reference(["tank", "weapon"]).primary_weapon_rule == "ground_vehicles"
    assert rule_reference(["van", "weapon"]).primary_weapon_rule == "weapons"


def test_weighted_fleiss_equals_unweighted_with_equal_weights():
    from zeroshot_detector.annotation import fleiss_kappa
    from zeroshot_detector.experiments.reference import _weighted_fleiss

    arr = np.array([[3, 0], [2, 1], [0, 3], [1, 2], [3, 0]])
    assert _weighted_fleiss(arr, np.ones(5)) == pytest.approx(fleiss_kappa(arr))


# -- RQ5 (exploratory temporal analysis) -----------------------------------------
def test_decomposition_and_glm_recover_a_within_batch_step():
    from zeroshot_detector.experiments.temporal import decomposition, glm_step, month_index

    rng = np.random.default_rng(0)
    e = month_index("2022-02")
    rows = []
    # batch "b1" straddles the event (step +0.3), "b0" before only, "b2" after only (no step)
    for batch, months in (("b0", range(e - 18, e - 6)), ("b1", range(e - 6, e + 7)), ("b2", range(e + 7, e + 19))):
        for t in months:
            if t == e:
                continue
            for _ in range(200):
                p = 0.1 + (0.3 if (batch == "b1" and t > e) else 0.0)
                rows.append({"batch": batch, "t": t, "label": "weapons" if rng.random() < p else "nonmilitary"})
    table = pd.DataFrame(rows)
    dec = decomposition(table, "label", "2022-02-24")
    assert dec["straddling_batches"] == ["b1"]
    assert dec["within_batch_change"] == pytest.approx(0.3, abs=0.03)
    step = glm_step(table, "label", e)
    assert step is not None and step > 1.0  # log-odds of 0.4 vs 0.1 is about 1.8
    # no straddling batch -> step not identified
    assert glm_step(table[table["batch"] != "b1"], "label", e) is None


def test_headline_rows_report_contrasts_and_predictions():
    from zeroshot_detector.experiments.render import _headline_rows

    def c(est, lo, hi):
        return {"estimate": est, "ci_low": lo, "ci_high": hi}

    s = {
        "rq2": {
            "lenient": {"total": {"contrasts": {"llm - pipeline: agreement": c(0.03, 0.02, 0.04)}}},
            "strict": {"total": {"contrasts": {"llm - pipeline: agreement": c(-0.04, -0.05, -0.03)}}},
        },
        "rq1_panel_vs_rule": {"lenient": c(0.999, 0.998, 1.0)},
    }
    rows = {r[0].split(":")[0]: r for r in _headline_rows(s)}
    assert rows["C1"][3] == "positive; interval excludes 0"
    assert rows["C2"][3].startswith("not computed")
    assert rows["P1"][3] == "confirmed"
    assert rows["P2"][3] == "confirmed"
    s["rq2"]["strict"]["total"]["contrasts"]["llm - pipeline: agreement"] = c(0.01, -0.01, 0.02)
    assert {r[0].split(":")[0]: r for r in _headline_rows(s)}["P2"][3] == "not confirmed"
