"""Assemble all experiment results (PLAN v2) into ``results/experiments/``."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..taxonomy import NONMILITARY
from .design import NO_OBJECT, estimate_design
from .grid import ZeroShotConfig, all_configs
from .reference import (
    READINGS,
    UNRESOLVED,
    adjudicate,
    agreement_stats,
    item_id,
    pass_readings,
    resampler,
)
from .rq4 import (
    THRESHOLDS,
    config_labels,
    evaluate_grid,
    nested_selection,
    record_metrics,
    supervised_reference,
    weak_supervision,
)


def refresh_rule_labels(strings: pd.DataFrame) -> dict:
    """Recompute the rule-reference columns with the current ``labels.rule_reference`` (in place).

    ``strings.csv.gz`` keeps the labels committed before annotation; the report uses
    the current implementation and records which strings changed (DEVIATIONS).
    """
    from .labels import rule_reference

    new = [rule_reference(tuple(o.split("; "))) for o in strings["objects"]]
    changes: dict = {}
    for r in ("equipment", "type", "lenient", "strict"):
        fresh = pd.Series([x.readings[r] for x in new], index=strings.index)
        changed = fresh != strings[f"rule_{r}"]
        changes[r] = {"strings": int(changed.sum()), "frames": int(strings.loc[changed, "frames"].sum())}
        strings[f"rule_{r}"] = fresh
    strings["rule_depiction"] = [x.depiction for x in new]
    return changes


DECISION_TABLE = (
    "helicopter",
    "airplane",
    "boat",
    "truck",
    "truckwheel",
    "car tiretruck",
    "tank",
    "weapon",
    "rifle",
    "handgun",
    "vehicle",
    "land vehicle",
)
CONFIRMATORY_ZS = ZeroShotConfig("deberta", "example", "military", "separated")
CORRECTED_PIPELINE = ZeroShotConfig("bart", "example", "ids", "separated")
HEAVY_N = 25


def with_deterministic(ref: pd.DataFrame) -> pd.DataFrame:
    """Add the never-annotated 'No object detected.' item (nonmilitary under every reading)."""
    noobj = item_id(NO_OBJECT)
    if noobj in set(ref["item_id"]):
        return ref
    row = {
        "item_id": noobj,
        "raw": NO_OBJECT,
        **{r: NONMILITARY for r in READINGS},
        **{f"{r}_agreement": 1.0 for r in READINGS},
        "n_votes": 0,
        "categories_2plus": [],
        "median_scores": {},
        "depiction": "real_or_unclear",
    }
    return pd.concat([ref, pd.DataFrame([row])], ignore_index=True)


def _ref_map(reference: pd.DataFrame, reading: str, text_of: Mapping[str, str]) -> dict[str, str]:
    return {text_of[i]: lab for i, lab in zip(reference["item_id"], reference[reading], strict=True)}


# -- embedded checks -----------------------------------------------------------------
def embedded_checks(passes: pd.DataFrame, items: pd.DataFrame) -> dict:
    kind = dict(zip(items["item_id"], items["kind"], strict=True))
    source = dict(zip(items["item_id"], items["source_item_id"], strict=True))
    out: dict = {}
    # test-retest within configuration
    rt = passes[passes["item_id"].map(kind) == "retest"]
    rows = []
    for _, r in rt.iterrows():
        orig = passes[(passes["item_id"] == source[r["item_id"]]) & (passes["config"] == r["config"])]
        if len(orig):
            rows.append(
                {f"same_{x}": orig.iloc[0][f"pass_{x}"] == r[f"pass_{x}"] for x in READINGS} | {"config": r["config"]}
            )
    rdf = pd.DataFrame(rows)
    out["test_retest"] = {
        "pairs": len(rdf),
        **({f"agreement_{x}": float(rdf[f"same_{x}"].mean()) for x in READINGS} if len(rdf) else {}),
        "by_config": {c: float(g["same_lenient"].mean()) for c, g in rdf.groupby("config")} if len(rdf) else {},
    }
    # synthetic rule compliance
    exp = {
        i: json.loads(e)
        for i, e in zip(items["item_id"], items.get("expected", pd.Series([""] * len(items))), strict=True)
        if isinstance(e, str) and e.startswith("{")
    }
    syn = passes[passes["item_id"].isin(exp)]
    comp = []
    for _, r in syn.iterrows():
        e = exp[r["item_id"]]
        comp.append(
            {
                "config": r["config"],
                "primary": r["primary"] == e["primary"],
                "categories": sorted(r["categories"]) == sorted(e["categories"]),
                "scores": {k: int(v) for k, v in r["scores"].items()} == {k: int(v) for k, v in e["scores"].items()},
                "lenient": r["pass_lenient"] == pass_readings(e["categories"], e["primary"], e["scores"])["lenient"],
            }
        )
    cdf = pd.DataFrame(comp)
    out["synthetic_compliance"] = (
        {
            c: {k: float(g[k].mean()) for k in ("primary", "categories", "scores", "lenient")} | {"n": len(g)}
            for c, g in cdf.groupby("config")
        }
        if len(cdf)
        else {}
    )
    # single-item vs batch (configuration A)
    a1 = passes[passes["config"] == "A1"]
    a = passes[(passes["config"] == "A") & passes["item_id"].isin(set(a1["item_id"]))].drop_duplicates("item_id")
    m = a1.merge(a, on="item_id", suffixes=("_single", "_batch"))
    out["single_item_vs_batch"] = (
        {
            "items": len(m),
            **{f"agreement_{x}": float((m[f"pass_{x}_single"] == m[f"pass_{x}_batch"]).mean()) for x in READINGS},
        }
        if len(m)
        else {}
    )
    return out


def position_effect(passes: pd.DataFrame, reference: pd.DataFrame) -> dict:
    ref = dict(zip(reference["item_id"], reference["lenient"], strict=True))
    p = passes[passes["config"].isin(("A", "B", "C")) & passes["item_id"].isin(ref)].copy()
    p["disagree"] = [lab != ref[i] for i, lab in zip(p["item_id"], p["pass_lenient"], strict=True)]
    bins = pd.cut(p["position"], [-1, 4, 9, 14, 19, 24], labels=["1-5", "6-10", "11-15", "16-20", "21-25"])
    return {str(k): float(v) for k, v in p.groupby(bins, observed=True)["disagree"].mean().items()}


def human_consistency(items: pd.DataFrame, reference: pd.DataFrame, frames: pd.DataFrame) -> dict:
    h = frames[frames["human_label"].notna()].drop_duplicates("florence_text")
    h = h.assign(item_id=h["florence_text"].map(item_id))
    m = h.merge(reference, on="item_id")
    out = {"strings": len(m)}
    for x in ("equipment", "lenient", "strict"):
        ok = m[x] == m["human_label"]
        out[f"agree_{x}"] = float(ok.mean()) if len(m) else float("nan")
        side = np.where(
            m["human_label"] == m["pipeline_label"],
            "human=pipeline",
            np.where(m["human_label"] == m["llm_label"], "human=llm", "neither"),
        )
        out[f"by_side_{x}"] = {s: float(ok[side == s].mean()) for s in np.unique(side)}
    return out


# -- sensitivity ---------------------------------------------------------------------
def weapon_first_passes(passes: pd.DataFrame, objects_of: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Pre-registered sensitivity (a): the primary chosen without the armament rule.

    Only passes where the armament rule can have applied change: a weapon and a
    vehicle category are present, no person is listed and no T3 object (tank)
    is listed (rule 2 keeps the tank primary). Their primary becomes ``weapons``.
    """
    from .labels import PERSONS

    p = passes.copy()
    prim = []
    for iid, cats, pr in zip(p["item_id"], p["categories"], p["primary"], strict=True):
        objs = set(objects_of.get(iid, ()))
        armament = "weapons" in cats and pr in ("aircraft", "maritime", "ground_vehicles") and not (objs & PERSONS)
        prim.append("weapons" if armament and "tank" not in objs else pr)
    for x in READINGS:
        p[f"pass_{x}"] = [
            pass_readings(c, pr, s)[x] for c, pr, s in zip(p["categories"], prim, p["scores"], strict=True)
        ]
    return p


def bare_score_override(ref: dict[str, str], reading: str, score: int) -> dict[str, str]:
    cats = {"helicopter": "aircraft", "airplane": "aircraft", "boat": "maritime", "truck": "ground_vehicles"}
    th = {"equipment": 0, "type": 1, "lenient": 2, "strict": 3}[reading]
    out = dict(ref)
    for t, c in cats.items():
        if t in out:
            out[t] = c if score >= th else NONMILITARY
    return out


def influence_table(
    records: pd.DataFrame,
    ref: dict[str, str],
    reference: pd.DataFrame,
    systems: Mapping[str, Mapping[str, str]],
    text_of: Mapping[str, str],
) -> list[dict]:
    w = records.groupby("text")["weight"].sum().sort_values(ascending=False)
    total = records["weight"].sum()
    agree = dict(zip(reference["item_id"].map(text_of), reference["lenient_agreement"], strict=True))
    base = {s: record_metrics(records, ref, lab)["agreement"] for s, lab in systems.items()}
    rows = []
    for t in w.index[:HEAVY_N]:
        flipped = dict(ref)
        cur = ref.get(t)
        alt = (
            NONMILITARY if cur != NONMILITARY else systems["llm"][t] if systems["llm"][t] != NONMILITARY else "aircraft"
        )
        flipped[t] = alt
        rows.append(
            {
                "text": t,
                "frame_share": float(w[t] / total),
                "lenient": cur,
                "vote_share": agree.get(t),
                **{
                    f"delta_{s}": record_metrics(records, flipped, lab)["agreement"] - base[s]
                    for s, lab in systems.items()
                },
            }
        )
    return rows


# -- main ----------------------------------------------------------------------------
def run(
    data_dir: Path,
    passes: pd.DataFrame,
    frames: pd.DataFrame,
    videos: pd.DataFrame,
    grid_scores_dir: Path | None,
    n_boot: int = 2000,
    seed: int = 20251128,
) -> dict:
    from .temporal import temporal_analysis

    records = pd.read_csv(data_dir / "design_records.csv", keep_default_na=False)
    records["fixed"] = records["fixed"].astype(str).str.lower().eq("true")
    strings = pd.read_csv(data_dir / "strings.csv.gz", keep_default_na=False)
    rule_changes = refresh_rule_labels(strings)
    items = pd.read_csv(data_dir / "items.csv", keep_default_na=False)
    text_of = dict(zip(items["item_id"], items["raw"], strict=False))
    text_of.update({item_id(t): t for t in strings["text"]})
    sample_ids = set(items.loc[items["kind"] == "sample", "item_id"])
    sample_passes = passes[passes["item_id"].isin(sample_ids)]
    reference = with_deterministic(adjudicate(sample_passes))
    item_weights = records.assign(iid=records["text"].map(item_id)).groupby("iid")["weight"].sum().to_dict()
    out: dict = {
        "design": {
            "records": len(records),
            "strings": int(records["text"].nunique()),
            "parts": records.groupby("part")["text"].nunique().to_dict(),
        },
        "rule_label_changes_since_commit": rule_changes,
    }
    out["annotation"] = {
        "answers": len(passes),
        "by_config": passes.groupby("config").size().to_dict(),
        "fallback_answers": int(passes["fallback"].sum()),
        "unresolved": {r: int((reference[r] == UNRESOLVED).sum()) for r in READINGS},
        "self_consistency_ABC": agreement_stats(sample_passes, sample_ids, weights=item_weights),
        "self_consistency_ABC_excluding_decision_table": agreement_stats(
            sample_passes, sample_ids - {item_id(t) for t in DECISION_TABLE}, weights=item_weights
        ),
    }
    out["checks"] = embedded_checks(passes, items)
    out["checks"]["position_effect_lenient_disagreement"] = position_effect(sample_passes, reference)
    hc_ids = sample_ids | set(items.loc[items["kind"] == "human_consistency", "item_id"])
    out["checks"]["human_consistency"] = human_consistency(
        items, with_deterministic(adjudicate(passes[passes["item_id"].isin(hc_ids)])), frames
    )

    # systems
    sys_labels: dict[str, dict[str, str]] = {
        "pipeline": dict(zip(strings["text"], strings["pipeline_label"], strict=True)),
        "llm": dict(zip(strings["text"], strings["llm_label"], strict=True)),
        "rule_reference": {},  # filled per reading
    }
    seg = {t: tuple(o.split("; ")) for t, o in zip(strings["text"], strings["objects"], strict=True)}
    grid_scores = {}
    if grid_scores_dir is not None:
        for c in all_configs():
            f = grid_scores_dir / f"{c.name}.csv.gz"
            if f.exists():
                grid_scores[c] = pd.read_csv(f, keep_default_na=False)
    if CORRECTED_PIPELINE in grid_scores:
        sys_labels["corrected_pipeline"] = config_labels(
            CORRECTED_PIPELINE, grid_scores[CORRECTED_PIPELINE], seg, True, 0.8
        )
    if CONFIRMATORY_ZS in grid_scores:
        sys_labels["deberta_military_separated"] = config_labels(
            CONFIRMATORY_ZS, grid_scores[CONFIRMATORY_ZS], seg, True, None
        )

    out["rq1_panel_vs_rule"] = {}
    out["rq2"] = {}
    for reading in READINGS:
        ref = _ref_map(reference, reading, text_of)
        systems = {k: v for k, v in sys_labels.items() if k != "rule_reference"}
        systems["rule_reference"] = dict(zip(strings["text"], strings[f"rule_{reading}"], strict=True))
        contrasts = [("llm", "pipeline")]
        if "deberta_military_separated" in systems:
            contrasts.append(("deberta_military_separated", "pipeline"))
        if "corrected_pipeline" in systems:
            contrasts.append(("corrected_pipeline", "pipeline"))
        total = estimate_design(
            records,
            lambda _r, ref=ref: ref,
            systems,
            n_boot,
            seed,
            contrasts,
            resample_passes=resampler(sample_passes, reading, text_of),
        )
        sampling = estimate_design(records, lambda _r, ref=ref: ref, systems, n_boot, seed, contrasts)
        out["rq2"][reading] = {"total": total, "sampling_only": sampling}
        out["rq1_panel_vs_rule"][reading] = total["systems"]["rule_reference"]["agreement"]
        # bounds for unresolved items: all-correct / all-wrong for each system
        unresolved = {text_of[i] for i in reference.loc[reference[reading] == UNRESOLVED, "item_id"]}
        if unresolved:
            bounds = {}
            for s, lab in systems.items():
                best = {t: (lab[t] if t in unresolved else v) for t, v in ref.items()}
                # all wrong: a valid label that differs from the system's label
                worst = {
                    t: ((NONMILITARY if lab[t] != NONMILITARY else "weapons") if t in unresolved else v)
                    for t, v in ref.items()
                }
                bounds[s] = {
                    "all_correct": record_metrics(records, best, lab)["agreement"],
                    "all_wrong": record_metrics(records, worst, lab)["agreement"],
                }
            out["rq2"][reading]["unresolved_bounds"] = bounds

    lenient = _ref_map(reference, "lenient", text_of)
    # sensitivity analyses (lenient, point estimates + sampling-only CIs)
    sens = {}
    objects_of = {item_id(t): o.split("; ") for t, o in zip(strings["text"], strings["objects"], strict=True)}
    wf = with_deterministic(adjudicate(weapon_first_passes(sample_passes, objects_of)))
    sens["weapon_first_primary"] = estimate_design(
        records, lambda _r: _ref_map(wf, "lenient", text_of), {k: sys_labels[k] for k in ("pipeline", "llm")}, 500, seed
    )
    for score in (1, 2, 3):
        r2 = bare_score_override(lenient, "lenient", score)
        sens[f"bare_dual_use_score_{score}"] = estimate_design(
            records, lambda _r, r2=r2: r2, {k: sys_labels[k] for k in ("pipeline", "llm")}, 500, seed
        )
    dep = dict(zip(reference["item_id"].map(text_of), reference["depiction"], strict=True))
    real = records[records["text"].map(lambda t: dep.get(t, "real_or_unclear") == "real_or_unclear")]
    sens["excluding_depictions"] = estimate_design(
        real, lambda _r: lenient, {k: sys_labels[k] for k in ("pipeline", "llm")}, 500, seed
    )
    for left_out in ("A", "B", "C"):
        lo = with_deterministic(adjudicate(sample_passes[sample_passes["config"] != left_out]))
        sens[f"without_config_{left_out}"] = estimate_design(
            records,
            lambda _r, lo=lo: _ref_map(lo, "lenient", text_of),
            {k: sys_labels[k] for k in ("pipeline", "llm")},
            500,
            seed,
        )
    out["sensitivity_lenient"] = sens
    out["influence_lenient"] = influence_table(
        records, lenient, reference, {k: sys_labels[k] for k in ("pipeline", "llm")}, text_of
    )

    # RQ4
    if grid_scores:
        grid = evaluate_grid(records, lenient, grid_scores, seg)
        base = record_metrics(records, lenient, sys_labels["pipeline"])["agreement"]
        grid["diff_vs_pipeline"] = grid["agreement"] - base
        out["rq4_grid_lenient"] = grid.to_dict("records")
        out["rq4_nested_selection"] = {
            k: v for k, v in nested_selection(records, lenient, grid_scores, seg).items() if k != "labels"
        }
        out["rq4_thresholds"] = [("argmax" if t is None else t) for t in THRESHOLDS]
    sup = supervised_reference(records, lenient, seg)
    out["rq4_supervised_reference"] = {k: v for k, v in sup.items() if k != "labels"}
    weak = weak_supervision(strings, records, lenient)
    out["rq4_weak_supervision"] = weak["out_of_fold"]
    out["rq4_baselines"] = {
        "all_nonmilitary": record_metrics(records, lenient, {t: NONMILITARY for t in records["text"]}),
        "rule_reference": record_metrics(
            records, lenient, dict(zip(strings["text"], strings["rule_lenient"], strict=True))
        ),
        "single_configuration_A": record_metrics(
            records,
            lenient,
            {
                **{t: NONMILITARY for t in records["text"]},
                **{
                    text_of[i]: lab
                    for i, lab in zip(
                        sample_passes.loc[sample_passes["config"] == "A", "item_id"],
                        sample_passes.loc[sample_passes["config"] == "A", "pass_lenient"],
                        strict=True,
                    )
                },
            },
        ),
    }
    out["rq5"] = temporal_analysis(frames, videos, ["pipeline_label", "llm_label"])
    return out


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
