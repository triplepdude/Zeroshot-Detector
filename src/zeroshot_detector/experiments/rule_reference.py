"""RQ1: the rule reference applied to the whole population (exact; no sampling)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..analysis.stats import add_post_ids
from ..evaluation.metrics import cohen_kappa, confusion_matrix, metrics_from_confusion
from ..segment import build_vocabulary, segment_all
from ..taxonomy import LABELS, NONMILITARY
from .labels import rule_reference

READINGS_ALL = ("equipment", "type", "lenient", "strict")


def population(frames: pd.DataFrame) -> pd.DataFrame:
    """Frames of one file per post (the study population)."""
    f = add_post_ids(frames) if "post_id" not in frames else frames
    return f[f["video"].isin(f.drop_duplicates("post_id")["video"])]


def string_table(frames: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct string in the population with frame count, system labels and rule labels."""
    pop = population(frames)
    counts = pop["florence_text"].value_counts()
    seg = segment_all(counts.index, build_vocabulary(counts.index))
    first = pop.drop_duplicates("florence_text").set_index("florence_text")
    rows = []
    for text, n in counts.items():
        r = rule_reference(seg[text])
        rows.append(
            {
                "text": text,
                "frames": int(n),
                "objects": "; ".join(seg[text]),
                "pipeline_label": first.loc[text, "pipeline_label"],
                "llm_label": first.loc[text, "llm_label"],
                "rule_primary": r.primary,
                "rule_categories": ";".join(r.categories),
                "rule_scores": ";".join(f"{k}={v}" for k, v in sorted(r.scores.items())),
                "rule_depiction": r.depiction,
                "rule_armament": r.armament,
                "rule_primary_weapon_first": r.primary_weapon_rule,
                **{f"rule_{k}": v for k, v in r.readings.items()},
            }
        )
    return pd.DataFrame(rows)


def exact_metrics(table: pd.DataFrame, reading: str, system: str) -> dict:
    w = table["frames"].to_numpy(dtype=float)
    cm = confusion_matrix(table[f"rule_{reading}"], table[system], LABELS, weights=w)
    m = metrics_from_confusion(cm, LABELS)
    present = [i for i in range(len(LABELS)) if cm[i, :].sum() > 0]
    f1_present = [m["per_class"][LABELS[i]]["f1"] for i in present]
    return {
        "agreement": m["accuracy"],
        "kappa": cohen_kappa(cm),
        "macro_f1_reference_classes": float(np.mean(f1_present)),
        "per_class": {lab: {k: m["per_class"][lab][k] for k in ("precision", "recall", "f1")} for lab in LABELS},
    }


def rq1_summary(table: pd.DataFrame) -> dict:
    w = table["frames"].to_numpy(dtype=float)
    total = w.sum()
    out: dict = {"frames": int(total), "strings": len(table), "readings": {}}
    for reading in READINGS_ALL:
        lab = table[f"rule_{reading}"]
        out["readings"][reading] = {
            "prevalence": {c: float(w[lab.to_numpy() == c].sum() / total) for c in LABELS},
            "military_share": float(w[lab.to_numpy() != NONMILITARY].sum() / total),
            "pipeline": exact_metrics(table, reading, "pipeline_label"),
            "llm": exact_metrics(table, reading, "llm_label"),
        }
    changes = {}
    for a, b in (("equipment", "type"), ("type", "lenient"), ("lenient", "strict"), ("equipment", "strict")):
        changes[f"{a}->{b}"] = float(w[(table[f"rule_{a}"] != table[f"rule_{b}"]).to_numpy()].sum() / total)
    out["frames_changing_between_readings"] = changes
    out["depiction"] = {
        d: float(w[(table["rule_depiction"] == d).to_numpy()].sum() / total)
        for d in ("real_or_unclear", "screen_or_print", "toy_or_model")
    }
    out["armament_rule_frames"] = float(w[table["rule_armament"].to_numpy()].sum() / total)
    return out


def object_group(row: pd.Series) -> str:
    """Equipment-reading category of a string, with ground vehicles split into tank / truck / other."""
    cat = row["rule_equipment"]
    if cat != "ground_vehicles":
        return cat
    objects = set(row["objects"].split("; "))
    return (
        "ground_vehicles: tank"
        if "tank" in objects
        else ("ground_vehicles: truck" if objects & {"truck", "pickup truck"} else "ground_vehicles: other")
    )


def contrast_by_group(table: pd.DataFrame, readings=("lenient", "strict"), a="llm_label", b="pipeline_label") -> dict:
    """Exact decomposition of the agreement difference a − b by object group (post hoc; adds up to the total)."""
    w = table["frames"].to_numpy(dtype=float)
    total = w.sum()
    group = table.apply(object_group, axis=1).to_numpy()
    out: dict = {"frame_share": {g: float(w[group == g].sum() / total) for g in sorted(set(group))}, "readings": {}}
    for reading in readings:
        ref = table[f"rule_{reading}"].to_numpy()
        gain = w * ((table[a].to_numpy() == ref).astype(float) - (table[b].to_numpy() == ref).astype(float))
        out["readings"][reading] = {
            "total": float(gain.sum() / total),
            "by_group": {g: float(gain[group == g].sum() / total) for g in sorted(set(group))},
        }
    return out
