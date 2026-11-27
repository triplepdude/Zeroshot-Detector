"""v2 sampling design (strings) and the matching estimator with annotation-aware bootstrap.

Design (``docs/experiments/PLAN.md`` §Design):

* **C** certainty stratum: the ``k_certainty`` most frequent strings; each
  enters with its exact frame count ``M_s``;
* **S** screened census: every ``nonmilitary|nonmilitary`` string outside C
  whose objects contain a weapon or watercraft token;
* **T** tail: the remaining frames, simple random sampling within
  pipeline × LLM strata (``nonmilitary|nonmilitary`` split by a vehicle-token
  screen), weight ``N_h / n_h``.

Estimates are totals over records: C and S records are fixed (weight ``M_s``),
tail records are resampled within their stratum in the bootstrap. In each
replicate the annotator passes of every string are also resampled with
replacement and re-adjudicated, so intervals include annotation uncertainty.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..taxonomy import LABELS, NONMILITARY

WEAPON_TOKENS = frozenset(
    {
        "weapon",
        "rifle",
        "handgun",
        "shotgun",
        "missile",
        "rocket",
        "cannon",
        "sword",
        "dagger",
        "bow and arrow",
        "knife",
    }
)
WATERCRAFT_TOKENS = frozenset({"boat", "canoe", "kayak", "jet ski", "watercraft"})
VEHICLE_TOKENS = frozenset(
    {
        "vehicle",
        "land vehicle",
        "car",
        "truck",
        "pickup truck",
        "van",
        "suv",
        "taxi",
        "limousine",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "ambulance",
        "golf cart",
        "snowplow",
        "cart",
        "tank",
        "helicopter",
        "airplane",
        "aircraft",
    }
)
NO_OBJECT = "No object detected."


def draw_string_design(
    strings: pd.DataFrame,
    pop_frames: pd.DataFrame,
    k_certainty: int,
    n_tail: int,
    n_tail_overrides: Mapping[str, int],
    seed: int,
) -> pd.DataFrame:
    """Return the design records.

    ``strings``: one row per string with ``text, frames, objects ('; '-joined),
    pipeline_label, llm_label``. ``pop_frames``: population frames
    (``florence_text, pipeline_label, llm_label``; one row per frame).
    Output columns: ``record_id, text, part (C/S/T), stratum, weight, fixed,
    N_h, n_h``.
    """
    s = strings.sort_values(["frames", "text"], ascending=[False, True]).reset_index(drop=True)
    certainty = set(s["text"].head(k_certainty))
    objs = {t: set(o.split("; ")) for t, o in zip(s["text"], s["objects"], strict=True)}
    nn = (s["pipeline_label"] == NONMILITARY) & (s["llm_label"] == NONMILITARY)
    screened = {
        t
        for t, is_nn in zip(s["text"], nn, strict=True)
        if is_nn and t not in certainty and objs[t] & (WEAPON_TOKENS | WATERCRAFT_TOKENS)
    }
    records = []
    for t, m in zip(s["text"], s["frames"], strict=True):
        if t in certainty or t in screened:
            part = "C" if t in certainty else "S"
            records.append(
                {"text": t, "part": part, "stratum": part, "weight": float(m), "fixed": True, "N_h": int(m), "n_h": 1}
            )
    # tail
    tail = pop_frames[~pop_frames["florence_text"].isin(certainty | screened)].copy()
    cell = tail["pipeline_label"] + "|" + tail["llm_label"]
    is_nn = cell == f"{NONMILITARY}|{NONMILITARY}"
    vehicle = tail["florence_text"].map(lambda t: bool(objs.get(t, set()) & VEHICLE_TOKENS))
    cell = cell.where(~is_nn, np.where(vehicle, "NN-vehicle-token", "NN-no-equipment-token"))
    tail["stratum"] = cell
    rng = np.random.default_rng(seed)
    for stratum, g in tail.groupby("stratum", sort=True):
        n_h = min(int(n_tail_overrides.get(stratum, n_tail)), len(g))
        idx = np.sort(rng.choice(len(g), size=n_h, replace=False))
        census = n_h == len(g)
        for t in g["florence_text"].iloc[idx]:
            records.append(
                {
                    "text": t,
                    "part": "T",
                    "stratum": stratum,
                    "weight": len(g) / n_h,
                    "fixed": census,
                    "N_h": len(g),
                    "n_h": n_h,
                }
            )
    out = pd.DataFrame(records)
    out.insert(0, "record_id", [f"r{i:05d}" for i in range(len(out))])
    return out


# -- estimation -------------------------------------------------------------------
def _weighted_cm(gold: np.ndarray, pred: np.ndarray, w: np.ndarray) -> np.ndarray:
    index = {lab: i for i, lab in enumerate(LABELS)}
    k = len(LABELS)
    gi = np.array([index.get(g, -1) for g in gold])
    pi = np.array([index[p] for p in pred])
    ok = gi >= 0  # unresolved gold labels are dropped
    return np.bincount(gi[ok] * k + pi[ok], weights=w[ok], minlength=k * k).reshape(k, k)


def metrics(cm: np.ndarray) -> dict:
    n = cm.sum()
    if n == 0:
        return {}
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n**2
    support = cm.sum(1)
    f1s = []
    out = {"agreement": float(po), "kappa": float((po - pe) / (1 - pe)) if pe < 1 else float("nan")}
    for i, lab in enumerate(LABELS):
        tp, pred, sup = cm[i, i], cm[:, i].sum(), support[i]
        prec = tp / pred if pred else float("nan")
        rec = tp / sup if sup else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if pred and sup and (prec + rec) else (0.0 if sup else float("nan"))
        out[f"precision_{lab}"] = float(prec)
        out[f"recall_{lab}"] = float(rec)
        out[f"prevalence_{lab}"] = float(sup / n)
        if sup:
            f1s.append(f1)
    out["macro_f1"] = float(np.mean(f1s)) if f1s else float("nan")
    mil = [i for i, lab in enumerate(LABELS) if lab != NONMILITARY]
    nm = LABELS.index(NONMILITARY)
    binary = np.array([[cm[np.ix_(mil, mil)].sum(), cm[mil, nm].sum()], [cm[nm, mil].sum(), cm[nm, nm]]])
    bo = np.trace(binary) / n
    be = (binary.sum(0) * binary.sum(1)).sum() / n**2
    out["binary_agreement"] = float(bo)
    out["binary_kappa"] = float((bo - be) / (1 - be)) if be < 1 else float("nan")
    out["military_share_reference"] = float(1 - support[nm] / n)
    out["military_share_system"] = float(1 - cm[:, nm].sum() / n)
    return out


def estimate_design(
    records: pd.DataFrame,
    reference: Callable[[Mapping[str, np.ndarray] | None], Mapping[str, str]],
    systems: Mapping[str, Mapping[str, str]],
    n_boot: int,
    seed: int,
    contrasts: Sequence[tuple[str, str]] = (),
    resample_passes: Callable[[np.random.Generator], Mapping[str, str]] | None = None,
    confidence: float = 0.95,
) -> dict:
    """Point estimates and bootstrap intervals for every system (and paired contrasts).

    ``reference(None)`` returns the point-estimate reference label per string;
    ``resample_passes(rng)`` returns a re-adjudicated reference per string from
    resampled passes (annotation-aware bootstrap). If it is ``None``, only
    sampling uncertainty is propagated.
    """
    texts = records["text"].to_numpy()
    w = records["weight"].to_numpy(dtype=float)
    ref0 = reference(None)
    gold0 = np.array([ref0.get(t, "unresolved") for t in texts], dtype=object)
    preds = {name: np.array([lab[t] for t in texts], dtype=object) for name, lab in systems.items()}
    point = {name: metrics(_weighted_cm(gold0, p, w)) for name, p in preds.items()}
    rng = np.random.default_rng(seed)
    groups = [
        np.flatnonzero((records["stratum"] == st).to_numpy() & ~records["fixed"].to_numpy())
        for st in records.loc[~records["fixed"], "stratum"].unique()
    ]
    fixed_idx = np.flatnonzero(records["fixed"].to_numpy())
    reps: dict[str, list[dict]] = {name: [] for name in preds}
    for _ in range(n_boot):
        idx = np.concatenate([fixed_idx, *[rng.choice(g, size=len(g), replace=True) for g in groups]])
        ref = resample_passes(rng) if resample_passes is not None else ref0
        gold = np.array([ref.get(t, "unresolved") for t in texts[idx]], dtype=object)
        for name, p in preds.items():
            reps[name].append(metrics(_weighted_cm(gold, p[idx], w[idx])))
    alpha = (1 - confidence) / 2

    def interval(values: list[float]) -> tuple[float, float, int]:
        arr = np.asarray(values, dtype=float)
        n_nan = int(np.isnan(arr).sum())
        arr = arr[~np.isnan(arr)]
        if not len(arr):
            return float("nan"), float("nan"), n_nan
        lo, hi = np.quantile(arr, [alpha, 1 - alpha])
        return float(lo), float(hi), n_nan

    out = {"systems": {}, "contrasts": {}}
    for name in preds:
        out["systems"][name] = {
            key: {
                "estimate": val,
                **dict(
                    zip(
                        ("ci_low", "ci_high", "nan_replicates"),
                        interval([r.get(key, np.nan) for r in reps[name]]),
                        strict=True,
                    )
                ),
            }
            for key, val in point[name].items()
        }
    for a, b in contrasts:
        for key in ("agreement", "kappa", "macro_f1", "binary_agreement"):
            diffs = [ra.get(key, np.nan) - rb.get(key, np.nan) for ra, rb in zip(reps[a], reps[b], strict=True)]
            lo, hi, n_nan = interval(diffs)
            out["contrasts"][f"{a} - {b}: {key}"] = {
                "estimate": point[a][key] - point[b][key],
                "ci_low": lo,
                "ci_high": hi,
                "nan_replicates": n_nan,
            }
    out["n_records"] = len(records)
    out["n_unresolved_records"] = int((gold0 == "unresolved").sum())
    out["n_boot"] = n_boot
    out["seed"] = seed
    return out
