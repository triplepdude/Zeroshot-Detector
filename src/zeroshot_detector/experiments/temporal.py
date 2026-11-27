"""RQ5 (exploratory): composition of the archive by batch around two dates.

Unit: one video file per post (first occurrence) with at least one frame; its
label is the plurality vote of its frame labels. Time: post creation month,
decoded from the post id. Batches (file-name prefixes) are chronological
collection batches, so a change "around" a date can reflect the collection
rather than the content. Therefore:

* the change is decomposed into a **within-batch** part (videos of batches
  that straddle the date: after vs before) and a **between-batch** part
  (batches entirely before vs entirely after);
* a binomial GLM with batch fixed effects, a linear month trend and a
  post-event step is fitted; the step is identified only from straddling
  batches;
* the event estimate is ranked among **placebo** estimates for every month from
  2020-06 to 2024-04. A change is reported only if it exceeds the 95th
  percentile of placebo |steps| and has the same sign within and between batches.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..aggregate import plurality_labels
from ..analysis.stats import add_post_ids, post_timestamp, video_counts
from ..data import split_video_id
from ..evaluation.metrics import wilson_interval
from ..taxonomy import NONMILITARY

EVENTS = {"2022-02-24": "Full-scale invasion of Ukraine", "2023-10-07": "7 October attack / Gaza war"}
WINDOW = 18
PLACEBO_RANGE = ("2020-06", "2024-04")


def month_index(month: str) -> int:
    y, m = month.split("-")
    return int(y) * 12 + int(m) - 1


def month_name(index: int) -> str:
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def video_table(frames: pd.DataFrame, videos: pd.DataFrame, label_columns: Sequence[str]) -> pd.DataFrame:
    """One row per post (first file): batch, month, plurality label per labelling."""
    if "post_id" not in frames:
        frames = add_post_ids(frames)
    # one file per post, deterministically preferring the copy in a dated batch
    files = frames.drop_duplicates("video")[["video", "batch", "post_id"]]
    files = files.assign(_unprefixed=files["batch"].isna()).sort_values(["_unprefixed", "video"], kind="stable")
    keep_files = set(files.drop_duplicates("post_id")["video"])
    out = pd.DataFrame({"video": videos["video"]})
    split = out["video"].map(split_video_id)
    out["batch"] = split.map(lambda s: s[0] or "none")
    ts = split.map(lambda s: post_timestamp(s[1]))
    out["month"] = ts.map(lambda t: t.strftime("%Y-%m") if t is not None and pd.notna(t) else None)
    for col in label_columns:
        out[col] = plurality_labels(video_counts(frames, videos, col))
    out = out[out["video"].isin(keep_files) & out["month"].notna()]
    out = out[out[label_columns[0]].notna()].reset_index(drop=True)
    out["t"] = out["month"].map(month_index)
    return out


def monthly_shares(table: pd.DataFrame, column: str) -> pd.DataFrame:
    g = table.assign(m=table[column] != NONMILITARY).groupby("month")["m"].agg(["sum", "size"])
    rows = []
    for month, r in g.iterrows():
        low, high = wilson_interval(r["sum"], r["size"])
        rows.append(
            {
                "month": month,
                "videos": int(r["size"]),
                "military": int(r["sum"]),
                "share": r["sum"] / r["size"],
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def decomposition(table: pd.DataFrame, column: str, event: str, window: int = WINDOW) -> dict:
    e = month_index(event[:7])
    w = table[(table["t"] >= e - window) & (table["t"] <= e + window) & (table["t"] != e)].copy()
    w["post"] = w["t"] > e
    w["mil"] = w[column] != NONMILITARY
    spans = w.groupby("batch")["post"].agg(["min", "max"])
    straddle = spans[(spans["min"] == False) & (spans["max"] == True)].index  # noqa: E712
    s = w[w["batch"].isin(straddle)]
    before_only = spans[spans["max"] == False].index  # noqa: E712
    after_only = spans[spans["min"] == True].index  # noqa: E712

    def share(df: pd.DataFrame) -> float:
        return float(df["mil"].mean()) if len(df) else float("nan")

    # within-batch change: after - before inside each straddling batch, weights n0*n1/(n0+n1) (CMH-style)
    per_batch = {}
    num = den = 0.0
    for b, g in s.groupby("batch"):
        n0, n1 = int((~g["post"]).sum()), int(g["post"].sum())
        diff = float(g.loc[g["post"], "mil"].mean() - g.loc[~g["post"], "mil"].mean())
        wt = n0 * n1 / (n0 + n1)
        per_batch[str(b)] = {"n_before": n0, "n_after": n1, "change": diff}
        num += wt * diff
        den += wt
    within = num / den if den else float("nan")
    between = share(w[w["batch"].isin(after_only)]) - share(w[w["batch"].isin(before_only)])
    return {
        "straddling_batches": sorted(map(str, straddle)),
        "videos_straddling": len(s),
        "within_by_batch": per_batch,
        "within_batch_change": within,
        "between_batch_change": between,
        "batches_before_only": sorted(map(str, before_only)),
        "batches_after_only": sorted(map(str, after_only)),
    }


def _logit_irls(x: np.ndarray, y: np.ndarray, iters: int = 50) -> np.ndarray:
    beta = np.zeros(x.shape[1])
    for _ in range(iters):
        eta = np.clip(x @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        wgt = np.maximum(p * (1 - p), 1e-9)
        z = eta + (y - p) / wgt
        xtw = x.T * wgt
        new = np.linalg.lstsq(xtw @ x + 1e-8 * np.eye(x.shape[1]), xtw @ z, rcond=None)[0]
        if np.max(np.abs(new - beta)) < 1e-8:
            return new
        beta = new
    return beta


def glm_step(table: pd.DataFrame, column: str, event_month: int, window: int = WINDOW) -> float | None:
    """Post-event step (log-odds) from a binomial GLM with batch fixed effects and a month trend."""
    w = table[(table["t"] >= event_month - window) & (table["t"] <= event_month + window) & (table["t"] != event_month)]
    if w.empty:
        return None
    post = (w["t"] > event_month).to_numpy(dtype=float)
    spans = w.assign(post=post).groupby("batch")["post"].agg(["min", "max"])
    if not ((spans["min"] == 0) & (spans["max"] == 1)).any():
        return None  # step not identified without a straddling batch
    batches = pd.get_dummies(w["batch"]).to_numpy(dtype=float)
    t = (w["t"].to_numpy(dtype=float) - event_month) / 12.0
    x = np.column_stack([batches, t, post])
    y = (w[column] != NONMILITARY).to_numpy(dtype=float)
    return float(_logit_irls(x, y)[-1])


def temporal_analysis(frames: pd.DataFrame, videos: pd.DataFrame, label_columns: Sequence[str]) -> dict:
    table = video_table(frames, videos, label_columns)
    lo, hi = (month_index(m) for m in PLACEBO_RANGE)
    out: dict = {"videos": len(table), "events": EVENTS, "window_months": WINDOW, "by_labelling": {}}
    for col in label_columns:
        placebo = {month_name(m): glm_step(table, col, m) for m in range(lo, hi + 1)}
        event_months = {ev[:7] for ev in EVENTS}
        valid = np.array([v for k, v in placebo.items() if v is not None and k not in event_months])
        q95 = float(np.quantile(np.abs(valid), 0.95)) if len(valid) else float("nan")
        events = {}
        for ev in EVENTS:
            em = month_index(ev[:7])
            step = glm_step(table, col, em)
            dec = decomposition(table, col, ev)
            others = valid
            rank = float((np.abs(others) < abs(step)).mean()) if step is not None and len(others) else float("nan")
            same_sign = (
                step is not None
                and np.sign(dec["within_batch_change"]) == np.sign(step)
                and np.sign(dec["between_batch_change"]) == np.sign(step)
            )
            events[ev] = {
                "glm_step_logodds": step,
                "placebo_percentile_of_abs_step": rank,
                "exceeds_placebo_95th": bool(step is not None and abs(step) > q95),
                "same_sign_within_and_between": bool(same_sign),
                "reported_as_change": bool(step is not None and abs(step) > q95 and same_sign),
                **dec,
            }
        out["by_labelling"][col] = {
            "monthly": monthly_shares(table, col).to_dict("records"),
            "placebo_steps": placebo,
            "placebo_abs_step_95th": q95,
            "events": events,
        }
    return out
