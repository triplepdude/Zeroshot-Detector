"""RQ5: share of military videos over time, around two events (descriptive).

Unit: one video file per post (first occurrence), with at least one frame; its
label is the plurality vote of its frame labels. Time: post creation month,
decoded from the post id (see ``analysis.stats.post_timestamp``).

Two summaries per event date:

* the difference between the military share in the ``window`` months before
  and after the date (videos as independent units; Newcombe hybrid-score
  interval);
* a weighted segmented regression on monthly shares (level and slope change
  at the event; weights = videos per month; Newey–West standard errors).

Both are descriptive: how videos were selected is undocumented and batches
are chronological, so changes may reflect collection rather than content.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..aggregate import plurality_labels
from ..analysis.stats import add_post_ids, post_timestamp, video_counts
from ..data import split_video_id
from ..evaluation.metrics import _z, wilson_interval
from ..taxonomy import NONMILITARY

EVENTS = {"2022-02-24": "Full-scale invasion of Ukraine", "2023-10-07": "7 October attack / Gaza war"}


def video_table(frames: pd.DataFrame, videos: pd.DataFrame, label_columns: Sequence[str]) -> pd.DataFrame:
    """One row per post (first file), with post month and plurality label per labelling."""
    if "post_id" not in frames:
        frames = add_post_ids(frames)
    keep_files = set(frames.drop_duplicates("post_id")["video"])
    out = pd.DataFrame({"video": videos["video"]})
    out["post_id"] = out["video"].map(lambda v: split_video_id(v)[1])
    ts = out["post_id"].map(post_timestamp)
    out["month"] = ts.map(lambda t: t.strftime("%Y-%m") if t is not None else None)
    for col in label_columns:
        counts = video_counts(frames, videos, col)
        out[col] = plurality_labels(counts)
    out = out[out["video"].isin(keep_files) & out["month"].notna()]
    return out[out[label_columns[0]].notna()].reset_index(drop=True)


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


def _month_index(month: str) -> int:
    y, m = month.split("-")
    return int(y) * 12 + int(m) - 1


def newcombe_difference(k1: int, n1: int, k2: int, n2: int, confidence: float = 0.95) -> dict:
    """Difference of proportions p2 - p1 with Newcombe's hybrid score interval (method 10)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_interval(k1, n1, confidence)
    l2, u2 = wilson_interval(k2, n2, confidence)
    d = p2 - p1
    low = d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    high = d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return {"before": p1, "after": p2, "difference": d, "ci_low": low, "ci_high": high, "n_before": n1, "n_after": n2}


def window_comparison(table: pd.DataFrame, column: str, event: str, months: int = 6) -> dict:
    e = _month_index(event[:7])
    idx = table["month"].map(_month_index)
    is_mil = table[column] != NONMILITARY
    before = (idx >= e - months) & (idx < e)
    after = (idx > e) & (idx <= e + months)  # the event month itself is excluded
    return newcombe_difference(int(is_mil[before].sum()), int(before.sum()), int(is_mil[after].sum()), int(after.sum()))


def newey_west(x: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """HAC covariance of OLS coefficients (Bartlett kernel)."""
    xtx_inv = np.linalg.inv(x.T @ x)
    u = x * resid[:, None]
    s = u.T @ u
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        g = u[lag:].T @ u[:-lag]
        s += w * (g + g.T)
    return xtx_inv @ s @ xtx_inv


def segmented_regression(monthly: pd.DataFrame, event: str, window: int = 18, lags: int = 3) -> dict:
    """WLS of monthly share on time, post-event level and post-event slope (±``window`` months)."""
    e = _month_index(event[:7])
    m = monthly.assign(t=monthly["month"].map(_month_index) - e)
    m = m[(m["t"] >= -window) & (m["t"] <= window) & (m["t"] != 0)]
    if len(m) < 8 or (m["t"] > 0).sum() < 3 or (m["t"] < 0).sum() < 3:
        return {"months": len(m), "note": "too few months on one side"}
    post = (m["t"] > 0).astype(float).to_numpy()
    t = m["t"].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(m)), t, post, post * t])
    w = np.sqrt(m["videos"].to_numpy(dtype=float))
    xw, yw = x * w[:, None], m["share"].to_numpy() * w
    beta, *_ = np.linalg.lstsq(xw, yw, rcond=None)
    resid = yw - xw @ beta
    cov = newey_west(xw, resid, lags)
    se = np.sqrt(np.diag(cov))
    z = _z(0.95)
    names = ["intercept", "pre_slope", "level_change", "slope_change"]
    return {
        "months": len(m),
        "window": window,
        "lags": lags,
        **{
            n: {"estimate": float(b), "ci_low": float(b - z * s), "ci_high": float(b + z * s)}
            for n, b, s in zip(names, beta, se, strict=True)
        },
    }


def temporal_analysis(frames: pd.DataFrame, videos: pd.DataFrame, label_columns: Sequence[str]) -> dict:
    table = video_table(frames, videos, label_columns)
    out = {"videos": len(table), "events": EVENTS, "by_labelling": {}}
    for col in label_columns:
        monthly = monthly_shares(table, col)
        out["by_labelling"][col] = {
            "monthly": monthly.to_dict("records"),
            "events": {
                ev: {
                    "window_6m": window_comparison(table, col, ev, 6),
                    "segmented_regression": segmented_regression(monthly, ev),
                }
                for ev in EVENTS
            },
        }
    return out
