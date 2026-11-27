"""Analyses of the archived study data.

Every function takes the canonical tables (see :mod:`zeroshot_detector.data`)
and returns plain, JSON-serialisable dictionaries. Uncertainty is quantified
with a cluster bootstrap over *posts* (a post id can occur in several files;
frames of the same post are treated as one cluster), because frames of one
video are strongly correlated and treating 1.1 M frames as independent would
give intervals that are far too narrow.

Terminology used throughout:

* ``pipeline`` - the label from Florence-2 + keyword rule + BART-MNLI
  (``pipeline_label``), i.e. the system under study;
* ``llm`` - the label from the OpenAI model used as a second annotator
  (``llm_label``). It is **not** ground truth; pipeline-vs-LLM figures are
  inter-system *agreement*;
* ``human`` - the label from the review web app (``human_label``), available
  only for a non-random subset of disagreements (see :func:`human_review`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..aggregate import legacy_labels, military_fraction_labels, plurality_labels
from ..data import split_video_id
from ..evaluation.bootstrap import (
    bootstrap_additive_many,
    cluster_confusion_tables,
    proportion_tables,
    ratio,
)
from ..evaluation.metrics import cohen_kappa, confusion_matrix, metrics_from_confusion, wilson_interval
from ..rules import KeywordRule, legacy_keyword_match
from ..taxonomy import LABELS, LEGACY_MILITARY_KEYWORDS, MILITARY_LABELS, NONMILITARY

BINARY_LABELS = ("military", NONMILITARY)


# -- helpers --------------------------------------------------------------------
def add_post_ids(frames: pd.DataFrame) -> pd.DataFrame:
    """Add ``batch`` (file-name prefix) and ``post_id`` columns."""
    split = frames["video"].map(split_video_id)
    return frames.assign(batch=split.map(lambda x: x[0]), post_id=split.map(lambda x: x[1]))


def post_timestamp(post_id: str) -> pd.Timestamp | None:
    """Creation time encoded in a TikTok post id (upper 32 bits = Unix seconds).

    This is an inference from the platform's public id scheme, not a field of
    the archived data; ids outside 2016-2030 are treated as undecodable.
    """
    try:
        seconds = int(post_id) >> 32
    except ValueError:
        return None
    if not 1_451_606_400 <= seconds <= 1_893_456_000:
        return None
    return pd.Timestamp(seconds, unit="s", tz="UTC")


def to_binary(labels: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(labels, dtype=object)
    return np.where(arr == NONMILITARY, NONMILITARY, "military")


def _cm_stats(labels: Sequence[str]) -> dict:
    k = len(labels)

    def acc(t: np.ndarray) -> float:
        cm = t.reshape(k, k)
        n = cm.sum()
        return float(np.trace(cm) / n) if n else float("nan")

    def kappa(t: np.ndarray) -> float:
        return cohen_kappa(t.reshape(k, k))

    def macro_f1(t: np.ndarray) -> float:
        return metrics_from_confusion(t.reshape(k, k), labels)["macro_f1"]

    return {"agreement": acc, "kappa": kappa, "macro_f1": macro_f1}


def compare_labelings(
    reference: Sequence[str],
    compared: Sequence[str],
    clusters: Sequence,
    labels: Sequence[str],
    n_boot: int,
    seed: int,
) -> dict:
    """Confusion matrix (rows = ``reference``), metrics and cluster-bootstrap CIs."""
    cm = confusion_matrix(reference, compared, labels)
    metrics = metrics_from_confusion(cm, labels)
    tables = cluster_confusion_tables(clusters, reference, compared, labels)
    boot = bootstrap_additive_many(tables, _cm_stats(labels), n_boot=n_boot, seed=seed)
    return {
        "labels": list(labels),
        "confusion_matrix": cm.astype(int).tolist(),
        "metrics": metrics,
        "ci": {name: res.to_dict() for name, res in boot.items()},
    }


def _proportion_with_ci(clusters: Sequence, successes: Sequence[bool], n_boot: int, seed: int) -> dict:
    tables = proportion_tables(clusters, successes)
    res = bootstrap_additive_many(tables, {"p": ratio}, n_boot=n_boot, seed=seed)["p"]
    k = float(np.sum(successes))
    n = float(len(successes))
    return {"successes": int(k), "n": int(n), **res.to_dict()}


def _wilson(k: float, n: float) -> dict:
    low, high = wilson_interval(k, n)
    return {"successes": int(k), "n": int(n), "estimate": k / n if n else float("nan"), "ci_low": low, "ci_high": high}


def _top_texts(df: pd.DataFrame, n: int = 10, extra: Sequence[str] = ()) -> list[dict]:
    cols = ["florence_text", *extra]
    g = df.groupby(cols, dropna=False).agg(frames=("video", "size"), videos=("post_id", "nunique"))
    g = g.sort_values("frames", ascending=False).head(n).reset_index()
    return [
        {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
        for row in g.to_dict("records")
    ]


def _distribution(series: pd.Series, labels: Sequence[str]) -> dict:
    counts = series.value_counts()
    total = int(counts.sum())
    return {
        label: {"n": int(counts.get(label, 0)), "share": float(counts.get(label, 0) / total) if total else float("nan")}
        for label in labels
    }


# -- 1. dataset description ---------------------------------------------------------
def dataset_summary(frames: pd.DataFrame, videos: pd.DataFrame) -> dict:
    per_video = frames.groupby("video").size()
    gaps = frames.sort_values(["video", "frame_number"]).groupby("video")["frame_number"].diff().dropna()
    skipped = float(((gaps / 30) - 1).clip(lower=0).sum())
    no_object = frames["florence_text"].eq("No object detected.")
    zero_frame_videos = videos.loc[videos[list(LABELS)].sum(axis=1) == 0, "video"].tolist()
    return {
        "n_frames": len(frames),
        "n_files": int(videos["video"].nunique()),
        "n_files_with_frames": int(frames["video"].nunique()),
        "n_files_without_frames": len(zero_frame_videos),
        "files_without_frames": zero_frame_videos,
        "n_posts": int(frames["post_id"].nunique()),
        "n_batches": int(frames["batch"].fillna("<none>").nunique()),
        "n_unique_texts": int(frames["florence_text"].nunique()),
        "frames_no_object_detected": int(no_object.sum()),
        "frames_per_file": {
            "mean": float(per_video.mean()),
            "median": float(per_video.median()),
            "p25": float(per_video.quantile(0.25)),
            "p75": float(per_video.quantile(0.75)),
            "p95": float(per_video.quantile(0.95)),
            "max": int(per_video.max()),
        },
        "all_frame_numbers_multiple_of_30": bool((frames["frame_number"] % 30 == 0).all()),
        "sampling_gaps": {
            "n_gaps": len(gaps),
            "n_gaps_longer_than_30": int((gaps > 30).sum()),
            "skipped_samples": skipped,
            "skipped_share": skipped / (len(frames) + skipped),
        },
        "texts_with_one_pipeline_label": bool((frames.groupby("florence_text")["pipeline_label"].nunique() == 1).all()),
        "texts_with_one_llm_label": bool((frames.groupby("florence_text")["llm_label"].nunique() == 1).all()),
        "post_dates": _post_dates(frames),
        "pipeline_distribution": _distribution(frames["pipeline_label"], LABELS),
        "llm_distribution": _distribution(frames["llm_label"], LABELS),
        "human_distribution": _distribution(frames["human_label"].dropna(), LABELS),
        "n_human_labelled_frames": int(frames["human_label"].notna().sum()),
        "n_human_labelled_texts": int(frames.loc[frames["human_label"].notna(), "florence_text"].nunique()),
    }


def _post_dates(frames: pd.DataFrame) -> dict:
    files = frames.drop_duplicates("video")[["video", "batch", "post_id"]]
    ts = files["post_id"].map(post_timestamp)
    ok = ts.notna()
    if not ok.any():
        return {"decodable_share": 0.0, "min": None, "median": None, "max": None, "files_per_year": {}, "by_batch": {}}
    by_batch = (
        files.assign(ts=ts)[ok].groupby(files["batch"].fillna("<none>"))["ts"].agg(["min", "median", "max", "size"])
    )
    return {
        "decodable_share": float(ok.mean()),
        "min": ts[ok].min().isoformat(),
        "median": ts[ok].median().isoformat(),
        "max": ts[ok].max().isoformat(),
        "files_per_year": {int(k): int(v) for k, v in ts[ok].dt.year.value_counts().sort_index().items()},
        "by_batch": {
            str(b): {"min": r["min"].date().isoformat(), "max": r["max"].date().isoformat(), "files": int(r["size"])}
            for b, r in by_batch.iterrows()
        },
    }


def duplicate_summary(frames: pd.DataFrame) -> dict:
    """Posts that were processed more than once (same post id under several batch prefixes)."""
    files_per_post = frames.groupby("post_id")["video"].nunique()
    dup_posts = files_per_post[files_per_post > 1]
    dup = frames[frames["post_id"].isin(dup_posts.index)]
    seq = (
        dup.sort_values(["video", "frame_number"])
        .groupby("video")
        .agg(
            post_id=("post_id", "first"),
            texts=("florence_text", tuple),
            labels=("pipeline_label", tuple),
        )
    )
    identical_texts = seq.groupby("post_id")["texts"].nunique().eq(1)
    identical_labels = seq.groupby("post_id")["labels"].nunique().eq(1)
    return {
        "n_posts_in_multiple_files": len(dup_posts),
        "n_files_involved": int(dup_posts.sum()),
        "max_files_per_post": int(dup_posts.max()) if len(dup_posts) else 0,
        "frames_in_duplicate_files": len(dup),
        "redundant_frames": int(len(dup) - dup["video"].isin(dup.groupby("post_id")["video"].first()).sum()),
        "posts_with_identical_texts_across_files": int(identical_texts.sum()),
        "posts_with_identical_labels_across_files": int(identical_labels.sum()),
        "duplicate_batch_pairs": _duplicate_batch_pairs(dup),
    }


def _duplicate_batch_pairs(dup: pd.DataFrame) -> dict[str, int]:
    pairs = (
        dup.drop_duplicates("video")
        .groupby("post_id")["batch"]
        .agg(lambda b: "+".join(sorted(x if isinstance(x, str) else "<none>" for x in b)))
    )
    return {k: int(v) for k, v in pairs.value_counts().head(10).items()}


# -- 2. keyword rule -------------------------------------------------------------
def keyword_audit(frames: pd.DataFrame, segmentation: dict[str, tuple[str, ...]] | None = None) -> dict:
    per_text = frames.groupby("florence_text").agg(frames=("video", "size"), label=("pipeline_label", "first"))
    matches = {}
    for kw in LEGACY_MILITARY_KEYWORDS:
        hit = per_text.index.str.lower().str.contains(kw, regex=False)
        matches[kw] = {"texts": int(hit.sum()), "frames": int(per_text.loc[hit, "frames"].sum())}
    first_hit = per_text.index.map(legacy_keyword_match)
    fired = pd.Series(first_hit, index=per_text.index).notna()
    gv = per_text["label"].eq("ground_vehicles")
    out = {
        "keywords": matches,
        "keywords_that_can_never_match": [kw for kw in LEGACY_MILITARY_KEYWORDS if kw != kw.lower()],
        "frames_keyword_fired": int(per_text.loc[fired, "frames"].sum()),
        "frames_pipeline_ground_vehicles": int(per_text.loc[gv, "frames"].sum()),
        "ground_vehicles_frames_from_keyword": int(per_text.loc[gv & fired, "frames"].sum()),
        "keyword_frames_not_ground_vehicles": int(per_text.loc[fired & ~gv, "frames"].sum()),
    }
    tank_top = per_text.index.str.contains("tank top", regex=False)
    out["tank_top"] = {"texts": int(tank_top.sum()), "frames": int(per_text.loc[tank_top, "frames"].sum())}
    if segmentation is not None:
        rule = KeywordRule.from_legacy_keywords()
        corrected = pd.Series(
            [rule.match(segmentation.get(t, (t,))) is not None for t in per_text.index], index=per_text.index
        )
        lost = fired & ~corrected
        gained = ~fired & corrected
        out["corrected_rule"] = {
            "frames_fired": int(per_text.loc[corrected, "frames"].sum()),
            "frames_no_longer_fired": int(per_text.loc[lost, "frames"].sum()),
            "texts_no_longer_fired": sorted(per_text.index[lost], key=lambda t: -per_text.loc[t, "frames"])[:20],
            "frames_newly_fired": int(per_text.loc[gained, "frames"].sum()),
        }
    return out


# -- 3. pipeline vs LLM ------------------------------------------------------------
def agreement(frames: pd.DataFrame, n_boot: int, seed: int) -> dict:
    out = {}
    out["frame_level"] = compare_labelings(
        frames["llm_label"], frames["pipeline_label"], frames["post_id"], LABELS, n_boot, seed
    )
    out["frame_level_binary"] = compare_labelings(
        to_binary(frames["llm_label"]),
        to_binary(frames["pipeline_label"]),
        frames["post_id"],
        BINARY_LABELS,
        n_boot,
        seed,
    )
    texts = frames.drop_duplicates("florence_text")
    out["text_level"] = compare_labelings(
        texts["llm_label"], texts["pipeline_label"], texts["florence_text"], LABELS, n_boot, seed
    )
    # disagreement cells
    dis = frames[frames["pipeline_label"] != frames["llm_label"]]
    cells = []
    for (p, q), g in dis.groupby(["pipeline_label", "llm_label"]):
        cells.append(
            {
                "pipeline_label": p,
                "llm_label": q,
                "frames": len(g),
                "texts": int(g["florence_text"].nunique()),
                "posts": int(g["post_id"].nunique()),
                "human_reviewed_frames": int(g["human_label"].notna().sum()),
                "top_texts": _top_texts(g, 8),
            }
        )
    cells.sort(key=lambda c: -c["frames"])
    out["disagreement_cells"] = cells
    out["n_disagreement_frames"] = len(dis)
    out["n_disagreement_frames_unreviewed"] = int(dis["human_label"].isna().sum())
    return out


# -- 4. human review --------------------------------------------------------------
def human_review(frames: pd.DataFrame, session: pd.DataFrame, n_boot: int, seed: int) -> dict:
    h = frames[frames["human_label"].notna()]
    cells_reviewed = sorted({(p, q) for p, q in zip(h["pipeline_label"], h["llm_label"], strict=True)})
    coverage = []
    for p, q in cells_reviewed:
        in_cell = (frames["pipeline_label"] == p) & (frames["llm_label"] == q)
        coverage.append(
            {
                "pipeline_label": p,
                "llm_label": q,
                "frames_in_cell": int(in_cell.sum()),
                "frames_reviewed": int((in_cell & frames["human_label"].notna()).sum()),
            }
        )
    reviewed_cells_all_disagreements = bool((h["pipeline_label"] != h["llm_label"]).all())

    def summarise(g: pd.DataFrame) -> dict:
        texts = g.drop_duplicates("florence_text")
        frame_level = {
            "sides_with_pipeline": _proportion_with_ci(
                g["post_id"], g["human_label"] == g["pipeline_label"], n_boot, seed
            ),
            "sides_with_llm": _proportion_with_ci(g["post_id"], g["human_label"] == g["llm_label"], n_boot, seed),
            "neither": _proportion_with_ci(
                g["post_id"],
                (g["human_label"] != g["llm_label"]) & (g["human_label"] != g["pipeline_label"]),
                n_boot,
                seed,
            ),
        }
        text_level = {
            "sides_with_pipeline": _wilson((texts["human_label"] == texts["pipeline_label"]).sum(), len(texts)),
            "sides_with_llm": _wilson((texts["human_label"] == texts["llm_label"]).sum(), len(texts)),
            "neither": _wilson(
                (
                    (texts["human_label"] != texts["llm_label"]) & (texts["human_label"] != texts["pipeline_label"])
                ).sum(),
                len(texts),
            ),
        }
        return {
            "frames": len(g),
            "texts": len(texts),
            "posts": int(g["post_id"].nunique()),
            "human_distribution_frames": _distribution(g["human_label"], LABELS),
            "human_distribution_texts": _distribution(texts["human_label"], LABELS),
            "frame_level": frame_level,
            "text_level": text_level,
        }

    by_cell = {f"{p}->{q}": summarise(g) for (p, q), g in h.groupby(["pipeline_label", "llm_label"])}
    neither = h[(h["human_label"] != h["llm_label"]) & (h["human_label"] != h["pipeline_label"])]

    # consistency of the raw review-session file with the merged labels
    sess = session.groupby("text")["user_category"].agg(lambda s: sorted(set(s)))
    merged = h.groupby("florence_text")["human_label"].first()
    common = sess.index.intersection(merged.index)
    consistent = [t for t in common if sess[t] == [merged[t]]]
    conflicting_within_session = {t: v for t, v in sess.items() if len(v) > 1}
    return {
        "n_frames": len(h),
        "n_texts": int(h["florence_text"].nunique()),
        "cells_reviewed": coverage,
        "only_disagreements_reviewed": reviewed_cells_all_disagreements,
        "pooled": summarise(h),
        "by_cell": by_cell,
        "neither_examples": _top_texts(neither, 12, extra=["pipeline_label", "llm_label", "human_label"]),
        "session_file": {
            "rows": len(session),
            "texts": int(session["text"].nunique()),
            "texts_also_in_merged_labels": len(common),
            "texts_consistent_with_merged_labels": len(consistent),
            "texts_with_conflicting_labels_within_session": conflicting_within_session,
        },
    }


def auto_correction(frames: pd.DataFrame) -> dict:
    """The original keyword "auto-correction" of disagreements (``prioritize_and_automate.py``).

    Disagreements matched by a substring rule were resolved automatically; the
    rest were sent to human review.
    """
    from ..data import LEGACY_AUTOCORRECT_RULES, _legacy_autocorrect

    dis = frames[frames["pipeline_label"] != frames["llm_label"]]
    texts = dis.groupby("florence_text").size().rename("frames").reset_index()
    texts["auto"] = _legacy_autocorrect(texts["florence_text"])
    auto = dis["florence_text"].map(texts.set_index("florence_text")["auto"])
    matched = auto.notna()
    reviewed = dis["human_label"].notna()
    keyword_hits = {}
    for category, keywords in LEGACY_AUTOCORRECT_RULES.items():
        for kw in keywords:
            hit = texts["florence_text"].str.lower().str.contains(kw, regex=False)
            if hit.any():
                top = texts[hit].sort_values("frames", ascending=False)["florence_text"].head(3).tolist()
                keyword_hits[kw] = {
                    "category": category,
                    "frames": int(texts.loc[hit, "frames"].sum()),
                    "examples": top,
                }
    return {
        "disagreement_frames": len(dis),
        "auto_resolved_frames": int(matched.sum()),
        "sent_to_humans_frames": int((~matched).sum()),
        "human_reviewed_equals_unmatched": bool((reviewed == ~matched).all()),
        "auto_label_equals_llm": int((auto[matched] == dis.loc[matched, "llm_label"]).sum()),
        "auto_label_equals_pipeline": int((auto[matched] == dis.loc[matched, "pipeline_label"]).sum()),
        "auto_label_distribution": {k: int(v) for k, v in auto[matched].value_counts().items()},
        "keyword_hits": keyword_hits,
    }


def discrepant_precision(frames: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Precision of each pipeline military label, using LLM agreement + human adjudication.

    For frames the pipeline labelled ``c``:

    * ``A`` - LLM agrees (assumed correct: *discrepant analysis*, optimistic);
    * ``R`` - LLM said nonmilitary and a human adjudicated (``H`` of them = ``c``);
    * ``U`` - LLM gave another military label, never adjudicated.

    ``point = (A + H) / (A + R)`` (``U`` excluded), ``lower = (A + H) / N``
    (all ``U`` wrong) and ``upper = (A + H + U) / N`` (all ``U`` right).
    """
    out = {}
    for c in MILITARY_LABELS:
        g = frames[frames["pipeline_label"] == c]
        if g.empty:
            continue
        agree = g["llm_label"] == c
        reviewed = g["human_label"].notna() & (g["llm_label"] == NONMILITARY)
        h_correct = reviewed & (g["human_label"] == c)
        unrev = ~agree & ~reviewed
        n, a, r, hc, u = len(g), int(agree.sum()), int(reviewed.sum()), int(h_correct.sum()), int(unrev.sum())
        num = (agree | h_correct).to_numpy(dtype=float)
        den = (agree | reviewed).to_numpy(dtype=float)
        _, idx = np.unique(g["post_id"].to_numpy(dtype=str), return_inverse=True)
        tables = np.column_stack([np.bincount(idx, weights=num), np.bincount(idx, weights=den)])
        boot = bootstrap_additive_many(tables, {"p": ratio}, n_boot=n_boot, seed=seed)["p"]
        out[c] = {
            "frames": n,
            "llm_agrees": a,
            "adjudicated": r,
            "adjudicated_correct": hc,
            "unadjudicated_disagreements": u,
            "point": boot.to_dict(),
            "lower_bound": (a + hc) / n,
            "upper_bound": (a + hc + u) / n,
        }
    return out


# -- 5. video level ---------------------------------------------------------------
def video_counts(frames: pd.DataFrame, videos: pd.DataFrame, column: str) -> np.ndarray:
    ct = frames.groupby(["video", column]).size().unstack(fill_value=0)
    ct = ct.reindex(index=videos["video"], columns=list(LABELS), fill_value=0)
    return ct.to_numpy()


def video_level(frames: pd.DataFrame, videos: pd.DataFrame, n_boot: int, seed: int) -> dict:
    counts = video_counts(frames, videos, "pipeline_label")
    stored = videos[list(LABELS)].to_numpy()
    total = counts.sum(axis=1)
    has = total > 0
    max_frame = frames.groupby("video")["frame_number"].max().reindex(videos["video"]).to_numpy()
    rules = {
        "legacy (as implemented)": legacy_labels(counts),
        "plurality": plurality_labels(counts),
        "any military frame": military_fraction_labels(counts, 1e-9),
        "military >= 10% of frames": military_fraction_labels(counts, 0.10),
        "military >= 25% of frames": military_fraction_labels(counts, 0.25),
        "military >= 50% of frames": military_fraction_labels(counts, 0.50),
    }
    distributions = {}
    for name, lab in rules.items():
        # The legacy rule also labels files without frames (as "weapons"); the
        # corrected rules leave them undetermined, so they are excluded there.
        s = pd.Series(lab if name.startswith("legacy") else lab[has])
        distributions[name] = _distribution(s, LABELS)
    nm = list(LABELS).index(NONMILITARY)
    best = counts.max(axis=1)
    tie_with_nm = has & (counts[:, nm] == best) & ((counts == best[:, None]).sum(axis=1) > 1)
    legacy = rules["legacy (as implemented)"]
    plur = rules["plurality"]
    changed = has & (legacy != plur)

    # duplicate-free variant: one file per post
    first_files = frames.drop_duplicates("post_id")["video"]
    keep = videos["video"].isin(first_files).to_numpy()
    dedup_dist = _distribution(pd.Series(plur[keep & has]), LABELS)

    # pipeline vs LLM at video level (plurality on both)
    llm_counts = video_counts(frames, videos, "llm_label")
    llm_plur = plurality_labels(llm_counts)
    post_of_video = videos["video"].map(lambda v: split_video_id(v)[1]).to_numpy()
    vid_agree = compare_labelings(llm_plur[has], plur[has], post_of_video[has], LABELS, n_boot, seed)
    vid_agree_bin = compare_labelings(
        to_binary(llm_plur[has]), to_binary(plur[has]), post_of_video[has], BINARY_LABELS, n_boot, seed
    )
    # descriptive trend: share of videos (one file per post) labelled military, by post year
    years = pd.Series(post_of_video).map(post_timestamp).map(lambda t: t.year if t is not None else None)
    trend = pd.DataFrame(
        {"year": years, "pipeline": plur != NONMILITARY, "llm": llm_plur != NONMILITARY, "use": keep & has}
    )
    trend = trend[trend["use"] & trend["year"].notna()]
    by_year = {}
    for year, g in trend.groupby("year"):
        by_year[int(year)] = {
            "videos": len(g),
            "pipeline_military": _wilson(g["pipeline"].sum(), len(g)),
            "llm_military": _wilson(g["llm"].sum(), len(g)),
        }
    return {
        "military_share_by_post_year": by_year,
        "counts_match_stored_summary": bool((counts == stored).all()),
        "n_files": len(videos),
        "n_files_with_frames": int(has.sum()),
        "legacy_label_of_files_without_frames": sorted(set(legacy[~has])),
        "n_files_without_frames": int((~has).sum()),
        "override_possible_files": int((has & (total > 0.5 * (max_frame + 1))).sum()),
        "ties_military_vs_nonmilitary": int(tie_with_nm.sum()),
        "legacy_vs_plurality_changed": int(changed.sum()),
        "distributions": distributions,
        "plurality_one_file_per_post": dedup_dist,
        "pipeline_vs_llm_plurality": vid_agree,
        "pipeline_vs_llm_plurality_binary": vid_agree_bin,
    }


# -- 6. analyses that need re-derived scores ----------------------------------------
def reproduction(frames: pd.DataFrame, relabel: pd.DataFrame) -> dict:
    per_text = frames.groupby("florence_text").agg(frames=("video", "size"), stored=("pipeline_label", "first"))
    merged = per_text.join(relabel.set_index("text")[["category", "source", "top_score"]], how="left")
    scored = merged["category"].notna()
    match = scored & (merged["category"] == merged["stored"])
    mism = merged[scored & ~match].sort_values("frames", ascending=False)
    near = scored & merged["top_score"].sub(0.8).abs().lt(1e-3)
    return {
        "texts_scored": int(scored.sum()),
        "texts_total": len(merged),
        "frames_covered": int(merged.loc[scored, "frames"].sum()),
        "frames_total": int(merged["frames"].sum()),
        "texts_matching": int(match.sum()),
        "frames_matching": int(merged.loc[match, "frames"].sum()),
        "text_match_rate": float(match.sum() / scored.sum()) if scored.any() else float("nan"),
        "frame_match_rate": float(merged.loc[match, "frames"].sum() / merged.loc[scored, "frames"].sum())
        if scored.any()
        else float("nan"),
        "mismatches": [
            {"text": t, "frames": int(r.frames), "stored": r.stored, "rederived": r.category, "top_score": r.top_score}
            for t, r in mism.head(25).iterrows()
        ],
        "texts_within_0.001_of_threshold": int(near.sum()),
    }


def _labels_at_threshold(relabel: pd.DataFrame, threshold: float | None) -> pd.Series:
    """Re-derive the pipeline label of every text at a different confidence threshold."""
    score_cols = [f"score_{label}" for label in LABELS]
    scores = relabel[score_cols].to_numpy(dtype=float)
    top = np.asarray(LABELS, dtype=object)[np.nanargmax(np.nan_to_num(scores, nan=-1), axis=1)]
    top_score = np.nanmax(np.nan_to_num(scores, nan=-1), axis=1)
    if threshold is None:
        lab = top
    else:
        lab = np.where(top_score >= threshold, top, NONMILITARY)
    lab = np.where(relabel["source"].eq("keyword"), "ground_vehicles", lab)
    return pd.Series(lab, index=relabel["text"].to_numpy())


def threshold_sensitivity(frames: pd.DataFrame, relabel: pd.DataFrame, thresholds: Sequence[float]) -> dict:
    texts_scored = set(relabel["text"])
    f = frames[frames["florence_text"].isin(texts_scored)]
    h = f[f["human_label"].notna()]
    rows = []
    for th in [*thresholds, None]:
        lab = _labels_at_threshold(relabel, th)
        pred = f["florence_text"].map(lab)
        cm = confusion_matrix(f["llm_label"], pred, LABELS)
        cmb = confusion_matrix(to_binary(f["llm_label"]), to_binary(pred), BINARY_LABELS)
        hp = h["florence_text"].map(lab)
        rows.append(
            {
                "threshold": th,
                "military_share": float((pred != NONMILITARY).mean()),
                "agreement_with_llm": float(np.trace(cm) / cm.sum()),
                "kappa_with_llm": cohen_kappa(cm),
                "binary_kappa_with_llm": cohen_kappa(cmb),
                "accuracy_on_human_reviewed": float((hp == h["human_label"]).mean()) if len(h) else float("nan"),
                "distribution": {label: float((pred == label).mean()) for label in LABELS},
            }
        )
    return {
        "frames_covered": len(f),
        "human_reviewed_frames_covered": len(h),
        "note": "accuracy_on_human_reviewed is computed only on the non-random adjudicated subset",
        "rows": rows,
    }


def keyword_counterfactual(frames: pd.DataFrame, relabel: pd.DataFrame, threshold: float = 0.8) -> dict:
    """What BART alone would have said about the frames the keyword rule forced to ground_vehicles."""
    kw = relabel[relabel["source"].eq("keyword")].copy()
    if kw.empty or kw["top_label"].isna().all():
        return {}
    kw["bart_only"] = np.where(kw["top_score"] >= threshold, kw["top_label"], NONMILITARY)
    weights = frames["florence_text"].value_counts()
    kw["frames"] = kw["text"].map(weights).fillna(0).astype(int)
    dist = kw.groupby("bart_only")["frames"].sum()
    return {
        "frames": int(kw["frames"].sum()),
        "bart_only_distribution": {label: int(dist.get(label, 0)) for label in LABELS},
    }


def segmentation_ablation(frames: pd.DataFrame, legacy: pd.DataFrame, segmented: pd.DataFrame) -> dict:
    common = set(legacy["text"]) & set(segmented["text"])
    f = frames[frames["florence_text"].isin(common)]
    leg = legacy.set_index("text")["category"]
    seg = segmented.set_index("text")["category"]
    a = f["florence_text"].map(leg)
    b = f["florence_text"].map(seg)
    h = f["human_label"].notna()
    cm_ab = confusion_matrix(a, b, LABELS)
    cm_la = confusion_matrix(f["llm_label"], a, LABELS)
    cm_lb = confusion_matrix(f["llm_label"], b, LABELS)
    texts = f.drop_duplicates("florence_text")
    return {
        "frames_covered": len(f),
        "texts_covered": len(common),
        "frames_changed": int((a != b).sum()),
        "texts_changed": int((texts["florence_text"].map(leg) != texts["florence_text"].map(seg)).sum()),
        "legacy_vs_segmented_confusion": cm_ab.astype(int).tolist(),
        "kappa_with_llm_legacy": cohen_kappa(cm_la),
        "kappa_with_llm_segmented": cohen_kappa(cm_lb),
        "agreement_with_llm_legacy": float(np.trace(cm_la) / cm_la.sum()),
        "agreement_with_llm_segmented": float(np.trace(cm_lb) / cm_lb.sum()),
        "accuracy_on_human_reviewed_legacy": float((a[h] == f.loc[h, "human_label"]).mean()) if h.any() else None,
        "accuracy_on_human_reviewed_segmented": float((b[h] == f.loc[h, "human_label"]).mean()) if h.any() else None,
        "distribution_legacy": {label: float((a == label).mean()) for label in LABELS},
        "distribution_segmented": {label: float((b == label).mean()) for label in LABELS},
        "largest_changes": (
            f[a != b]
            .assign(legacy=a[a != b], segmented=b[a != b])
            .groupby(["florence_text", "legacy", "segmented"])
            .size()
            .sort_values(ascending=False)
            .head(15)
            .reset_index(name="frames")
            .to_dict("records")
        ),
    }
