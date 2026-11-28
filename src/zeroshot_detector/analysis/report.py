"""``zsd analyze``: run every analysis and write ``results/``.

Outputs
-------
``results/summary.json``
    every number reported, with confidence intervals and provenance.
``results/REPORT.md``
    human-readable report generated from ``summary.json`` (do not edit by hand).
``results/figures/*.png``
    figures referenced by the report (when matplotlib is installed).
"""

from __future__ import annotations

import json
import logging
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__
from ..data import FRAMES_FILE, REVIEW_SESSION_FILE, VIDEOS_FILE, load_frames, load_review_session, load_videos, sha256
from ..segment import build_vocabulary, default_base_vocabulary, segment_all
from ..taxonomy import LABELS
from . import stats as S

log = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = tuple(round(x, 2) for x in np.arange(0.30, 0.96, 0.05))
LEGACY_RELABEL = "legacy_bart-large-mnli.csv.gz"
SEGMENTED_RELABEL = "segmented_bart-large-mnli.csv.gz"


def _load_relabel(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, keep_default_na=False, na_values={"matched_keyword": [""], "top_label": [""]})
    for col in [c for c in df.columns if c.startswith("score_")] + ["top_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _git_commit(repo: Path) -> str | None:
    import subprocess

    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def run_analysis(
    data_dir: str | Path,
    out_dir: str | Path,
    relabel_dir: str | Path | None = None,
    n_boot: int = 2000,
    seed: int = 2025,
    figures: bool = True,
) -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    relabel_dir = Path(relabel_dir) if relabel_dir else out_dir / "relabel"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = S.add_post_ids(load_frames(data_dir))
    videos = load_videos(data_dir)
    session = load_review_session(data_dir)
    texts = frames["florence_text"].value_counts()
    segmentation = segment_all(texts.index, build_vocabulary(texts.index))
    base = default_base_vocabulary()
    verified = base | set(texts.index)
    seg_ok = [t for t, parts in segmentation.items() if all(p in verified for p in parts)]

    summary: dict = {
        "provenance": {
            "package_version": __version__,
            "git_commit": _git_commit(Path(__file__).resolve().parents[3]),
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "n_boot": n_boot,
            "seed": seed,
            "confidence": 0.95,
            "cluster_unit": "post id (files with the same post id form one cluster)",
            "data_sha256": {name: sha256(data_dir / name) for name in (FRAMES_FILE, VIDEOS_FILE, REVIEW_SESSION_FILE)},
        }
    }
    log.info("dataset summary")
    summary["dataset"] = S.dataset_summary(frames, videos)
    summary["duplicates"] = S.duplicate_summary(frames)
    summary["segmentation"] = {
        "texts": len(segmentation),
        "texts_split": sum(1 for p in segmentation.values() if len(p) > 1),
        "frames_all_parts_known": int(texts[seg_ok].sum()),
        "frames_all_parts_known_share": float(texts[seg_ok].sum() / texts.sum()),
        "distinct_object_labels": len({p for parts in segmentation.values() for p in parts}),
    }
    summary["keyword_rule"] = S.keyword_audit(frames, segmentation)
    log.info("agreement")
    summary["agreement"] = S.agreement(frames, n_boot, seed)
    log.info("human review")
    summary["auto_correction"] = S.auto_correction(frames)
    summary["human_review"] = S.human_review(frames, session, n_boot, seed)
    summary["precision_estimates"] = S.discrepant_precision(frames, n_boot, seed)
    log.info("video level")
    summary["video_level"] = S.video_level(frames, videos, n_boot, seed)

    legacy = _load_relabel(relabel_dir / LEGACY_RELABEL)
    segmented = _load_relabel(relabel_dir / SEGMENTED_RELABEL)
    if legacy is not None:
        log.info("reproduction / threshold sensitivity")
        summary["reproduction"] = S.reproduction(frames, legacy)
        summary["threshold_sensitivity"] = S.threshold_sensitivity(frames, legacy, DEFAULT_THRESHOLDS)
        summary["keyword_counterfactual"] = S.keyword_counterfactual(frames, legacy)
        meta = relabel_dir / (LEGACY_RELABEL.split(".")[0] + ".meta.json")
        if meta.exists():
            summary["reproduction"]["run"] = json.loads(meta.read_text())
    if legacy is not None and segmented is not None:
        log.info("separator ablation")
        summary["separator_ablation"] = S.segmentation_ablation(frames, legacy, segmented)
        meta = relabel_dir / (SEGMENTED_RELABEL.split(".")[0] + ".meta.json")
        if meta.exists():
            summary["separator_ablation"]["run"] = json.loads(meta.read_text())

    figure_paths = make_figures(summary, out_dir / "figures") if figures else {}
    (out_dir / "summary.json").write_text(json.dumps(_clean(summary), indent=2, sort_keys=False) + "\n")
    (out_dir / "REPORT.md").write_text(render_markdown(summary, figure_paths, out_dir))
    return summary


# -- figures -----------------------------------------------------------------------
def make_figures(summary: dict, fig_dir: Path) -> dict[str, Path]:
    try:
        import matplotlib  # noqa: F401
    except ImportError:  # pragma: no cover - optional dependency
        log.warning("matplotlib not installed; skipping figures")
        return {}
    from . import figures as F

    paths = {}
    fl = summary["agreement"]["frame_level"]
    paths["confusion"] = F.confusion_figure(
        fl["confusion_matrix"],
        fl["labels"],
        fig_dir / "confusion_llm_vs_pipeline.png",
        "Frame labels: LLM (rows) vs. pipeline (columns)",
        "LLM label",
        "Pipeline label",
    )
    hr = summary["human_review"]
    rows = []
    for key, name in [
        ("weapons->nonmilitary", "Weapons vs. non-military"),
        ("maritime->nonmilitary", "Maritime vs. non-military"),
    ]:
        if key in hr["by_cell"]:
            for level, level_name in [("frame_level", "frames"), ("text_level", "distinct texts")]:
                d = hr["by_cell"][key][level]
                rows.append((f"{name} ({level_name})", {k: v["estimate"] for k, v in d.items()}))
    for level, level_name in [("frame_level", "frames"), ("text_level", "distinct texts")]:
        d = hr["pooled"][level]
        rows.append((f"All adjudicated ({level_name})", {k: v["estimate"] for k, v in d.items()}))
    paths["human"] = F.stacked_share_figure(
        rows,
        [
            ("sides_with_pipeline", "Human agrees with pipeline", F.CATEGORY_COLORS["weapons"]),
            ("sides_with_llm", "Human agrees with LLM", F.CATEGORY_COLORS["aircraft"]),
            ("neither", "Neither", F.NEUTRAL),
        ],
        fig_dir / "human_adjudication.png",
        "Human adjudication of pipeline-military / LLM-non-military disagreements",
        "Pipeline label → LLM label. Reviewers saw only the Florence-2 text and both labels (not the image).",
    )
    by_year = summary["video_level"]["military_share_by_post_year"]
    if sum(r["videos"] >= 100 for r in by_year.values()) >= 2:
        paths["trend"] = F.trend_figure(by_year, fig_dir / "military_share_by_year.png")
    paths["video_rules"] = F.video_rules_figure(
        summary["video_level"]["distributions"], fig_dir / "video_level_rules.png"
    )
    if "threshold_sensitivity" in summary:
        paths["threshold"] = F.threshold_figure(
            summary["threshold_sensitivity"]["rows"], fig_dir / "threshold_sensitivity.png"
        )
    return paths


# -- markdown --------------------------------------------------------------------------
def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def pct(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{100 * x:.{digits}f}%"


def num(x: float | None, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{digits}f}"


def ci(d: dict, fmt=num) -> str:
    return f"{fmt(d['estimate'])} ({fmt(d['ci_low'])}–{fmt(d['ci_high'])})"


def ci_pct(d: dict) -> str:
    return ci(d, lambda x: pct(x))


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _code(text: str) -> str:
    return "`" + text.replace("|", "\\|") + "`"


def render_markdown(s: dict, figures: dict[str, Path], out_dir: Path) -> str:
    def fig(key: str, alt: str) -> str:
        if key not in figures:
            return ""
        return f"\n![{alt}]({figures[key].relative_to(out_dir).as_posix()})\n"

    d, dup, kw = s["dataset"], s["duplicates"], s["keyword_rule"]
    ag, hr, pe, vl = s["agreement"], s["human_review"], s["precision_estimates"], s["video_level"]
    prov = s["provenance"]
    L: list[str] = []
    w = L.append
    w("# Analysis of the archived study data")
    w("")
    w(
        "> Generated by `zsd analyze` from `data/` — do not edit by hand. Every number below is also in "
        "[`summary.json`](summary.json). Interpretation, caveats and the definitions of each quantity are in "
        "[`docs/methodology.md`](../docs/methodology.md)."
    )
    w("")
    w(
        f"Package {prov['package_version']} · bootstrap: {prov['n_boot']} replicates, seed {prov['seed']}, "
        f"95% percentile intervals, resampling {prov['cluster_unit']} · "
        f"`frames.csv.gz` sha256 `{prov['data_sha256'][FRAMES_FILE][:16]}…`"
    )
    w("")
    w(
        "Terminology: **pipeline** = Florence-2 `<OD>` + keyword rule + BART-MNLI (the system studied); "
        "**LLM** = the OpenAI-model label used as a second annotator (*not* ground truth); "
        "**human** = labels from the review web app (only for a non-random subset)."
    )

    # 1 data
    w("\n## 1. Data\n")
    fpf = d["frames_per_file"]
    w(
        table(
            ["Quantity", "Value"],
            [
                ["Sampled frames (rows)", f"{d['n_frames']:,}"],
                ["Video files processed", f"{d['n_files']:,} ({d['n_files_without_frames']} yielded no frames)"],
                ["Distinct posts (after removing batch prefix)", f"{d['n_posts']:,}"],
                ["Collection batches (file-name prefixes)", d["n_batches"]],
                ["Distinct Florence-2 strings", f"{d['n_unique_texts']:,}"],
                [
                    "Frames per file: median (IQR), max",
                    f"{fpf['median']:.0f} ({fpf['p25']:.0f}–{fpf['p75']:.0f}), {fpf['max']:,}",
                ],
                ["Frames with `No object detected.`", f"{d['frames_no_object_detected']:,}"],
                [
                    "Sampling interval",
                    "every 30th decoded frame"
                    + (" (all frame numbers are multiples of 30)" if d["all_frame_numbers_multiple_of_30"] else ""),
                ],
                [
                    "Samples skipped by de-duplication / errors",
                    f"{d['sampling_gaps']['skipped_samples']:,.0f} ({pct(d['sampling_gaps']['skipped_share'], 2)})",
                ],
                [
                    "Each string always has the same pipeline / LLM label",
                    f"{d['texts_with_one_pipeline_label']} / {d['texts_with_one_llm_label']}",
                ],
                [
                    "Human-labelled frames (distinct strings)",
                    f"{d['n_human_labelled_frames']:,} ({d['n_human_labelled_texts']})",
                ],
            ],
        )
    )
    pdates = d["post_dates"]
    if pdates["min"]:
        w("")
        w(
            f"**Post dates.** Posts were created between {pdates['min'][:10]} and {pdates['max'][:10]} "
            f"(median {pdates['median'][:10]}). Creation times are not stored in the data; they are decoded from the "
            "post ids (the upper 32 bits of a TikTok id are the Unix creation time). The numeric file-name prefixes are "
            "chronological collection batches: "
            + ", ".join(f"`{b}` {r['min'][:7]}–{r['max'][:7]}" for b, r in pdates["by_batch"].items() if b != "<none>")
            + "; files without a prefix span the whole period."
        )
    w("")
    w(
        f"**Duplicate posts.** {dup['n_posts_in_multiple_files']:,} posts were processed twice (same post id under "
        f"two batch prefixes); in all {dup['posts_with_identical_texts_across_files']:,} cases the frame-by-frame "
        f"outputs are identical, so {dup['redundant_frames']:,} frames "
        f"({pct(dup['redundant_frames'] / d['n_frames'])}) are exact duplicates. Duplicated pairs: "
        + ", ".join(f"{k} ({v})" for k, v in list(dup["duplicate_batch_pairs"].items())[:4])
        + " — files without a prefix that were also collected in a dated batch. Bootstrap clusters are formed by "
        "post, so duplicates do not narrow the intervals."
    )
    w("")
    w("Label distributions (frames):\n")
    w(
        table(
            ["Label", "Pipeline", "LLM", "Human (adjudicated subset)"],
            [
                [
                    label,
                    f"{d['pipeline_distribution'][label]['n']:,} ({pct(d['pipeline_distribution'][label]['share'])})",
                    f"{d['llm_distribution'][label]['n']:,} ({pct(d['llm_distribution'][label]['share'])})",
                    f"{d['human_distribution'][label]['n']:,}",
                ]
                for label in LABELS
            ],
        )
    )

    # 2 reproduction
    if "reproduction" in s:
        r = s["reproduction"]
        w("\n## 2. Reproducibility of the stored pipeline labels\n")
        w(
            f"The stored label of every frame is a deterministic function of its Florence-2 string, so the legacy "
            f"decision rule (`FrameClassifier(legacy=True)`, BART-MNLI, threshold 0.8) was re-run on the distinct "
            f"strings. Coverage: {r['texts_scored']:,}/{r['texts_total']:,} strings "
            f"({pct(r['frames_covered'] / r['frames_total'])} of frames)."
        )
        w("")
        w(
            f"* Re-derived label equals the stored label for **{r['texts_matching']:,}/{r['texts_scored']:,} strings "
            f"({pct(r['text_match_rate'], 2)})**, i.e. {pct(r['frame_match_rate'], 3)} of covered frames."
        )
        w(
            f"* Strings whose top score is within 0.001 of the threshold (numerically fragile): {r['texts_within_0.001_of_threshold']}."
        )
        if r.get("run"):
            w(
                f"* Model: `{r['run'].get('model')}` @ `{r['run'].get('revision')}`; transformers {r['run'].get('transformers')}, torch {r['run'].get('torch')}."
            )
        if r["mismatches"]:
            w("\nLargest mismatches:\n")
            w(
                table(
                    ["String", "Frames", "Stored", "Re-derived", "Top score"],
                    [
                        [_code(m["text"]), m["frames"], m["stored"], m["rederived"], num(m["top_score"])]
                        for m in r["mismatches"][:10]
                    ],
                )
            )

    # 3 keyword rule
    w("\n## 3. Keyword rule audit\n")
    fired = {k: v for k, v in kw["keywords"].items() if v["frames"] > 0}
    w(
        f"The original rule forces `ground_vehicles` when any of {len(kw['keywords'])} keywords occurs as a substring "
        f"of the lower-cased string. Only {len(fired)} keyword(s) ever fired: "
        + ", ".join(f"`{k}` ({v['frames']:,} frames, {v['texts']} strings)" for k, v in fired.items())
        + f". {len(kw['keywords_that_can_never_match'])} keywords contain capitals and can never match "
        f"({', '.join('`' + k + '`' for k in kw['keywords_that_can_never_match'])}). "
        f"All {kw['frames_pipeline_ground_vehicles']:,} pipeline `ground_vehicles` frames come from the keyword rule "
        f"({kw['ground_vehicles_frames_from_keyword']:,}); BART-MNLI never assigned that class at ≥ 0.8. "
        f"`tank top` (clothing) triggers the rule in {kw['tank_top']['frames']} frames."
    )
    if "corrected_rule" in kw:
        c = kw["corrected_rule"]
        w("")
        w(
            f"The corrected rule (whole-word, per object label, with exclusions such as `tank top`) applied to the "
            f"re-segmented strings fires on {c['frames_fired']:,} frames: {c['frames_no_longer_fired']:,} frames no "
            f"longer fire and {c['frames_newly_fired']:,} newly fire."
        )
    if s.get("keyword_counterfactual"):
        kc = s["keyword_counterfactual"]
        w("")
        w(
            f"Without the keyword rule, BART-MNLI (threshold 0.8) would have labelled the {kc['frames']:,} keyword frames: "
            + ", ".join(f"{k} {v:,}" for k, v in kc["bart_only_distribution"].items() if v)
            + "."
        )

    # 4 agreement
    fl, fb, tl = ag["frame_level"], ag["frame_level_binary"], ag["text_level"]
    vp, vb = vl["pipeline_vs_llm_plurality"], vl["pipeline_vs_llm_plurality_binary"]
    w("\n## 4. Agreement between pipeline and LLM labels\n")
    w("Agreement is not accuracy: the LLM labelled the same Florence-2 strings and can be wrong in the same way.\n")
    w(
        table(
            ["Unit", "n", "Observed agreement", "Cohen's κ", "Macro-F1 (LLM as reference)"],
            [
                [
                    "Frames, 5 classes",
                    f"{int(fl['metrics']['n']):,}",
                    ci_pct(fl["ci"]["agreement"]),
                    ci(fl["ci"]["kappa"]),
                    ci(fl["ci"]["macro_f1"]),
                ],
                [
                    "Frames, military vs. not",
                    f"{int(fb['metrics']['n']):,}",
                    ci_pct(fb["ci"]["agreement"]),
                    ci(fb["ci"]["kappa"]),
                    ci(fb["ci"]["macro_f1"]),
                ],
                [
                    "Distinct strings, 5 classes",
                    f"{int(tl['metrics']['n']):,}",
                    ci_pct(tl["ci"]["agreement"]),
                    ci(tl["ci"]["kappa"]),
                    ci(tl["ci"]["macro_f1"]),
                ],
                [
                    "Videos (plurality vote), 5 classes",
                    f"{int(vp['metrics']['n']):,}",
                    ci_pct(vp["ci"]["agreement"]),
                    ci(vp["ci"]["kappa"]),
                    ci(vp["ci"]["macro_f1"]),
                ],
                [
                    "Videos (plurality vote), military vs. not",
                    f"{int(vb['metrics']['n']):,}",
                    ci_pct(vb["ci"]["agreement"]),
                    ci(vb["ci"]["kappa"]),
                    ci(vb["ci"]["macro_f1"]),
                ],
            ],
        )
    )
    w("")
    w(
        "Frame-level agreement is dominated by a few very frequent, easy strings; weighting every distinct string "
        "once gives a much lower κ."
    )
    w("\nPer class (frames; precision = share of pipeline-`c` frames the LLM also called `c`):\n")
    w(
        table(
            ["Class", "Pipeline frames", "LLM frames", "Precision", "Recall", "F1"],
            [
                [
                    label,
                    f"{int(m['predicted']):,}",
                    f"{int(m['support']):,}",
                    pct(m["precision"]),
                    pct(m["recall"]),
                    num(m["f1"]),
                ]
                for label, m in fl["metrics"]["per_class"].items()
            ],
        )
    )
    w(fig("confusion", "Confusion matrix of LLM vs pipeline frame labels"))
    w("\n### Disagreements\n")
    w(
        f"{ag['n_disagreement_frames']:,} frames disagree; {ag['n_disagreement_frames_unreviewed']:,} of them "
        f"({pct(ag['n_disagreement_frames_unreviewed'] / ag['n_disagreement_frames'])}) were never seen by a human.\n"
    )
    w(
        table(
            ["Pipeline → LLM", "Frames", "Strings", "Posts", "Human-reviewed", "Most frequent strings (frames)"],
            [
                [
                    f"{c['pipeline_label']} → {c['llm_label']}",
                    f"{c['frames']:,}",
                    c["texts"],
                    c["posts"],
                    f"{c['human_reviewed_frames']:,}",
                    "; ".join(f"{_code(t['florence_text'])} ({t['frames']:,})" for t in c["top_texts"][:3]),
                ]
                for c in ag["disagreement_cells"]
                if c["frames"] >= 20
            ],
        )
    )

    # 5 auto-correction + human
    ac = s["auto_correction"]
    w("\n## 5. How the original study resolved disagreements\n")
    w(
        f'Disagreements were first passed through keyword "auto-correction" rules (`legacy/post_processing/'
        f"prioritize_and_automate.py`, substring matching): {ac['auto_resolved_frames']:,} of "
        f"{ac['disagreement_frames']:,} disagreement frames ({pct(ac['auto_resolved_frames'] / ac['disagreement_frames'])}) "
        f"were resolved automatically and the remaining {ac['sent_to_humans_frames']:,} went to human review "
        f"(the reviewed set is exactly the unmatched set: {ac['human_reviewed_equals_unmatched']}). "
        f"The automatic label equals the **LLM** label for {ac['auto_label_equals_llm']:,} frames and the pipeline "
        f"label for {ac['auto_label_equals_pipeline']:,}: the rules never sided with the pipeline. They also encode a "
        "taxonomy decision — every truck, boat or plane is military:\n"
    )
    w(
        table(
            ["Rule keyword", "→ label", "Frames resolved", "Most frequent strings"],
            [
                [f"`{kw}`", h["category"], f"{h['frames']:,}", "; ".join(_code(t) for t in h["examples"])]
                for kw, h in sorted(ac["keyword_hits"].items(), key=lambda kv: -kv[1]["frames"])
            ],
        )
    )

    w("\n## 6. Human adjudication\n")
    cov = ", ".join(
        f"{c['pipeline_label']} → {c['llm_label']} ({c['frames_reviewed']:,}/{c['frames_in_cell']:,} frames)"
        for c in hr["cells_reviewed"]
    )
    w(
        f"Human labels exist for {hr['n_frames']:,} frames ({hr['n_texts']} distinct strings). They cover exactly the "
        f"disagreement cells where the pipeline said *military* and the LLM said *non-military*: {cov}. No agreement "
        "cell and no other disagreement cell was reviewed, so these labels cannot estimate overall accuracy or recall. "
        "Reviewers judged the Florence-2 string (not the image) and saw both system labels."
    )
    w("")
    rows = []
    for key, cell in hr["by_cell"].items():
        for level, unit in [("frame_level", "frames"), ("text_level", "strings")]:
            x = cell[level]
            rows.append(
                [
                    key.replace("->", " → "),
                    f"{cell['frames' if level == 'frame_level' else 'texts']:,} {unit}",
                    ci_pct(x["sides_with_pipeline"]),
                    ci_pct(x["sides_with_llm"]),
                    ci_pct(x["neither"]),
                ]
            )
    for level, unit in [("frame_level", "frames"), ("text_level", "strings")]:
        x = hr["pooled"][level]
        rows.append(
            [
                "**all adjudicated**",
                f"{hr['pooled']['frames' if level == 'frame_level' else 'texts']:,} {unit}",
                ci_pct(x["sides_with_pipeline"]),
                ci_pct(x["sides_with_llm"]),
                ci_pct(x["neither"]),
            ]
        )
    w(table(["Cell", "n", "Human = pipeline", "Human = LLM", "Neither"], rows))
    w("")
    w("Frame-level intervals: post-cluster bootstrap. String-level intervals: Wilson score.")
    w(fig("human", "Human adjudication shares"))
    sf = hr["session_file"]
    conflicts = sf["texts_with_conflicting_labels_within_session"]
    w(
        f"\n**Label quality.** The raw session file `reviewed_mismatches.csv` ({sf['rows']:,} rows, {sf['texts']} strings) "
        f"agrees with the merged labels for {sf['texts_consistent_with_merged_labels']}/{sf['texts_also_in_merged_labels']} strings; "
        + (
            "strings with conflicting labels inside the session: "
            + ", ".join(f"{_code(t)} ({', '.join(v)})" for t, v in conflicts.items())
            if conflicts
            else "no string has conflicting labels inside the session"
        )
        + ". Examples where the human label matches neither system (several look implausible, e.g. `paddle` → aircraft):\n"
    )
    w(
        table(
            ["String", "Pipeline", "LLM", "Human", "Frames"],
            [
                [_code(e["florence_text"]), e["pipeline_label"], e["llm_label"], e["human_label"], e["frames"]]
                for e in hr["neither_examples"][:8]
            ],
        )
    )

    # 6 precision
    w("\n## 7. Estimated precision of the pipeline's military labels\n")
    w(
        "Discrepant-analysis estimate: frames where the LLM agrees are *assumed* correct, adjudicated disagreements "
        "use the human label, unadjudicated disagreements are excluded (point) or counted as all-wrong / all-right "
        "(bounds). Because agreement is assumed correct, these estimates are biased upwards.\n"
    )
    w(
        table(
            [
                "Pipeline label",
                "Frames",
                "LLM agrees",
                "Adjudicated (human = pipeline)",
                "Unadjudicated",
                "Precision (95% CI)",
                "Bounds",
            ],
            [
                [
                    c,
                    f"{x['frames']:,}",
                    f"{x['llm_agrees']:,}",
                    f"{x['adjudicated']:,} ({x['adjudicated_correct']:,})",
                    f"{x['unadjudicated_disagreements']:,}",
                    ci_pct(x["point"]),
                    f"{pct(x['lower_bound'])}–{pct(x['upper_bound'])}",
                ]
                for c, x in pe.items()
            ],
        )
    )
    w("")
    w(
        "`ground_vehicles` shows 100% only because the LLM never contradicted a `tank` frame and none was reviewed; "
        "the `tank top` frames in §3 are known false positives."
    )

    # 7 video level
    w("\n## 8. Video-level labels and the aggregation rule\n")
    w(
        f"Per-file frame counts recomputed from `frames.csv.gz` match the stored per-video summary: "
        f"{vl['counts_match_stored_summary']}. Findings about the original aggregation code:\n"
    )
    w(
        f"* The {vl['n_files_without_frames']} files that yielded no frames are labelled "
        f"`{', '.join(vl['legacy_label_of_files_without_frames'])}` by the original rule (`max` over all-zero counts returns the first label)."
    )
    w(
        f"* {vl['ties_military_vs_nonmilitary']} files have a tie between a military label and `nonmilitary`; the original "
        f"rule resolves every such tie towards the military label (label order). This is the only difference between the "
        f"original rule and plurality voting with non-military tie-breaking ({vl['legacy_vs_plurality_changed']} files)."
    )
    w(
        f'* The "more than half non-military → non-military" override compares below-threshold *sampled* frames with half '
        f"of *all decoded* frames (≈30× more), so it can only have fired for files with one or two sampled frames "
        f"(at most {vl['override_possible_files']} files)."
    )
    w("")
    rules = list(vl["distributions"])
    w(
        table(
            ["Rule", *LABELS],
            [
                [
                    r,
                    *[
                        f"{vl['distributions'][r][label]['n']:,} ({pct(vl['distributions'][r][label]['share'])})"
                        for label in LABELS
                    ],
                ]
                for r in rules
            ]
            + [
                [
                    "plurality, one file per post",
                    *[
                        f"{vl['plurality_one_file_per_post'][label]['n']:,} ({pct(vl['plurality_one_file_per_post'][label]['share'])})"
                        for label in LABELS
                    ],
                ]
            ],
        )
    )
    w("")
    w(
        "The share of videos called military ranges from "
        f"{pct(1 - vl['distributions']['plurality']['nonmilitary']['share'])} (plurality) to "
        f"{pct(1 - vl['distributions']['any military frame']['nonmilitary']['share'])} (any military frame): the "
        "video-level conclusion depends more on the aggregation rule than on the frame classifier."
    )
    w(fig("video_rules", "Video-level military share by aggregation rule"))
    w("\n### Military share of videos by post year (descriptive)\n")
    by_year = vl["military_share_by_post_year"]
    w(
        table(
            ["Post year", "Videos", "Pipeline: military", "LLM: military"],
            [
                [y, f"{r['videos']:,}", ci_pct(r["pipeline_military"]), ci_pct(r["llm_military"])]
                for y, r in by_year.items()
            ],
        )
    )
    w("")
    w(
        "Plurality vote per video, one file per post, 95% Wilson intervals. How videos were selected for collection "
        "is not documented, so these shares describe this archive, not the platform."
    )
    w(fig("trend", "Military share by post year"))

    # 8 threshold
    if "threshold_sensitivity" in s:
        ts = s["threshold_sensitivity"]
        w("\n## 9. Sensitivity to the zero-shot confidence threshold\n")
        w(
            f"Labels re-derived from the stored BART-MNLI scores at other thresholds (keyword rule unchanged). "
            f"Coverage: {ts['frames_covered']:,} frames. The last column is computed on the "
            f"{ts['human_reviewed_frames_covered']:,} adjudicated frames only (non-random subset).\n"
        )
        w(
            table(
                [
                    "Threshold",
                    "Military share",
                    "Agreement with LLM",
                    "κ with LLM",
                    "κ military vs. not",
                    "Accuracy on adjudicated",
                ],
                [
                    [
                        "argmax (none)" if r["threshold"] is None else f"{r['threshold']:.2f}",
                        pct(r["military_share"]),
                        pct(r["agreement_with_llm"]),
                        num(r["kappa_with_llm"]),
                        num(r["binary_kappa_with_llm"]),
                        pct(r["accuracy_on_human_reviewed"]),
                    ]
                    for r in ts["rows"]
                ],
            )
        )
        w(fig("threshold", "Threshold sensitivity"))

    # 9 ablation
    if "separator_ablation" in s:
        ab = s["separator_ablation"]
        w("\n## 10. Ablation: label separator and corrected keyword rule\n")
        seg = s["segmentation"]
        w(
            f"The archived strings were re-segmented into object labels ({pct(seg['frames_all_parts_known_share'], 2)} of "
            f"frames split into known labels; see `zeroshot_detector.segment`) and classified with the corrected rule "
            f"(labels joined with `, `, whole-word keyword rule). Coverage: {ab['frames_covered']:,} frames, "
            f"{ab['texts_covered']:,} strings.\n"
        )
        w(
            table(
                ["", "Original (glued strings)", "Corrected (separated labels)"],
                [
                    ["κ with LLM", num(ab["kappa_with_llm_legacy"]), num(ab["kappa_with_llm_segmented"])],
                    [
                        "Agreement with LLM",
                        pct(ab["agreement_with_llm_legacy"]),
                        pct(ab["agreement_with_llm_segmented"]),
                    ],
                    [
                        "Accuracy on adjudicated frames",
                        pct(ab["accuracy_on_human_reviewed_legacy"]),
                        pct(ab["accuracy_on_human_reviewed_segmented"]),
                    ],
                    *[
                        [
                            f"share {label}",
                            pct(ab["distribution_legacy"][label]),
                            pct(ab["distribution_segmented"][label]),
                        ]
                        for label in LABELS
                    ],
                ],
            )
        )
        w(f"\n{ab['frames_changed']:,} frames ({ab['texts_changed']:,} strings) change label. Largest changes:\n")
        w(
            table(
                ["String", "Original", "Corrected", "Frames"],
                [
                    [_code(x["florence_text"]), x["legacy"], x["segmented"], x["frames"]]
                    for x in ab["largest_changes"][:10]
                ],
            )
        )

    w("\n## Caveats\n")
    w(
        "* There is no image-level ground truth. LLM labels are a second opinion produced from the same text; human labels "
        "exist only for one kind of disagreement and were made from text, with both system labels visible.\n"
        "* The prompt and model version used to produce the LLM labels are not recorded in the repository.\n"
        "* Frames are highly correlated within a video; all intervals resample whole posts.\n"
        "* See `docs/methodology.md` (threats to validity) and `docs/audit.md` (defects found in the original code)."
    )
    return "\n".join(L) + "\n"
