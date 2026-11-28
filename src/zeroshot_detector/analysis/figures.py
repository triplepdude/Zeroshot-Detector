"""Static figures for the analysis report (matplotlib, light theme).

Colours follow one fixed entity → colour mapping across every figure
(validated categorical palette: adjacent-pair CVD ΔE ≥ 11.8); segments are
separated by 2 px surface gaps and labelled directly, so identity never relies
on colour alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..taxonomy import LABELS

SURFACE = "#ffffff"
INK = "#191919"
INK_2 = "#666666"
GRID = "#e7e7e7"
AXIS = "#c2c2c2"
NEUTRAL = "#b4b4b4"
CATEGORY_COLORS = {
    "weapons": "#257de0",
    "aircraft": "#f47230",
    "maritime": "#00b894",
    "ground_vehicles": "#f2a600",
    "nonmilitary": NEUTRAL,
}
#: Fill colours that need dark text for labels drawn inside them.
DARK_TEXT_FILLS = {"#f2a600", NEUTRAL, "#00b894", "#f47230"}
SEQUENTIAL = ["#ffffff", "#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]
PRETTY = {
    "weapons": "Weapons",
    "aircraft": "Aircraft",
    "maritime": "Maritime",
    "ground_vehicles": "Ground vehicles",
    "nonmilitary": "Non-military",
}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def confusion_figure(
    cm: list[list[int]], labels: list[str], path: Path, title: str, row_name: str, col_name: str
) -> Path:
    from matplotlib.colors import LinearSegmentedColormap

    plt = _plt()
    cm_arr = np.asarray(cm, dtype=float)
    row_share = cm_arr / np.maximum(cm_arr.sum(axis=1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.imshow(row_share, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            share = row_share[i, j]
            color = "#ffffff" if share > 0.55 else INK
            ax.text(j, i - 0.12, f"{int(cm_arr[i, j]):,}", ha="center", va="center", fontsize=8, color=color)
            ax.text(j, i + 0.2, f"{share:.1%}", ha="center", va="center", fontsize=7, color=color)
    names = [PRETTY.get(label, label) for label in labels]
    ax.set_xticks(range(len(labels)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), names)
    ax.set_xlabel(col_name)
    ax.set_ylabel(row_name)
    ax.set_title(title, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.text(
        0,
        -0.32,
        "Cell: frame count and share of the row. Darker = larger share of the row.",
        transform=ax.transAxes,
        color=INK_2,
        fontsize=7,
    )
    return _save(fig, path)


def stacked_share_figure(
    rows: list[tuple[str, dict[str, float]]],
    segments: list[tuple[str, str, str]],
    path: Path,
    title: str,
    subtitle: str,
    xlabel: str = "Share",
    xmax: float = 1.0,
) -> Path:
    """Horizontal 100 %-or-less stacked bars. ``segments`` = [(key, display name, colour)]."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(rows) + 1.4))
    y = np.arange(len(rows))[::-1]
    for (_, shares), yy in zip(rows, y, strict=True):
        left = 0.0
        for key, _name, color in segments:
            w = shares.get(key, 0.0)
            if w <= 0:
                continue
            ax.barh(yy, w, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2)
            if w >= 0.06:
                ax.text(
                    left + w / 2,
                    yy,
                    f"{w:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=INK if color in DARK_TEXT_FILLS else "#ffffff",
                )
            left += w
        ax.text(left + 0.01, yy, f"{left:.1%}" if left < 0.995 else "", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, [name for name, _ in rows])
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title(title, loc="left", pad=24)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in segments]
    ax.legend(
        handles,
        [n for _, n, _ in segments],
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=len(segments),
        frameon=False,
        fontsize=8,
        handlelength=1.0,
        borderaxespad=0.2,
    )
    fig.text(0.01, -0.03, subtitle, color=INK_2, fontsize=7)
    return _save(fig, path)


def threshold_figure(rows: list[dict], path: Path, study_threshold: float = 0.8) -> Path:
    plt = _plt()
    pts = [r for r in rows if r["threshold"] is not None]
    th = [r["threshold"] for r in pts]
    panels = [
        ("military_share", "Frames labelled military", lambda v: f"{v:.0%}"),
        ("kappa_with_llm", "Cohen's κ with LLM labels (5 classes)", lambda v: f"{v:.2f}"),
        ("accuracy_on_human_reviewed", "Accuracy on adjudicated frames*", lambda v: f"{v:.0%}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
    for ax, (key, name, fmt) in zip(axes, panels, strict=True):
        vals = [r[key] for r in pts]
        ax.plot(th, vals, color=CATEGORY_COLORS["weapons"], linewidth=2)
        ax.scatter(th, vals, s=14, color=CATEGORY_COLORS["weapons"], zorder=3, edgecolor=SURFACE, linewidth=1)
        ax.axvline(study_threshold, color=INK_2, linestyle=(0, (3, 3)), linewidth=1)
        ax.text(study_threshold, ax.get_ylim()[1], " study\n setting", va="top", fontsize=7, color=INK_2)
        ax.set_title(name, loc="left", fontsize=9)
        ax.set_xlabel("Zero-shot confidence threshold")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _, f=fmt: f(v)))
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    fig.text(
        0.01,
        -0.06,
        "*Adjudicated frames are the non-random subset where the pipeline said military and the LLM said non-military.",
        color=INK_2,
        fontsize=7,
    )
    fig.tight_layout()
    return _save(fig, path)


def video_rules_figure(distributions: dict[str, dict[str, dict]], path: Path) -> Path:
    military = [label for label in LABELS if label != "nonmilitary"]
    rows = [(name, {label: d[label]["share"] for label in military}) for name, d in distributions.items()]
    segments = [(label, PRETTY[label], CATEGORY_COLORS[label]) for label in military]
    return stacked_share_figure(
        rows,
        segments,
        path,
        title="Share of videos labelled military, by aggregation rule",
        subtitle="Same frame labels, different video-level rules. Number at bar end = total military share.",
        xlabel="Share of video files",
        xmax=min(1.0, max(sum(r.values()) for _, r in rows) + 0.08),
    )


def trend_figure(by_year: dict, path: Path, min_videos: int = 100) -> Path:
    """Share of videos labelled military by post year (pipeline and LLM), with 95 % Wilson intervals."""
    plt = _plt()
    years = [int(y) for y, r in by_year.items() if r["videos"] >= min_videos]
    series = [
        ("pipeline_military", "Pipeline", CATEGORY_COLORS["weapons"]),
        ("llm_military", "LLM", CATEGORY_COLORS["aircraft"]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for key, name, color in series:
        est = [by_year[str(y) if str(y) in by_year else y][key]["estimate"] for y in years]
        lo = [by_year[str(y) if str(y) in by_year else y][key]["ci_low"] for y in years]
        hi = [by_year[str(y) if str(y) in by_year else y][key]["ci_high"] for y in years]
        ax.fill_between(years, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(years, est, color=color, linewidth=2, label=name)
        ax.scatter(years, est, s=16, color=color, zorder=3, edgecolor=SURFACE, linewidth=1)
        ax.text(years[-1] + 0.12, est[-1], name, color=INK, fontsize=8, va="center")
    ax.set_xticks(years, [str(y) for y in years])
    ax.set_xlim(years[0] - 0.3, years[-1] + 0.9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, None)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Post year (decoded from the post id)")
    ax.set_title("Share of videos labelled military, by post year", loc="left", pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8, borderaxespad=0.2)
    fig.text(
        0.01,
        -0.04,
        f"Plurality vote per video, one file per post, years with >= {min_videos} videos. Bands: 95% Wilson intervals. "
        "Descriptive only: the sampling of videos is undocumented.",
        color=INK_2,
        fontsize=7,
    )
    return _save(fig, path)
