"""Canonical study data: loading, legacy export and integrity checks.

The original repository stored the same 1,099,951 frame records in several
overlapping CSV files (one column added per processing stage) plus notebook
checkpoint copies. They are consolidated into two canonical, gzip-compressed
tables plus the raw output of one review session:

``data/frames.csv.gz``
    one row per sampled frame, in the original row order:
    ``video, frame_number, florence_text, pipeline_label, llm_label, human_label``
``data/videos.csv.gz``
    one row per processed video (including videos that yielded no frames),
    in the original order, with per-category frame counts.
``data/reviewed_mismatches.csv``
    raw output of the Flask review app for one session (kept unchanged).

:func:`export_legacy` regenerates every original CSV from the canonical
tables, and :func:`verify_legacy_checksums` checks the result is
byte-identical to the files that were committed originally (SHA-256 recorded
in ``data/legacy_checksums.sha256``).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .taxonomy import LABELS

FRAMES_FILE = "frames.csv.gz"
VIDEOS_FILE = "videos.csv.gz"
REVIEW_SESSION_FILE = "reviewed_mismatches.csv"
LEGACY_CHECKSUMS_FILE = "legacy_checksums.sha256"

FRAME_COLUMNS = ["video", "frame_number", "florence_text", "pipeline_label", "llm_label", "human_label"]

#: Placeholder the original ``*_updated.csv`` used for frames without a human label.
LEGACY_NONE = "None"

#: Auto-correction rules copied from ``post_processing/prioritize_and_automate.py``.
LEGACY_AUTOCORRECT_RULES: dict[str, list[str]] = {
    "ground_vehicles": ["tank", "jeep", "armored vehicle", "truck", "humvee"],
    "maritime": ["ship", "submarine", "boat", "naval", "vessel"],
    "weapons": ["gun", "rifle", "missile", "grenade", "artillery"],
    "aircraft": ["plane", "helicopter", "jet", "drone", "aircraft"],
}
#: Minimum number of mismatched frames for a text to be written to ``frequent_mismatches.csv``.
LEGACY_FREQUENT_THRESHOLD = 10
#: The human-labelled file was re-saved with Windows line endings (probably by a spreadsheet tool).
LEGACY_CRLF_FILES = frozenset({"text_with_openAI_Validation_updated.csv"})

_GZIP = {"method": "gzip", "mtime": 0}  # deterministic archives


# -- loading -----------------------------------------------------------------
def default_data_dir() -> Path:
    """``data/`` at the repository root (or ``$ZSD_DATA_DIR``)."""
    import os

    env = os.environ.get("ZSD_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data"


def load_frames(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Frame table; ``human_label`` is ``NaN`` where no human label exists."""
    path = Path(data_dir or default_data_dir()) / FRAMES_FILE
    df = pd.read_csv(
        path,
        dtype={"video": str, "florence_text": str, "pipeline_label": str, "llm_label": str, "human_label": str},
        keep_default_na=False,
        na_values={"human_label": [""]},
    )
    df["frame_number"] = df["frame_number"].astype("int64")
    return df


def load_videos(data_dir: str | Path | None = None) -> pd.DataFrame:
    path = Path(data_dir or default_data_dir()) / VIDEOS_FILE
    return pd.read_csv(path, dtype={"video": str}, keep_default_na=False)


def load_review_session(data_dir: str | Path | None = None) -> pd.DataFrame:
    path = Path(data_dir or default_data_dir()) / REVIEW_SESSION_FILE
    return pd.read_csv(path, keep_default_na=False, na_values={"auto_corrected": [""]})


def split_video_id(video: str) -> tuple[str | None, str]:
    """Split ``"10_6896654168296099074.mp4"`` into ``("10", "6896654168296099074")``.

    The numeric prefix is the batch/folder the file came from; the remainder is
    the platform's post id. Files without a prefix return ``None``.
    """
    stem = video[:-4] if video.lower().endswith(".mp4") else video
    prefix, sep, rest = stem.partition("_")
    if sep and rest:
        return prefix, rest
    return None, stem


# -- migration from the original layout ----------------------------------------
def build_canonical(legacy_dir: str | Path, out_dir: str | Path) -> dict[str, int]:
    """Create the canonical tables from the original ``post_processing/`` files.

    Kept for provenance; this was run once to create ``data/``.
    """
    legacy_dir, out_dir = Path(legacy_dir), Path(out_dir)
    ckpt = legacy_dir / ".ipynb_checkpoints"
    frames = pd.read_csv(ckpt / "counts_summary_frames-checkpoint.csv", dtype=str, keep_default_na=False)
    updated = pd.read_csv(legacy_dir / "text_with_openAI_Validation_updated.csv", dtype=str, keep_default_na=False)
    if len(frames) != len(updated):
        raise ValueError("frame table and label table have different lengths")
    if not (frames["label"].values == updated["text"].values).all():
        raise ValueError("frame table and label table are not row-aligned (text differs)")
    if not (frames["original_classification"].values == updated["label"].values).all():
        raise ValueError("frame table and label table are not row-aligned (label differs)")
    canonical = pd.DataFrame(
        {
            "video": frames["video"],
            "frame_number": frames["frame_number"].astype("int64"),
            "florence_text": updated["text"],
            "pipeline_label": updated["label"],
            "llm_label": updated["predicted_category"],
            "human_label": updated["user_cat"].where(updated["user_cat"] != LEGACY_NONE, ""),
        }
    )
    summary = pd.read_csv(ckpt / "counts_summary_summary-checkpoint.csv", dtype={"video": str}, keep_default_na=False)
    order = list(dict.fromkeys(summary["video"]))
    videos = summary.pivot(index="video", columns="category", values="count").reindex(order)[list(LABELS)]
    videos = videos.reset_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(out_dir / FRAMES_FILE, index=False, compression=_GZIP)
    videos.to_csv(out_dir / VIDEOS_FILE, index=False, compression=_GZIP)
    return {"frames": len(canonical), "videos": len(videos)}


# -- legacy export --------------------------------------------------------------
def _legacy_autocorrect(texts: Iterable[str]) -> list[str | None]:
    out: list[str | None] = []
    for text in texts:
        lowered = text.lower()
        hit = None
        for category, keywords in LEGACY_AUTOCORRECT_RULES.items():
            if any(keyword in lowered for keyword in keywords):
                hit = category
                break
        out.append(hit)
    return out


def legacy_tables(frames: pd.DataFrame, videos: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Rebuild every original CSV (as DataFrames) from the canonical tables."""
    base = pd.DataFrame(
        {
            "text": frames["florence_text"],
            "label": frames["pipeline_label"],
            "predicted_category": frames["llm_label"],
        }
    )
    tables: dict[str, pd.DataFrame] = {}
    tables["text_and_labels.csv"] = base[["text", "label"]]
    tables["text_with_openAI_Validation.csv"] = base
    tables["text_with_openAI_Validation_updated.csv"] = base.assign(
        user_cat=frames["human_label"].fillna(LEGACY_NONE).replace("", LEGACY_NONE)
    )
    mismatches = base[base["label"] != base["predicted_category"]].reset_index(drop=True)
    tables["mismatches.csv"] = mismatches
    auto = mismatches.assign(auto_corrected=_legacy_autocorrect(mismatches["text"]))
    tables["auto_corrected_mismatches.csv"] = auto
    counts = Counter(mismatches["text"])
    frequent = {t for t, c in counts.items() if c >= LEGACY_FREQUENT_THRESHOLD}
    tables["frequent_mismatches.csv"] = auto[auto["text"].isin(frequent)]
    tables["counts_summary_frames.csv"] = pd.DataFrame(
        {
            "video": frames["video"],
            "frame_number": frames["frame_number"],
            "label": frames["florence_text"],
            "original_classification": frames["pipeline_label"],
        }
    )
    long = videos.melt(id_vars="video", value_vars=list(LABELS), var_name="category", value_name="count")
    long["_order"] = long["video"].map({v: i for i, v in enumerate(videos["video"])})
    long["_cat"] = long["category"].map({c: i for i, c in enumerate(LABELS)})
    tables["counts_summary_summary.csv"] = long.sort_values(["_order", "_cat"])[["video", "category", "count"]]
    return tables


def export_legacy(data_dir: str | Path | None, out_dir: str | Path) -> dict[str, Path]:
    """Write the original CSV files to ``out_dir``; returns ``{name: path}``."""
    frames = load_frames(data_dir)
    videos = load_videos(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, table in legacy_tables(frames, videos).items():
        path = out_dir / name
        table.to_csv(path, index=False, lineterminator="\r\n" if name in LEGACY_CRLF_FILES else "\n")
        paths[name] = path
    return paths


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_checksums(path: str | Path) -> dict[str, str]:
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest
    return out


def verify_legacy_checksums(data_dir: str | Path | None, out_dir: str | Path) -> dict[str, bool]:
    """Export the legacy CSVs and compare with the recorded original SHA-256 hashes."""
    data_dir = Path(data_dir or default_data_dir())
    expected = read_checksums(data_dir / LEGACY_CHECKSUMS_FILE)
    paths = export_legacy(data_dir, out_dir)
    return {name: sha256(path) == expected[name] for name, path in paths.items() if name in expected}
