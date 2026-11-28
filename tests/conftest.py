from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zeroshot_detector.taxonomy import LABELS

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"


class FakeScorer:
    """Deterministic zero-shot scorer: looks texts up in a table, else uniform scores."""

    def __init__(self, table: dict[str, dict[str, float]] | None = None) -> None:
        self.table = table or {}
        self.calls: list[list[str]] = []

    def score(self, texts: Sequence[str], labels: Sequence[str]) -> list[dict[str, float]]:
        self.calls.append(list(texts))
        out = []
        for text in texts:
            if text in self.table:
                out.append({label: self.table[text].get(label, 0.0) for label in labels})
            else:
                out.append({label: 1.0 / len(labels) for label in labels})
        return out


def peaked(label: str, score: float) -> dict[str, float]:
    rest = (1 - score) / (len(LABELS) - 1)
    return {lab: (score if lab == label else rest) for lab in LABELS}


@pytest.fixture
def fake_scorer() -> FakeScorer:
    return FakeScorer(
        {
            "weapon": peaked("weapons", 0.96),
            "human facemantrousers": peaked("weapons", 0.85),
            "boatperson": peaked("maritime", 0.77),
            "airplane": peaked("aircraft", 0.99),
            "car, door, wheel": peaked("ground_vehicles", 0.54),
            "human face, man, trousers": peaked("nonmilitary", 0.9),
        }
    )


@pytest.fixture
def synthetic_frames() -> pd.DataFrame:
    """Small frame table with the canonical schema (3 batches, one duplicated post)."""
    rng = np.random.default_rng(0)
    rows = []
    texts = {
        "weapons": ["weapon", "personweapon"],
        "aircraft": ["airplane", "helicopter"],
        "maritime": ["boat"],
        "ground_vehicles": ["tank"],
        "nonmilitary": ["person", "car tiretruck", "human face"],
    }
    llm_map = {"car tiretruck": "ground_vehicles", "personweapon": "nonmilitary"}
    for v in range(40):
        video = f"{10 + v % 3}_{1000 + v}.mp4"
        main = LABELS[v % len(LABELS)]
        for k in range(int(rng.integers(3, 9))):
            label = main if rng.random() < 0.6 else "nonmilitary"
            text = texts[label][int(rng.integers(len(texts[label])))]
            llm = llm_map.get(text, label)
            human = None
            if text == "personweapon":
                human = "weapons" if rng.random() < 0.5 else "nonmilitary"
            rows.append((video, 30 * k, text, label, llm, human))
    df = pd.DataFrame(
        rows, columns=["video", "frame_number", "florence_text", "pipeline_label", "llm_label", "human_label"]
    )
    # make each text map to exactly one human label, as in the real data
    first = df.dropna(subset=["human_label"]).groupby("florence_text")["human_label"].first()
    df["human_label"] = np.where(df["human_label"].notna(), df["florence_text"].map(first), None)
    # one post processed twice (identical rows under another prefix)
    dup = df[df["video"] == "10_1000.mp4"].assign(video="11_1000.mp4")
    return pd.concat([df, dup], ignore_index=True)


@pytest.fixture
def synthetic_data_dir(tmp_path: Path, synthetic_frames: pd.DataFrame) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    synthetic_frames.to_csv(d / "frames.csv.gz", index=False)
    counts = synthetic_frames.groupby(["video", "pipeline_label"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list(LABELS), fill_value=0)
    videos = counts.reset_index()
    empty = pd.DataFrame([{"video": "12_9999.mp4", **{label: 0 for label in LABELS}}])
    pd.concat([videos, empty], ignore_index=True).to_csv(d / "videos.csv.gz", index=False)
    h = synthetic_frames.dropna(subset=["human_label"])
    session = pd.DataFrame(
        {
            "text": h["florence_text"],
            "label": h["pipeline_label"],
            "predicted_category": h["llm_label"],
            "auto_corrected": "",
            "user_category": h["human_label"],
        }
    )
    session.to_csv(d / "reviewed_mismatches.csv", index=False)
    return d


def has_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None
