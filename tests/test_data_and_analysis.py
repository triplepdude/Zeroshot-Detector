import json

import pandas as pd
import pytest
from conftest import DATA, has_module

from zeroshot_detector.analysis import stats as S
from zeroshot_detector.analysis.report import run_analysis
from zeroshot_detector.data import (
    export_legacy,
    legacy_tables,
    load_frames,
    load_videos,
    sha256,
    split_video_id,
    verify_legacy_checksums,
)
from zeroshot_detector.taxonomy import LABELS


def test_split_video_id():
    assert split_video_id("10_6896654168296099074.mp4") == ("10", "6896654168296099074")
    assert split_video_id("6786698084266102022.mp4") == (None, "6786698084266102022")


def test_legacy_tables_from_synthetic(synthetic_data_dir):
    frames, videos = load_frames(synthetic_data_dir), load_videos(synthetic_data_dir)
    t = legacy_tables(frames, videos)
    assert list(t["text_and_labels.csv"].columns) == ["text", "label"]
    mm = t["mismatches.csv"]
    assert (mm["label"] != mm["predicted_category"]).all()
    upd = t["text_with_openAI_Validation_updated.csv"]
    assert set(upd["user_cat"]) >= {"None"}
    summary = t["counts_summary_summary.csv"]
    assert len(summary) == len(videos) * len(LABELS)
    assert list(summary["category"].iloc[: len(LABELS)]) == list(LABELS)
    paths = export_legacy(synthetic_data_dir, synthetic_data_dir / "export")
    assert b"\r\n" in paths["text_with_openAI_Validation_updated.csv"].read_bytes()


def test_analysis_end_to_end_on_synthetic(synthetic_data_dir, tmp_path):
    out = tmp_path / "results"
    s = run_analysis(synthetic_data_dir, out, n_boot=50, seed=1, figures=has_module("matplotlib"))
    assert (out / "REPORT.md").exists() and (out / "summary.json").exists()
    loaded = json.loads((out / "summary.json").read_text())
    assert loaded["dataset"]["n_frames"] == s["dataset"]["n_frames"]
    assert s["duplicates"]["n_posts_in_multiple_files"] == 1
    assert s["video_level"]["counts_match_stored_summary"]
    assert s["video_level"]["legacy_label_of_files_without_frames"] == ["weapons"]
    fl = s["agreement"]["frame_level"]
    assert fl["ci"]["kappa"]["ci_low"] <= fl["metrics"]["kappa"] <= fl["ci"]["kappa"]["ci_high"]
    report = (out / "REPORT.md").read_text()
    assert "## 4. Agreement between pipeline and LLM labels" in report


def test_threshold_sensitivity_with_relabel(synthetic_data_dir):
    frames = S.add_post_ids(load_frames(synthetic_data_dir))
    texts = frames["florence_text"].unique()
    rows = []
    for t in texts:
        lab = frames.loc[frames["florence_text"] == t, "pipeline_label"].iloc[0]
        scores = {f"score_{label}": (0.85 if label == lab else 0.0375) for label in LABELS}
        rows.append(
            {
                "text": t,
                "category": lab,
                "source": "keyword" if lab == "ground_vehicles" else "zero_shot",
                "top_label": lab,
                "top_score": 0.85,
                **scores,
            }
        )
    relabel = pd.DataFrame(rows)
    rep = S.reproduction(frames, relabel)
    assert rep["text_match_rate"] == 1.0
    ts = S.threshold_sensitivity(frames, relabel, [0.5, 0.9])
    at_09 = ts["rows"][1]
    assert at_09["distribution"]["nonmilitary"] > ts["rows"][0]["distribution"]["nonmilitary"]


@pytest.mark.data
@pytest.mark.skipif(not (DATA / "frames.csv.gz").exists(), reason="study data not present")
class TestStudyData:
    def test_checksums(self):
        for line in (DATA / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split()
            assert sha256(DATA / name) == digest, name

    def test_original_csvs_regenerate_byte_identically(self, tmp_path):
        result = verify_legacy_checksums(DATA, tmp_path)
        assert len(result) == 8
        assert all(result.values()), result

    def test_invariants(self):
        f = load_frames(DATA)
        assert len(f) == 1_099_951
        assert (f.groupby("florence_text")["pipeline_label"].nunique() == 1).all()
        assert set(f["pipeline_label"]) <= set(LABELS) and set(f["llm_label"]) <= set(LABELS)
        assert (f["frame_number"] % 30 == 0).all()
