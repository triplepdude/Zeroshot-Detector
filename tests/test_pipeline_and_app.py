from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import REPO, FakeScorer, has_module, peaked

from zeroshot_detector.cli import build_parser
from zeroshot_detector.config import config_from_dict, load_config
from zeroshot_detector.florence import FrameDetections, parse_od
from zeroshot_detector.video import l1_distance, sampling_step

needs_cv2 = pytest.mark.skipif(not (has_module("cv2") and has_module("PIL")), reason="opencv/pillow not installed")


class TestConfig:
    def test_shipped_configs_load(self):
        default = load_config(REPO / "configs" / "default.yaml")
        assert not default.classifier.legacy and default.aggregation.rule == "plurality"
        original = load_config(REPO / "configs" / "original_study.yaml")
        assert original.classifier.legacy and original.aggregation.rule == "legacy"
        assert default.fingerprint() != original.fingerprint()

    def test_unknown_keys_rejected(self):
        with pytest.raises(ValueError):
            config_from_dict({"source": {"kind": "local", "path": "x", "bucket": "typo"}})
        with pytest.raises(ValueError):
            config_from_dict({"sources": {}})

    def test_validation(self):
        with pytest.raises(ValueError):
            config_from_dict({"source": {"kind": "minio"}})
        with pytest.raises(ValueError):
            config_from_dict({"source": {"kind": "local", "path": "x"}, "aggregation": {"rule": "military_fraction"}})

    def test_overrides(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump({"source": {"kind": "local", "path": "v"}}))
        cfg = load_config(p, {"classifier.threshold": 0.5})
        assert cfg.classifier.threshold == 0.5


def test_sampling_step():
    assert sampling_step(30.0, 30, None) == 30
    assert sampling_step(25.0, 30, 2.0) == 50
    with pytest.raises(ValueError):
        sampling_step(None, 30, 1.0)


def test_l1_distance():
    a = np.zeros((2, 2, 3), dtype=np.uint8)
    b = a.copy()
    b[0, 0, 0] = 255
    assert l1_distance(a, b) == 255
    assert l1_distance(a, np.zeros((3, 2, 3), dtype=np.uint8)) == float("inf")


def make_video(path: Path, n_frames=95, size=(64, 48), switch_at=60):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), 40 if i < switch_at else 200, dtype=np.uint8)
        writer.write(frame)
    writer.release()


@needs_cv2
def test_iter_sampled_frames_and_dedup(tmp_path):
    from zeroshot_detector.video import SamplingStats, iter_sampled_frames, read_frame

    video = tmp_path / "v.avi"
    make_video(video)
    stats = SamplingStats()
    kept = [f.index for f in iter_sampled_frames(video, 30, dedup_l1_threshold=5000, stats=stats)]
    # candidates 0, 30, 60, 90; 30 and 90 duplicate the previously kept frame
    assert stats.candidates == 4 and stats.decoded_frames == 95
    assert kept == [0, 60] and stats.dropped_as_duplicates == [30, 90]
    assert [f.index for f in iter_sampled_frames(video, 30, dedup_l1_threshold=None)] == [0, 30, 60, 90]
    assert read_frame(video, 61).mean() > 150


class FakeDetector:
    def __init__(self):
        self.resolved_revision = "fake"

    def detect(self, images):
        out = []
        for im in images:
            bright = np.asarray(im).mean() > 100
            raw = "<s>tank<loc_1><loc_2><loc_3><loc_4></s>" if bright else "<s>weapon<loc_1><loc_2><loc_3><loc_4></s>"
            out.append(parse_od(raw, im.size))
        return out


@needs_cv2
@pytest.mark.parametrize("legacy", [False, True])
def test_pipeline_run_and_resume(tmp_path, monkeypatch, legacy):
    import zeroshot_detector.pipeline as P
    from zeroshot_detector.classify import FrameClassifier

    make_video(tmp_path / "videos" / "a" / "10_1.avi", switch_at=200)
    make_video(tmp_path / "videos" / "b" / "11_2.avi", switch_at=0)
    (tmp_path / "videos" / "broken.mp4").write_bytes(b"not a video")
    cfg = config_from_dict(
        {
            "source": {"kind": "local", "path": str(tmp_path / "videos")},
            "sampling": {"dedup_l1_threshold": None},
            "classifier": {"legacy": legacy},
            "aggregation": {"rule": "legacy" if legacy else "plurality"},
            "output": {"copy_videos": True},
        }
    )
    scorer = FakeScorer({"weapon": peaked("weapons", 0.95)})
    monkeypatch.setattr(P, "build_components", lambda c: (FakeDetector(), FrameClassifier(scorer, legacy=legacy)))
    run_dir = tmp_path / "run"
    manifest = P.run(cfg, run_dir)
    assert manifest["errors_this_session"] == 1 and manifest["videos_processed_this_session"] == 3
    import pandas as pd

    videos = pd.read_csv(run_dir / "videos.csv").set_index("key")
    assert videos.loc["a/10_1.avi", "category"] == "weapons"
    assert videos.loc["b/11_2.avi", "category"] == "ground_vehicles"
    assert videos.loc["broken.mp4", "status"] == "error"
    frames = pd.read_json(run_dir / "frames.jsonl", lines=True)
    assert len(frames) == 8 and set(frames["source"]) == {"zero_shot", "keyword"}
    assert (run_dir / "sorted" / "weapons" / "videos" / "a" / "10_1.avi").exists()
    # resume: only the failed video is retried
    manifest = P.run(cfg, run_dir)
    assert manifest["videos_processed_this_session"] == 1
    # a different configuration may not write into the same run directory
    cfg.classifier.threshold = 0.5
    with pytest.raises(RuntimeError):
        P.run(cfg, run_dir)


@pytest.mark.skipif(not has_module("flask"), reason="flask not installed")
def test_review_app_blind_flow(tmp_path):
    import pandas as pd

    from zeroshot_detector.review.app import create_app

    sheet = tmp_path / "sheet.csv"
    pd.DataFrame({"sample_id": ["a1", "b2"], "video": ["x.mp4", "y.mp4"], "frame_number": [0, 30]}).to_csv(
        sheet, index=False
    )
    answers = tmp_path / "answers.csv"
    client = create_app(sheet, answers).test_client()
    assert client.get("/").status_code == 200
    r = client.post("/start", data={"annotator": "ann1"})
    assert r.status_code == 302
    page = client.get("/annotate/ann1").get_data(as_text=True)
    assert "pipeline" not in page.lower() and "predicted" not in page.lower()
    for _ in range(2):
        page = client.get("/annotate/ann1").get_data(as_text=True)
        sid = page.split('name="sample_id" value="')[1].split('"')[0]
        assert (
            client.post("/annotate/ann1", data={"sample_id": sid, "label": "weapons", "started": "0"}).status_code
            == 302
        )
    assert "All done" in client.get("/annotate/ann1").get_data(as_text=True)
    assert client.post("/annotate/ann1", data={"sample_id": "zz", "label": "weapons"}).status_code == 400
    saved = pd.read_csv(answers)
    assert sorted(saved["sample_id"]) == ["a1", "b2"] and set(saved["annotator"]) == {"ann1"}


def test_cli_parser():
    p = build_parser()
    assert p.parse_args(["analyze", "--n-boot", "10"]).n_boot == 10
    assert p.parse_args(["data", "verify"]).data_command == "verify"
    assert p.parse_args(["annotate", "sample", "--n-per-stratum", "5"]).n_per_stratum == 5


def test_frame_detections_labels_are_unique_in_order():
    det = FrameDetections(raw_text="", detections=[])
    assert det.labels == []
    det = parse_od("b<loc_1><loc_1><loc_1><loc_1>a<loc_1><loc_1><loc_1><loc_1>b<loc_1><loc_1><loc_1><loc_1>")
    assert det.labels == ["b", "a"]
