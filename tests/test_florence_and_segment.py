import pytest

from zeroshot_detector.florence import legacy_text, parse_od
from zeroshot_detector.segment import build_vocabulary, segment, segment_all

RAW = (
    "</s><s>car<loc_54><loc_332><loc_933><loc_775>door<loc_709><loc_198><loc_910><loc_547>"
    "<loc_352><loc_181><loc_522><loc_342>wheel<loc_703><loc_576><loc_869><loc_771></s>"
)


class TestParseOD:
    def test_labels_and_multiple_boxes_per_label(self):
        det = parse_od(RAW, image_size=(640, 480))
        assert det.labels == ["car", "door", "wheel"]
        assert [d.label for d in det.detections] == ["car", "door", "door", "wheel"]
        x1, _, _, y2 = det.detections[0].box
        assert x1 == pytest.approx((54 + 0.5) / 1000 * 640)
        assert y2 == pytest.approx((775 + 0.5) / 1000 * 480)

    def test_legacy_text_glues_labels(self):
        # This is exactly the format of the archived strings.
        assert legacy_text(RAW) == "cardoorwheel"
        assert parse_od(RAW).legacy_text == "cardoorwheel"

    def test_multiword_labels_and_no_objects(self):
        raw = "<s>human face<loc_1><loc_2><loc_3><loc_4>land vehicle<loc_5><loc_6><loc_7><loc_8></s>"
        assert parse_od(raw).labels == ["human face", "land vehicle"]
        assert legacy_text(raw) == "human faceland vehicle"
        assert parse_od("</s><s></s>").labels == []

    def test_incomplete_box_group_ignored(self):
        assert parse_od("car<loc_1><loc_2><loc_3>").labels == []


class TestSegment:
    vocab = frozenset({"car", "car tire", "truck", "human face", "man", "trousers", "wheel", "wheelchair", "chair"})

    def test_fewest_parts(self):
        assert segment("wheelchair", self.vocab) == ("wheelchair",)
        assert segment("car tiretruck", self.vocab) == ("car tire", "truck")
        assert segment("human facemantrousers", self.vocab) == ("human face", "man", "trousers")

    def test_unsegmentable(self):
        assert segment("persontank", self.vocab) is None

    def test_vocabulary_learns_standalone_labels(self):
        texts = ["car tire", "car tiretruck", "truck", "carcar tire", "car"]
        vocab = build_vocabulary(texts, base={"truck", "car"})
        seg = segment_all(texts, vocab)
        assert seg["car tiretruck"] == ("car tire", "truck")
        assert seg["carcar tire"] == ("car", "car tire")
        assert seg["car tire"] == ("car tire",)

    def test_residual_mining(self):
        # "human beard" never occurs alone; it is recovered from >= 3 distinct contexts
        base = {"human face", "man", "woman", "person", "glasses"}
        texts = [
            f"{x}human beardhuman face{y}" for x, y in [("", "man"), ("", "woman"), ("", "person"), ("glasses", "man")]
        ]
        vocab = build_vocabulary(texts, base=base, min_support=3)
        assert "human beard" in vocab
        seg = segment_all(texts, vocab)
        assert seg["human beardhuman faceman"] == ("human beard", "human face", "man")
        assert seg["glasseshuman beardhuman faceman"] == ("glasses", "human beard", "human face", "man")
        assert build_vocabulary(texts, base=base, min_support=4) >= {"human beardhuman faceman"}
