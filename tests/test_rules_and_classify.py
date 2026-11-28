import pytest
from conftest import peaked

from zeroshot_detector.classify import (
    SOURCE_BELOW_THRESHOLD,
    SOURCE_EMPTY,
    SOURCE_KEYWORD,
    SOURCE_ZERO_SHOT,
    FrameClassifier,
    decide,
)
from zeroshot_detector.rules import KeywordRule, legacy_keyword_match
from zeroshot_detector.taxonomy import LEGACY_MILITARY_KEYWORDS, validate_labels


class TestLegacyKeywordMatch:
    def test_substring_inside_glued_labels(self):
        assert legacy_keyword_match("persontankwheel") == "tank"

    def test_capitalised_keywords_never_match(self):
        # The original compared raw keywords against lower-cased text.
        for kw in ("APC", "MRAP", "Stryker", "Bradley"):
            assert legacy_keyword_match(kw) is None
            assert legacy_keyword_match(f"a {kw} on the road") is None

    def test_false_positive_on_clothing(self):
        assert legacy_keyword_match("tank topwoman") == "tank"

    def test_first_keyword_in_list_order(self):
        assert legacy_keyword_match("military convoy") == "convoy"


class TestKeywordRule:
    rule = KeywordRule.from_legacy_keywords()

    def test_whole_word_case_insensitive(self):
        assert self.rule.match(["Tank"]).keyword == "tank"
        assert self.rule.match(["MRAP"]).keyword == "MRAP"
        assert self.rule.match(["tanker"]) is None

    def test_exclusions(self):
        assert self.rule.match(["tank top"]) is None
        assert self.rule.match(["fish tank", "person"]) is None
        assert self.rule.match(["tank top", "tank"]).object_label == "tank"

    def test_longest_phrase_reported(self):
        assert self.rule.match(["military truck"]).keyword == "military truck"

    def test_multiword_whitespace(self):
        assert self.rule.match(["patrol   vehicle"]).keyword == "patrol vehicle"


def test_decide_threshold_is_inclusive():
    scores = peaked("weapons", 0.8)
    assert decide(scores, 0.8)[:2] == ("weapons", SOURCE_ZERO_SHOT)
    assert decide(scores, 0.81)[:2] == ("nonmilitary", SOURCE_BELOW_THRESHOLD)


class TestFrameClassifier:
    def test_legacy_reproduces_original_rule(self, fake_scorer):
        clf = FrameClassifier(fake_scorer, legacy=True)
        out = clf.classify_texts(["weapon", "human facemantrousers", "boatperson", "persontank", "", "unknown"])
        assert [d.category for d in out] == [
            "weapons",
            "weapons",
            "nonmilitary",
            "ground_vehicles",
            None,
            "nonmilitary",
        ]
        assert [d.source for d in out] == [
            SOURCE_ZERO_SHOT,
            SOURCE_ZERO_SHOT,
            SOURCE_BELOW_THRESHOLD,
            SOURCE_KEYWORD,
            SOURCE_EMPTY,
            SOURCE_BELOW_THRESHOLD,
        ]
        # keyword frames are not scored unless always_score
        assert all("persontank" not in call for call in fake_scorer.calls)

    def test_always_score_keeps_keyword_decision(self, fake_scorer):
        clf = FrameClassifier(fake_scorer, legacy=True, always_score=True)
        (d,) = clf.classify_texts(["persontank"])
        assert d.category == "ground_vehicles" and d.scores and d.matched_keyword == "tank"

    def test_corrected_mode_joins_labels(self, fake_scorer):
        clf = FrameClassifier(fake_scorer, legacy=False)
        out = clf.classify_objects([["human face", "man", "trousers"], ["car", "door", "wheel"], ["tank top", "woman"]])
        assert out[0].text == "human face, man, trousers"
        assert out[0].category == "nonmilitary"
        assert out[1].category == "nonmilitary" and out[1].top_label == "ground_vehicles"
        assert out[2].source != SOURCE_KEYWORD

    def test_legacy_object_input_is_glued(self, fake_scorer):
        clf = FrameClassifier(fake_scorer, legacy=True)
        (d,) = clf.classify_objects([["human face", "man", "trousers"]])
        assert d.text == "human facemantrousers" and d.category == "weapons"

    def test_validation(self, fake_scorer):
        with pytest.raises(ValueError):
            FrameClassifier(fake_scorer, threshold=1.5)
        with pytest.raises(ValueError):
            FrameClassifier(fake_scorer, keyword_category="tanks")
        with pytest.raises(ValueError):
            validate_labels(["a", "b"])


def test_keyword_list_is_verbatim():
    assert len(LEGACY_MILITARY_KEYWORDS) == 18
    assert LEGACY_MILITARY_KEYWORDS[4] == "APC"
