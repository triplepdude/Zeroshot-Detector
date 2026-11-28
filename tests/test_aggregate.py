import numpy as np
import pytest

from zeroshot_detector.aggregate import (
    aggregate_video,
    count_matrix,
    legacy_labels,
    military_fraction_labels,
    plurality_labels,
)
from zeroshot_detector.taxonomy import LABELS

W, A, M, G, N = LABELS


def legacy_reference(frame_labels, below=0, total_frames=None):
    """Literal transcription of the original aggregation code."""
    from collections import Counter

    label_counts = Counter({label: 0 for label in LABELS})
    for c in frame_labels:
        label_counts[c] += 1
    complete = {label: label_counts[label] for label in LABELS}
    most = max(complete, key=complete.get) if label_counts else "nonmilitary"
    if total_frames is not None and below > total_frames * 0.5:
        most = "nonmilitary"
    return most


@pytest.mark.parametrize(
    "frames",
    [[], [N], [W, N], [N, W], [A, A, N], [N, N, G], [M, W], [G, A, A, G, N, N], [W] * 3 + [N] * 3],
)
def test_legacy_matches_original_code(frames):
    counts = count_matrix([frames])
    assert legacy_labels(counts)[0] == legacy_reference(frames)


def test_legacy_empty_video_is_weapons():
    # The original returns the first label for an all-zero Counter.
    assert legacy_labels(np.zeros((1, 5), dtype=int))[0] == "weapons"


def test_legacy_override_uses_total_decoded_frames():
    counts = count_matrix([[W, N, N]])
    # 2 below-threshold sampled frames vs. 90 decoded frames: override cannot fire
    assert legacy_labels(counts, below_threshold_counts=np.array([2]), total_frames=np.array([90]))[0] == "nonmilitary"
    counts = count_matrix([[W, W, N]])
    assert legacy_labels(counts, below_threshold_counts=np.array([1]), total_frames=np.array([90]))[0] == "weapons"
    assert legacy_labels(counts, below_threshold_counts=np.array([1]), total_frames=np.array([1]))[0] == "nonmilitary"


def test_plurality_tie_goes_to_nonmilitary_and_empty_is_none():
    counts = count_matrix([[W, N], [W, A], [], [A, A, N]])
    assert list(plurality_labels(counts)) == ["nonmilitary", "weapons", None, "aircraft"]


def test_military_fraction():
    counts = count_matrix([[W, N, N, N], [A, A, W, N], [N, N]])
    assert list(military_fraction_labels(counts, 0.25)) == ["weapons", "aircraft", "nonmilitary"]
    assert list(military_fraction_labels(counts, 0.5)) == ["nonmilitary", "aircraft", "nonmilitary"]
    with pytest.raises(ValueError):
        military_fraction_labels(counts, 0.0)


def test_aggregate_video_ignores_none_and_reports_ties():
    d = aggregate_video([W, None, N], rule="plurality")
    assert d.category == "nonmilitary" and d.tie and d.n_frames == 2
    d = aggregate_video([A, A, N], rule="military_fraction", min_fraction=0.5)
    assert d.category == "aircraft" and not d.tie
    with pytest.raises(ValueError):
        aggregate_video([A], rule="majority")
