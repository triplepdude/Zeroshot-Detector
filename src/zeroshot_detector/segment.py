"""Recover object-label lists from archived, separator-less Florence-2 strings.

The original pipeline decoded Florence-2 ``<OD>`` output with
``batch_decode(..., skip_special_tokens=True)``, which strips the
``<loc_*>`` box tokens and thereby glues the object labels together
(``"human face<loc_..>...man<loc_..>..."`` → ``"human facemantrousers"``).
Only these glued strings were archived.

This module re-segments them so that the effect of the missing separator can
be measured (see the ``segmented`` relabelling in the analysis). The
vocabulary is the Open Images V7 boxable class list (which Florence-2's
detection labels largely follow) plus every archived string that occurs on its
own and cannot itself be split into known labels. Segmentation uses as *few*
parts as possible, so that genuine compound class names (``"wheelchair"``,
``"sunglasses"``) are not split. Strings that cannot be fully covered are left
unsplit.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from importlib import resources

_OI_RESOURCE = "open_images_v7_boxable_classes.txt"

#: Florence-2 labels that occur in the archive only inside longer strings and are
#: not Open Images class names (found by inspecting unsplittable strings).
SUPPLEMENTARY_LABELS: frozenset[str] = frozenset({"human beard", "suv"})


def open_images_classes() -> frozenset[str]:
    text = resources.files("zeroshot_detector.resources").joinpath(_OI_RESOURCE).read_text()
    return frozenset(line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#"))


def default_base_vocabulary() -> frozenset[str]:
    return open_images_classes() | SUPPLEMENTARY_LABELS


def segment(
    text: str, vocabulary: frozenset[str] | set[str], max_label_len: int | None = None
) -> tuple[str, ...] | None:
    """Split ``text`` into the fewest vocabulary items that concatenate to it.

    Returns ``None`` when no full segmentation exists. Among segmentations with
    the same number of parts, the one whose parts are in alphabetical order is
    preferred (Florence-2 emits labels mostly in that order).
    """
    n = len(text)
    longest = max_label_len or max((len(v) for v in vocabulary), default=0)

    @functools.cache
    def best(i: int) -> tuple[int, int, tuple[str, ...]] | None:
        # returns (n_parts, n_order_violations, parts)
        if i == n:
            return (0, 0, ())
        result = None
        for j in range(i + 1, min(n, i + longest) + 1):
            word = text[i:j]
            if word not in vocabulary:
                continue
            rest = best(j)
            if rest is None:
                continue
            violation = 1 if rest[2] and rest[2][0] < word else 0
            candidate = (1 + rest[0], violation + rest[1], (word, *rest[2]))
            if result is None or candidate[:2] < result[:2]:
                result = candidate
        return result

    found = best(0)
    return None if found is None else found[2]


def _reachable(text: str, vocab: set[str], longest: int) -> tuple[list[bool], list[bool]]:
    """Positions reachable from the start / from which the end is reachable, using ``vocab``."""
    n = len(text)
    fwd = [False] * (n + 1)
    fwd[0] = True
    for i in range(n):
        if fwd[i]:
            for j in range(i + 1, min(n, i + longest) + 1):
                if text[i:j] in vocab:
                    fwd[j] = True
    bwd = [False] * (n + 1)
    bwd[n] = True
    for j in range(n, 0, -1):
        if bwd[j]:
            for i in range(max(0, j - longest), j):
                if text[i:j] in vocab:
                    bwd[i] = True
    return fwd, bwd


def _shortest_residual(text: str, vocab: set[str], longest: int, min_len: int) -> str | None:
    """Shortest unknown piece ``text[i:j]`` such that the rest of ``text`` is segmentable."""
    fwd, bwd = _reachable(text, vocab, longest)
    best = None
    for i in range(len(text)):
        if not fwd[i]:
            continue
        for j in range(i + min_len, len(text) + 1):
            if bwd[j]:
                piece = text[i:j]
                if piece != text and piece.strip() == piece and (best is None or len(piece) < len(best)):
                    best = piece
                break  # longer j only gives longer pieces from this i
    return best


def build_vocabulary(
    texts: Iterable[str],
    base: Iterable[str] | None = None,
    min_support: int = 10,
    min_len: int = 4,
    max_rounds: int = 5,
) -> frozenset[str]:
    """Build the segmentation vocabulary from ``base`` and the data.

    ``base`` defaults to the Open Images classes plus :data:`SUPPLEMENTARY_LABELS`.

    1. Archived strings are visited from shortest to longest. A string that
       cannot be split into two or more items of the current vocabulary is
       taken to be a single label and added (so ``"car tire"``, seen on its
       own, is known before ``"car tiretruck"`` is visited).
    2. *Residual mining* for labels that never occur alone: for each string
       that is still unsplittable, the shortest unknown piece whose removal
       leaves a segmentable remainder is collected; pieces found in at least
       ``min_support`` distinct strings are added and step 1 is repeated.
    """
    from collections import Counter

    texts = sorted(set(texts), key=lambda t: (len(t), t))
    base_vocab = frozenset(base) if base is not None else default_base_vocabulary()
    mined: set[str] = set()
    for _ in range(max_rounds):
        vocab = set(base_vocab) | mined
        unsplit = []
        for text in texts:
            if text in vocab:
                continue
            longest = max(len(v) for v in vocab)
            parts = segment(text, vocab, longest)
            if parts is None or len(parts) < 2:
                vocab.add(text)
                unsplit.append(text)
        longest = max(len(v) for v in vocab)
        support: Counter[str] = Counter()
        for text in unsplit:
            others = vocab - {text}
            piece = _shortest_residual(text, others, longest, min_len)
            if piece is not None and piece not in others:
                support[piece] += 1
        new = {piece for piece, n in support.items() if n >= min_support} - mined
        if not new:
            break
        mined |= new
    return frozenset(vocab)


def segment_all(texts: Iterable[str], vocabulary: frozenset[str] | None = None) -> dict[str, tuple[str, ...]]:
    """Segment unique ``texts``; unsplittable strings map to a 1-tuple of themselves."""
    texts = list(dict.fromkeys(texts))
    vocab = vocabulary if vocabulary is not None else build_vocabulary(texts)
    longest = max(len(v) for v in vocab)
    out = {}
    for text in texts:
        parts = segment(text, vocab - {text}, longest)
        if parts is None or len(parts) < 2:
            parts = (text,)
        out[text] = parts
    return out
