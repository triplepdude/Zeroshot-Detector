"""Re-derive frame labels from stored Florence-2 descriptions.

The archived study data store, for every sampled frame, the Florence-2 output
string and the label the pipeline assigned to it. Because the label is a
deterministic function of the string (keyword rule, then thresholded
zero-shot), re-running the classifier on the *unique* strings and comparing
with the stored labels checks that the archived data were produced by the code
in this repository, and yields the full score vectors needed for threshold
sensitivity analysis.

Results are checkpointed after every chunk, so an interrupted run resumes.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from .classify import FrameClassifier

log = logging.getLogger(__name__)


def decisions_to_frame(
    classifier: FrameClassifier,
    texts: Sequence[str],
    objects: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Classify ``texts``; with ``objects``, classify their segmented object lists instead."""
    if objects is None:
        decisions = classifier.classify_texts(texts)
    else:
        decisions = classifier.classify_objects([objects[t] for t in texts])
    rows = []
    for text, d in zip(texts, decisions, strict=True):
        row = {
            "text": text,
            "model_input": d.text,
            "category": d.category,
            "source": d.source,
            "matched_keyword": d.matched_keyword,
            "top_label": d.top_label,
            "top_score": d.top_score,
        }
        for label in classifier.labels:
            row[f"score_{label}"] = d.scores.get(label)
        rows.append(row)
    return pd.DataFrame(rows)


def relabel_texts(
    classifier: FrameClassifier,
    texts: Sequence[str],
    output: str | Path,
    chunk_size: int = 512,
    objects: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Classify unique ``texts`` and write/extend ``output`` (CSV, optionally ``.gz``).

    ``objects`` optionally maps each text to its object-label list (see
    :mod:`zeroshot_detector.segment`); the classifier then sees the objects.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    unique = list(dict.fromkeys(t for t in texts if isinstance(t, str)))
    done: pd.DataFrame | None = None
    if output.exists():
        done = pd.read_csv(output, keep_default_na=False, na_values=[""])
        finished = set(done["text"])
        log.info("resuming: %d/%d texts already scored", len(finished), len(unique))
        unique = [t for t in unique if t not in finished]
    parts = [done] if done is not None else []
    for start in range(0, len(unique), chunk_size):
        chunk = unique[start : start + chunk_size]
        parts.append(decisions_to_frame(classifier, chunk, objects))
        result = pd.concat(parts, ignore_index=True)
        tmp = output.with_name(output.name + ".tmp")
        result.to_csv(tmp, index=False, compression="gzip" if output.suffix == ".gz" else None)
        tmp.replace(output)
        log.info("scored %d/%d new texts", min(start + chunk_size, len(unique)), len(unique))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
