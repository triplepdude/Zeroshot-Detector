"""RQ4: alternative zero-shot set-ups and a supervised baseline, scored on the reference sample.

Each zero-shot configuration is scored once per distinct string and cached as
``<cache_dir>/<config name>.csv.gz`` (columns ``text`` + one score per label),
so thresholds can be swept without re-running the model.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..rules import KeywordRule, legacy_keyword_match
from ..taxonomy import LABELS, NONMILITARY

log = logging.getLogger(__name__)

MODELS = {
    "bart": "facebook/bart-large-mnli",
    "deberta": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
}
TEMPLATES = {"example": "This example is {}.", "image": "This image shows {}."}
VERBALIZERS: dict[str, dict[str, str]] = {
    "ids": {},
    "military": {
        "weapons": "weapons",
        "aircraft": "military aircraft",
        "maritime": "military ships or boats",
        "ground_vehicles": "military ground vehicles",
        "nonmilitary": "nothing military",
    },
}
INPUTS = ("glued", "separated")


@dataclass(frozen=True)
class ZeroShotConfig:
    model: str
    template: str
    verbalizer: str
    input: str

    @property
    def name(self) -> str:
        return f"{self.model}-{self.template}-{self.verbalizer}-{self.input}"


def all_configs() -> list[ZeroShotConfig]:
    return [ZeroShotConfig(*c) for c in itertools.product(MODELS, TEMPLATES, VERBALIZERS, INPUTS)]


def model_input(text: str, segmentation: Mapping[str, Sequence[str]], mode: str) -> str:
    return text if mode == "glued" else ", ".join(segmentation.get(text, (text,)))


def score_config(
    config: ZeroShotConfig,
    texts: Sequence[str],
    segmentation: Mapping[str, Sequence[str]],
    cache_dir: str | Path,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Scores (single-label softmax over labels) for every text; cached on disk."""
    from ..zero_shot import NLIZeroShotScorer

    cache = Path(cache_dir) / f"{config.name}.csv.gz"
    if cache.exists():
        cached = pd.read_csv(cache, keep_default_na=False)
        if set(texts) <= set(cached["text"]):
            return cached.set_index("text").loc[list(texts)].reset_index()
    scorer = NLIZeroShotScorer(
        model=MODELS[config.model],
        hypothesis_template=TEMPLATES[config.template],
        verbalizers=VERBALIZERS[config.verbalizer],
        batch_size=batch_size,
    )
    inputs = [model_input(t, segmentation, config.input) for t in texts]
    matrix = scorer.score_matrix(inputs, LABELS)
    out = pd.DataFrame(matrix, columns=[f"score_{label}" for label in LABELS])
    out.insert(0, "text", list(texts))
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False, compression={"method": "gzip", "mtime": 0})
    log.info("scored %s (%d texts), revision %s", config.name, len(texts), scorer.resolved_revision)
    return out


def keyword_fires(texts: Sequence[str], segmentation: Mapping[str, Sequence[str]], mode: str) -> np.ndarray:
    """Original (glued) or corrected (separated) keyword rule, per text."""
    if mode == "glued":
        return np.array([legacy_keyword_match(t) is not None for t in texts])
    rule = KeywordRule.from_legacy_keywords()
    return np.array([rule.match(segmentation.get(t, (t,))) is not None for t in texts])


def decide_labels(scores: pd.DataFrame, threshold: float | None, keyword: np.ndarray | None) -> pd.Series:
    """Thresholded arg-max (``None`` = plain arg-max); keyword hits forced to ground_vehicles."""
    mat = scores[[f"score_{label}" for label in LABELS]].to_numpy()
    top = np.asarray(LABELS, dtype=object)[mat.argmax(axis=1)]
    lab = top if threshold is None else np.where(mat.max(axis=1) >= threshold, top, NONMILITARY)
    if keyword is not None:
        lab = np.where(keyword, "ground_vehicles", lab)
    return pd.Series(lab, index=scores["text"].to_numpy())


# -- supervised baseline ---------------------------------------------------------
@dataclass
class BagOfLabels:
    """Binary bag-of-object-labels features from segmented strings."""

    vocabulary: list[str] = field(default_factory=list)

    def fit(self, objects: Sequence[Sequence[str]], min_count: int = 1) -> BagOfLabels:
        from collections import Counter

        counts = Counter(o for objs in objects for o in set(objs))
        self.vocabulary = sorted(o for o, c in counts.items() if c >= min_count)
        return self

    def transform(self, objects: Sequence[Sequence[str]]) -> np.ndarray:
        index = {o: i for i, o in enumerate(self.vocabulary)}
        x = np.zeros((len(objects), len(self.vocabulary)), dtype=np.float32)
        for r, objs in enumerate(objects):
            for o in objs:
                if o in index:
                    x[r, index[o]] = 1.0
        return x


def fit_logistic(x: np.ndarray, y: Sequence[str], weights: np.ndarray | None = None, c: float = 1.0):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=c, max_iter=2000)
    model.fit(x, np.asarray(y), sample_weight=weights)
    return model
