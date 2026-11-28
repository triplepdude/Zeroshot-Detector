"""NLI-based zero-shot text scoring (BART-MNLI by default).

:class:`NLIZeroShotScorer` implements the same scoring rule as the Hugging Face
``zero-shot-classification`` pipeline used in the original study:

* every (text, label) pair is scored by the NLI model with the hypothesis
  ``hypothesis_template.format(label)``;
* single-label mode (the original setting) applies a softmax over the
  *entailment* logits of all candidate labels;
* multi-label mode applies, per label, a softmax over (contradiction,
  entailment).

It batches pairs by length and can be checked against the reference pipeline
with :meth:`NLIZeroShotScorer.compare_with_pipeline`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np

DEFAULT_NLI_MODEL = "facebook/bart-large-mnli"
#: Default hypothesis template of the Hugging Face zero-shot pipeline (used by the original study).
DEFAULT_HYPOTHESIS_TEMPLATE = "This example is {}."


class ZeroShotScorer(Protocol):
    """Anything that returns one ``{label: score}`` dict per input text."""

    def score(self, texts: Sequence[str], labels: Sequence[str]) -> list[dict[str, float]]: ...


def entailment_softmax(entail_logits: np.ndarray) -> np.ndarray:
    """Single-label zero-shot scores from an ``(n_texts, n_labels)`` entailment-logit matrix."""
    shifted = entail_logits - entail_logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def multilabel_scores(contradiction_logits: np.ndarray, entail_logits: np.ndarray) -> np.ndarray:
    """Independent per-label scores: softmax over (contradiction, entailment)."""
    return 1.0 / (1.0 + np.exp(contradiction_logits - entail_logits))


class NLIZeroShotScorer:
    """Batched NLI zero-shot scorer.

    Parameters
    ----------
    model:
        Hugging Face model id of an NLI sequence-classification model.
    revision:
        Optional model revision (commit hash) for exact reproducibility.
    hypothesis_template:
        Template with a single ``{}`` placeholder for the (verbalised) label.
    verbalizers:
        Optional mapping from label to the text inserted into the hypothesis
        (e.g. ``{"ground_vehicles": "military ground vehicles"}``). Labels not
        in the mapping are inserted verbatim, as in the original study.
    multi_label:
        Score labels independently instead of competing (original: ``False``).
    """

    def __init__(
        self,
        model: str = DEFAULT_NLI_MODEL,
        revision: str | None = None,
        hypothesis_template: str = DEFAULT_HYPOTHESIS_TEMPLATE,
        verbalizers: Mapping[str, str] | None = None,
        multi_label: bool = False,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        if "{}" not in hypothesis_template:
            raise ValueError("hypothesis_template must contain '{}'")
        self.model_id = model
        self.revision = revision
        self.hypothesis_template = hypothesis_template
        self.verbalizers = dict(verbalizers or {})
        self.multi_label = multi_label
        self.batch_size = batch_size
        self._device = device
        self._model = None
        self._tokenizer = None

    # -- model loading -----------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id, revision=self.revision)
        self._model.to(device).eval()
        self._device = device
        label2id = {k.lower(): v for k, v in self._model.config.label2id.items()}
        entail = [v for k, v in label2id.items() if k.startswith("entail")]
        contra = [v for k, v in label2id.items() if k.startswith("contra")]
        if not entail or not contra:
            raise ValueError(f"cannot find entailment/contradiction ids in {label2id}")
        self.entailment_id, self.contradiction_id = entail[0], contra[0]

    @property
    def resolved_revision(self) -> str | None:
        """Commit hash of the loaded model weights (``None`` before loading)."""
        if self._model is None:
            return None
        return getattr(self._model.config, "_commit_hash", None)

    def hypothesis(self, label: str) -> str:
        return self.hypothesis_template.format(self.verbalizers.get(label, label))

    # -- scoring -----------------------------------------------------------
    def logits(self, texts: Sequence[str], labels: Sequence[str]) -> np.ndarray:
        """Raw NLI logits with shape ``(n_texts, n_labels, n_classes)``."""
        self._load()
        torch = self._torch
        hypotheses = [self.hypothesis(label) for label in labels]
        pairs = [(i, j) for i in range(len(texts)) for j in range(len(labels))]
        # Sort by premise length so batches have similar padding.
        order = sorted(range(len(pairs)), key=lambda k: len(texts[pairs[k][0]]))
        n_classes = self._model.config.num_labels
        out = np.zeros((len(texts), len(labels), n_classes), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                chunk = [pairs[k] for k in order[start : start + self.batch_size]]
                enc = self._tokenizer(
                    [texts[i] for i, _ in chunk],
                    [hypotheses[j] for _, j in chunk],
                    return_tensors="pt",
                    padding=True,
                    truncation="only_first",
                ).to(self._device)
                logits = self._model(**enc).logits.float().cpu().numpy()
                for (i, j), row in zip(chunk, logits, strict=True):
                    out[i, j] = row
        return out

    def score_matrix(self, texts: Sequence[str], labels: Sequence[str]) -> np.ndarray:
        """Zero-shot scores with shape ``(n_texts, n_labels)``."""
        if len(texts) == 0:
            return np.zeros((0, len(labels)))
        logits = self.logits(texts, labels)
        entail = logits[..., self.entailment_id].astype(np.float64)
        if self.multi_label:
            return multilabel_scores(logits[..., self.contradiction_id].astype(np.float64), entail)
        return entailment_softmax(entail)

    def score(self, texts: Sequence[str], labels: Sequence[str]) -> list[dict[str, float]]:
        matrix = self.score_matrix(texts, labels)
        return [dict(zip(labels, map(float, row), strict=True)) for row in matrix]

    def compare_with_pipeline(self, texts: Sequence[str], labels: Sequence[str]) -> float:
        """Max absolute score difference vs. ``transformers.pipeline('zero-shot-classification')``."""
        from transformers import pipeline

        self._load()
        pipe = pipeline("zero-shot-classification", model=self._model, tokenizer=self._tokenizer, device=self._device)
        ours = self.score_matrix(texts, labels)
        worst = 0.0
        verbalised = [self.verbalizers.get(label, label) for label in labels]
        for i, text in enumerate(texts):
            ref = pipe(
                text,
                candidate_labels=verbalised,
                hypothesis_template=self.hypothesis_template,
                multi_label=self.multi_label,
            )
            ref_scores = dict(zip(ref["labels"], ref["scores"], strict=True))
            for j, label in enumerate(verbalised):
                worst = max(worst, abs(ref_scores[label] - ours[i, j]))
        return worst
