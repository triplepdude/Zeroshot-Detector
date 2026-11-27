"""Keyword rules applied before zero-shot classification.

Two implementations are provided:

* :func:`legacy_keyword_match` reproduces the original behaviour exactly: a raw
  substring test of each keyword against the lower-cased, *concatenated*
  Florence-2 output. Capitalised keywords therefore never match, and matches
  can straddle two object labels (``"persontank"``) or hit unrelated phrases
  (``"tank top"``).
* :class:`KeywordRule` is the corrected rule: case-insensitive, whole-word
  matching against each detected object label separately, with an explicit
  exclusion list.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .taxonomy import DEFAULT_KEYWORD_EXCLUSIONS, LEGACY_MILITARY_KEYWORDS


def legacy_keyword_match(text: str, keywords: Sequence[str] = LEGACY_MILITARY_KEYWORDS) -> str | None:
    """Return the first keyword that the original script would have matched.

    Mirrors ``any(word in text.lower() for word in military_keywords)``.
    """
    lowered = text.lower()
    for keyword in keywords:
        if keyword in lowered:
            return keyword
    return None


def _phrase_pattern(phrase: str) -> str:
    words = [re.escape(word) for word in phrase.lower().split()]
    return r"\s+".join(words)


@dataclass(frozen=True)
class KeywordMatch:
    category: str
    keyword: str
    object_label: str


class KeywordRule:
    """Whole-word, case-insensitive keyword rule over detected object labels.

    Parameters
    ----------
    rules:
        Mapping from category to the keywords that force that category.
    exclusions:
        Phrases that veto a match when they occur in the same object label
        (for example ``"tank top"`` vetoes ``"tank"``).
    """

    def __init__(
        self,
        rules: dict[str, Sequence[str]],
        exclusions: Iterable[str] = DEFAULT_KEYWORD_EXCLUSIONS,
    ) -> None:
        self.rules = {category: tuple(keywords) for category, keywords in rules.items()}
        self._patterns: list[tuple[str, str, re.Pattern[str]]] = []
        for category, keywords in self.rules.items():
            # Longer phrases first so "military truck" is reported over "military".
            for keyword in sorted(keywords, key=len, reverse=True):
                pattern = re.compile(rf"(?<![a-z0-9]){_phrase_pattern(keyword)}(?![a-z0-9])", re.IGNORECASE)
                self._patterns.append((category, keyword, pattern))
        self.exclusions = tuple(exclusions)
        self._exclusion_patterns = [
            re.compile(rf"(?<![a-z0-9]){_phrase_pattern(p)}(?![a-z0-9])", re.IGNORECASE) for p in self.exclusions
        ]

    def match(self, object_labels: Sequence[str]) -> KeywordMatch | None:
        """Return the first rule that fires for any object label, else ``None``."""
        for label in object_labels:
            if any(p.search(label) for p in self._exclusion_patterns):
                continue
            for category, keyword, pattern in self._patterns:
                if pattern.search(label):
                    return KeywordMatch(category=category, keyword=keyword, object_label=label)
        return None

    @classmethod
    def from_legacy_keywords(cls, category: str = "ground_vehicles", **kwargs) -> KeywordRule:
        """Corrected rule using the original keyword list, all mapped to ``category``."""
        return cls({category: LEGACY_MILITARY_KEYWORDS}, **kwargs)
