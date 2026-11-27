"""Label taxonomy and the keyword list used by the original study."""

from __future__ import annotations

#: Category labels, in the order used by the original script. The order
#: matters for legacy tie-breaking (see :mod:`zeroshot_detector.aggregate`).
LABELS: tuple[str, ...] = ("weapons", "aircraft", "maritime", "ground_vehicles", "nonmilitary")

NONMILITARY = "nonmilitary"
MILITARY_LABELS: tuple[str, ...] = tuple(label for label in LABELS if label != NONMILITARY)

#: Keyword list copied verbatim from ``Recursive Florence 2 + Bart + Confidence.py``
#: (order and capitalisation preserved). In the original code, any keyword found
#: in the lower-cased Florence-2 output forces the frame to ``ground_vehicles``.
#: The capitalised entries ("APC", "MRAP", "Stryker", "Bradley") can never match a
#: lower-cased string; this is reproduced in legacy mode on purpose.
LEGACY_MILITARY_KEYWORDS: tuple[str, ...] = (
    "armored",
    "convoy",
    "military",
    "humvee",
    "APC",
    "tank",
    "infantry",
    "MRAP",
    "Stryker",
    "Bradley",
    "military truck",
    "artillery",
    "howitzer",
    "patrol vehicle",
    "combat vehicle",
    "troop carrier",
    "military convoy",
    "tracked vehicle",
)

#: Phrases that contain a keyword but are not military vehicles. Only used by the
#: corrected (non-legacy) keyword rule.
DEFAULT_KEYWORD_EXCLUSIONS: tuple[str, ...] = (
    "tank top",
    "fish tank",
    "water tank",
    "gas tank",
    "fuel tank",
    "septic tank",
    "storage tank",
    "think tank",
)


def validate_labels(labels: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return ``labels`` as a tuple after checking it is usable as a taxonomy."""
    labels = tuple(labels)
    if len(labels) < 2:
        raise ValueError("a taxonomy needs at least two labels")
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate labels in taxonomy: {labels}")
    if NONMILITARY not in labels:
        raise ValueError(f"taxonomy must contain the fallback label {NONMILITARY!r}")
    return labels
