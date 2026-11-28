"""Researcher-decided label table and the deterministic *rule reference*.

Every entry below is a decision by the study authors (see
``docs/experiments/CODEBOOK.md``), not a property of the data. The table maps
detector labels (as produced by :mod:`zeroshot_detector.segment`) to an
equipment category, a military *tier* and a role:

* tiers for vehicle labels: **T3** inherently military, **T2** military-typical
  dual-use, **T1** everyday civilian; weapons carry no tier and are never gated;
* roles: ``equipment``, ``parent`` (generic name of a family; adds nothing when
  a more specific label of the same family is present), ``part`` (never creates
  a category), ``not_equipment`` (look-alikes such as ``paddle``),
  ``cue_military`` / ``cue_civilian`` (change scores only), ``depiction``
  (equipment shown on a screen/print), ``toy``, ``person``.

Labels not in the table are not equipment and not cues.

:func:`rule_reference` applies the codebook rules mechanically to a list of
object labels and returns, for every category present, a 0–3 military score,
the primary category and the label under each reading.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import pandas as pd

from ..taxonomy import NONMILITARY

VEHICLE_CATEGORIES = ("aircraft", "maritime", "ground_vehicles")
CATEGORY_ORDER = ("weapons", "aircraft", "maritime", "ground_vehicles")  # the study's label order

# label -> (category, tier, role)
TABLE: dict[str, tuple[str | None, int | None, str]] = {
    # weapons (never gated)
    "weapon": ("weapons", None, "parent"),
    "rifle": ("weapons", None, "equipment"),
    "handgun": ("weapons", None, "equipment"),
    "shotgun": ("weapons", None, "equipment"),
    "missile": ("weapons", None, "equipment"),
    "cannon": ("weapons", None, "equipment"),
    "sword": ("weapons", None, "equipment"),
    "dagger": ("weapons", None, "equipment"),
    "bow and arrow": ("weapons", None, "equipment"),
    "rocket": ("weapons", None, "conditional_weapon"),  # weapon only with missile, weapon or tank
    "knife": ("weapons", None, "conditional_weapon"),  # weapon only with another weapon or a military cue
    "kitchen knife": (None, None, "not_equipment"),
    # aircraft
    "aircraft": ("aircraft", 2, "parent"),
    "helicopter": ("aircraft", 2, "equipment"),
    "airplane": ("aircraft", 2, "equipment"),
    # maritime
    "watercraft": ("maritime", 2, "parent"),
    "boat": ("maritime", 2, "equipment"),
    "canoe": ("maritime", 1, "equipment"),
    "jet ski": ("maritime", 1, "equipment"),
    "kayak": ("maritime", 1, "equipment"),
    # ground vehicles
    "tank": ("ground_vehicles", 3, "equipment"),
    "truck": ("ground_vehicles", 2, "equipment"),
    "pickup truck": ("ground_vehicles", 2, "equipment"),
    "land vehicle": ("ground_vehicles", 1, "parent"),
    "vehicle": (None, None, "parent"),  # alone: no category
    **{
        lab: ("ground_vehicles", 1, "equipment")
        for lab in (
            "car",
            "van",
            "suv",
            "taxi",
            "limousine",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
            "ambulance",
            "golf cart",
            "snowplow",
            "cart",
            "wheelchair",
            "segway",
            "snowmobile",
        )
    },
    # parts (never create a category)
    **{
        lab: (None, None, "part")
        for lab in (
            "wheel",
            "car tire",
            "tire",
            "vehicle registration plate",
            "seat belt",
            "steering wheel",
            "propeller",
            "taillight",
            "headlight",
            "windshield wiper",
            "rearview mirror",
            "bicycle wheel",
        )
    },
    # look-alikes that are not equipment
    **{
        lab: (None, None, "not_equipment")
        for lab in (
            "parachute",
            "balloon",
            "kite",
            "life jacket",
            "paddle",
            "marine mammal",
            "swimming pool",
            "surfboard",
            "tank top",
            "harbor seal",
            "dolphin",
            "whale",
            "shark",
        )
    },
    # cues
    "helmet": (None, None, "cue_military_with_weapon"),
    **{
        lab: (None, None, "cue_civilian")
        for lab in (
            "luggage and bags",
            "suitcase",
            "swimming pool",
            "fish",
            "ing rod",
            "surfboard",
            "life jacket",
            "swimwear",
        )
    },
    **{
        lab: (None, None, "depiction")
        for lab in (
            "poster",
            "television",
            "computer monitor",
            "laptop",
            "mobile phone",
            "picture frame",
            "billboard",
            "tablet computer",
        )
    },
    "toy": (None, None, "toy"),
    **{lab: (None, None, "person") for lab in ("person", "man", "woman", "boy", "girl", "human face")},
}
# A few labels have two roles (e.g. swimming pool is a look-alike and a civilian cue).
CIVILIAN_CUES = frozenset(k for k, v in TABLE.items() if v[2] == "cue_civilian") | {
    "swimming pool",
    "surfboard",
    "life jacket",
}
DEPICTIONS = frozenset(k for k, v in TABLE.items() if v[2] == "depiction")
PERSONS = frozenset(k for k, v in TABLE.items() if v[2] == "person")
INHERENT_MILITARY = frozenset({"tank", "missile", "cannon"})
SPECIFIC_WEAPONS = frozenset(k for k, v in TABLE.items() if v[0] == "weapons" and v[2] == "equipment")


@dataclass
class RuleLabel:
    """The rule reference for one object list."""

    categories: list[str] = field(default_factory=list)
    scores: dict[str, int] = field(default_factory=dict)  # vehicle categories only
    primary: str = "none"
    depiction: str = "real_or_unclear"
    armament: bool = False
    readings: dict[str, str] = field(default_factory=dict)
    primary_weapon_rule: str = "none"  # sensitivity: primary without the armament rule


def _weapons_present(objs: set[str]) -> bool:
    if objs & SPECIFIC_WEAPONS or "weapon" in objs:
        return True
    return any(_conditional_weapon_qualifies(o, objs) for o in objs & {"rocket", "knife"})


def _conditional_weapon_qualifies(o: str, objs: set[str]) -> bool:
    if o == "rocket":
        return bool(objs & {"missile", "weapon", "tank"})
    if o == "knife":
        return bool(objs & (SPECIFIC_WEAPONS | {"weapon"}) or objs & INHERENT_MILITARY)
    return False


def _equipment_items(objs: set[str]) -> list[str]:
    """Distinct equipment items after the parent, part and conditional rules (input to the toy rule).

    A generic parent (``weapon``, ``land vehicle``, ``aircraft``, ``watercraft``)
    names the same object as a specific label of its family and is not a separate
    item; a conditional weapon (``rocket``, ``knife``) is an item only when its
    condition holds; parts, cues and ``vehicle`` never are.
    """
    specific = {
        o
        for o in objs
        if o in TABLE
        and TABLE[o][0] is not None
        and (
            TABLE[o][2] == "equipment"
            or (TABLE[o][2] == "conditional_weapon" and _conditional_weapon_qualifies(o, objs))
        )
    }
    items = []
    for o in sorted(objs):
        if o not in TABLE:
            continue
        cat, _, role = TABLE[o]
        if cat is None:
            continue
        if role == "parent" and any(TABLE[s][0] == cat for s in specific):
            continue
        if role == "conditional_weapon" and o not in specific:
            continue
        items.append(o)
    return items


def _vehicle_tiers(objs: set[str]) -> dict[str, int]:
    """Max tier per vehicle category, applying the parent rules."""
    tiers: dict[str, int] = {}
    specific = {o for o in objs if o in TABLE and TABLE[o][2] == "equipment" and TABLE[o][0] in VEHICLE_CATEGORIES}
    for o in objs:
        if o not in TABLE:
            continue
        cat, tier, role = TABLE[o]
        if cat not in VEHICLE_CATEGORIES:
            continue
        if role == "parent" and any(TABLE[s][0] == cat for s in specific):
            continue  # the parent names the same object as a specific label
        tiers[cat] = max(tiers.get(cat, 0), tier)
    return tiers


def _pick(order_tiers: dict[str, int], qualifying: Iterable[str], armament: bool, weapons: bool) -> str:
    """Most salient qualifying category, following the codebook's ordered rules."""
    q = [c for c in CATEGORY_ORDER if c in set(qualifying)]
    if not q:
        return NONMILITARY
    vehicles = [c for c in q if c in VEHICLE_CATEGORIES]
    if armament and vehicles:  # rule 1: a weapon without a person is the vehicle's armament
        return max(vehicles, key=lambda c: (order_tiers.get(c, 0), -CATEGORY_ORDER.index(c)))
    t3 = [c for c in vehicles if order_tiers.get(c, 0) == 3]
    if t3:  # rule 2: an inherently military object is primary
        return t3[0]
    if "weapons" in q and weapons:  # rule 3: otherwise weapons
        return "weapons"
    return max(vehicles, key=lambda c: (order_tiers.get(c, 0), -CATEGORY_ORDER.index(c)))


def rule_reference(objects: Sequence[str]) -> RuleLabel:
    objs = set(objects)
    if objs == {"No object detected."}:
        return RuleLabel(readings={r: NONMILITARY for r in ("equipment", "type", "lenient", "strict")})
    weapons = _weapons_present(objs)
    tiers = _vehicle_tiers(objs)
    # toys: "toy" with exactly one equipment item (counted after the parent and conditional rules) -> it is the toy
    depiction = "real_or_unclear"
    if "toy" in objs and len(_equipment_items(objs)) == 1:
        depiction = "toy_or_model"
        weapons, tiers = False, {}
    elif objs & DEPICTIONS and (weapons or tiers):
        depiction = "screen_or_print"
    persons = bool(objs & PERSONS)
    armament = weapons and bool(tiers) and not persons
    military_cue = bool(objs & INHERENT_MILITARY) or ("helmet" in objs and weapons)
    civilian_cue = bool(objs & CIVILIAN_CUES)
    scores = {}
    for cat, tier in tiers.items():
        if tier == 3 or armament:
            scores[cat] = 3
        elif tier == 2:
            scores[cat] = 3 if military_cue else (1 if civilian_cue else 2)
        else:  # T1
            scores[cat] = 2 if military_cue else 0
    categories = [c for c in CATEGORY_ORDER if (c == "weapons" and weapons) or c in tiers]
    primary = _pick(tiers, categories, armament, weapons)
    readings = {
        "equipment": primary if primary != "none" else NONMILITARY,
        "type": _pick(tiers, [c for c in categories if c == "weapons" or scores[c] >= 1], armament, weapons),
        "lenient": _pick(tiers, [c for c in categories if c == "weapons" or scores[c] >= 2], armament, weapons),
        "strict": _pick(tiers, [c for c in categories if c == "weapons" or scores[c] == 3], armament, weapons),
    }
    # sensitivity: the ordered rules without rule 1 (armament); a tank stays primary
    weapon_first = _pick(tiers, categories, False, weapons)
    return RuleLabel(
        categories=categories,
        scores=scores,
        primary=primary if primary != NONMILITARY else "none",
        depiction=depiction,
        armament=armament,
        readings=readings,
        primary_weapon_rule=weapon_first if weapon_first != NONMILITARY else "none",
    )


def label_table_frame(label_frames: dict[str, int] | None = None) -> pd.DataFrame:
    """The label table as a DataFrame (optionally with frame counts)."""
    rows = []
    for label, (cat, tier, role) in sorted(TABLE.items()):
        rows.append(
            {
                "label": label,
                "category": cat or "",
                "tier": "" if tier is None else f"T{tier}",
                "role": role,
                "frames": (label_frames or {}).get(label, ""),
                "civilian_cue": label in CIVILIAN_CUES,
            }
        )
    return pd.DataFrame(rows)
