"""Build the annotation materials for the v2 reference standard (PLAN §5–§6).

Writes to ``data/reference/``:

* ``strings.csv.gz``: every population string with frame count, objects, system
  labels and the rule reference;
* ``design_records.csv``: the sample (C/S/T records with weights);
* ``items.csv``: everything shown to annotators (sample strings + embedded checks);
* ``batches.json``: per configuration, batches of 25 items (item order and
  object order already applied), plus single-item and extra-pass batches;
* ``instruction.md``: the exact instruction text (header + codebook v2);
* ``MANIFEST.json``: seed, counts, SHA-256 of codebook, instruction and files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import sha256
from .design import NO_OBJECT, draw_string_design
from .labels import rule_reference
from .reference import item_id
from .rule_reference import population, string_table

SEED = 20251128
K_CERTAINTY = 500
N_TAIL = 30
TAIL_OVERRIDES = {"NN-vehicle-token": 120, "NN-no-equipment-token": 40}
BATCH_SIZE = 25
HEAVY_SHARE = 0.005
N_HUMAN_EXTRA = 150
RETEST_SHARE = 0.10
N_SINGLE_ITEM = 100

#: Synthetic strings (absent from the archive) whose answers follow from single rules.
SYNTHETIC = [
    ("tankboat", ["tank", "boat"]),
    ("truckflag", ["truck", "flag"]),
    ("boatrifle", ["boat", "rifle"]),
    ("pickup truckrifle", ["pickup truck", "rifle"]),
    ("canoeweapon", ["canoe", "weapon"]),
    ("helicopterluggage and bags", ["helicopter", "luggage and bags"]),
    ("boatfishsurfboard", ["boat", "fish", "surfboard"]),
    ("carlimousine", ["car", "limousine"]),
    ("truckwheelweapon", ["truck", "wheel", "weapon"]),
    ("airplanerocket", ["airplane", "rocket"]),
    ("helmetmotorcycleperson", ["helmet", "motorcycle", "person"]),
    ("helicopterpersonsword", ["helicopter", "person", "sword"]),
]

CONFIGS = {
    "A": {"tier": "primary", "objects": "archived"},
    "B": {"tier": "secondary", "objects": "archived"},
    "C": {"tier": "primary", "objects": "shuffled"},
}
EXTRA_CONFIGS = {
    "D": {"tier": "secondary", "objects": "shuffled"},
    "E": {"tier": "primary", "objects": "reversed"},
}

HEADER = (
    "You are annotating items for a research study. Follow the codebook below exactly. "
    "Do not use any tools, do not read or search any files, and do not consult anything other than this "
    "message. For every item in the JSON list at the end, return one answer that echoes its item_id and raw "
    "exactly.\n\n"
)


def instruction_text(codebook: str) -> str:
    return HEADER + codebook.strip() + "\n"


def prompt_for(instruction: str, items: list[dict]) -> str:
    """The exact prompt sent for one batch (the workflow concatenates the same way)."""
    return instruction + "\n## Items\n\n" + json.dumps(items, ensure_ascii=False, indent=1) + "\n"


def _fresh_id(text: str, salt: str) -> str:
    return hashlib.sha1(f"{salt}:{text}".encode()).hexdigest()[:10]


def _arrange(objects: list[str], mode: str, rng: np.random.Generator) -> list[str]:
    if mode == "shuffled":
        return [objects[i] for i in rng.permutation(len(objects))]
    if mode == "reversed":
        return list(reversed(objects))
    return list(objects)


def build(frames: pd.DataFrame, codebook_path: str | Path, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pop = population(frames)
    strings = string_table(frames)
    records = draw_string_design(strings, pop, K_CERTAINTY, N_TAIL, TAIL_OVERRIDES, SEED)
    records["item_id"] = records["text"].map(item_id)
    total_frames = len(pop)

    objects_of = dict(zip(strings["text"], strings["objects"], strict=True))
    sample_texts = [t for t in dict.fromkeys(records["text"]) if t != NO_OBJECT]
    heavy = [t for t, m in zip(strings["text"], strings["frames"], strict=True) if m / total_frames > HEAVY_SHARE]
    heavy = [t for t in heavy if t != NO_OBJECT]

    human = pop.loc[pop["human_label"].notna(), "florence_text"].drop_duplicates()
    human_outside = sorted(set(human) - set(sample_texts))
    human_extra = list(rng.choice(human_outside, size=min(N_HUMAN_EXTRA, len(human_outside)), replace=False))

    items = []
    for t in sample_texts:
        items.append(
            {"item_id": item_id(t), "raw": t, "objects": objects_of[t], "kind": "sample", "source_item_id": ""}
        )
    for t in human_extra:
        items.append(
            {
                "item_id": item_id(t),
                "raw": t,
                "objects": objects_of[t],
                "kind": "human_consistency",
                "source_item_id": "",
            }
        )
    for raw, objs in SYNTHETIC:
        r = rule_reference(objs)
        items.append(
            {
                "item_id": item_id(raw),
                "raw": raw,
                "objects": "; ".join(objs),
                "kind": "synthetic",
                "source_item_id": "",
                "expected": json.dumps(
                    {"categories": r.categories, "primary": r.primary, "scores": r.scores, "depiction": r.depiction}
                ),
            }
        )
    items_df = pd.DataFrame(items)
    if items_df["item_id"].duplicated().any():  # pragma: no cover
        raise ValueError("duplicate item ids")

    # retest items per configuration (fresh ids, half heavy)
    n_retest = round(RETEST_SHARE * len(sample_texts))
    batches = []
    retest_rows = []
    instruction = instruction_text(Path(codebook_path).read_text())
    base_items = items_df[items_df["kind"] != "retest"]
    for cfg_name, cfg in CONFIGS.items():
        crng = np.random.default_rng(SEED + sum(map(ord, cfg_name)))
        heavy_pick = list(crng.choice(heavy, size=min(len(heavy), n_retest // 2), replace=False))
        others = [t for t in sample_texts if t not in heavy_pick]
        other_pick = list(crng.choice(others, size=n_retest - len(heavy_pick), replace=False))
        retest = [
            {
                "item_id": _fresh_id(t, f"retest-{cfg_name}"),
                "raw": t,
                "objects": objects_of[t],
                "kind": "retest",
                "source_item_id": item_id(t),
                "config": cfg_name,
            }
            for t in heavy_pick + other_pick
        ]
        retest_rows.extend(retest)
        pool = [dict(r) for r in base_items.to_dict("records")] + retest
        order = crng.permutation(len(pool))
        for b, start in enumerate(range(0, len(pool), BATCH_SIZE)):
            chunk = [pool[i] for i in order[start : start + BATCH_SIZE]]
            shown = [
                {
                    "item_id": c["item_id"],
                    "raw": c["raw"],
                    "objects": "; ".join(_arrange(c["objects"].split("; "), cfg["objects"], crng)),
                }
                for c in chunk
            ]
            batches.append({"config": cfg_name, "tier": cfg["tier"], "batch_id": f"{cfg_name}-{b:03d}", "items": shown})
    # extra passes for heavy strings
    for cfg_name, cfg in EXTRA_CONFIGS.items():
        crng = np.random.default_rng(SEED + sum(map(ord, cfg_name)))
        order = crng.permutation(len(heavy))
        shown = [
            {
                "item_id": item_id(heavy[i]),
                "raw": heavy[i],
                "objects": "; ".join(_arrange(objects_of[heavy[i]].split("; "), cfg["objects"], crng)),
            }
            for i in order
        ]
        for b, start in enumerate(range(0, len(shown), BATCH_SIZE)):
            batches.append(
                {
                    "config": cfg_name,
                    "tier": cfg["tier"],
                    "batch_id": f"{cfg_name}-{b:03d}",
                    "items": shown[start : start + BATCH_SIZE],
                }
            )
    # single-item calls (configuration A1)
    srng = np.random.default_rng(SEED + 99)
    single = list(srng.choice(sample_texts, size=min(N_SINGLE_ITEM, len(sample_texts)), replace=False))
    for i, t in enumerate(single):
        batches.append(
            {
                "config": "A1",
                "tier": "primary",
                "batch_id": f"A1-{i:03d}",
                "items": [{"item_id": item_id(t), "raw": t, "objects": objects_of[t]}],
            }
        )

    items_all = pd.concat([items_df, pd.DataFrame(retest_rows)], ignore_index=True)
    strings.to_csv(out / "strings.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    records.to_csv(out / "design_records.csv", index=False)
    items_all.to_csv(out / "items.csv", index=False)
    (out / "batches.json").write_text(json.dumps(batches, ensure_ascii=False, indent=1) + "\n")
    (out / "instruction.md").write_text(instruction)
    manifest = {
        "seed": SEED,
        "k_certainty": K_CERTAINTY,
        "n_tail": N_TAIL,
        "tail_overrides": TAIL_OVERRIDES,
        "batch_size": BATCH_SIZE,
        "population_frames": int(total_frames),
        "population_strings": len(strings),
        "records": len(records),
        "sample_strings_annotated": len(sample_texts),
        "heavy_strings": len(heavy),
        "human_extra": len(human_extra),
        "synthetic": len(SYNTHETIC),
        "retest_per_config": n_retest,
        "single_item_calls": len(single),
        "batches": {c: sum(1 for b in batches if b["config"] == c) for c in [*CONFIGS, *EXTRA_CONFIGS, "A1"]},
        "configs": {**CONFIGS, **EXTRA_CONFIGS, "A1": {"tier": "primary", "objects": "archived", "single_item": True}},
        "sha256": {
            "CODEBOOK.md": sha256(codebook_path),
            **{
                p.name: sha256(p)
                for p in (
                    out / "instruction.md",
                    out / "batches.json",
                    out / "items.csv",
                    out / "design_records.csv",
                    out / "strings.csv.gz",
                )
            },
        },
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
