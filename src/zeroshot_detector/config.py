"""Run configuration (YAML ↔ dataclasses).

Secrets are never read from the YAML file: MinIO/S3 credentials come from the
``MINIO_ACCESS_KEY`` / ``MINIO_SECRET_KEY`` environment variables.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .taxonomy import LABELS, LEGACY_MILITARY_KEYWORDS, validate_labels
from .zero_shot import DEFAULT_HYPOTHESIS_TEMPLATE, DEFAULT_NLI_MODEL

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


@dataclass
class SourceConfig:
    kind: str = "local"  #: ``local`` or ``minio``
    path: str | None = None  #: directory of videos (``local``)
    endpoint: str | None = None  #: ``host:port`` (``minio``)
    buckets: list[str] = field(default_factory=list)
    prefix: str = ""
    secure: bool = True
    extensions: list[str] = field(default_factory=lambda: list(VIDEO_EXTENSIONS))

    def validate(self) -> None:
        if self.kind == "local" and not self.path:
            raise ValueError("source.path is required for a local source")
        if self.kind == "minio" and (not self.endpoint or not self.buckets):
            raise ValueError("source.endpoint and source.buckets are required for a minio source")
        if self.kind not in ("local", "minio"):
            raise ValueError(f"unknown source.kind {self.kind!r}")


@dataclass
class SamplingConfig:
    every_n_frames: int = 30  #: original study: every 30th decoded frame
    every_seconds: float | None = None  #: if set, the step is round(fps * every_seconds)
    #: Skip a sample whose L1 pixel distance to the previous kept sample is <= this
    #: (original: 5000, which only removes near-identical frames). ``None`` disables.
    dedup_l1_threshold: float | None = 5000.0
    max_samples_per_video: int | None = None

    def validate(self) -> None:
        if self.every_seconds is None and self.every_n_frames < 1:
            raise ValueError("sampling.every_n_frames must be >= 1")
        if self.every_seconds is not None and self.every_seconds <= 0:
            raise ValueError("sampling.every_seconds must be > 0")


@dataclass
class DetectorConfig:
    model: str = "microsoft/Florence-2-large"
    revision: str | None = None  #: pin a commit hash for exact reproducibility
    task: str = "<OD>"
    num_beams: int = 3
    max_new_tokens: int = 1024
    batch_size: int = 8
    dtype: str = "float32"
    device: str | None = None
    #: ``auto`` (native for ``florence-community/*`` ids, else remote), ``remote`` or ``native``
    backend: str = "auto"


@dataclass
class ClassifierConfig:
    model: str = DEFAULT_NLI_MODEL
    revision: str | None = None
    labels: list[str] = field(default_factory=lambda: list(LABELS))
    hypothesis_template: str = DEFAULT_HYPOTHESIS_TEMPLATE
    verbalizers: dict[str, str] = field(default_factory=dict)
    multi_label: bool = False
    threshold: float = 0.8
    #: ``True`` reproduces the original study exactly (glued labels, substring keywords).
    legacy: bool = False
    keyword_category: str = "ground_vehicles"
    keywords: list[str] = field(default_factory=lambda: list(LEGACY_MILITARY_KEYWORDS))
    always_score: bool = True
    batch_size: int = 32
    device: str | None = None

    def validate(self) -> None:
        validate_labels(self.labels)
        if not 0 <= self.threshold <= 1:
            raise ValueError("classifier.threshold must be in [0, 1]")


@dataclass
class AggregationConfig:
    rule: str = "plurality"  #: legacy | plurality | military_fraction
    min_fraction: float | None = None  #: required for ``military_fraction``

    def validate(self) -> None:
        if self.rule == "military_fraction" and not self.min_fraction:
            raise ValueError("aggregation.min_fraction is required for rule military_fraction")
        if self.rule not in ("legacy", "plurality", "military_fraction"):
            raise ValueError(f"unknown aggregation.rule {self.rule!r}")

    def kwargs(self) -> dict:
        return {"min_fraction": self.min_fraction} if self.rule == "military_fraction" else {}


@dataclass
class OutputConfig:
    dir: str = "runs"
    #: Also copy each video into ``<dir>/<run>/sorted/<category>/`` like the original script.
    copy_videos: bool = False
    #: Keep Florence-2's raw output (with location tokens) in frames.jsonl.
    keep_raw_text: bool = True


@dataclass
class Config:
    source: SourceConfig = field(default_factory=SourceConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    seed: int = 0

    def validate(self) -> Config:
        self.source.validate()
        self.sampling.validate()
        self.classifier.validate()
        self.aggregation.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        """Short hash of the full configuration (recorded in run manifests)."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


_SECTIONS = {
    "source": SourceConfig,
    "sampling": SamplingConfig,
    "detector": DetectorConfig,
    "classifier": ClassifierConfig,
    "aggregation": AggregationConfig,
    "output": OutputConfig,
}


def config_from_dict(raw: dict[str, Any]) -> Config:
    unknown = set(raw) - set(_SECTIONS) - {"seed"}
    if unknown:
        raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        section = raw.get(name) or {}
        allowed = {f.name for f in dataclasses.fields(cls)}
        bad = set(section) - allowed
        if bad:
            raise ValueError(f"unknown keys in [{name}]: {sorted(bad)}")
        kwargs[name] = cls(**section)
    if "seed" in raw:
        kwargs["seed"] = int(raw["seed"])
    return Config(**kwargs).validate()


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for dotted, value in (overrides or {}).items():
        section, _, key = dotted.partition(".")
        if key:
            raw.setdefault(section, {})[key] = value
        else:
            raw[section] = value
    return config_from_dict(raw)
