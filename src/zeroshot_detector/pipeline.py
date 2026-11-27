"""End-to-end video classification: sample → detect → classify → aggregate.

Outputs (in ``<output.dir>/<run_name>/``):

``frames.jsonl``
    one JSON object per sampled frame (objects, boxes, model input, all
    zero-shot scores, decision source, ...).
``videos.csv``
    one row per video (frame counts per label, decided category, sampling
    statistics, status). Used to resume an interrupted run.
``manifest.json``
    configuration, its fingerprint, package versions, model revisions, git
    commit and timing.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from pathlib import Path

from .aggregate import aggregate_video
from .classify import SOURCE_BELOW_THRESHOLD, SOURCE_EMPTY, FrameClassifier
from .config import Config
from .florence import FlorenceDetector, FrameDetections
from .manifest import base_manifest, finish
from .sources import VideoItem, make_source
from .video import SamplingStats, iter_sampled_frames
from .zero_shot import NLIZeroShotScorer

log = logging.getLogger(__name__)

VIDEO_FIELDS = [
    "video_id",
    "container",
    "key",
    "status",
    "error",
    "category",
    "rule",
    "tie",
    "n_frames",
    "fps",
    "decoded_frames",
    "reported_frame_count",
    "dropped_as_duplicates",
    "seconds",
]


def build_components(cfg: Config) -> tuple[FlorenceDetector, FrameClassifier]:
    d, c = cfg.detector, cfg.classifier
    detector = FlorenceDetector(
        model=d.model,
        revision=d.revision,
        task=d.task,
        num_beams=d.num_beams,
        max_new_tokens=d.max_new_tokens,
        dtype=d.dtype,
        device=d.device,
        backend=d.backend,
    )
    scorer = NLIZeroShotScorer(
        model=c.model,
        revision=c.revision,
        hypothesis_template=c.hypothesis_template,
        verbalizers=c.verbalizers,
        multi_label=c.multi_label,
        batch_size=c.batch_size,
        device=c.device,
    )
    classifier = FrameClassifier(
        scorer,
        labels=c.labels,
        threshold=c.threshold,
        legacy=c.legacy,
        keyword_category=c.keyword_category,
        keywords=c.keywords,
        always_score=c.always_score,
    )
    return detector, classifier


def _detect_batched(detector: FlorenceDetector, images: list, batch_size: int) -> list[FrameDetections]:
    out: list[FrameDetections] = []
    i = 0
    while i < len(images):
        batch = images[i : i + batch_size]
        try:
            out.extend(detector.detect(batch))
            i += len(batch)
        except RuntimeError as exc:  # CUDA OOM → retry with smaller batches
            if "out of memory" not in str(exc).lower() or batch_size == 1:
                raise
            import torch

            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            log.warning("out of memory; retrying with batch size %d", batch_size)
    return out


def process_video(
    path: Path,
    video_id: str,
    cfg: Config,
    detector: FlorenceDetector,
    classifier: FrameClassifier,
    chunk_frames: int = 64,
) -> tuple[list[dict], dict]:
    """Classify one local video file; returns (frame records, video record).

    Frames are streamed in chunks of ``chunk_frames`` so memory use does not
    grow with video length.
    """
    from PIL import Image

    stats = SamplingStats()
    s = cfg.sampling
    frame_iter = iter_sampled_frames(
        path,
        every_n_frames=s.every_n_frames,
        every_seconds=s.every_seconds,
        dedup_l1_threshold=s.dedup_l1_threshold,
        max_samples=s.max_samples_per_video,
        stats=stats,
    )
    frame_records: list[dict] = []
    categories: list[str | None] = []
    sources: list[str] = []

    def flush(chunk: list) -> None:
        images = [Image.fromarray(f.rgb) for f in chunk]
        detections = _detect_batched(detector, images, cfg.detector.batch_size)
        if classifier.legacy:
            decisions = classifier.classify_texts([d.legacy_text for d in detections])
        else:
            decisions = classifier.classify_objects([d.labels for d in detections])
        for frame, det, dec in zip(chunk, detections, decisions, strict=True):
            record = {
                "video_id": video_id,
                "frame_index": frame.index,
                "timestamp": frame.timestamp,
                "objects": det.labels,
                "boxes": [[d.label, *d.box] for d in det.detections],
                "legacy_text": det.legacy_text,
                **{k: v for k, v in dec.to_dict().items() if k != "text"},
                "model_input": dec.text,
            }
            if cfg.output.keep_raw_text:
                record["raw_text"] = det.raw_text
            frame_records.append(record)
            categories.append(dec.category)
            sources.append(dec.source)

    chunk: list = []
    for frame in frame_iter:
        chunk.append(frame)
        if len(chunk) >= chunk_frames:
            flush(chunk)
            chunk = []
    if chunk:
        flush(chunk)

    kwargs = cfg.aggregation.kwargs()
    if cfg.aggregation.rule == "legacy":
        # Original override: below-threshold (and empty) frames vs. half of *all decoded* frames.
        import numpy as np

        n_below = sum(src in (SOURCE_BELOW_THRESHOLD, SOURCE_EMPTY) for src in sources)
        kwargs = {"below_threshold_counts": np.array([n_below]), "total_frames": np.array([stats.decoded_frames])}
    decision = aggregate_video(categories, rule=cfg.aggregation.rule, labels=classifier.labels, **kwargs)
    video_record = {
        "category": decision.category,
        "rule": decision.rule,
        "tie": decision.tie,
        "n_frames": decision.n_frames,
        **{f"n_{label}": n for label, n in decision.counts.items()},
        "fps": stats.fps,
        "decoded_frames": stats.decoded_frames,
        "reported_frame_count": stats.reported_frame_count,
        "dropped_as_duplicates": len(stats.dropped_as_duplicates),
    }
    return frame_records, video_record


def _completed(videos_csv: Path) -> set[str]:
    if not videos_csv.exists():
        return set()
    with videos_csv.open(newline="") as fh:
        return {row["video_id"] for row in csv.DictReader(fh) if row["status"] == "ok"}


def run(cfg: Config, run_dir: str | Path, limit: int | None = None) -> dict:
    """Process every video of the configured source; resumable."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = base_manifest("pipeline_run")
    manifest["config"] = cfg.to_dict()
    manifest["config_fingerprint"] = cfg.fingerprint()
    previous = run_dir / "manifest.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        if old.get("config_fingerprint") != manifest["config_fingerprint"]:
            raise RuntimeError(
                f"{run_dir} was produced with a different configuration "
                f"({old.get('config_fingerprint')} != {manifest['config_fingerprint']}); use a new run directory"
            )

    source = make_source(cfg.source)
    detector, classifier = build_components(cfg)
    labels = classifier.labels
    videos_csv = run_dir / "videos.csv"
    done = _completed(videos_csv)
    fields = VIDEO_FIELDS + [f"n_{label}" for label in labels]
    new_file = not videos_csv.exists()
    processed = errors = 0
    with videos_csv.open("a", newline="") as vf, (run_dir / "frames.jsonl").open("a") as ff:
        writer = csv.DictWriter(vf, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for item in source.list():
            if item.video_id in done:
                continue
            if limit is not None and processed >= limit:
                break
            start = time.perf_counter()
            row = {"video_id": item.video_id, "container": item.container, "key": item.key}
            try:
                with source.open(item) as path:
                    frame_records, video_record = process_video(path, item.video_id, cfg, detector, classifier)
                    if cfg.output.copy_videos:
                        _copy_sorted(path, item, video_record["category"], run_dir)
                for rec in frame_records:
                    ff.write(json.dumps(rec) + "\n")
                row.update(video_record, status="ok", error="")
            except Exception as exc:  # recorded and skipped; the run continues
                log.exception("failed: %s", item.video_id)
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                errors += 1
            row["seconds"] = round(time.perf_counter() - start, 2)
            writer.writerow(row)
            vf.flush()
            ff.flush()
            processed += 1
            log.info("%s → %s (%.1fs)", item.video_id, row.get("category"), row["seconds"])

    finish(
        manifest,
        videos_processed_this_session=processed,
        errors_this_session=errors,
        models={
            "detector": {
                "id": cfg.detector.model,
                "revision": getattr(detector, "resolved_revision", None) or cfg.detector.revision,
            },
            "classifier": {
                "id": cfg.classifier.model,
                "revision": getattr(classifier.scorer, "resolved_revision", None) or cfg.classifier.revision,
            },
        },
    )
    previous.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _copy_sorted(path: Path, item: VideoItem, category: str | None, run_dir: Path) -> None:
    dest = run_dir / "sorted" / (category or "undetermined") / item.container / item.key
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
