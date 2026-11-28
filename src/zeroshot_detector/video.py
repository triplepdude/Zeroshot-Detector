"""Frame sampling from video files (OpenCV)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SampledFrame:
    index: int  #: 0-based decoded-frame index
    timestamp: float | None  #: seconds, from the container's fps (None if unknown)
    rgb: np.ndarray


@dataclass
class SamplingStats:
    fps: float | None = None
    reported_frame_count: int | None = None
    decoded_frames: int = 0
    candidates: int = 0  #: frames on the sampling grid
    kept: int = 0
    dropped_as_duplicates: list[int] = field(default_factory=list)


def sampling_step(fps: float | None, every_n_frames: int, every_seconds: float | None) -> int:
    if every_seconds is None:
        return max(1, int(every_n_frames))
    if not fps or fps <= 0:
        raise ValueError("every_seconds requires a known frame rate")
    return max(1, round(fps * every_seconds))


def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Sum of absolute pixel differences (what ``cv2.norm(a, b, cv2.NORM_L1)`` returns)."""
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a.astype(np.int64) - b.astype(np.int64)).sum())


def iter_sampled_frames(
    path: str | Path,
    every_n_frames: int = 30,
    every_seconds: float | None = None,
    dedup_l1_threshold: float | None = 5000.0,
    max_samples: int | None = None,
    stats: SamplingStats | None = None,
) -> Iterator[SampledFrame]:
    """Decode ``path`` sequentially and yield every ``step``-th frame as RGB.

    Mirrors the original loop (sequential ``cap.read()``; a frame on the grid
    is kept when it is the first one or its L1 distance to the previously
    *kept* frame exceeds ``dedup_l1_threshold``). ``stats`` is filled in place.
    """
    import cv2

    stats = stats if stats is not None else SamplingStats()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        stats.fps = fps
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stats.reported_frame_count = reported or None
        step = sampling_step(fps, every_n_frames, every_seconds)
        previous = None
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                stats.candidates += 1
                if (
                    previous is None
                    or dedup_l1_threshold is None
                    or previous.shape != frame.shape
                    or cv2.norm(frame, previous, cv2.NORM_L1) > dedup_l1_threshold
                ):
                    previous = frame
                    stats.kept += 1
                    yield SampledFrame(
                        index=index,
                        timestamp=index / fps if fps else None,
                        rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    )
                    if max_samples is not None and stats.kept >= max_samples:
                        index += 1
                        break
                else:
                    stats.dropped_as_duplicates.append(index)
            index += 1
        stats.decoded_frames = index
    finally:
        cap.release()


def read_frame(path: str | Path, index: int) -> np.ndarray | None:
    """Return frame ``index`` (RGB) by sequential decoding, or ``None``.

    Sequential decoding is slower than seeking but returns exactly the frame
    the pipeline saw (seeking can land on a different frame for some codecs).
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                return None
            if i == index:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()
