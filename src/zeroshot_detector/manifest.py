"""Provenance records written next to every run's outputs."""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

from . import __version__

_PACKAGES = (
    "numpy",
    "pandas",
    "torch",
    "transformers",
    "tokenizers",
    "opencv-python-headless",
    "pillow",
    "minio",
    "timm",
)


def git_state(path: str | Path | None = None) -> dict:
    cwd = str(path or Path(__file__).resolve().parent)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def package_versions() -> dict[str, str | None]:
    out = {}
    for name in _PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def hardware() -> dict:
    info = {"machine": platform.machine(), "processor": platform.processor() or None}
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    return info


def base_manifest(kind: str) -> dict:
    return {
        "kind": kind,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "zeroshot_detector": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git": git_state(),
        "packages": package_versions(),
        "hardware": hardware(),
    }


def finish(manifest: dict, **extra) -> dict:
    manifest.update(extra)
    manifest["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    return manifest
