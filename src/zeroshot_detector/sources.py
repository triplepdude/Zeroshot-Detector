"""Where videos come from: a local directory or MinIO / S3 buckets."""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .config import SourceConfig


@dataclass(frozen=True)
class VideoItem:
    key: str  #: path relative to the source root (or object name); unique within a source
    container: str  #: directory or bucket
    size: int | None = None

    @property
    def video_id(self) -> str:
        return f"{self.container}/{self.key}"


class LocalSource:
    def __init__(self, root: str | Path, extensions: tuple[str, ...]) -> None:
        self.root = Path(root)
        self.extensions = tuple(e.lower() for e in extensions)
        if not self.root.is_dir():
            raise FileNotFoundError(f"video directory not found: {self.root}")

    def list(self) -> Iterator[VideoItem]:
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.suffix.lower() in self.extensions:
                yield VideoItem(
                    key=path.relative_to(self.root).as_posix(), container=self.root.name, size=path.stat().st_size
                )

    @contextlib.contextmanager
    def open(self, item: VideoItem) -> Iterator[Path]:
        yield self.root / item.key


class MinioSource:
    """Lists and downloads objects; credentials come from the environment.

    ``MINIO_ACCESS_KEY`` / ``MINIO_SECRET_KEY`` (both optional, for anonymous
    buckets as in the original study).
    """

    def __init__(self, endpoint: str, buckets: list[str], prefix: str, secure: bool, extensions: tuple[str, ...]):
        from minio import Minio

        self.client = Minio(
            endpoint,
            access_key=os.environ.get("MINIO_ACCESS_KEY"),
            secret_key=os.environ.get("MINIO_SECRET_KEY"),
            secure=secure,
        )
        self.buckets = buckets
        self.prefix = prefix
        self.extensions = tuple(e.lower() for e in extensions)

    def list(self) -> Iterator[VideoItem]:
        for bucket in self.buckets:
            if not self.client.bucket_exists(bucket):
                raise FileNotFoundError(f"bucket does not exist: {bucket}")
            for obj in self.client.list_objects(bucket, prefix=self.prefix or None, recursive=True):
                if Path(obj.object_name).suffix.lower() in self.extensions:
                    yield VideoItem(key=obj.object_name, container=bucket, size=obj.size)

    @contextlib.contextmanager
    def open(self, item: VideoItem) -> Iterator[Path]:
        suffix = Path(item.key).suffix
        fd, name = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            self.client.fget_object(item.container, item.key, name)
            yield Path(name)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(name)


def make_source(cfg: SourceConfig) -> LocalSource | MinioSource:
    exts = tuple(cfg.extensions)
    if cfg.kind == "local":
        return LocalSource(cfg.path, exts)
    return MinioSource(cfg.endpoint, list(cfg.buckets), cfg.prefix, cfg.secure, exts)
