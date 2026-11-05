"""Local input content identity, outside measured work."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path


def _fingerprint(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@dataclass
class SourceIdentity:
    sha256: str | None = None
    size_bytes: int | None = None
    _path: Path | None = None
    _fingerprint: tuple[int, ...] | None = None

    def verify(self) -> None:
        if self._path is None:
            return
        try:
            unchanged = _fingerprint(self._path.stat()) == self._fingerprint
        except OSError:
            unchanged = False
        if not unchanged:
            raise ValueError("local source changed during the benchmark")

    def metadata(self) -> dict:
        return {"content_sha256": self.sha256, "size_bytes": self.size_bytes}


def identify_source(url: str) -> SourceIdentity:
    """Hash regular local paths; URL content and nonregular inputs stay unknown."""
    if "://" in url:
        return SourceIdentity()
    path = Path(url)
    try:
        initial = path.stat()
        if not stat.S_ISREG(initial.st_mode):
            return SourceIdentity()
        fingerprint = _fingerprint(initial)
        digest = hashlib.sha256()
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as stream:
            if _fingerprint(os.fstat(stream.fileno())) != fingerprint:
                raise ValueError("local source changed while hashing")
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            if _fingerprint(os.fstat(stream.fileno())) != fingerprint:
                raise ValueError("local source changed while hashing")
    except OSError:
        raise ValueError("could not fingerprint local source") from None
    identity = SourceIdentity(digest.hexdigest(), initial.st_size, path, fingerprint)
    identity.verify()
    return identity
