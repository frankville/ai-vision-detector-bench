"""Atomic, collision-safe JSON result persistence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def write_json(directory: Path | str, stem: str, data: dict) -> Path:
    """Publish complete finite JSON, choosing a suffix when a name exists."""
    if not stem or stem in {".", ".."} or "/" in stem or "\\" in stem:
        raise ValueError("stem must be a nonempty filename without path separators")
    serialized = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, prefix=".result-", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(serialized + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        suffix = 0
        while True:
            name = f"{stem}.json" if suffix == 0 else f"{stem}__{suffix}.json"
            path = directory / name
            try:
                os.link(temporary, path)
            except FileExistsError:
                suffix += 1
            else:
                return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
