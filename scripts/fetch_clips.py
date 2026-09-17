#!/usr/bin/env python3
"""Download the sample clips listed in clips/sources.json.

Video files are not committed. This keeps the repository small and, more
importantly, makes it structurally impossible to leak footage from a real
deployment into a public repo by accident.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

CLIPS = Path(__file__).resolve().parent.parent / "clips"


def main() -> int:
    manifest = json.loads((CLIPS / "sources.json").read_text())
    failures = 0

    for clip in manifest["clips"]:
        target = CLIPS / clip["name"]
        if target.exists():
            print(f"  have  {clip['name']} ({target.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"  fetch {clip['name']} from {clip['url']}")
        try:
            urllib.request.urlretrieve(clip["url"], target)
        except Exception as exc:
            print(f"  FAILED {clip['name']}: {exc}", file=sys.stderr)
            target.unlink(missing_ok=True)
            failures += 1
            continue
        print(f"  ok    {clip['name']} ({target.stat().st_size / 1e6:.1f} MB)")
        print(f"        {clip['license']} - {clip['attribution']}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
