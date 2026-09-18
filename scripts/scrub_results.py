#!/usr/bin/env python3
"""Strip identifying fields from result JSON files before publishing them.

Result files are meant to be committed, and a benchmark result has no reason to
carry the name of the machine that produced it. Hostnames routinely contain a
person's name or an employer's, so they are removed here and no longer collected
at all. `profile` is the field that identifies a machine, and you choose it.

Run over the whole tree, or a subdirectory:

    python scripts/scrub_results.py
    python scripts/scrub_results.py benchmarks/some-host
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: Keys removed from the `host` block wherever they appear.
DENYLIST = {"hostname", "node", "fqdn", "user", "username"}


def scrub_file(path: Path) -> list[str]:
    """Remove denylisted keys in place. Returns the keys that were removed."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []

    removed: list[str] = []
    host = data.get("host")
    if isinstance(host, dict):
        for key in list(host):
            if key.lower() in DENYLIST:
                host.pop(key)
                removed.append(key)

    if removed:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return removed


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path("benchmarks")
    if not root.exists():
        print(f"nothing to scrub: {root} does not exist")
        return 0

    total = 0
    for path in sorted(root.rglob("*.json")):
        removed = scrub_file(path)
        if removed:
            total += 1
            print(f"  scrubbed {', '.join(removed)} from {path}")

    print(f"{total} file(s) changed" if total else "already clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
