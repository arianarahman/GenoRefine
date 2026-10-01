# Purpose: Create or verify the SHA-256 manifest for the public release tree.
# Author: Ariana Rahman (Arizona State University)

"""Create or verify the SHA-256 manifest for the public release tree."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"


def files() -> list[Path]:
    return sorted(
        (
            path
            for path in ROOT.rglob("*")
            if path.is_file()
            and ".git" not in path.relative_to(ROOT).parts
            and path != MANIFEST
        ),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def rendered_manifest() -> str:
    return "".join(
        f"{digest(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in files()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = rendered_manifest()
    if args.check:
        if not MANIFEST.exists() or MANIFEST.read_text(encoding="utf-8") != expected:
            print("MANIFEST.sha256 is missing or stale.")
            return 1
        print(f"Verified {len(files())} release files against MANIFEST.sha256.")
        return 0
    MANIFEST.write_text(expected, encoding="utf-8", newline="\n")
    print(f"Wrote SHA-256 entries for {len(files())} release files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
