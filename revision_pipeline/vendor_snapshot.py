"""Preserve the small, unmodified cartography dependency from the supplied source.

No copied environment is activated and no scientific package is imported here.
Original license/attribution accompanies the source; review release permissions
before publishing this local research bundle. This is not a PyPI replacement.
"""

from pathlib import Path
import shutil

from .integrity import file_fingerprint
from .runs import write_json


FILES = (
    "genomap/__init__.py", "genomap/genomap.py",
    "genomap/genomapOPT/__init__.py", "genomap/genomapOPT/genomapOPT.py",
    "genomap/bregman_genomap/__init__.py", "genomap/bregman_genomap/bregman_genomap.py",
)


def main():
    package = Path(__file__).resolve().parent
    source = package.parent / "venv_pancreas/Lib/site-packages"
    destination = package / "vendor"
    paths = list(FILES) + ["genomap-1.3.6.dist-info/LICENSE.txt"]
    records = {}
    for relative in paths:
        original = source / relative
        fingerprint = file_fingerprint(original)
        target = destination / relative
        if target.exists():
            if file_fingerprint(target) != fingerprint:
                raise FileExistsError(f"Refusing to replace changed vendor file: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
        if file_fingerprint(target) != fingerprint:
            raise RuntimeError(f"Source changed during copy: {relative}")
        records[relative] = fingerprint
    manifest = destination / "source_manifest.json"
    if not manifest.exists():
        write_json(manifest, {"source": "supplied venv_pancreas; genomap 1.3.6",
                              "modifications": "none; byte-for-byte copies", "files": records})
    print(f"Preserved {len(records)} unmodified cartography/license files in {destination}")


if __name__ == "__main__":
    main()
