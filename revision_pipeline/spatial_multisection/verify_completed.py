"""Deep validation for a Package 4 run accepted by resumable orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..integrity import canonical_hash, project_path
from .common import ROOT, read_json, require_run, source_snapshot


def validate(path: Path, kind: str) -> dict:
    candidate = project_path(ROOT, Path(path).as_posix())
    manifest = require_run(candidate, kind)
    expected_source = canonical_hash(source_snapshot())
    if not manifest.get("source_tree_sha256") or manifest["source_tree_sha256"] != expected_source:
        raise ValueError("Completed run was produced from a different or null scientific source tree")
    if kind == "spatial_multisection_panel":
        config = read_json(candidate / "config.json")
        from .consolidate import _load_all
        _load_all(config["prefix"])
    return {"run_id": manifest["run_id"], "kind": kind, "source_tree_sha256": expected_source,
            "artifacts_verified": len(manifest["artifacts"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--kind", required=True)
    args = parser.parse_args()
    print(validate(args.run, args.kind))


if __name__ == "__main__":
    main()
