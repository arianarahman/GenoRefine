# Purpose: Deep semantic validation for resumable GraphST Package 4b runs.
# Author: Ariana Rahman (Arizona State University)

"""Deep semantic validation for resumable GraphST Package 4b runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..integrity import canonical_hash, file_fingerprint, project_path
from .common import ROOT, SPEC_PATH, load_alignment, load_k_selection, read_json, require_run, source_snapshot


def validate(path: Path, kind: str) -> dict:
    candidate = project_path(ROOT, Path(path).as_posix())
    manifest = require_run(candidate, kind)
    expected_source = canonical_hash(source_snapshot())
    if manifest.get("source_tree_sha256") != expected_source:
        raise ValueError("Completed run was produced from a different Package 4b source tree")
    config = read_json(candidate / "config.json")
    if kind == "spatial_graphst_preflight":
        receipt = read_json(candidate / "preflight.json")
        if (receipt.get("passed") is not True or receipt.get("scientific_data_used") is not False
                or receipt.get("specification") != file_fingerprint(SPEC_PATH)
                or receipt.get("source_tree_sha256") != expected_source
                or receipt.get("paste", {}).get("passed") is not True
                or receipt.get("graphst", {}).get("passed") is not True):
            raise ValueError("Preflight receipt does not pass the frozen acceptance gates")
    elif kind == "spatial_graphst_alignment":
        load_alignment(candidate, config["donor"])
    elif kind == "spatial_graphst_k_selection":
        load_k_selection(candidate)
    elif kind == "spatial_graphst_training":
        from .score import load_training
        parents = config.get("parent_runs", {})
        load_training(
            candidate, ROOT / parents["alignment"], ROOT / parents["k_selection"],
            config["donor"], int(config["seed"]),
        )
    elif kind in {"spatial_graphst_section_score", "spatial_graphst_donor_score"}:
        from .score import load_training
        parents = config.get("parent_runs", {})
        trained = load_training(
            ROOT / parents["training"], ROOT / parents["alignment"], ROOT / parents["k_selection"],
            config["donor"], int(config["algorithmic_seed"]),
        )
        summary = read_json(candidate / "summary.json")
        if (summary.get("donor") != config["donor"]
                or summary.get("algorithmic_seed") != config["algorithmic_seed"]
                or config.get("input", {}).get("training_manifest") != trained["manifest"]):
            raise ValueError("Score summary/parent binding changed")
        if kind == "spatial_graphst_section_score" and summary.get("section") != config.get("section"):
            raise ValueError("Section-score identity changed")
    elif kind == "spatial_graphst_panel":
        from .consolidate import _load_all
        _load_all(config["prefix"])
    else:
        raise ValueError(f"Unsupported GraphST Package 4b run kind: {kind}")
    return {
        "run_id": manifest["run_id"], "kind": kind,
        "source_tree_sha256": expected_source,
        "artifacts_verified": len(manifest["artifacts"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--kind", required=True)
    args = parser.parse_args()
    print(validate(args.run, args.kind), flush=True)


if __name__ == "__main__":
    main()
