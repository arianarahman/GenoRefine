"""Add descriptive paired safeguards to a VERIFIED saved evaluation, without refitting."""

import argparse
from pathlib import Path

from ..data.store import read_json
from ..integrity import file_fingerprint, canonical_hash
from ..runs import RunDirectory
from .config import EvaluationConfig
from .contrasts import paired_clustering
from .runner import snapshot
from .verify_repeat import inspect


def compare_saved(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    manifest, record = inspect(path)
    if len(record["embeddings"]) != 2:
        raise ValueError("Exactly one baseline/refined evaluation pair required")
    ids = read_json(path/"embedding_0/cell_ids.json")
    if ids != read_json(path/"embedding_1/cell_ids.json"):
        raise ValueError("Saved pair has different populations or row order")
    config = EvaluationConfig.from_dict(record["evaluation"])
    summary = read_json(path/"summary.json")
    verified = (summary.get("pair") or {}).get("pairing_status") == "exact_parent_reference_verified"
    sources = snapshot(root)
    original = file_fingerprint(path/"run.json")
    result = paired_clustering(read_json(path/"embedding_0/grid.json"),
                               read_json(path/"embedding_1/grid.json"), config, verified=verified)
    with RunDirectory(root/"revision_pipeline/runs", kind="paired_clustering_postprocess", config={
            "source_run": str(path), "source_manifest": original, "source_evaluation_config": record,
            "scope": "Postprocess existing scores only; not a new graph, refiner or primary experiment"}) as run:
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.write_json("clustering_comparison.json", result)
        inspect(path)  # Reject input mutation even if its manifest was not changed.
        if file_fingerprint(path/"run.json") != original or snapshot(root) != sources:
            raise RuntimeError("Sources changed while comparing saved results")
    return run.final_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evaluation_run", type=Path)
    args = p.parse_args()
    print(compare_saved(args.evaluation_run, Path(__file__).resolve().parents[2]))
