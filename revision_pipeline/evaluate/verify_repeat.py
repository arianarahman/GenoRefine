"""Compare two completed fresh-process evaluations without ignoring failures."""

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

from ..data.store import read_json
from ..integrity import file_fingerprint, project_path
from ..runs import RunDirectory


def inspect(path):
    manifest = read_json(path / "run.json")
    if manifest["status"] != "succeeded" or manifest["kind"] != "step3b_evaluation":
        raise ValueError("Expected a completed evaluator run")
    for relative, fingerprint in manifest["artifacts"].items():
        if file_fingerprint(project_path(path, relative)) != fingerprint:
            raise ValueError("Evaluator artifact integrity failure")
    return manifest, read_json(path / "config.json")


def exact_array(a, b):
    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes(order="C") != b.tobytes(order="C"):
        raise AssertionError("Array shape, dtype or bytes differ between processes")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("first", type=Path)
    p.add_argument("second", type=Path)
    args = p.parse_args()
    first, config = inspect(args.first)
    second, second_config = inspect(args.second)
    if config != second_config or first["source_tree_sha256"] != second["source_tree_sha256"]:
        raise ValueError("Settings or source snapshots differ")
    checks = []
    for file in ("pair/clustering_comparison.json",):
        if (args.first/file).exists() or (args.second/file).exists():
            if read_json(args.first/file) != read_json(args.second/file):
                raise AssertionError("Paired clustering safeguards differ between processes")
    summaries = [read_json(path/"summary.json") for path in (args.first, args.second)]
    if [r.get("cluster_counts_at_0_5") for r in summaries[0]["evaluations"]] != [
            r.get("cluster_counts_at_0_5") for r in summaries[1]["evaluations"]]:
        raise AssertionError("Anchor cluster counts differ between processes")
    for i, name in enumerate(config["embeddings"]):
        prefix = f"embedding_{i}"
        for file in ("partitions.npy",):
            exact_array(np.load(args.first/prefix/file), np.load(args.second/prefix/file))
        if read_json(args.first/prefix/"cell_ids.json") != read_json(args.second/prefix/"cell_ids.json"):
            raise AssertionError("Ordered IDs differ")
        for file in ("knn_indices.npy", "knn_distances.npy"):
            if (args.first/prefix/file).exists() or (args.second/prefix/file).exists():
                exact_array(np.load(args.first/prefix/file), np.load(args.second/prefix/file))
        for file in ("connectivities.npz", "distances.npz"):
            a, b = load_npz(args.first/prefix/file), load_npz(args.second/prefix/file)
            assert a.shape == b.shape
            for key in ("data", "indices", "indptr"):
                exact_array(getattr(a, key), getattr(b, key))
        for file, ignore in (("grid.json", {"leiden_and_agreement_seconds"}),
                             ("selected.json", {"leiden_and_agreement_seconds"}),
                             ("metrics.json", {"seconds"})):
            rows = [[{k: v for k, v in row.items() if k not in ignore} for row in read_json(path/prefix/file)]
                    for path in (args.first, args.second)]
            if rows[0] != rows[1]:
                raise AssertionError(f"{file} differs between processes")
        if read_json(args.first/prefix/"asw_sampling.json") != read_json(args.second/prefix/"asw_sampling.json"):
            raise AssertionError("Sampling IDs or coverage differs")
        for relative in first["artifacts"]:
            if relative.startswith(prefix+"/metric_") and relative.endswith(".npy"):
                exact_array(np.load(args.first/relative), np.load(args.second/relative))
        checks.append({"embedding": name, "partitions_identical": True, "graphs_identical": True,
                       "scores_identical": True, "per_cell_metrics_identical": True, "sampling_identical": True})
    root = Path(__file__).resolve().parents[2]
    with RunDirectory(root / "revision_pipeline/runs", kind="step3b_fresh_process_repeat", config={
            "first": first["run_id"], "second": second["run_id"],
            "run_manifests": [file_fingerprint(path/"run.json") for path in (args.first, args.second)]}) as run:
        run.write_json("checks.json", {"passed": True, "checks": checks,
                       "scope": "Evaluator reproducibility only; no model fitting and no historical pancreas reconstruction"})
    print(run.final_path)


if __name__ == "__main__":
    main()
