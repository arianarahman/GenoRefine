# Purpose: Score one completed genoMOI-core seed with the frozen primary evaluator.
# Author: Ariana Rahman (Arizona State University)

"""Score one completed genoMOI-core seed with the frozen primary evaluator."""

import argparse
from pathlib import Path
import statistics
import time

from threadpoolctl import threadpool_limits

from ..audit import source_paths
from ..data.store import Store, runtime_inventory
from ..evaluate.config import EvaluationConfig
from ..evaluate.engine import graph_and_grid, metric_records
from ..evaluate.inputs import load_refined_bundle
from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import completed, read
from ..runs import RunDirectory
from ..step4.scoring import metric_map
from .common import ROOT, protocol


PRIMARY = ROOT / "revision_pipeline/configs/evaluation_primary_v1.json"


def snapshot():
    return {p.relative_to(ROOT).as_posix(): file_fingerprint(p) for p in source_paths(ROOT)}


def main():
    """Score one validated genoMOI-core run with the frozen primary evaluator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    spec, k_record, protocol_fp, _ = protocol()
    if not args.execute:
        parser.error("Explicit execution flag required")
    if args.seed not in spec["replicate_seeds"]:
        parser.error("Seed is outside the frozen panel")
    training = args.training_run.resolve()
    completed(training, "step5a_genomoi_core")
    training_manifest = file_fingerprint(training / "run.json")
    if read(training / "summary.json")["seed"] != args.seed:
        raise ValueError("Training seed/run mismatch")

    store = Store(ROOT / spec["store"])
    parent = store.embedding(spec["dataset"], spec["embedding"])
    dataset = store.dataset(spec["dataset"])
    if parent.parent_reference() != k_record["parent_reference"] or parent.cell_ids != dataset.cell_ids:
        raise ValueError("Frozen parent or canonical IDs changed")
    refined = load_refined_bundle(training / "bundle", expected_parent=parent.parent_reference(),
                                  output_cell_ids=dataset.cell_ids)
    config = EvaluationConfig.from_dict(read(PRIMARY))
    sources = snapshot()
    run_config = {
        "protocol_file": protocol_fp,
        "evaluation": config.to_dict(),
        "seed": args.seed,
        "training_run": str(training),
        "training_manifest": training_manifest,
        "input": refined.metadata,
    }
    with RunDirectory(ROOT / "revision_pipeline/runs", kind="step5a_genomoi_scoring",
                      config=run_config, run_id=args.run_id) as run:
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.manifest.update(scientific_experiment=True,
                            experiment_role="post_step4_matched_legacy_core_diagnostic_scoring")
        run.write_json("runtime_start.json", runtime_inventory())
        reference, interpretation = dataset.reference_partition()
        with threadpool_limits(limits=1):
            graph = graph_and_grid(refined.values, reference, dataset.cell_ids, config, run=run,
                                   prefix="evaluation", training_label_use="label_free")
            started = time.perf_counter()
            metrics = metric_records(refined.values, dataset, config, grid=graph, run=run,
                                     prefix="evaluation")
            geometry_seconds = time.perf_counter() - started
        values = metric_map(metrics)
        selected = graph["selected"]
        summary = {
            "seed": args.seed,
            "mean_ARI_at_0_5": statistics.mean(row["ARI"] for row in selected),
            "mean_RI_at_0_5": statistics.mean(row["RI"] for row in selected),
            "K_at_0_5": {str(row["leiden_seed"]): row["n_clusters"] for row in selected},
            "reference_ASW_subsample": values["reference_ASW_subsample"],
            "predicted_cluster_ASW_subsample_mean": statistics.mean(
                value for key, value in values.items() if key.startswith("predicted_cluster_ASW_subsample_seed")),
            "iLISI_scib_metrics": values["iLISI_scib_metrics"],
            "D_batch_fixed90_including_self": values["D_batch_fixed90_including_self"],
            "reference_knn_purity": values["reference_knn_purity"],
            "grid_rows": len(graph["grid"]),
            "graph_seconds": graph["timing"]["graph_seconds"],
            "grid_seconds": graph["timing"]["grid_seconds"],
            "geometry_seconds": geometry_seconds,
            "reference_interpretation": interpretation,
            "not_independent_architecture_comparator": True,
        }
        run.write_json("summary.json", summary)
        run.write_json("runtime_end.json", runtime_inventory())
        if snapshot() != sources or file_fingerprint(training / "run.json") != training_manifest:
            raise RuntimeError("Source or immutable training run changed during scoring")
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
