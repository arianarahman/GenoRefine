# Purpose: Score one independent IDEC output with the frozen primary evaluator.
# Author: Ariana Rahman (Arizona State University)

"""Score one independent IDEC output with the frozen primary evaluator."""

from __future__ import annotations

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
from .run_idec import protocol_path, select_case

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "revision_pipeline/configs/independent_idec_v1.json"


def snapshot():
    return {p.relative_to(ROOT).as_posix(): file_fingerprint(p) for p in source_paths(ROOT)}


def main():
    """Score one verified IDEC stage with the frozen graph, clustering, and metric evaluator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROTOCOL))
    parser.add_argument("--case")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", choices=["pretrain", "joint"], required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required")
    protocol = protocol_path(args.config)
    spec = read(protocol)
    case = select_case(spec, args.case)
    training = args.training_run.resolve()
    completed(training, "independent_idec_training")
    summary = read(training / "summary.json")
    if (summary["seed"] != args.seed or args.seed not in spec["replicate_seeds"]
            or summary.get("case_id", case["id"]) != case["id"]):
        raise ValueError("Seed/training-run mismatch")
    store = Store(ROOT / case.get("store", spec.get("default_store", spec.get("store"))))
    parent = store.embedding(case["dataset"], case["embedding"])
    dataset = store.dataset(case["dataset"])
    refined = load_refined_bundle(training / "bundles" / args.stage,
                                  expected_parent=parent.parent_reference(),
                                  output_cell_ids=dataset.cell_ids)
    evaluation = EvaluationConfig.from_dict(read(ROOT / spec["evaluation_config"]))
    sources = snapshot()
    config = {"protocol_file": file_fingerprint(protocol), "case": case, "seed": args.seed,
              "stage": args.stage, "training_run": str(training),
              "training_manifest": file_fingerprint(training / "run.json"),
              "evaluation": evaluation.to_dict(), "input": refined.metadata}
    with RunDirectory(ROOT / "revision_pipeline/runs", kind="independent_idec_scoring",
                      config=config, run_id=args.run_id) as run:
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.manifest.update(scientific_experiment=True,
                            experiment_role="independent_refinement_comparator_scoring")
        run.write_json("runtime_start.json", runtime_inventory())
        reference, interpretation = dataset.reference_partition()
        with threadpool_limits(limits=1):
            graph = graph_and_grid(refined.values, reference, dataset.cell_ids, evaluation,
                                   run=run, prefix="evaluation",
                                   training_label_use="label_free")
            started = time.perf_counter()
            metrics = metric_records(refined.values, dataset, evaluation, grid=graph,
                                     run=run, prefix="evaluation")
            geometry_seconds = time.perf_counter() - started
        values = metric_map(metrics)
        selected = graph["selected"]
        result = {
            "case_id": case["id"], "display_dataset": case["display_dataset"],
            "dataset": case["dataset"], "embedding": case["embedding"],
            "seed": args.seed, "stage": args.stage,
            "mean_ARI_at_0_5": statistics.mean(row["ARI"] for row in selected),
            "mean_RI_at_0_5": statistics.mean(row["RI"] for row in selected),
            "K_at_0_5": {str(row["leiden_seed"]): row["n_clusters"] for row in selected},
            "reference_ASW_subsample": values["reference_ASW_subsample"],
            "predicted_cluster_ASW_subsample_mean": statistics.mean(
                value for key, value in values.items()
                if key.startswith("predicted_cluster_ASW_subsample_seed")),
            "iLISI_scib_metrics": values["iLISI_scib_metrics"],
            "D_batch_fixed90_including_self": values["D_batch_fixed90_including_self"],
            "reference_knn_purity": values["reference_knn_purity"],
            "grid_rows": len(graph["grid"]),
            "graph_seconds": graph["timing"]["graph_seconds"],
            "grid_seconds": graph["timing"]["grid_seconds"],
            "geometry_seconds": geometry_seconds,
            "reference_interpretation": interpretation,
            "independent_architecture_comparator": True}
        run.write_json("summary.json", result)
        run.write_json("runtime_end.json", runtime_inventory())
        if snapshot() != sources:
            raise RuntimeError("Source changed during scoring")
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
