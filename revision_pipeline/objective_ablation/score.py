"""Score one objective-ablation embedding with the frozen primary evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics

from threadpoolctl import threadpool_limits

from ..data.store import Store, runtime_inventory
from ..evaluate.config import EvaluationConfig
from ..evaluate.engine import graph_and_grid, metric_records
from ..evaluate.inputs import load_refined_bundle
from ..integrity import file_fingerprint
from ..pilot.common import completed, read
from ..runs import RunDirectory
from ..step4.scoring import metric_map
from . import VARIANTS

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "revision_pipeline/configs/objective_ablation_v1.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required")
    spec = read(PROTOCOL)
    training = args.training_run.resolve()
    completed(training, "objective_ablation_training")
    saved = read(training / "summary.json")
    if saved["seed"] != args.seed or saved["variant"] != args.variant:
        raise ValueError("Training run binding mismatch")
    store = Store(ROOT / spec["store"])
    parent = store.embedding(spec["dataset"], spec["embedding"])
    dataset = store.dataset(spec["dataset"])
    refined = load_refined_bundle(training / "bundles/joint",
                                  expected_parent=parent.parent_reference(),
                                  output_cell_ids=dataset.cell_ids)
    config = EvaluationConfig.from_dict(read(ROOT / spec["evaluation_config"]))
    with RunDirectory(ROOT / "revision_pipeline/runs", kind="objective_ablation_scoring",
                      config={"variant": args.variant, "seed": args.seed,
                              "training_run": str(training),
                              "training_manifest": file_fingerprint(training / "run.json"),
                              "evaluation": config.to_dict()}, run_id=args.run_id) as run:
        run.write_json("runtime_start.json", runtime_inventory())
        reference, interpretation = dataset.reference_partition()
        with threadpool_limits(limits=1):
            graph = graph_and_grid(refined.values, reference, dataset.cell_ids, config,
                                   run=run, prefix="evaluation",
                                   training_label_use="label_free")
            metrics = metric_records(refined.values, dataset, config, grid=graph,
                                     run=run, prefix="evaluation")
        values = metric_map(metrics)
        selected = graph["selected"]
        run.write_json("summary.json", {
            "variant": args.variant, "seed": args.seed,
            "ARI": statistics.mean(row["ARI"] for row in selected),
            "RI": statistics.mean(row["RI"] for row in selected),
            "SIL_cluster": statistics.mean(
                value for key, value in values.items()
                if key.startswith("predicted_cluster_ASW_subsample_seed")),
            "SIL_reference": values["reference_ASW_subsample"],
            "iLISI": values["iLISI_scib_metrics"],
            "purity": values["reference_knn_purity"],
            "K": {str(row["leiden_seed"]): row["n_clusters"] for row in selected},
            "reference_interpretation": interpretation,
        })
        run.write_json("runtime_end.json", runtime_inventory())
        run.manifest.update(scientific_experiment=True, experiment_role=spec["role"])
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
