# Purpose: Score one focused-ablation joint embedding with the frozen evaluator.
# Author: Ariana Rahman (Arizona State University)

"""Score one focused-ablation joint embedding with the frozen evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import time

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
PROTOCOL = ROOT / "revision_pipeline/configs/focused_ablation_v1.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--training-run", type=Path, required=True)
    p.add_argument("--run-id", required=True); p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if not a.execute: p.error("Explicit --execute is required")
    spec = read(PROTOCOL); training = a.training_run.resolve()
    completed(training, "focused_ablation_training")
    saved = read(training / "summary.json")
    if saved["seed"] != a.seed or saved["variant"] != a.variant:
        raise ValueError("Training run binding mismatch")
    store = Store(ROOT / spec["store"]); parent = store.embedding(spec["dataset"], spec["embedding"])
    dataset = store.dataset(spec["dataset"])
    refined = load_refined_bundle(training / "bundles/joint", expected_parent=parent.parent_reference(),
                                  output_cell_ids=dataset.cell_ids)
    config = EvaluationConfig.from_dict(read(ROOT / spec["evaluation_config"]))
    with RunDirectory(ROOT / "revision_pipeline/runs", kind="focused_ablation_scoring",
                      config={"variant": a.variant, "seed": a.seed,
                              "training_run": str(training),
                              "training_manifest": file_fingerprint(training / "run.json"),
                              "evaluation": config.to_dict()}, run_id=a.run_id) as run:
        run.write_json("runtime_start.json", runtime_inventory())
        reference, interpretation = dataset.reference_partition()
        with threadpool_limits(limits=1):
            graph = graph_and_grid(refined.values, reference, dataset.cell_ids, config,
                                   run=run, prefix="evaluation", training_label_use="label_free")
            start = time.perf_counter()
            metrics = metric_records(refined.values, dataset, config, grid=graph, run=run,
                                     prefix="evaluation")
            metric_seconds = time.perf_counter() - start
        values = metric_map(metrics); selected = graph["selected"]
        run.write_json("summary.json", {
            "variant": a.variant, "seed": a.seed,
            "ARI": statistics.mean(x["ARI"] for x in selected),
            "RI": statistics.mean(x["RI"] for x in selected),
            "SIL_cluster": statistics.mean(v for k, v in values.items()
                                           if k.startswith("predicted_cluster_ASW_subsample_seed")),
            "SIL_reference": values["reference_ASW_subsample"],
            "iLISI": values["iLISI_scib_metrics"],
            "purity": values["reference_knn_purity"],
            "K": {str(x["leiden_seed"]): x["n_clusters"] for x in selected},
            "timing": {**graph["timing"], "metric_seconds": metric_seconds},
            "reference_interpretation": interpretation})
        run.write_json("runtime_end.json", runtime_inventory())
        run.manifest.update(scientific_experiment=True,
                            experiment_role="focused_ablation_scoring")
    print(run.final_path, flush=True)


if __name__ == "__main__": main()
