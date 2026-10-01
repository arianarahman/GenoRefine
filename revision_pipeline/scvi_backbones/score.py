# Purpose: Score a standalone scVI embedding with the frozen primary evaluator.
# Author: Ariana Rahman (Arizona State University)

"""Score a standalone scVI embedding with the frozen primary evaluator."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from ..data.store import Store
from ..evaluate.engine import graph_and_grid, metric_records
from ..inductive_validation.common import evaluation_config
from ..runs import RunDirectory
from ..step4.scoring import metric_map

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "revision_pipeline/runs"
SPEC = json.loads((ROOT / "revision_pipeline/configs/scvi_other_datasets_v1.json").read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["mouse_senis", "pancreas_five_study"], required=True)
    p.add_argument("--training-run", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=range(5), required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if not a.execute:
        p.error("Explicit --execute is required")

    dataset = Store(ROOT / SPEC["store"]).dataset(a.dataset)
    reference, interpretation = dataset.reference_partition()
    with np.load(a.training_run / "embedding.npz", allow_pickle=False) as saved:
        values = np.asarray(saved["values"], dtype=np.float64)
        ids = tuple(str(v) for v in saved["ids"])
    if ids != tuple(dataset.cell_ids):
        raise ValueError("scVI cell order differs from frozen canonical order")
    config = evaluation_config()
    start = time.perf_counter()
    with RunDirectory(
        RUNS,
        kind="scvi_standalone_backbone_score",
        run_id=a.run_id,
        config={"dataset": a.dataset, "seed": a.seed, "evaluation": config.to_dict()},
    ) as run:
        with threadpool_limits(limits=1):
            grid = graph_and_grid(values, reference, ids, config, run=run, prefix="evaluation", training_label_use="none; labels evaluation only")
            metrics = metric_records(values, dataset, config, grid=grid, run=run, prefix="evaluation")
        mapped = metric_map(metrics)
        selected = grid["selected"]
        run.write_json("summary.json", {
            "dataset": a.dataset,
            "seed": a.seed,
            "method": "scVI",
            "ARI": statistics.mean(x["ARI"] for x in selected),
            "SIL_cluster": statistics.mean(v for k, v in mapped.items() if k.startswith("predicted_cluster_ASW_subsample_seed")),
            "SIL_reference": mapped["reference_ASW_subsample"],
            "iLISI": mapped["iLISI_scib_metrics"],
            "purity": mapped["reference_knn_purity"],
            "reference_interpretation": interpretation,
            "wall_seconds": time.perf_counter() - start,
            "training_run": str(a.training_run),
            "evidence_status": "primary_comparator" if a.dataset == "mouse_senis" else "sensitivity_only",
        })
        run.manifest.update(scientific_experiment=True, experiment_role="standalone_scvi_scoring")
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
