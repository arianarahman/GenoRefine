# Purpose: Score one representation only on the genuinely held-out HP-CB cells.
# Author: Ariana Rahman (Arizona State University)

"""Score one representation only on the genuinely held-out HP-CB cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from ..broader_validation.common import EvaluationDataset, selected_summary
from ..evaluate.engine import graph_and_grid, metric_records
from ..integrity import file_fingerprint
from ..runs import RunDirectory
from .common import RUNS, completed, evaluation_config, specification


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--training", type=Path, required=True)
    p.add_argument("--representation", choices=["baseline", "pretrain", "reconstruction", "joint"], required=True)
    p.add_argument("--run-id", required=True); p.add_argument("--execute", action="store_true"); a = p.parse_args()
    if not a.execute: p.error("Explicit --execute is required")
    training = completed(a.training, "inductive_refiner_training")
    context = json.loads((training / "config.json").read_text(encoding="utf-8"))
    with np.load(training / "heldout_outputs.npz", allow_pickle=False) as saved:
        values = np.asarray(saved[a.representation]); ids = tuple(str(v) for v in saved["ids"])
        reference = np.asarray(saved["reference"], dtype=np.int64); batches = tuple(str(v) for v in saved["batches"])
    dataset = EvaluationDataset(ids, reference, batches,
        "Agreement with supplied HP-CB annotations on cells excluded from all fitting", True)
    config = evaluation_config(); seed = context["seed"]
    with RunDirectory(RUNS, kind="inductive_validation_score", run_id=a.run_id,
                      config={"protocol": specification()["protocol_id"], "representation": a.representation,
                              "seed": seed if a.representation != "baseline" else None,
                              "training": file_fingerprint(training / "run.json"), "evaluation": config.to_dict()}) as run:
        start = time.perf_counter()
        with threadpool_limits(limits=1):
            grid = graph_and_grid(values, reference, ids, config, run=run, prefix="heldout",
                                  training_label_use="none; evaluation labels only")
            metrics = metric_records(values, dataset, config, grid=grid, run=run, prefix="heldout")
        result = selected_summary(grid, metrics)
        result.update(representation=a.representation,
                      seed=seed if a.representation != "baseline" else None,
                      n_cells=len(values), wall_seconds=time.perf_counter() - start,
                      claim_boundary=specification()["claim_boundary"])
        run.write_json("summary.json", result)
        run.manifest.update(scientific_experiment=True, experiment_role="fully_inductive_heldout_scoring")
    print(run.final_path, flush=True)


if __name__ == "__main__": main()
