# Purpose: Select GenoRefine K from the training upstream without reference labels.
# Author: Ariana Rahman (Arizona State University)

"""Select GenoRefine K from the training upstream without reference labels."""

import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from ..evaluate.engine import graph_and_grid
from ..integrity import file_fingerprint
from ..runs import RunDirectory
from .common import RUNS, completed, evaluation_config, specification


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--run-id", required=True); p.add_argument("--execute", action="store_true"); a = p.parse_args()
    if not a.execute: p.error("Explicit --execute is required")
    prepared = completed(a.prepared, "inductive_upstream_preparation")
    with np.load(prepared / "inputs.npz", allow_pickle=False) as saved:
        x = np.asarray(saved["x_train"]); ids = tuple(str(v) for v in saved["ids_train"])
    with RunDirectory(RUNS, kind="inductive_k_calibration", run_id=a.run_id,
                      config={"protocol": specification()["protocol_id"],
                              "prepared": file_fingerprint(prepared / "run.json")}) as run:
        grid = graph_and_grid(x, np.zeros(len(x), dtype=np.int64), ids, evaluation_config(), run=run,
                              prefix="training_baseline", training_label_use="none_dummy_reference")
        counts = [int(v["n_clusters"]) for v in grid["selected"]]
        run.write_json("selection.json", {"n_clusters": int(statistics.median(counts)),
            "counts_by_leiden_seed": dict(zip(("0", "1", "2"), counts)), "resolution": 0.5,
            "reference_labels_used": False, "rule": "median fixed-resolution training-baseline cluster count"})
        run.manifest.update(scientific_experiment=True, experiment_role="fully_inductive_label_free_k_calibration")
    print(run.final_path, flush=True)


if __name__ == "__main__": main()
