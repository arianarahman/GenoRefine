# Purpose: Train GenoRefine on the fully train-fitted upstream and transform held-out cells.
# Author: Ariana Rahman (Arizona State University)

"""Train GenoRefine on the fully train-fitted upstream and transform held-out cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..data.readers import array_hash
from ..data.store import EmbeddingView
from ..integrity import canonical_hash, file_fingerprint
from ..main_benchmark.fast_runtime import configure
from ..refine.config import LayoutConfig, RefinerConfig
from ..runs import RunDirectory
from ..step4.train import fit_paired
from ..step4_policy import planned_training_config
from .common import ROOT, RUNS, completed, specification


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True); p.add_argument("--seed", type=int, choices=range(5), required=True)
    p.add_argument("--run-id", required=True); p.add_argument("--runtime-profile", choices=["fast_gpu"], default="fast_gpu")
    p.add_argument("--execute", action="store_true"); a = p.parse_args()
    if not a.execute: p.error("Explicit --execute is required")
    prepared = completed(a.prepared, "inductive_upstream_preparation")
    calibration = completed(a.calibration, "inductive_k_calibration")
    decision = json.loads((calibration / "selection.json").read_text(encoding="utf-8"))
    with np.load(prepared / "inputs.npz", allow_pickle=False) as saved:
        arrays = {name: np.asarray(saved[name]) for name in saved.files}
    x_train = arrays["x_train"].astype(np.float64); ids_train = tuple(str(v) for v in arrays["ids_train"])
    x_test = arrays["x_test"].astype(np.float64); ids_test = tuple(str(v) for v in arrays["ids_test"])
    features = [str(v) for v in arrays["features"]]
    parent = EmbeddingView(x_train, ids_train, {
        "dataset_fingerprint": canonical_hash({"protocol": specification()["protocol_id"],
                                                "prepared": file_fingerprint(prepared / "run.json")}),
        "id": "train_fitted_pca_batch_centering", "stored_values_file_sha256": file_fingerprint(prepared / "inputs.npz")["sha256"],
        "coordinate_names": features})
    config = RefinerConfig(training=planned_training_config(len(ids_train), decision["n_clusters"], a.seed),
                           layout=LayoutConfig(requested_side=36, scaling="none", transport_iterations=200, epsilon=0.0))
    runtime = configure(a.runtime_profile)
    context = {"protocol": specification()["protocol_id"], "seed": a.seed,
               "prepared": str(prepared.resolve()), "calibration": str(calibration.resolve()),
               "K_binding": decision, "runtime": runtime, "effective_refiner": config.to_dict()}
    with RunDirectory(RUNS, kind="inductive_refiner_training", config=context, run_id=a.run_id) as run:
        run.write_json("input.json", {"train_cells": len(x_train), "heldout_cells": len(x_test),
            "parent_reference": parent.parent_reference(), "training_label_use": "none",
            "fit_boundary": "Held-out cells unseen by upstream, layout and network fitting"})
        outputs = fit_paired(parent, config, run)
        from ..refine.staged import StagedGenoDR
        evaluation = {"baseline": x_test}
        for stage in ("pretrain", "reconstruction", "joint"):
            model = StagedGenoDR.load(run.path / "models" / stage)
            evaluation[stage] = model.transform(x_test, cell_ids=ids_test, feature_ids=features)
        np.savez_compressed(run.artifact_path("heldout_outputs.npz"), ids=arrays["ids_test"],
            reference=arrays["reference_test"], batches=arrays["batch_test"], **evaluation)
        run.write_json("heldout_output.json", {"mode": "frozen train-fitted upstream, layout and encoder",
            "representations": {name: {"shape": list(value.shape), "sha256": array_hash(value)}
                                for name, value in evaluation.items()},
            "training_output_hashes": {name: array_hash(value) for name, value in outputs.items()},
            "claim_boundary": specification()["claim_boundary"]})
        run.manifest.update(scientific_experiment=True, experiment_role="fully_inductive_refiner_training")
    print(run.final_path, flush=True)


if __name__ == "__main__": main()
