# Purpose: Train a pinned unsupervised scVI backbone on HP-CB counts.
# Author: Ariana Rahman (Arizona State University)

"""Train a pinned unsupervised scVI backbone on HP-CB counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scvi

from ..data.readers import array_hash
from ..data.store import Store
from ..integrity import file_fingerprint
from ..runs import RunDirectory

ROOT = Path(__file__).resolve().parents[2]; RUNS = ROOT / "revision_pipeline/runs"
SPEC_PATH = ROOT / "revision_pipeline/configs/scvi_comparator_v1.json"


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--seed", type=int, choices=range(5), required=True)
    p.add_argument("--run-id", required=True); p.add_argument("--execute", action="store_true"); a = p.parse_args()
    if not a.execute: p.error("Explicit --execute is required")
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8")); scvi.settings.seed = a.seed
    selection_run = ROOT / spec["feature_selection_run"]
    with np.load(selection_run / "inputs.npz", allow_pickle=False) as saved:
        selected = np.asarray(saved["selected_feature_indices"], dtype=np.int64)
    if selected.shape != (2000,) or np.any(np.diff(selected) <= 0): raise ValueError("Frozen feature selection changed")
    store = Store(ROOT / spec["store"]); dataset = store.dataset(spec["dataset"]); source = ROOT / spec["source_h5ad"]
    ids = np.asarray(dataset.cell_ids, dtype="U"); batches = np.asarray(dataset.batch_labels(), dtype="U")
    feature_ids = np.asarray(dataset.record["feature_ids"], dtype="U")[selected]
    with h5py.File(source, "r") as handle:
        counts = np.asarray(handle["layers/counts"][:, selected], dtype=np.float32)
    if counts.shape != (len(ids), 2000) or np.any(counts < 0) or not np.isfinite(counts).all():
        raise ValueError("Invalid scVI count input")
    noninteger_fraction = float(np.mean(counts != np.rint(counts)))
    row_sums = counts.sum(axis=1, dtype=np.float64)
    adata = ad.AnnData(X=counts, obs=pd.DataFrame({spec["batch_key"]: batches}, index=ids),
                       var=pd.DataFrame(index=feature_ids))
    scvi.model.SCVI.setup_anndata(adata, batch_key=spec["batch_key"])
    model = scvi.model.SCVI(adata, n_hidden=spec["hidden"], n_latent=spec["latent_dim"],
                            n_layers=spec["layers"], gene_likelihood=spec["gene_likelihood"],
                            dispersion=spec["dispersion"])
    model.train(max_epochs=spec["max_epochs"], early_stopping=spec["early_stopping"],
                accelerator="gpu", devices=1, batch_size=spec["batch_size"], enable_progress_bar=False)
    latent = np.asarray(model.get_latent_representation(), dtype=np.float64)
    context = {"protocol": spec, "seed": a.seed, "source": file_fingerprint(source),
               "selection": file_fingerprint(selection_run / "run.json"), "labels_used": False,
               "input_audit": {"noninteger_fraction": noninteger_fraction,
                   "minimum": float(counts.min()), "maximum": float(counts.max()),
                   "row_sum_minimum": float(row_sums.min()),
                   "row_sum_median": float(np.median(row_sums)),
                   "row_sum_maximum": float(row_sums.max()),
                   "interpretation": spec["input_limitation"]}}
    with RunDirectory(RUNS, kind="scvi_backbone_training", config=context, run_id=a.run_id) as run:
        np.savez_compressed(run.artifact_path("embedding.npz"), values=latent, ids=ids,
                            features=np.asarray([f"scVI{i+1}" for i in range(latent.shape[1])], dtype="U"))
        model.save(run.artifact_path("model"), overwrite=True, save_anndata=False)
        history = {name: [float(v) for v in frame.iloc[:, 0].to_numpy()]
                   for name, frame in model.history.items()}
        run.write_json("summary.json", {"seed": a.seed, "shape": list(latent.shape),
            "sha256": array_hash(latent), "epochs_run": max(len(v) for v in history.values()),
            "history_final": {k: v[-1] for k, v in history.items() if v},
            "device": str(model.device), "labels_used": False,
            "input_audit": context["input_audit"]})
        run.write_json("history.json", history)
        run.manifest.update(scientific_experiment=True, experiment_role="contemporary_unsupervised_scvi_backbone")
    print(run.final_path, flush=True)


if __name__ == "__main__": main()
