# Purpose: Train standalone scVI backbones for mouse or five-study pancreas.
# Author: Ariana Rahman (Arizona State University)

"""Train standalone scVI backbones for mouse or five-study pancreas."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import scvi

from ..data.readers import array_hash
from ..data.store import Store
from ..integrity import file_fingerprint
from ..runs import RunDirectory

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "revision_pipeline/runs"
SPEC_PATH = ROOT / "revision_pipeline/configs/scvi_other_datasets_v1.json"


def _mouse(spec: dict, dataset):
    source = ROOT / spec["source"]
    h5 = ad.read_h5ad(source)
    if tuple(map(str, h5.obs_names)) != tuple(dataset.cell_ids):
        raise ValueError("Mouse H5AD order differs from the frozen canonical order")
    features = [x.strip() for x in (ROOT / spec["feature_selection"]).read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(features) != 2000 or len(set(features)) != 2000:
        raise ValueError("Frozen mouse HVG list must contain 2000 unique genes")
    raw_names = pd.Index(h5.raw.var_names.astype(str))
    selected = raw_names.get_indexer(features)
    if np.any(selected < 0):
        raise ValueError("Frozen mouse HVG list contains genes absent from raw/X")
    counts = h5.raw.X[:, selected]
    counts = counts.tocsr().astype(np.float32) if sp.issparse(counts) else sp.csr_matrix(np.asarray(counts, dtype=np.float32))
    source_files = [source, ROOT / spec["feature_selection"]]
    return counts, np.asarray(features, dtype="U"), source_files


def _pancreas(spec: dict, dataset):
    parts = dataset.record["loader"]["parts"]
    matrices = []
    source_files = []
    for item in parts:
        path = ROOT / item["path"]
        values = np.asarray(sio.loadmat(path)[item["key"]], dtype=np.float32)
        matrices.append(values)
        source_files.append(path)
    counts = sp.csr_matrix(np.vstack(matrices), dtype=np.float32)
    features = np.asarray([f"supplied_feature_{i + 1}" for i in range(counts.shape[1])], dtype="U")
    return counts, features, source_files


def _audit(values: sp.csr_matrix, batches: np.ndarray, status: str):
    data = values.data
    row_sums = np.asarray(values.sum(axis=1), dtype=np.float64).ravel()
    by_batch = {}
    for batch in sorted(set(map(str, batches))):
        mask = batches == batch
        sums = row_sums[mask]
        by_batch[batch] = {
            "cells": int(mask.sum()),
            "row_sum_minimum": float(sums.min()),
            "row_sum_median": float(np.median(sums)),
            "row_sum_maximum": float(sums.max()),
        }
    return {
        "shape": list(values.shape),
        "noninteger_fraction_nonzero": float(np.mean(data != np.rint(data))),
        "minimum": float(data.min()) if data.size else 0.0,
        "maximum": float(data.max()) if data.size else 0.0,
        "zero_fraction": float(1.0 - values.nnz / np.prod(values.shape)),
        "row_sum_minimum": float(row_sums.min()),
        "row_sum_median": float(np.median(row_sums)),
        "row_sum_maximum": float(row_sums.max()),
        "by_batch": by_batch,
        "interpretation": status,
    }


def main():
    """Audit the input, train one label-free scVI backbone, and persist its latent model."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["mouse_senis", "pancreas_five_study"], required=True)
    p.add_argument("--seed", type=int, choices=range(5), required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if not a.execute:
        p.error("Explicit --execute is required")

    protocol = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    spec = protocol["datasets"][a.dataset]
    store = Store(ROOT / protocol["store"])
    dataset = store.dataset(a.dataset)
    ids = np.asarray(dataset.cell_ids, dtype="U")
    batches = np.asarray(dataset.batch_labels(), dtype="U")
    values, feature_ids, source_files = (_mouse(spec, dataset) if a.dataset == "mouse_senis" else _pancreas(spec, dataset))
    if values.shape != (len(ids), 2000) or values.data.size == 0 or np.any(values.data < 0) or not np.isfinite(values.data).all():
        raise ValueError("Invalid scVI input matrix")

    audit = _audit(values, batches, spec["input_status"])
    if a.dataset == "mouse_senis" and audit["noninteger_fraction_nonzero"] != 0.0:
        raise ValueError("Mouse raw/X is no longer integer-valued")
    scvi.settings.seed = a.seed
    adata = ad.AnnData(
        X=values,
        obs=pd.DataFrame({spec["batch_key"]: batches}, index=ids),
        var=pd.DataFrame(index=feature_ids),
    )
    scvi.model.SCVI.setup_anndata(adata, batch_key=spec["batch_key"])
    model = scvi.model.SCVI(
        adata,
        n_hidden=protocol["hidden"],
        n_latent=protocol["latent_dim"],
        n_layers=protocol["layers"],
        gene_likelihood=protocol["gene_likelihood"],
        dispersion=protocol["dispersion"],
    )
    start = time.perf_counter()
    model.train(
        max_epochs=protocol["max_epochs"],
        early_stopping=protocol["early_stopping"],
        accelerator="gpu",
        devices=1,
        batch_size=protocol["batch_size"],
        enable_progress_bar=False,
    )
    train_seconds = time.perf_counter() - start
    latent = np.asarray(model.get_latent_representation(), dtype=np.float64)
    context = {
        "protocol": protocol,
        "dataset": a.dataset,
        "seed": a.seed,
        "sources": [file_fingerprint(path) for path in source_files],
        "labels_used": False,
        "input_audit": audit,
    }
    with RunDirectory(RUNS, kind="scvi_standalone_backbone_training", config=context, run_id=a.run_id) as run:
        np.savez_compressed(
            run.artifact_path("embedding.npz"),
            values=latent,
            ids=ids,
            features=np.asarray([f"scVI{i + 1}" for i in range(latent.shape[1])], dtype="U"),
        )
        model.save(run.artifact_path("model"), overwrite=True, save_anndata=False)
        history = {name: [float(v) for v in frame.iloc[:, 0].to_numpy()] for name, frame in model.history.items()}
        run.write_json("history.json", history)
        run.write_json("summary.json", {
            "dataset": a.dataset,
            "seed": a.seed,
            "shape": list(latent.shape),
            "sha256": array_hash(latent),
            "epochs_run": max(len(v) for v in history.values()),
            "training_wall_seconds": train_seconds,
            "history_final": {k: v[-1] for k, v in history.items() if v},
            "device": str(model.device),
            "labels_used": False,
            "input_audit": audit,
        })
        run.manifest.update(
            scientific_experiment=True,
            experiment_role="standalone_unsupervised_scvi_backbone",
            evidence_status="primary_comparator" if a.dataset == "mouse_senis" else "sensitivity_only",
        )
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
