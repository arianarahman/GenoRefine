# Purpose: Fit the complete upstream representation without seeing held-out HP-CB cells.
# Author: Ariana Rahman (Arizona State University)

"""Fit the complete upstream representation without seeing held-out HP-CB cells."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA

from ..data.readers import array_hash
from ..data.store import Store
from ..integrity import file_fingerprint
from ..runs import RunDirectory
from .common import ROOT, RUNS, specification


def heldout_mask(ids, batches, seed):
    mask = np.asarray([int.from_bytes(hashlib.sha256(f"{cell}|{seed}".encode()).digest()[:8], "big") % 5 == 0
                       for cell in ids], dtype=bool)
    for batch in sorted(set(batches)):
        here = np.asarray(batches) == batch
        if mask[here].sum() < 20 or (~mask & here).sum() < 20:
            raise ValueError(f"Insufficient deterministic split coverage for {batch}")
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True); parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute: parser.error("Explicit --execute is required")
    spec = specification(); store = Store(ROOT / spec["store"]); store.verify(ROOT)
    dataset = store.dataset(spec["dataset"])
    ids = np.asarray(dataset.cell_ids, dtype="U")
    batches = np.asarray(dataset.batch_labels(), dtype="U")
    reference, interpretation = dataset.reference_partition()
    test = heldout_mask(ids, batches, spec["split"]["seed"]); train = ~test
    source = ROOT / spec["source_h5ad"]
    n, d = len(ids), len(dataset.record["feature_ids"])
    sums = np.zeros(d, dtype=np.float64); squares = np.zeros(d, dtype=np.float64); count = 0
    with h5py.File(source, "r") as handle:
        x = handle["X"]
        if x.shape != (n, d): raise ValueError("Source expression shape changed")
        for start in range(0, n, 512):
            stop = min(n, start + 512); block = np.asarray(x[start:stop], dtype=np.float64)
            keep = train[start:stop]
            sums += block[keep].sum(axis=0); squares += np.square(block[keep]).sum(axis=0); count += int(keep.sum())
    variance = np.maximum(squares / count - np.square(sums / count), 0.0)
    ranked = np.lexsort((np.arange(d), -variance))[:2000]
    selected = np.sort(ranked)
    values = np.empty((n, len(selected)), dtype=np.float32)
    with h5py.File(source, "r") as handle:
        x = handle["X"]
        for start in range(0, n, 512):
            stop = min(n, start + 512); values[start:stop] = x[start:stop, selected]
    mean = values[train].mean(axis=0, dtype=np.float64)
    scale = values[train].std(axis=0, dtype=np.float64); scale[scale == 0] = 1.0
    standardized = ((values.astype(np.float64) - mean) / scale).astype(np.float32)
    upstream = spec["upstream"]
    pca = PCA(n_components=upstream["pca_components"], svd_solver=upstream["pca_solver"],
              random_state=upstream["pca_seed"])
    z_train = pca.fit_transform(standardized[train]).astype(np.float64)
    z_test = pca.transform(standardized[test]).astype(np.float64)
    global_mean = z_train.mean(axis=0)
    train_batches, test_batches = batches[train], batches[test]
    batch_means = {}
    for batch in sorted(set(batches)):
        batch_means[batch] = z_train[train_batches == batch].mean(axis=0)
        z_train[train_batches == batch] += global_mean - batch_means[batch]
        z_test[test_batches == batch] += global_mean - batch_means[batch]
    config = {"protocol": spec, "source": file_fingerprint(source)}
    with RunDirectory(RUNS, kind="inductive_upstream_preparation", config=config, run_id=args.run_id) as run:
        np.savez_compressed(run.artifact_path("inputs.npz"),
            x_train=z_train, x_test=z_test, ids_train=ids[train], ids_test=ids[test],
            reference_train=reference[train], reference_test=reference[test],
            batch_train=train_batches, batch_test=test_batches,
            features=np.asarray([f"PC{i+1}" for i in range(z_train.shape[1])], dtype="U"),
            selected_feature_indices=selected)
        run.write_json("upstream_model.json", {
            "train_cells": int(train.sum()), "heldout_cells": int(test.sum()),
            "split_by_batch": {b: {"train": int(((batches == b) & train).sum()),
                                    "heldout": int(((batches == b) & test).sum())} for b in sorted(set(batches))},
            "label_use": "none in splitting, feature selection, scaling, PCA and batch centering",
            "selected_feature_ids_sha256": array_hash(np.asarray(dataset.record["feature_ids"], dtype="U")[selected]),
            "selected_feature_indices_sha256": array_hash(selected),
            "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "reference_interpretation": interpretation,
            "fit_boundary": "All fitted upstream quantities use training cells only",
            "claim_boundary": spec["claim_boundary"]})
        run.manifest.update(scientific_experiment=True, experiment_role="fully_inductive_upstream_preparation")
    print(run.final_path, flush=True)


if __name__ == "__main__": main()
