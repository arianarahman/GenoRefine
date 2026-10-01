# Purpose: Cross-fitted expression signatures for embedding-neighborhood validation.
# Author: Ariana Rahman (Arizona State University)

"""Cross-fitted expression signatures for embedding-neighborhood validation.

Signatures are learned from one deterministic half of the cells and assigned to
the other half, then vice versa.  They are independent of the evaluated
embeddings but annotation-coupled because supplied classes define the signatures.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import statistics

import numpy as np
from scipy import sparse
from scipy.io import loadmat

from ..data.readers import array_hash
from ..data.store import Store, runtime_inventory
from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import read
from ..runs import RunDirectory

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "revision_pipeline/runs/20260916T221434Z-a0884ffbd933"
DATASETS = ("pancreas_five_study", "hpcb", "mouse_senis")


def deterministic_folds(cell_ids):
    return np.asarray([hashlib.sha256(value.encode("utf-8")).digest()[0] & 1
                       for value in cell_ids], dtype=np.int8)


def normalized_log1p(values):
    x = sparse.csr_matrix(values, dtype=np.float32)
    if x.data.size and (not np.isfinite(x.data).all() or x.data.min() < 0):
        raise ValueError("Expression must be finite and nonnegative")
    totals = np.asarray(x.sum(axis=1)).ravel()
    if np.any(totals <= 0):
        raise ValueError("Zero-total expression row")
    x = sparse.diags((10000.0 / totals).astype(np.float32)) @ x
    x.data = np.log1p(x.data)
    return x.tocsr()


def load_expression(dataset, expected_ids):
    import anndata as ad
    if dataset == "pancreas_five_study":
        items = (("dataBaronX.mat", "dataBaron"), ("dataMuraroX.mat", "dataMuraro"),
                 ("dataScapleX.mat", "dataScaple"), ("dataWangX.mat", "dataWang"),
                 ("dataXinX.mat", "dataXin"))
        arrays = [np.asarray(loadmat(ROOT / "PancreasDataset/Dataset" / name)[key])
                  for name, key in items]
        x = sparse.csr_matrix(np.vstack(arrays), dtype=np.float32)
        names = [str(i) for i in range(x.shape[1])]
        sources = [ROOT / "PancreasDataset/Dataset" / name for name, _ in items]
        gene_status = "positional_features_only_no_gene_symbols_marker_identity_not_assessable"
    else:
        relative = ("HPCBDataset/Dataset/human_pancreas_norm_complexBatch.h5ad"
                    if dataset == "hpcb" else
                    "MouseDataset/Dataset/tabula-muris_sub50k_combined.h5ad")
        path = ROOT / relative
        a = ad.read_h5ad(path)
        if list(map(str, a.obs_names)) != list(expected_ids):
            raise ValueError("H5AD order differs from canonical store")
        if dataset == "hpcb":
            x = a.layers["counts"]
            names = list(map(str, a.var_names))
        else:
            x = a.raw.X if a.raw is not None else a.X
            names = list(map(str, a.raw.var_names if a.raw is not None else a.var_names))
        x = sparse.csr_matrix(x, dtype=np.float32)
        sources = [path]
        gene_status = "gene_symbols_present_cross_fitted_annotation_coupled_signatures"
    if x.shape[0] != len(expected_ids) or len(names) != x.shape[1]:
        raise ValueError("Expression shape/ID mismatch")
    return normalized_log1p(x), names, sources, gene_status


def crossfit_signatures(x, labels, cell_ids, gene_names, *, top_n=10, min_train=5,
                        min_test=5, chunk_size=5000):
    labels = np.asarray(labels).astype(str)
    classes = np.asarray(sorted(set(labels)))
    true_codes = np.searchsorted(classes, labels).astype(np.int32)
    folds = deterministic_folds(cell_ids)
    predictions = np.full(len(labels), -1, dtype=np.int32)
    eligible_any = np.zeros(len(classes), dtype=bool)
    fold_records = []
    excluded = np.asarray([str(g).upper().startswith(("MT-", "RPL", "RPS"))
                           for g in gene_names])
    for test_fold in (0, 1):
        train = folds != test_fold
        test = folds == test_fold
        train_counts = np.bincount(true_codes[train], minlength=len(classes))
        test_counts = np.bincount(true_codes[test], minlength=len(classes))
        eligible = np.flatnonzero((train_counts >= min_train) & (test_counts >= min_test))
        if len(eligible) < 2:
            raise ValueError("Too few eligible signature classes")
        eligible_any[eligible] = True
        train_all_mean = np.asarray(x[train].mean(axis=0)).ravel()
        marker_indices = []
        for code in eligible:
            in_class = train & (true_codes == code)
            class_mean = np.asarray(x[in_class].mean(axis=0)).ravel()
            rest_n = int(train.sum() - in_class.sum())
            rest_mean = (train_all_mean * train.sum() - class_mean * in_class.sum()) / rest_n
            effect = class_mean - rest_mean
            effect[excluded] = -np.inf
            order = np.lexsort((np.arange(len(effect)), -effect))
            selected = order[:top_n]
            if not np.isfinite(effect[selected]).all() or np.any(effect[selected] <= 0):
                raise ValueError("Insufficient positive signature features")
            marker_indices.append(selected)
        union = np.unique(np.concatenate(marker_indices))
        lookup = {int(value): i for i, value in enumerate(union)}
        marker_positions = [np.asarray([lookup[int(v)] for v in row])
                            for row in marker_indices]
        train_union = x[train][:, union]
        mean = np.asarray(train_union.mean(axis=0)).ravel()
        second = np.asarray(train_union.power(2).mean(axis=0)).ravel()
        scale = np.sqrt(np.maximum(second - mean ** 2, 1e-6))
        test_indices = np.flatnonzero(test)
        for begin in range(0, len(test_indices), chunk_size):
            cell_index = test_indices[begin:begin + chunk_size]
            dense = x[cell_index][:, union].toarray().astype(np.float32, copy=False)
            dense = (dense - mean) / scale
            scores = np.column_stack([dense[:, positions].mean(axis=1)
                                      for positions in marker_positions])
            predictions[cell_index] = eligible[np.argmax(scores, axis=1)]
        fold_records.append({
            "test_fold": test_fold,
            "train_cells": int(train.sum()), "test_cells": int(test.sum()),
            "eligible_classes": [classes[i] for i in eligible],
            "markers": {classes[code]: [gene_names[i] for i in indices]
                        for code, indices in zip(eligible, marker_indices)}})
    if np.any(predictions < 0):
        raise ValueError("Cross-fitting left cells unassigned")
    eligible_cells = eligible_any[true_codes]
    per_class = {}
    recalls = []
    for code in np.flatnonzero(eligible_any):
        mask = true_codes == code
        recall = float(np.mean(predictions[mask] == code))
        recalls.append(recall)
        per_class[classes[code]] = {"cells": int(mask.sum()), "recall": recall}
    return {"predictions": predictions, "true_codes": true_codes,
            "eligible_cells": eligible_cells, "folds": folds,
            "classes": classes.tolist(), "fold_records": fold_records,
            "accuracy_eligible": float(np.mean(predictions[eligible_cells] == true_codes[eligible_cells])),
            "balanced_accuracy_eligible": statistics.mean(recalls),
            "eligible_class_count": int(eligible_any.sum()),
            "total_class_count": len(classes), "per_class": per_class}


def annotation_audit(dataset, view):
    record = {"dataset": dataset, "cells": len(view.cell_ids),
              "unique_cell_ids": len(set(view.cell_ids)) == len(view.cell_ids),
              "annotation_policy": view.annotation_policy}
    if dataset == "pancreas_five_study":
        items = (("dataBaronX.mat", "GT_b"), ("dataMuraroX.mat", "GT_m"),
                 ("dataScapleX.mat", "GT_s"), ("dataWangX.mat", "GT_w"),
                 ("dataXinX.mat", "GT_x"))
        combined = np.concatenate([np.asarray(loadmat(ROOT / "PancreasDataset/Dataset" / p)[k]).ravel()
                                   for p, k in items])
        labels = np.asarray(loadmat(ROOT / "PancreasDataset/Dataset/classLabel.mat")["classLabel"]).ravel()
        record.update(per_study_GT_concatenation_matches_classLabel=bool(np.array_equal(combined, labels)),
                      numeric_codes=sorted(map(int, np.unique(labels))),
                      code15_cells=int(np.sum(labels == 15)),
                      code15_study="Baron",
                      authoritative_code15_name_available=False,
                      gene_symbols_available=False)
    elif dataset == "mouse_senis":
        names = np.asarray(view.record["obs"]["cell_ontology_class"]).astype(str)
        ontology = np.asarray(view.record["obs"]["cell_ontology_id"]).astype(str)
        valid = ~np.isin(ontology, ["NA", "nan", "None", ""])
        pairs = set(zip(names[valid], ontology[valid]))
        record.update(missing_class_names=int(np.sum(names == "")),
                      supplied_class_count=len(set(names)), ontology_id_valid_cells=int(valid.sum()),
                      ontology_id_missing_cells=int((~valid).sum()),
                      distinct_name_id_pairs=len(pairs),
                      ontology_mapping_one_to_one=(len(pairs) == len(set(names[valid])) == len(set(ontology[valid]))),
                      ontology_id_policy="Do not use as an authoritative one-to-one mapping")
    else:
        names = np.asarray(view.record["obs"]["celltype"]).astype(str)
        record.update(missing_class_names=int(np.sum(names == "")),
                      supplied_class_count=len(set(names)), gene_symbols_available=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required")
    store = Store(STORE)
    config = {"datasets": list(DATASETS), "fold_rule": "sha256(cell_id)[0] parity",
              "top_features_per_class": 10, "minimum_cells_per_fold": 5,
              "normalization": "library_size_10000_log1p",
              "scope": "embedding-independent but annotation-coupled cross-fitted expression signatures"}
    with RunDirectory(ROOT / "revision_pipeline/runs", kind="biological_marker_crossfit",
                      config=config, run_id=args.run_id) as run:
        run.write_json("runtime_start.json", runtime_inventory())
        summaries = {}
        for dataset in DATASETS:
            view = store.dataset(dataset)
            reference, interpretation = view.reference_partition()
            raw_labels = np.asarray(view.record["obs"][view.record["loader"]["label_key"]]).astype(str)
            x, genes, sources, gene_status = load_expression(dataset, view.cell_ids)
            result = crossfit_signatures(x, raw_labels, view.cell_ids, genes)
            prefix = run.artifact_path(dataset)
            prefix.mkdir()
            np.save(prefix / "predicted_codes.npy", result.pop("predictions"), allow_pickle=False)
            np.save(prefix / "true_codes.npy", result.pop("true_codes"), allow_pickle=False)
            np.save(prefix / "eligible_cells.npy", result.pop("eligible_cells"), allow_pickle=False)
            np.save(prefix / "folds.npy", result.pop("folds"), allow_pickle=False)
            audit = annotation_audit(dataset, view)
            summary = {**result, "dataset": dataset, "cells": len(view.cell_ids),
                       "features": x.shape[1], "gene_identity_status": gene_status,
                       "reference_interpretation": interpretation,
                       "annotation_audit": audit,
                       "source_files": {str(p.relative_to(ROOT)): file_fingerprint(p) for p in sources},
                       "predicted_codes_sha256": array_hash(np.load(prefix / "predicted_codes.npy", allow_pickle=False)),
                       "independent_of_evaluated_embeddings": True,
                       "independent_of_supplied_annotations": False}
            run.write_json(f"{dataset}/summary.json", summary)
            summaries[dataset] = {k: summary[k] for k in
                                  ("cells", "features", "gene_identity_status",
                                   "accuracy_eligible", "balanced_accuracy_eligible",
                                   "eligible_class_count", "total_class_count", "annotation_audit")}
        run.write_json("summary.json", {"datasets": summaries,
                       "claim_boundary": "Cross-fitted expression evidence, not independent annotation ground truth"})
        run.write_json("runtime_end.json", runtime_inventory())
        run.manifest.update(scientific_experiment=True,
                            experiment_role="biological_expression_signature_validation")
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()

