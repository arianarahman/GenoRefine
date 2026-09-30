"""Select pooled GenoRefine K and per-section SpaGCN K without annotations."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import igraph
import leidenalg
import numba
import numpy as np
import scanpy as sc
from scipy import sparse

from ..data.readers import array_hash
from ..evaluate.config import HISTORICAL_STACK
from ..evaluate.exact import exact_connectivities
from ..integrity import canonical_hash, file_fingerprint, validate_cell_ids
from ..runs import RunDirectory
from ..step4_policy import derive_training_k
from .common import (
    RUNS, evaluation_config, evaluation_dict, load_fixed_harmony, source_snapshot, specification,
    validate_evaluation_runtime,
)


def _sparse_hash(matrix: sparse.spmatrix) -> str:
    value = sparse.csr_matrix(matrix)
    value.sort_indices()
    digest = hashlib.sha256()
    for part in (np.asarray(value.shape, dtype=np.int64), value.indptr, value.indices, value.data):
        digest.update(memoryview(np.ascontiguousarray(part)).cast("B"))
    return digest.hexdigest()


def fixed_partitions(values: np.ndarray, cell_ids: list[str] | tuple[str, ...], config=None) -> dict:
    """Build one exact graph and cluster it at resolution 0.5 for seeds 0--2."""
    config = evaluation_config() if config is None else config
    ids = validate_cell_ids(cell_ids)
    values = np.asarray(values)
    if values.shape[0] != len(ids) or values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Invalid label-free K-selection embedding")
    if (config.neighbor_backend != "exact_stable_id" or config.fixed_resolution != 0.5
            or tuple(config.leiden_seeds) != (0, 1, 2) or config.n_neighbors != 15):
        raise ValueError("K selection requires the frozen primary exact graph anchor")
    expected = {name: HISTORICAL_STACK[name] for name in ("scanpy", "leidenalg", "igraph", "numba")}
    if {name: version(name) for name in expected} != expected:
        raise RuntimeError("K selection requires the recorded exact-evaluator stack")
    distances, connectivities, knn, knn_distances = exact_connectivities(
        values, ids, config.n_neighbors, working_memory_mb=config.working_memory_mb,
    )
    data = ad.AnnData(np.zeros((len(ids), 1), dtype=np.float32))
    data.obs_names = ids
    data.obsp["distances"] = distances
    data.obsp["connectivities"] = connectivities
    data.uns["neighbors"] = {
        "distances_key": "distances",
        "connectivities_key": "connectivities",
        "params": {"n_neighbors": config.n_neighbors + 1, "method": "umap", "metric": "euclidean"},
    }
    partitions = {}
    previous_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        for seed in config.leiden_seeds:
            sc.tl.leiden(
                data, resolution=0.5, random_state=seed, key_added="prediction", directed=True,
                use_weights=True, n_iterations=-1,
                partition_type=leidenalg.RBConfigurationVertexPartition,
            )
            partitions[int(seed)] = data.obs["prediction"].cat.codes.to_numpy(dtype=np.int32)
    finally:
        numba.set_num_threads(previous_threads)
    return {
        "partitions": partitions,
        "knn_indices": knn,
        "knn_distances": knn_distances,
        "provenance": {
            "cell_ids_sha256": canonical_hash(ids),
            "embedding_sha256": array_hash(values),
            "connectivities_sha256": _sparse_hash(connectivities),
            "distances_sha256": _sparse_hash(distances),
            "knn_indices_sha256": array_hash(knn),
            "knn_distances_sha256": array_hash(knn_distances),
            "resolution": 0.5,
            "leiden_seeds": [0, 1, 2],
            "reference_labels_used": False,
            "tie_policy": "distance then canonical cell ID, including cutoff ties",
            "igraph_native_version": igraph.__igraph_version__,
        },
    }


def execute(harmony: Path, run_id: str) -> Path:
    values, metadata, harmony_record = load_fixed_harmony(harmony)
    spec = specification()
    config = evaluation_config()
    frozen_evaluation = evaluation_dict()
    runtime = validate_evaluation_runtime()
    sources = source_snapshot()
    context = {
        "protocol_id": spec["protocol_id"],
        "harmony_manifest": file_fingerprint(Path(harmony) / "run.json"),
        "selection": spec["cluster_count_selection"],
        "evaluation": frozen_evaluation,
        "runtime": runtime,
        "reference_labels_used": False,
    }
    with RunDirectory(RUNS, kind="spatial_multisection_k_selection", run_id=run_id, config=context) as run:
        pooled = fixed_partitions(values, metadata["cell_id"].astype(str).tolist(), config)
        pooled_decision = derive_training_k(
            pooled["partitions"], metadata["cell_id"].astype(str).tolist(),
            metadata["cell_id"].astype(str).tolist(), frozen_evaluation,
        )
        np.savez_compressed(
            run.artifact_path("pooled_partitions.npz"),
            **{f"seed{seed}": part for seed, part in pooled["partitions"].items()},
        )
        np.save(run.artifact_path("pooled_knn_indices.npy"), pooled["knn_indices"], allow_pickle=False)
        np.save(run.artifact_path("pooled_knn_distances.npy"), pooled["knn_distances"], allow_pickle=False)
        sections = {}
        for section in spec["sections"]:
            take = np.flatnonzero(metadata["section"].astype(str).to_numpy() == section)
            ids = metadata.iloc[take]["cell_id"].astype(str).tolist()
            result = fixed_partitions(values[take], ids, config)
            decision = derive_training_k(result["partitions"], ids, ids, frozen_evaluation)
            np.savez_compressed(
                run.artifact_path(f"sections/{section}_partitions.npz"),
                **{f"seed{seed}": part for seed, part in result["partitions"].items()},
            )
            np.save(run.artifact_path(f"sections/{section}_knn_indices.npy"), result["knn_indices"], allow_pickle=False)
            sections[section] = {**decision, "n_cells": len(take), "graph": result["provenance"]}
        record = {
            "protocol_id": spec["protocol_id"],
            "harmony_manifest": file_fingerprint(Path(harmony) / "run.json"),
            "harmony_embedding_sha256": harmony_record["embedding_sha256"],
            "reference_labels_used": False,
            "pooled": {**pooled_decision, "n_cells": len(values), "graph": pooled["provenance"]},
            "sections": sections,
        }
        run.write_json("k_selection.json", record)
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        if source_snapshot() != sources:
            raise RuntimeError("Scientific source changed during K selection")
        run.manifest.update(
            scientific_experiment=True,
            experiment_role="label_free_spatial_panel_cluster_count_selection",
            training_performed=False,
            scoring_performed=False,
        )
    return run.final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmony", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required")
    print(execute(args.harmony, args.run_id), flush=True)


if __name__ == "__main__":
    main()
