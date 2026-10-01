# Purpose: Graph/grid engine and metric records.
# Author: Ariana Rahman (Arizona State University)

"""Graph/grid engine and metric records. All coordinates remain ID aligned."""

from importlib.metadata import version
import time
import warnings

import numpy as np
from scipy.sparse import save_npz
from sklearn.metrics import adjusted_rand_score, rand_score

from ..data.readers import array_hash
from ..integrity import canonical_hash, validate_cell_ids
from .config import HISTORICAL_STACK
from .exact import exact_connectivities
from .metrics import asw, d_batch, ilisi, isolated_asw, labels, matrix, neighbors, purity


def assert_historical_stack():
    observed = {k: version(k) for k in HISTORICAL_STACK}
    if observed != HISTORICAL_STACK:
        raise RuntimeError(f"Historical regression stack mismatch: {observed}")
    return observed


def select_rows(rows, config, reference_count):
    """Selection is separate from fitting/scoring. Stable tie -> first resolution."""
    if config.selection == "grid_only":
        return []
    selected = []
    for seed in config.leiden_seeds:
        candidates = [r for r in rows if r["leiden_seed"] == seed]
        if config.selection == "fixed_resolution":
            chosen = next(r for r in candidates if r["resolution"] == config.fixed_resolution)
        elif config.selection == "matched_reference_count":
            chosen = min(candidates, key=lambda r: abs(r["n_clusters"] - reference_count))
        else:
            chosen = max(candidates, key=lambda r: r["ARI"])
        difference = abs(chosen["n_clusters"]-reference_count) / reference_count
        calibration = None
        if config.selection == "matched_reference_count":
            calibration = {
                "status": "exact_match" if difference == 0 else (
                    "within_tolerance" if difference <= config.calibration_relative_tolerance else "failed"),
                "relative_count_error": difference,
                "relative_tolerance": config.calibration_relative_tolerance,
                "grid_cluster_count_min": min(r["n_clusters"] for r in candidates),
                "grid_cluster_count_max": max(r["n_clusters"] for r in candidates),
                "selected_at_grid_boundary": chosen["resolution"] in (config.resolutions[0], config.resolutions[-1]),
                "interpretation": "Closest available count, not necessarily a successful calibration; label-informed secondary only"}
        selected.append(dict(chosen, selected=True, calibration=calibration,
                             reference_count_difference=(chosen["n_clusters"]-reference_count)
                             if config.selection == "matched_reference_count" else None))
    return selected


def graph_representation(values, config):
    """Private writable buffer for PyNNDescent, preserving stored input bytes.

    ascontiguousarray alone can return a read-only view unchanged. PyNNDescent
    0.6.0's Numba signatures reject such a native float32 input on large graphs.
    """
    x = matrix(values)
    dims = min(config.dimensions, x.shape[1]) if config.dimensions else x.shape[1]
    return np.array(x[:, :dims], dtype=np.float32 if config.precision == "float32" else x.dtype,
                    order="C", copy=True)


def graph_and_grid(values, reference, cell_ids, config, *, run=None, prefix="evaluation", training_label_use="unknown"):
    """Build the exact neighbor graph and evaluate the declared Leiden resolution grid."""
    import anndata as ad
    import scanpy as sc
    import leidenalg
    import numba
    import igraph
    x = matrix(values)
    ids = validate_cell_ids(cell_ids)
    if len(ids) != len(x) or len(x) <= config.n_neighbors:
        raise ValueError("ID coverage mismatch or n <= graph n_neighbors")
    y = labels(reference, len(x))
    if config.purpose == "historical_regression":
        assert_historical_stack()
    if config.purpose == "primary_evaluation":
        expected = {name: HISTORICAL_STACK[name] for name in ("scanpy", "leidenalg", "igraph", "numba")}
        if {name: version(name) for name in expected} != expected:
            raise RuntimeError("pre3c_exact_v1 requires the recorded validated Leiden/graph stack")
    rep = graph_representation(x, config)
    dims = rep.shape[1]
    a = ad.AnnData(np.zeros((len(x), 1), dtype=np.float32))
    a.obs_names = list(ids)
    a.obsm["X_evaluation"] = rep
    t0 = time.perf_counter()
    captured = []
    previous_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(config.neighbor_threads)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            if config.neighbor_backend == "exact_stable_id":
                distances, connectivities, knn_ids, knn_distances = exact_connectivities(
                    rep, ids, config.n_neighbors, working_memory_mb=config.working_memory_mb)
                a.obsp["distances"], a.obsp["connectivities"] = distances, connectivities
                a.uns["neighbors"] = {"distances_key": "distances", "connectivities_key": "connectivities",
                                      "params": {"n_neighbors": config.n_neighbors+1, "method": "umap",
                                                 "metric": "euclidean", "use_rep": "X_evaluation"}}
            else:
                sc.pp.neighbors(a, use_rep="X_evaluation", n_neighbors=config.n_neighbors,
                                metric=config.metric, random_state=config.graph_seed, method="umap")
            captured.extend(str(w.message) for w in records)
    finally:
        numba.set_num_threads(previous_threads)
    graph_seconds = time.perf_counter()-t0
    rows, partitions = [], []
    clustering_started = time.perf_counter()
    for seed in config.leiden_seeds:
        if run:
            print(f"  {prefix}: Leiden seed {seed}, {len(config.resolutions)} resolutions", flush=True)
        for resolution in config.resolutions:
            start = time.perf_counter()
            sc.tl.leiden(a, resolution=float(resolution), random_state=seed, key_added="prediction",
                         directed=True, use_weights=True, n_iterations=-1,
                         partition_type=leidenalg.RBConfigurationVertexPartition)
            pred = a.obs["prediction"].cat.codes.to_numpy(dtype=np.int32)
            partitions.append(pred.copy())
            row = {"profile": config.name, "purpose": config.purpose,
                   "resolution": float(resolution), "leiden_seed": seed, "graph_seed": config.graph_seed,
                   "n_clusters": int(len(np.unique(pred))), "ARI": float(adjusted_rand_score(y, pred)),
                   "RI": float(rand_score(y, pred)), "selection_rule": config.selection,
                   "selection_label_informed": config.label_informed,
                   "training_label_use": training_label_use,
                   "reference_count_target": int(len(np.unique(y))) if config.selection == "matched_reference_count" else None,
                   "graph_label_informed": False, "ARI_uses_reference_for_scoring": True,
                   "n_cells": len(x), "dimensions_used": dims, "distance": config.metric,
                   "n_neighbors": config.n_neighbors, "partition_index": len(rows),
                   "leiden_and_agreement_seconds": time.perf_counter()-start}
            rows.append(row)
    timing = {"graph_seconds": graph_seconds, "grid_seconds": time.perf_counter()-clustering_started}
    partitions = np.asarray(partitions, dtype=np.int32)
    graph_info = {"effective_dimensions": dims, "dtype": str(rep.dtype), "representation_sha256": array_hash(rep),
                  "cell_order_sha256": canonical_hash(list(ids)), "graph_method": "scanpy_umap_connectivities",
                  "directed": True, "use_weights": True, "n_iterations": -1,
                  "partition_type": "leidenalg.RBConfigurationVertexPartition", "warnings": captured,
                  "neighbor_numba_threads": config.neighbor_threads, "native_igraph_version": igraph.__igraph_version__,
                  "private_writable_working_copy": True,
                  "neighbor_search": "scanpy_dense_pairwise" if len(x) < (8192 if config.metric == "euclidean" else 4096) else "pynndescent",
                  "dense_graph_distance_upper_bound_bytes": len(x)**2 * 8 if len(x) < (8192 if config.metric == "euclidean" else 4096) else 0,
                  "self_policy": "Scanpy graph construction (separate from identity-excluded geometry metrics)"}
    graph_info["neighbor_backend"] = config.neighbor_backend
    if config.neighbor_backend == "exact_stable_id":
        graph_info.update(neighbor_search="scipy_cdist_exact_blocked", distance_arithmetic="float64_direct_euclidean",
                          tie_policy="distance_then_lexicographic_unique_cell_ID_including_cutoff_ties",
                          self_policy="Non-self neighbors; explicit self prepended only for UMAP affinity",
                          nonself_neighbors=config.n_neighbors, umap_neighbor_slots=config.n_neighbors+1,
                          affinity="UMAP fuzzy union; set_op_mix_ratio=1; local_connectivity=1",
                          dense_graph_distance_upper_bound_bytes=0,
                          distance_block_budget_bytes=config.working_memory_mb*1024**2,
                          exact_neighbor_indices_sha256=array_hash(knn_ids),
                          exact_neighbor_distances_sha256=array_hash(knn_distances))
    selected = select_rows(rows, config, len(np.unique(y)))
    if run:
        run.write_json(f"{prefix}/cell_ids.json", list(ids))
        run.write_json(f"{prefix}/graph.json", graph_info)
        run.write_json(f"{prefix}/grid.json", rows)
        run.write_json(f"{prefix}/selected.json", selected)
        np.save(run.artifact_path(f"{prefix}/partitions.npy"), partitions, allow_pickle=False)
        save_npz(run.artifact_path(f"{prefix}/connectivities.npz"), a.obsp["connectivities"])
        save_npz(run.artifact_path(f"{prefix}/distances.npz"), a.obsp["distances"])
        if config.neighbor_backend == "exact_stable_id":
            np.save(run.artifact_path(f"{prefix}/knn_indices.npy"), knn_ids, allow_pickle=False)
            np.save(run.artifact_path(f"{prefix}/knn_distances.npy"), knn_distances, allow_pickle=False)
    return {"grid": rows, "selected": selected, "partitions": partitions,
            "graph": graph_info, "timing": timing}


def metric_records(values, dataset, config, *, grid, run=None, prefix="evaluation"):
    """Geometry uses full coordinates; graph truncation is never inherited silently.

    ASW sampling, if needed, defines a SUBSAMPLE statistic, not a full-data score.
    The sampled IDs and missing groups are saved. No NxN matrix is retained.
    """
    x = matrix(values)
    y, interpretation = dataset.reference_partition()
    y = labels(y, len(x))
    batches = dataset.batch_labels()
    b = labels(batches, len(x))
    batch_applicable = bool(dataset.record["registry"]["batch_evaluation"]) and len(np.unique(b)) > 1
    take = np.sort(np.random.default_rng(config.sampling_seed).choice(
        len(x), min(len(x), config.silhouette_max_cells), replace=False))
    sampled = len(take) != len(x)
    scope = "subsample" if sampled else "full"
    sampling = {"scope": scope, "max_cells": config.silhouette_max_cells, "seed": config.sampling_seed,
                "indices": take.tolist(), "cell_ids": [dataset.cell_ids[i] for i in take],
                "full_reference_groups": int(len(np.unique(y))), "sample_reference_groups": int(len(np.unique(y[take]))),
                "warning": "Subsample-defined ASW and isolation; not full-population estimates" if sampled else None}
    sampling["anonymous_group_coverage"] = [{"group_code": int(c), "full_cells": int(np.sum(y == c)),
        "sample_cells": int(np.sum(y[take] == c)), "full_batches": int(len(np.unique(b[y == c]))),
        "sample_batches": int(len(np.unique(b[take][y[take] == c])))} for c in np.unique(y)]
    if run:
        run.write_json(f"{prefix}/asw_sampling.json", sampling)
    rows = []

    def record(name, function, *, batch=False, context=None):
        start = time.perf_counter()
        row = {"metric": name, "value": None, "status": "not_run", "dimensions_used": x.shape[1],
               "n_cells_full": len(x),
               "reference_interpretation": interpretation, "selection_label_informed": config.label_informed,
               **(context or {})}
        if batch and not batch_applicable:
            row.update(status="not_applicable", reason="Single-batch or registry-disallowed batch evaluation")
        else:
            try:
                value, extra, per_cell = function()
                row.update(extra)
                row.update(value=float(value) if value is not None else None,
                           status="ok" if value is not None else "undefined")
                if row["value"] is not None and not np.isfinite(row["value"]):
                    raise ValueError("Nonfinite metric")
                if run and per_cell is not None:
                    np.save(run.artifact_path(f"{prefix}/metric_{len(rows):03d}.npy"), per_cell, allow_pickle=False)
                    row["per_cell_path"] = f"{prefix}/metric_{len(rows):03d}.npy"
                    row["per_cell_order"] = "asw_sampling.json" if row.get("scope") in {"subsample", "full"} else "cell_ids.json"
            except ValueError as error:
                row.update(value=None, status="undefined", reason=str(error))
        row["seconds"] = time.perf_counter()-start
        rows.append(row)

    def db():
        per, effective = d_batch(x, b, k=90, working_memory_mb=config.working_memory_mb)
        return per.mean(), {"k_requested": 90, "k_effective": effective, "includes_self": True,
                            "distance": "euclidean", "weighting": "uniform", "aggregation": "mean"}, per

    def li():
        score, per = ilisi(x, b, k=config.lisi_k, perplexity=config.lisi_perplexity,
                          metric=config.metric, working_memory_mb=config.working_memory_mb)
        return score, {"implementation": "scib-metrics==0.5.8", "k": config.lisi_k,
                       "perplexity": config.lisi_perplexity, "includes_self": False,
                       "distance": config.metric, "aggregation": "median_then_batch_count_scale",
                       "geometry": "embedding_kNN_not_graph_geodesic"}, per

    def ref_asw():
        per = asw(x[take], y[take], working_memory_mb=config.working_memory_mb)
        return per.mean(), {"scope": scope, "distance": "euclidean", "scale": "raw_minus1_to1"}, per

    def iso():
        score, classes, per = isolated_asw(x[take], y[take], b[take], threshold=config.isolated_batch_threshold,
                                          working_memory_mb=config.working_memory_mb)
        return score, {"scope": scope, "threshold": config.isolated_batch_threshold,
                       "isolated_group_count": len(classes), "scale": "(ASW+1)/2",
                       "singleton_isolated_group_count": sum(int(np.sum(labels(y[take], len(take)) == c) == 1) for c in classes),
                       "coverage_note": "See anonymous_group_coverage; singleton ASW=0 becomes 0.5 after rescaling. Not a rare-cell validation result.",
                       "definition": "scib-metrics_0.5.8_multiclass_ASW_equal_group_mean; NOT F1",
                       "reason": None if score is not None else "No isolated labels at chosen threshold"}, per

    if "D_batch" in config.metrics:
        record("D_batch_fixed90_including_self", db, batch=True)
    if "iLISI_scib_metrics" in config.metrics:
        record("iLISI_scib_metrics", li, batch=True)
    if "reference_ASW" in config.metrics:
        record("reference_ASW_"+scope, ref_asw)
    if "isolated_label_ASW" in config.metrics:
        record("isolated_label_ASW_"+scope, iso, batch=True)
    if "reference_knn_purity" in config.metrics:
        def local_purity():
            idx, _ = neighbors(x, config.geometry_k, metric=config.metric, working_memory_mb=config.working_memory_mb)
            per = purity(idx, y)
            return per.mean(), {"k": config.geometry_k, "distance": config.metric, "includes_self": False}, per
        record("reference_knn_purity", local_purity)
    if "predicted_cluster_ASW" in config.metrics:
        if not grid["selected"]:
            rows.append({"metric": "predicted_cluster_ASW_"+scope, "value": None, "status": "not_run",
                         "selection_label_informed": config.label_informed,
                         "reason": "grid_only: no partition selected; does not silently choose best ARI"})
        for selected in grid["selected"]:
            def pred_asw(selected=selected):
                pred = grid["partitions"][selected["partition_index"]]
                per = asw(x[take], pred[take], working_memory_mb=config.working_memory_mb)
                return per.mean(), {"scope": scope, "distance": "euclidean", "scale": "raw_minus1_to1"}, per
            record("predicted_cluster_ASW_"+scope, pred_asw, context={"leiden_seed": selected["leiden_seed"],
                   "resolution": selected["resolution"], "partition_index": selected["partition_index"]})
    if run:
        run.write_json(f"{prefix}/metrics.json", rows)
    return rows
