"""Immutable result run orchestration; neither integrates nor trains."""

import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from ..audit import source_paths
from ..data.store import Store, read_json, runtime_inventory
from ..integrity import canonical_hash, file_fingerprint
from ..runs import RunDirectory
from .engine import graph_and_grid, metric_records
from .cache import reuse_baseline
from .contrasts import anchor_counts, paired_clustering
from .inputs import load_refined_bundle
from .metrics import neighbors, overlap, purity


def runtime():
    record = runtime_inventory()
    record.update(scanpy_or_leiden_required=True, hardware={"machine": platform.machine(),
                 "processor": platform.processor(), "logical_cpu_count": os.cpu_count()},
                 threadpools=threadpool_info(), thread_limits_requested=1,
                 process_environment={key: os.environ.get(key) for key in (
                     "PYTHONHASHSEED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMBA_NUM_THREADS", "JAX_PLATFORMS")})
    return record


def snapshot(root):
    return {p.relative_to(root).as_posix(): file_fingerprint(p) for p in source_paths(root)}


def require_primary_convergence(dataset_id, name, metadata, config):
    if config.purpose == "primary_evaluation" and dataset_id == "pancreas_five_study" and name == "Harmony":
        convergence = metadata.get("primary_convergence", {})
        policy_id = convergence.get("policy_id")
        valid = (
            policy_id in {
                    "pancreas_harmony_convergence_v1",
                    "pancreas_harmony_convergence_v2",
                }
            and convergence.get("status") == "passed"
        ) or (
            policy_id == "pancreas_harmony_fixed10_descriptive_v3"
            and convergence.get("status") == "fixed_budget_completed"
            and convergence.get("numerical_convergence_claimed") is False
        )
        if not valid:
            raise ValueError("Primary pancreas Harmony requires a NEW convergence-verified baseline; retain the 10-iteration result only for historical/development work")


def paired_geometry(before, after, cell_ids, reference, config, *, run, prefix="pair", verified=False):
    if before.shape[0] != after.shape[0] or before.shape[0] != len(cell_ids):
        raise ValueError("Paired geometry requires complete ID-aligned populations")
    start = time.perf_counter()
    a, _ = neighbors(before, config.geometry_k, metric=config.metric, working_memory_mb=config.working_memory_mb)
    b, _ = neighbors(after, config.geometry_k, metric=config.metric, working_memory_mb=config.working_memory_mb)
    stability, delta = overlap(a, b), purity(b, reference)-purity(a, reference)
    row = {"metric": "neighbor_Jaccard_and_purity_delta", "k": config.geometry_k, "distance": config.metric,
           "includes_self": False, "neighbor_selection": "identity_exclusion", "n_cells": len(cell_ids),
           "before_dimensions": before.shape[1], "after_dimensions": after.shape[1],
           "mean_Jaccard": float(stability.mean()), "mean_reference_purity_delta": float(delta.mean()),
           "pairing_status": "exact_parent_reference_verified" if verified else "historical_input_pairing_unverified",
           "interpretation": "Paired coordinate comparison; no biological improvement claim from overlap alone",
           "cell_order_sha256": canonical_hash(list(cell_ids)), "seconds": time.perf_counter()-start}
    run.write_json(f"{prefix}/summary.json", row)
    np.save(run.artifact_path(f"{prefix}/jaccard.npy"), stability, allow_pickle=False)
    np.save(run.artifact_path(f"{prefix}/purity_delta.npy"), delta, allow_pickle=False)
    return row


def evaluate_store(root, store_path, dataset_id, embedding_name, config, *, compare_to=None,
                   refined_bundle=None, parent_order="canonical", baseline_evaluation=None):
    root, store_path = Path(root).resolve(), Path(store_path).resolve()
    if compare_to and refined_bundle:
        raise ValueError("Choose historical comparison OR verified bundle")
    if parent_order not in {"canonical", "historical_source"}:
        raise ValueError("Unknown actual parent order")
    sources = snapshot(root)
    store = Store(store_path)
    store_hash = file_fingerprint(store_path / "run.json")
    if config.row_order == "historical_source":
        view = store.historical_input(dataset_id, embedding_name)
        primary, dataset = view.embedding, view.dataset
        reference = view.parent_reference()
    else:
        primary, dataset = store.embedding(dataset_id, embedding_name), store.dataset(dataset_id)
        reference = primary.parent_reference()
    items = [(embedding_name, primary, reference)]
    verified = False
    if refined_bundle:
        parent = (store.historical_input(dataset_id, embedding_name).parent_reference()
                  if parent_order == "historical_source" else store.embedding(dataset_id, embedding_name).parent_reference())
        refined = load_refined_bundle(refined_bundle, expected_parent=parent, output_cell_ids=primary.cell_ids)
        items.append(("refined_bundle", refined, refined.metadata))
        verified = True
    elif compare_to:
        if config.row_order != "canonical":
            raise ValueError("Historical pairs must be ID-aligned in canonical order for geometry")
        other = store.embedding(dataset_id, compare_to)
        items.append((compare_to, other, other.parent_reference()))
    if any(item[1].cell_ids != dataset.cell_ids for item in items):
        raise ValueError("Dataset/embedding IDs differ")
    for name, embedding, _ in items:
        require_primary_convergence(dataset_id, name, embedding.metadata, config)
    run_config = {"evaluation": config.to_dict(), "dataset": dataset_id,
                              "embeddings": [item[0] for item in items], "store_manifest": store_hash,
                              "parent_order": parent_order, "refined_bundle": str(refined_bundle) if refined_bundle else None,
                              "baseline_evaluation": str(baseline_evaluation) if baseline_evaluation else None}
    with RunDirectory(root / "revision_pipeline/runs", kind="step3b_evaluation", config=run_config) as run:
        run.write_json("source_manifest.json", sources)
        start_runtime = runtime()
        run.write_json("runtime_start.json", start_runtime)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.write_json("annotation_policy.json", dataset.annotation_policy)
        reference_labels, interpretation = dataset.reference_partition()
        summaries = []
        with threadpool_limits(limits=1):
            for index, (name, embedding, input_ref) in enumerate(items):
                print(f"Evaluating {dataset_id}/{name}: full grid {len(config.resolutions)} x {len(config.leiden_seeds)}", flush=True)
                prefix = f"embedding_{index}"
                input_record = {"name": name, "reference": input_ref,
                    "kind": embedding.metadata["kind"], "pairing_status": embedding.metadata.get("pairing_status"),
                    "training_label_use": embedding.metadata.get("training_label_use", "unknown_historical_provenance"),
                    "reference_interpretation": interpretation}
                if index == 0 and baseline_evaluation:
                    summaries.append(reuse_baseline(baseline_evaluation, run=run, expected_config=run_config,
                        expected_input=input_record, source_hash=canonical_hash(sources), current_runtime=start_runtime))
                    print("  Verified baseline results reused; no repeated baseline graph/grid", flush=True)
                    continue
                run.write_json(f"{prefix}/input.json", input_record)
                graph = graph_and_grid(embedding.values, reference_labels, embedding.cell_ids, config, run=run, prefix=prefix,
                                       training_label_use=embedding.metadata.get("training_label_use", "unknown_historical_provenance"))
                # Historical precision is an explicit profile conversion only.
                coords = embedding.values.astype(np.float32) if config.precision == "float32" else embedding.values
                start = time.perf_counter()
                metrics = metric_records(coords, dataset, config, grid=graph, run=run, prefix=prefix)
                summaries.append({"name": name, "grid_rows": len(graph["grid"]), "selected": graph["selected"],
                                  "cluster_counts_at_0_5": anchor_counts(graph["grid"], config) if .5 in config.resolutions else None,
                                  "metrics": metrics, "timing": {**graph["timing"], "geometry_seconds": time.perf_counter()-start}})
            pair = None
            clustering_comparison = None
            if len(items) == 2:
                pair = paired_geometry(items[0][1].values, items[1][1].values, dataset.cell_ids,
                                       reference_labels, config, run=run, verified=verified)
                if config.selection == "fixed_resolution" and config.fixed_resolution == .5:
                    clustering_comparison = paired_clustering(
                        read_json(run.path / "embedding_0/grid.json"), read_json(run.path / "embedding_1/grid.json"),
                        config, verified=verified)
                    run.write_json("pair/clustering_comparison.json", clustering_comparison)
        run.write_json("summary.json", {"purpose": config.purpose, "dataset": dataset_id,
                                      "reference_interpretation": interpretation, "evaluations": summaries, "pair": pair,
                                      "clustering_comparison": clustering_comparison})
        run.write_json("runtime_end.json", runtime())
        if snapshot(root) != sources or file_fingerprint(store_path / "run.json") != store_hash:
            raise RuntimeError("Sources or store manifest changed during evaluation")
        # ru_maxrss on Linux is KiB and is a whole-process high-water mark,
        # including imports; it is NOT per-stage incremental memory.
        run.manifest.update(peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform != "darwin" else 1),
                            peak_memory_status="process_lifetime_ru_maxrss_including_imports",
                            scientific_experiment=False)
    return run.final_path
