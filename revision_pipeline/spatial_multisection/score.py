"""Score section geometry/spatial preservation or donor-pair mixing."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits

from ..evaluate.engine import graph_and_grid, metric_records
from ..evaluate.inputs import load_refined_bundle
from ..evaluate.metrics import d_batch, ilisi, neighbors, overlap, purity
from ..integrity import canonical_hash, file_fingerprint
from ..runs import RunDirectory
from .common import (
    ROOT, RUNS, evaluation_config, foundation_path, load_fixed_harmony, load_metadata,
    load_k_selection, read_json, require_run, section_dataset, source_snapshot, specification,
    validate_evaluation_runtime,
)


def _metric_values(rows: list[dict], prefix: str) -> list[float]:
    return [float(row["value"]) for row in rows
            if row.get("status") == "ok" and row.get("metric", "").startswith(prefix)]


def validate_training_binding(
    config: dict,
    *,
    method: str,
    harmony_manifest: dict,
    k_selection_manifest: dict,
    decisions: dict,
    section: str | None = None,
) -> None:
    expected_k = decisions["pooled"] if method == "genorefine" else decisions["sections"][section]
    if (config.get("harmony_manifest") != harmony_manifest
            or config.get("k_selection_manifest") != k_selection_manifest
            or config.get("K_binding") != expected_k
            or (method == "spagcn" and config.get("section") != section)):
        raise ValueError(f"{method} training is bound to another section/Harmony/K-selection run")


def _load_representation(
    method: str,
    *,
    harmony: Path,
    k_selection: Path,
    training: Path | None,
    section: str | None = None,
) -> tuple[np.ndarray, object, dict, dict | None]:
    spec = specification()
    metadata = load_metadata()
    decisions = load_k_selection(k_selection, harmony)
    harmony_manifest = file_fingerprint(Path(harmony) / "run.json")
    k_selection_manifest = file_fingerprint(Path(k_selection) / "run.json")
    native = None
    if method == "harmony_fixed":
        values, _, record = load_fixed_harmony(harmony)
        provenance = {"method": method, "harmony_manifest": harmony_manifest,
                      "k_selection_manifest": k_selection_manifest,
                      "embedding_sha256": record["embedding_sha256"], "algorithmic_seed": None}
    elif method == "harmony_native_sensitivity":
        values = np.load(foundation_path() / "harmony50.npy", allow_pickle=False)
        provenance = {"method": method, "role": "native-stop sensitivity only",
                      "foundation_manifest": file_fingerprint(foundation_path() / "run.json"),
                      "harmony_manifest": harmony_manifest,
                      "k_selection_manifest": k_selection_manifest,
                      "algorithmic_seed": None}
    elif method == "genorefine":
        if training is None:
            raise ValueError("GenoRefine scoring requires a training run")
        require_run(training, "spatial_multisection_genorefine_training")
        cfg = read_json(Path(training) / "config.json")
        validate_training_binding(
            cfg, method=method, harmony_manifest=harmony_manifest,
            k_selection_manifest=k_selection_manifest, decisions=decisions,
        )
        parent = read_json(Path(training) / "input.json")["parent_reference"]
        bundle = load_refined_bundle(
            Path(training) / "bundles/joint", expected_parent=parent,
            output_cell_ids=tuple(metadata["cell_id"].astype(str)),
        )
        values = bundle.values
        provenance = {"method": method, "training_manifest": file_fingerprint(Path(training) / "run.json"),
                      "algorithmic_seed": int(cfg["seed"]), "stage": "joint",
                      "bundle_manifest": bundle.metadata["bundle_manifest"]}
    elif method == "spagcn":
        if training is None or section is None:
            raise ValueError("SpaGCN scoring requires one section training run")
        require_run(training, "spatial_multisection_spagcn_training")
        cfg = read_json(Path(training) / "config.json")
        validate_training_binding(
            cfg, method=method, harmony_manifest=harmony_manifest,
            k_selection_manifest=k_selection_manifest, decisions=decisions, section=section,
        )
        with np.load(Path(training) / "outputs.npz", allow_pickle=False) as saved:
            ids = [str(x) for x in saved["ids"]]
            take = np.flatnonzero(metadata["section"].astype(str).to_numpy() == section)
            expected = metadata.iloc[take]["cell_id"].astype(str).tolist()
            if ids != expected:
                raise ValueError("SpaGCN output ID/order mismatch")
            values = np.asarray(saved["latent"])
            native = {"predicted": np.asarray(saved["predicted"]), "refined": np.asarray(saved["refined"])}
        provenance = {"method": method, "training_manifest": file_fingerprint(Path(training) / "run.json"),
                      "algorithmic_seed": int(cfg["seed"]), "section": section,
                      "native_prediction_role": "secondary"}
        return values, metadata.iloc[take].reset_index(drop=True), provenance, native
    else:
        raise ValueError("Unknown spatial-panel method")
    if values.shape[0] != len(metadata) or not np.isfinite(values).all():
        raise ValueError("Pooled representation is invalid")
    if section is not None:
        take = np.flatnonzero(metadata["section"].astype(str).to_numpy() == section)
        return np.asarray(values[take]), metadata.iloc[take].reset_index(drop=True), provenance, native
    return np.asarray(values), metadata, provenance, native


def spatial_metrics(values: np.ndarray, coordinates: np.ndarray, k: int, working_memory_mb: int) -> tuple[dict, dict]:
    latent_idx, latent_distance = neighbors(values, k, working_memory_mb=working_memory_mb)
    spatial_idx, spatial_distance = neighbors(coordinates, k, working_memory_mb=working_memory_mb)
    jaccard = overlap(spatial_idx, latent_idx)
    physical = np.linalg.norm(coordinates[latent_idx] - coordinates[:, None, :], axis=2)
    per_query_physical = physical.mean(axis=1)
    local_scale = spatial_distance.mean(axis=1)
    if np.any(local_scale <= 0):
        raise ValueError("Degenerate physical coordinates")
    ratio = per_query_physical / local_scale
    summary = {
        "k": k,
        "self_excluded_by_index": True,
        "spatial_latent_knn_jaccard": float(jaccard.mean()),
        "spatial_latent_knn_jaccard_median": float(np.median(jaccard)),
        "physical_distance_among_latent_neighbors_mean_fullres_pixels": float(per_query_physical.mean()),
        "physical_distance_among_latent_neighbors_median_fullres_pixels": float(np.median(per_query_physical)),
        "physical_distance_ratio_to_local_spatial_knn_mean": float(ratio.mean()),
    }
    arrays = {"latent_neighbors": latent_idx, "spatial_neighbors": spatial_idx,
              "jaccard_per_spot": jaccard, "physical_distance_per_spot": per_query_physical,
              "physical_distance_ratio_per_spot": ratio, "latent_neighbor_distance": latent_distance}
    return summary, arrays


def score_section(method: str, section: str, harmony: Path, k_selection: Path, training: Path | None, run_id: str) -> Path:
    spec = specification()
    if section not in spec["sections"]:
        raise ValueError("Unplanned DLPFC section")
    load_k_selection(k_selection, harmony)
    values, metadata, provenance, native = _load_representation(
        method, harmony=harmony, k_selection=k_selection, training=training, section=section,
    )
    dataset = section_dataset(metadata)
    config = evaluation_config()
    runtime = validate_evaluation_runtime()
    sources = source_snapshot()
    context = {"protocol_id": spec["protocol_id"], "scope": "within_section", "section": section,
               "method": method, "input": provenance,
               "harmony_manifest": file_fingerprint(Path(harmony) / "run.json"),
               "k_selection_manifest": file_fingerprint(Path(k_selection) / "run.json"),
               "evaluation": config.to_dict(), "runtime": runtime}
    with RunDirectory(RUNS, kind="spatial_multisection_section_score", run_id=run_id, config=context) as run:
        start = time.perf_counter()
        with threadpool_limits(limits=1):
            grid = graph_and_grid(values, dataset.reference, dataset.cell_ids, config, run=run,
                                  prefix="evaluation", training_label_use="none; evaluation labels only")
            metrics = metric_records(values, dataset, config, grid=grid, run=run, prefix="evaluation")
            spatial, arrays = spatial_metrics(
                values,
                metadata[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=np.float64),
                spec["evaluation"]["spatial_neighbor_k"], config.working_memory_mb,
            )
        selected = grid["selected"]
        if len(selected) != 3 or {row["leiden_seed"] for row in selected} != {0, 1, 2}:
            raise ValueError("Incomplete fixed-resolution section partitions")
        nmi = [float(normalized_mutual_info_score(dataset.reference, grid["partitions"][row["partition_index"]]))
               for row in selected]
        ari = [float(row["ARI"]) for row in selected]
        cluster_sil = _metric_values(metrics, "predicted_cluster_ASW_")
        reference_sil = _metric_values(metrics, "reference_ASW_")
        label_purity = _metric_values(metrics, "reference_knn_purity")
        if len(cluster_sil) != 3 or len(reference_sil) != 1 or len(label_purity) != 1:
            raise ValueError("Incomplete section evaluation endpoints")
        native_summary = None
        if native is not None:
            native_summary = {}
            for name, pred in native.items():
                native_summary[name] = {
                    "ARI": float(adjusted_rand_score(dataset.reference, pred)),
                    "NMI": float(normalized_mutual_info_score(dataset.reference, pred)),
                    "clusters": int(len(np.unique(pred))),
                    "role": "secondary task-native SpaGCN partition",
                }
        summary = {
            "section": section,
            "donor": str(metadata["donor"].iloc[0]),
            "method": method,
            "algorithmic_seed": provenance["algorithmic_seed"],
            "n_spots": len(values),
            "fixed_resolution": 0.5,
            "leiden_seeds": [0, 1, 2],
            "clusters": [int(row["n_clusters"]) for row in selected],
            "ARI": float(statistics.mean(ari)),
            "ARI_by_leiden_seed": ari,
            "NMI": float(statistics.mean(nmi)),
            "NMI_by_leiden_seed": nmi,
            "predicted_cluster_silhouette": float(statistics.mean(cluster_sil)),
            "predicted_cluster_silhouette_by_leiden_seed": cluster_sil,
            "reference_label_silhouette": reference_sil[0],
            "label_neighbor_purity": label_purity[0],
            **spatial,
            "native_spagcn_partitions": native_summary,
            "label_policy": spec["label_policy"],
            "wall_seconds": time.perf_counter() - start,
        }
        for name, array in arrays.items():
            np.save(run.artifact_path(f"spatial/{name}.npy"), array, allow_pickle=False)
        run.write_json("summary.json", summary)
        run.write_json("input.json", provenance)
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        if source_snapshot() != sources:
            raise RuntimeError("Scientific source changed during section scoring")
        run.manifest.update(scientific_experiment=True, experiment_role="multi_section_spatial_scoring",
                            training_performed=False, scoring_performed=True)
    return run.final_path


def score_donor(method: str, donor: str, harmony: Path, k_selection: Path, training: Path | None, run_id: str) -> Path:
    spec = specification()
    if donor not in spec["donors"] or method not in {"harmony_fixed", "harmony_native_sensitivity", "genorefine"}:
        raise ValueError("Unplanned donor-pair score")
    load_k_selection(k_selection, harmony)
    values, metadata, provenance, _ = _load_representation(
        method, harmony=harmony, k_selection=k_selection, training=training,
    )
    take = np.flatnonzero(metadata["donor"].astype(str).to_numpy() == donor)
    x = values[take]
    batches = metadata.iloc[take]["section"].astype(str).to_numpy(dtype="U")
    if set(batches) != set(spec["donors"][donor]):
        raise ValueError("Donor pair section coverage changed")
    config = evaluation_config()
    runtime = validate_evaluation_runtime()
    sources = source_snapshot()
    context = {"protocol_id": spec["protocol_id"], "scope": "donor_pair", "donor": donor,
               "method": method, "input": provenance,
               "harmony_manifest": file_fingerprint(Path(harmony) / "run.json"),
               "k_selection_manifest": file_fingerprint(Path(k_selection) / "run.json"),
               "iLISI_k": config.lisi_k,
               "iLISI_perplexity": config.lisi_perplexity, "D_batch_k": 90,
               "runtime": runtime}
    with RunDirectory(RUNS, kind="spatial_multisection_donor_score", run_id=run_id, config=context) as run:
        with threadpool_limits(limits=1):
            i_score, i_per = ilisi(x, batches, k=config.lisi_k, perplexity=config.lisi_perplexity,
                                   metric="euclidean", working_memory_mb=config.working_memory_mb)
            d_per, d_k = d_batch(x, batches, k=90, working_memory_mb=config.working_memory_mb)
        np.save(run.artifact_path("ilisi_per_spot.npy"), i_per, allow_pickle=False)
        np.save(run.artifact_path("d_batch_per_spot.npy"), d_per, allow_pickle=False)
        run.write_json("summary.json", {
            "donor": donor, "sections": spec["donors"][donor], "method": method,
            "algorithmic_seed": provenance["algorithmic_seed"], "n_spots": len(x),
            "iLISI": float(i_score), "iLISI_definition": "scib-metrics 0.5.8 embedding-kNN, scaled",
            "D_batch": float(np.mean(d_per)), "D_batch_effective_k_including_self": int(d_k),
            "interpretation_boundary": "section mixing within one donor; anatomical layer composition may differ between sections",
        })
        run.write_json("input.json", provenance)
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        if source_snapshot() != sources:
            raise RuntimeError("Scientific source changed during donor scoring")
        run.manifest.update(scientific_experiment=True, experiment_role="donor_pair_section_mixing_scoring",
                            training_performed=False, scoring_performed=True)
    return run.final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["section", "donor"])
    parser.add_argument("--method", choices=["harmony_fixed", "harmony_native_sensitivity", "genorefine", "spagcn"], required=True)
    parser.add_argument("--section", choices=specification()["sections"])
    parser.add_argument("--donor", choices=sorted(specification()["donors"]))
    parser.add_argument("--harmony", type=Path, required=True)
    parser.add_argument("--k-selection", type=Path, required=True)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required")
    if args.action == "section":
        if args.section is None:
            parser.error("--section is required for section scoring")
        result = score_section(args.method, args.section, args.harmony, args.k_selection, args.training, args.run_id)
    else:
        if args.donor is None:
            parser.error("--donor is required for donor scoring")
        result = score_donor(args.method, args.donor, args.harmony, args.k_selection, args.training, args.run_id)
    print(result, flush=True)


if __name__ == "__main__":
    main()
