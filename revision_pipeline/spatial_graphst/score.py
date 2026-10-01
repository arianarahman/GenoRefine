# Purpose: Score GraphST Package 4b with the frozen common evaluator and native endpoints.
# Author: Ariana Rahman (Arizona State University)

"""Score GraphST Package 4b with the frozen common evaluator and native endpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits

from ..data.readers import array_hash
from ..evaluate.engine import graph_and_grid, metric_records
from ..evaluate.metrics import d_batch, ilisi
from ..integrity import canonical_hash, file_fingerprint
from ..runs import RunDirectory
from ..spatial_multisection.common import section_dataset
from ..spatial_multisection.score import spatial_metrics
from .common import (
    RUNS, evaluation_config, load_alignment, load_k_selection, load_metadata, read_json,
    require_run, source_snapshot, specification, validate_evaluation_runtime,
)


def _metric_values(rows: list[dict], prefix: str) -> list[float]:
    return [float(row["value"]) for row in rows
            if row.get("status") == "ok" and row.get("metric", "").startswith(prefix)]


def load_training(training: Path, alignment_path: Path, k_selection_path: Path,
                  donor: str, seed: int) -> dict:
    """Load one completed GraphST fit and verify its source and runtime bindings."""
    spec = specification()
    training = Path(training)
    require_run(training, "spatial_graphst_training")
    alignment = load_alignment(alignment_path, donor)
    decisions = load_k_selection(k_selection_path)
    decision = decisions["donors"][donor]
    config = read_json(training / "config.json")
    record = read_json(training / "training_record.json")
    if (config.get("protocol_id") != spec["protocol_id"]
            or config.get("donor") != donor
            or config.get("sections") != spec["donors"][donor]
            or config.get("seed") != seed
            or config.get("alignment_manifest") != alignment["manifest"]
            or config.get("k_selection_manifest") != file_fingerprint(Path(k_selection_path) / "run.json")
            or config.get("K_binding") != decision
            or config.get("settings") != spec["graphst"]
            or config.get("reference_labels_loaded") is not False
            or config.get("reference_labels_used") is not False):
        raise ValueError("GraphST training is not bound to the requested frozen parents")
    with np.load(training / "outputs.npz", allow_pickle=False) as saved:
        wanted = {
            "ids", "sections", "official_emb", "common_embedding", "native_mclust", "native_refined",
            "aligned_coordinates", "hvg_mask",
        }
        if set(saved.files) != wanted:
            raise ValueError("GraphST training output schema changed")
        output = {name: np.asarray(saved[name]) for name in wanted}
    ids = output["ids"].astype(str)
    sections = output["sections"].astype(str)
    official_embedding = np.asarray(output["official_emb"], dtype=np.float32)
    common_embedding = np.asarray(output["common_embedding"], dtype=np.float32)
    raw = np.asarray(output["native_mclust"], dtype=np.int64)
    refined = np.asarray(output["native_refined"], dtype=np.int64)
    coordinates = np.asarray(output["aligned_coordinates"], dtype=np.float64)
    hvg = np.asarray(output["hvg_mask"], dtype=bool)
    if (ids.tolist() != alignment["ids"].tolist()
            or sections.tolist() != alignment["sections"].tolist()
            or official_embedding.shape != (
                len(ids), int(spec["graphst"]["continuous_representation"]["official_train_output_dimensions"])
            )
            or common_embedding.shape != (
                len(ids), int(spec["graphst"]["continuous_representation"]["common_evaluator_dimensions"])
            )
            or raw.shape != (len(ids),) or refined.shape != (len(ids),)
            or coordinates.shape != (len(ids), 2)
            or not np.array_equal(coordinates, alignment["aligned_coordinates"])
            or not np.isfinite(official_embedding).all() or not np.isfinite(common_embedding).all()
            or len(np.unique(raw)) != int(decision["n_clusters"])
            or int(hvg.sum()) != int(spec["graphst"]["preprocessing"]["n_top_genes"])):
        raise ValueError("GraphST training output violates the frozen representation/K contract")
    if (record.get("cell_ids_sha256") != canonical_hash(ids.tolist())
            or record.get("official_emb_sha256") != array_hash(official_embedding)
            or record.get("common_embedding_sha256") != array_hash(common_embedding)
            or record.get("native_mclust_sha256") != array_hash(raw)
            or record.get("native_refined_sha256") != array_hash(refined)
            or record.get("aligned_coordinates_sha256") != array_hash(coordinates)
            or record.get("hvg_mask_sha256") != array_hash(hvg)
            or record.get("reference_labels_loaded") is not False
            or record.get("reference_labels_used") is not False):
        raise ValueError("GraphST training receipt differs from its outputs")
    return {
        "ids": ids, "sections": sections, "official_emb": official_embedding,
        "common_embedding": common_embedding,
        "native_mclust": raw, "native_refined": refined,
        "aligned_coordinates": coordinates, "record": record,
        "manifest": file_fingerprint(training / "run.json"),
        "alignment_manifest": alignment["manifest"],
        "k_selection_manifest": file_fingerprint(Path(k_selection_path) / "run.json"),
    }


def score_section(section: str, donor: str, seed: int, training: Path,
                  alignment: Path, k_selection: Path, run_id: str) -> Path:
    """Score common and task-native GraphST outputs for one tissue section."""
    spec = specification()
    if donor not in spec["donors"] or section not in spec["donors"][donor]:
        raise ValueError("Section does not belong to the requested donor")
    trained = load_training(training, alignment, k_selection, donor, seed)
    take = np.flatnonzero(trained["sections"] == section)
    values = trained["common_embedding"][take]
    ids = trained["ids"][take].tolist()
    metadata = load_metadata(include_labels=True)
    lookup = metadata.set_index("cell_id", drop=False)
    frame = lookup.loc[ids].reset_index(drop=True)
    if frame["cell_id"].astype(str).tolist() != ids:
        raise ValueError("Evaluation label registry order differs from GraphST output")
    dataset = section_dataset(frame)
    config = evaluation_config()
    runtime = validate_evaluation_runtime()
    sources = source_snapshot()
    provenance = {
        "method": "graphst",
        "training_manifest": trained["manifest"],
        "alignment_manifest": trained["alignment_manifest"],
        "k_selection_manifest": trained["k_selection_manifest"],
        "donor": donor,
        "section": section,
        "algorithmic_seed": seed,
        "continuous_endpoint": "official GraphST clustering PCA20 representation (emb_pca)",
        "native_endpoint_role": "secondary task-native mclust/refined partitions",
    }
    context = {
        "protocol_id": spec["protocol_id"], "scope": "within_section",
        "section": section, "donor": donor, "method": "graphst",
        "algorithmic_seed": seed, "input": provenance,
        "parent_runs": {
            "training": Path(training).resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix(),
            "alignment": Path(alignment).resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix(),
            "k_selection": Path(k_selection).resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix(),
        },
        "evaluation": config.to_dict(), "runtime": runtime,
        "label_policy": spec["label_policy"],
    }
    with RunDirectory(RUNS, kind="spatial_graphst_section_score", run_id=run_id, config=context) as run:
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            grid = graph_and_grid(
                values, dataset.reference, dataset.cell_ids, config, run=run,
                prefix="evaluation", training_label_use="none; evaluation labels only",
            )
            metrics = metric_records(values, dataset, config, grid=grid, run=run, prefix="evaluation")
            spatial, arrays = spatial_metrics(
                values,
                frame[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=np.float64),
                int(spec["evaluation"]["spatial_neighbor_k"]), config.working_memory_mb,
            )
        selected = grid["selected"]
        if len(selected) != 3 or {row["leiden_seed"] for row in selected} != {0, 1, 2}:
            raise ValueError("Incomplete fixed-resolution common-evaluator partitions")
        ari = [float(row["ARI"]) for row in selected]
        nmi = [float(normalized_mutual_info_score(
            dataset.reference, grid["partitions"][row["partition_index"]],
        )) for row in selected]
        cluster_sil = _metric_values(metrics, "predicted_cluster_ASW_")
        reference_sil = _metric_values(metrics, "reference_ASW_")
        label_purity = _metric_values(metrics, "reference_knn_purity")
        if len(cluster_sil) != 3 or len(reference_sil) != 1 or len(label_purity) != 1:
            raise ValueError("Incomplete common-evaluator section endpoints")
        native = {}
        for name, partition in (
            ("mclust", trained["native_mclust"][take]),
            ("refined", trained["native_refined"][take]),
        ):
            native[name] = {
                "ARI": float(adjusted_rand_score(dataset.reference, partition)),
                "NMI": float(normalized_mutual_info_score(dataset.reference, partition)),
                "clusters": int(len(np.unique(partition))),
                "role": "secondary task-native GraphST partition",
            }
        summary = {
            "section": section, "donor": donor, "method": "graphst",
            "algorithmic_seed": seed, "n_spots": len(values),
            "fixed_resolution": 0.5, "leiden_seeds": [0, 1, 2],
            "clusters": [int(row["n_clusters"]) for row in selected],
            "ARI": float(statistics.mean(ari)), "ARI_by_leiden_seed": ari,
            "NMI": float(statistics.mean(nmi)), "NMI_by_leiden_seed": nmi,
            "predicted_cluster_silhouette": float(statistics.mean(cluster_sil)),
            "predicted_cluster_silhouette_by_leiden_seed": cluster_sil,
            "reference_label_silhouette": reference_sil[0],
            "label_neighbor_purity": label_purity[0],
            **spatial,
            "native_graphst_partitions": native,
            "label_policy": spec["label_policy"],
            "wall_seconds": time.perf_counter() - started,
        }
        for name, array in arrays.items():
            np.save(run.artifact_path(f"spatial/{name}.npy"), array, allow_pickle=False)
        run.write_json("summary.json", summary)
        run.write_json("input.json", provenance)
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.manifest.update(
            scientific_experiment=True, experiment_role="graphst_common_and_native_section_scoring",
            training_performed=False, scoring_performed=True,
        )
        if source_snapshot() != sources:
            raise RuntimeError("Scientific source changed during GraphST section scoring")
    return run.final_path


def score_donor(donor: str, seed: int, training: Path, alignment: Path,
                k_selection: Path, run_id: str) -> Path:
    """Score the aligned GraphST representation for one donor pair."""
    spec = specification()
    if donor not in spec["donors"]:
        raise ValueError("Unplanned donor")
    trained = load_training(training, alignment, k_selection, donor, seed)
    batches = trained["sections"]
    if set(batches) != set(spec["donors"][donor]):
        raise ValueError("Donor-pair output section coverage changed")
    config = evaluation_config()
    runtime = validate_evaluation_runtime()
    sources = source_snapshot()
    provenance = {
        "method": "graphst", "training_manifest": trained["manifest"],
        "alignment_manifest": trained["alignment_manifest"],
        "k_selection_manifest": trained["k_selection_manifest"],
        "donor": donor, "algorithmic_seed": seed,
    }
    context = {
        "protocol_id": spec["protocol_id"], "scope": "donor_pair", "donor": donor,
        "method": "graphst", "algorithmic_seed": seed, "input": provenance,
        "parent_runs": {
            "training": Path(training).resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix(),
            "alignment": Path(alignment).resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix(),
            "k_selection": Path(k_selection).resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix(),
        },
        "iLISI_k": config.lisi_k, "iLISI_perplexity": config.lisi_perplexity,
        "D_batch_k": 90, "runtime": runtime,
    }
    with RunDirectory(RUNS, kind="spatial_graphst_donor_score", run_id=run_id, config=context) as run:
        with threadpool_limits(limits=1):
            i_score, i_per = ilisi(
                trained["common_embedding"], batches, k=config.lisi_k,
                perplexity=config.lisi_perplexity, metric="euclidean",
                working_memory_mb=config.working_memory_mb,
            )
            d_per, d_k = d_batch(
                trained["common_embedding"], batches, k=90,
                working_memory_mb=config.working_memory_mb,
            )
        np.save(run.artifact_path("ilisi_per_spot.npy"), i_per, allow_pickle=False)
        np.save(run.artifact_path("d_batch_per_spot.npy"), d_per, allow_pickle=False)
        run.write_json("summary.json", {
            "donor": donor, "sections": spec["donors"][donor], "method": "graphst",
            "algorithmic_seed": seed, "n_spots": len(trained["common_embedding"]),
            "iLISI": float(i_score),
            "iLISI_definition": "scib-metrics 0.5.8 embedding-kNN, scaled",
            "D_batch": float(np.mean(d_per)),
            "D_batch_effective_k_including_self": int(d_k),
            "interpretation_boundary": "section mixing within one donor; anatomical layer composition may differ between sections",
        })
        run.write_json("input.json", provenance)
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.manifest.update(
            scientific_experiment=True, experiment_role="graphst_donor_pair_section_mixing_scoring",
            training_performed=False, scoring_performed=True,
        )
        if source_snapshot() != sources:
            raise RuntimeError("Scientific source changed during GraphST donor scoring")
    return run.final_path


def main() -> None:
    """Parse a section or donor GraphST scoring request and dispatch it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["section", "donor"])
    parser.add_argument("--section")
    parser.add_argument("--donor", choices=sorted(specification()["donors"]), required=True)
    parser.add_argument("--seed", type=int, choices=range(5), required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--k-selection", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required")
    if args.action == "section":
        if args.section is None:
            parser.error("--section is required")
        result = score_section(
            args.section, args.donor, args.seed, args.training,
            args.alignment, args.k_selection, args.run_id,
        )
    else:
        if args.section is not None:
            parser.error("--section is not valid for donor scoring")
        result = score_donor(
            args.donor, args.seed, args.training,
            args.alignment, args.k_selection, args.run_id,
        )
    print(result, flush=True)


if __name__ == "__main__":
    main()
