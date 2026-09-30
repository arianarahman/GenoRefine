"""Score one corrupted baseline or trained artifact-validation representation."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import time

import numpy as np
from threadpoolctl import threadpool_limits

from ..data.readers import array_hash
from ..data.store import Store
from ..evaluate.config import EvaluationConfig
from ..evaluate.engine import graph_and_grid, metric_records
from ..evaluate.inputs import load_refined_bundle
from ..evaluate.runner import runtime
from ..integrity import canonical_hash, file_fingerprint, project_path
from ..pilot.common import completed, read, snapshot
from ..runs import RunDirectory, write_json
from ..step4_policy import load_policy
from .common import ROOT, load_prepared, specification
from .metrics import (
    clean_neighbor_recovery,
    clean_neighbor_recovery_from_reference,
    counterfactual_batch_sensitivity,
    preservation_metrics_from_neighbors,
)
from .source_compatibility import (
    expected_observed_hash,
    verify_scoring_source_transition,
)


METHODS = ("corrupted_baseline", "genorefine", "idec")


def _relative(path: Path | str) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def evaluation_config() -> EvaluationConfig:
    policy = load_policy()
    return EvaluationConfig.from_dict(read(ROOT / policy["primary_evaluation_config"]))


def resolve_representation(prepared, artifact_id, training=None):
    artifact = prepared.artifact(artifact_id)
    if training is None:
        values = artifact.values
        clean_output = artifact.clean
        counterfactual_outputs = artifact.counterfactuals
        provenance = {
            "method": "corrupted_baseline",
            "replicate_seed": None,
            "training_label_use": "not_applicable_no_training",
            "parent_reference": artifact.parent_reference(),
        }
    else:
        training = Path(training).resolve()
        manifest = completed(training, "artifact_validation_v2_training")
        config = read(training / "config.json")
        if (
            config.get("prepared") != _relative(prepared.path)
            or config.get("prepared_manifest")
            != file_fingerprint(prepared.path / "run.json")
            or config.get("case_id") != prepared.case["id"]
            or config.get("artifact_id") != artifact_id
            or config.get("method") not in {"genorefine", "idec"}
            or config.get("replicate_seed") not in specification()["replicate_seeds"]
            or config.get("artifact_parent_reference") != artifact.parent_reference()
        ):
            raise ValueError("Training run does not match this prepared artifact")
        values = load_refined_bundle(
            training / "bundle",
            expected_parent=artifact.parent_reference(),
            output_cell_ids=artifact.cell_ids,
        ).values
        clean_output = np.load(
            training / "inference/clean.npy", mmap_mode="r", allow_pickle=False
        )
        counterfactual_outputs = np.load(
            training / "inference/counterfactuals.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        inference = read(training / "inference/checks.json")
        binding = config.get("K_binding")
        if (
            array_hash(clean_output) != inference["clean_sha256"]
            or array_hash(counterfactual_outputs)
            != inference["counterfactuals_sha256"]
            or expected_observed_hash(values)
            != inference["expected_observed_sha256"]
            or inference.get("observed_reloaded_sha256")
            != inference["expected_observed_sha256"]
            or counterfactual_outputs.shape
            != (len(artifact.batch_levels), len(artifact.cell_ids), 32)
            or not isinstance(binding, dict)
            or not isinstance(binding.get("baseline_score"), str)
            or binding.get("baseline_manifest") is None
        ):
            raise ValueError("Training counterfactual inference artifacts changed")
        provenance = {
            "method": config["method"],
            "replicate_seed": config["replicate_seed"],
            "training_label_use": "label_free",
            "parent_reference": artifact.parent_reference(),
            "training": _relative(training),
            "training_manifest": file_fingerprint(training / "run.json"),
            "training_source_tree_sha256": manifest.get("source_tree_sha256"),
            "baseline_score": binding["baseline_score"],
            "baseline_manifest": binding["baseline_manifest"],
            "baseline_source_tree_sha256": binding.get(
                "baseline_source_tree_sha256"
            ),
            "native_scoring_dtype": np.asarray(values).dtype.str,
            "native_scoring_values_sha256": array_hash(values),
            "integrity_validation_dtype": np.dtype(np.float64).str,
            "expected_observed_integrity_sha256": expected_observed_hash(values),
        }
    expected_dimensions = artifact.values.shape[1] if training is None else 32
    if (
        values.shape != (len(artifact.cell_ids), expected_dimensions)
        or clean_output.shape != values.shape
        or counterfactual_outputs.shape
        != (len(artifact.batch_levels), len(artifact.cell_ids), expected_dimensions)
        or not np.isfinite(values).all()
        or not np.isfinite(clean_output).all()
        or not np.isfinite(counterfactual_outputs).all()
    ):
        raise ValueError("Scoring representation or counterfactual outputs are invalid")
    provenance.update(
        shape=list(values.shape),
        values_sha256=array_hash(values),
        clean_output_sha256=array_hash(clean_output),
        counterfactual_outputs_sha256=array_hash(counterfactual_outputs),
        canonical_cell_ids_sha256=canonical_hash(list(artifact.cell_ids)),
    )
    return values, clean_output, counterfactual_outputs, provenance


def _clean_neighbor_reference(prepared, artifact_id, provenance):
    """Load the paired clean-neighbor oracle from the K-binding baseline run."""
    if provenance["method"] == "corrupted_baseline":
        return None, None
    baseline = project_path(ROOT, provenance["baseline_score"])
    manifest = completed(baseline, "artifact_validation_v2_score")
    config = read(baseline / "config.json")
    if (
        file_fingerprint(baseline / "run.json") != provenance["baseline_manifest"]
        or manifest.get("source_tree_sha256")
        != provenance.get("baseline_source_tree_sha256")
        or config.get("prepared") != _relative(prepared.path)
        or config.get("prepared_manifest")
        != file_fingerprint(prepared.path / "run.json")
        or config.get("case_id") != prepared.case["id"]
        or config.get("artifact_id") != artifact_id
        or config.get("method") != "corrupted_baseline"
        or config.get("replicate_seed") is not None
    ):
        raise ValueError("Clean-neighbor reference is not the paired baseline score")
    path = baseline / "artifact/clean_neighbors.npy"
    neighbors = np.load(path, mmap_mode="r", allow_pickle=False)
    return neighbors, {
        "baseline_score": _relative(baseline),
        "baseline_manifest": file_fingerprint(baseline / "run.json"),
        "clean_neighbors": file_fingerprint(path),
        "source_tree_sha256": manifest.get("source_tree_sha256"),
        "role": "paired_clean_neighbor_oracle_reused_without_recomputation",
    }


def _without_arrays(mapping):
    return {
        key: value
        for key, value in mapping.items()
        if not isinstance(value, np.ndarray)
    }


def _metric_lookup(rows):
    result = {}
    for row in rows:
        if row.get("status") == "ok":
            result.setdefault(row["metric"], []).append(row["value"])
    return {
        name: {
            "values": values,
            "mean": float(statistics.mean(values)),
            "minimum": float(min(values)),
            "maximum": float(max(values)),
        }
        for name, values in result.items()
    }


def run_score(prepared_path, artifact_id, run_id, training=None):
    spec = specification()
    sources = snapshot(ROOT)
    source_tree_sha256 = canonical_hash(sources)
    prepared = load_prepared(project_path(ROOT, _relative(prepared_path)))
    artifact = prepared.artifact(artifact_id)
    values, clean_output, counterfactual_outputs, provenance = resolve_representation(
        prepared, artifact_id, training
    )
    clean_reference, clean_reference_receipt = _clean_neighbor_reference(
        prepared, artifact_id, provenance
    )
    config = evaluation_config()
    source_compatibility = {}
    if provenance["method"] != "corrupted_baseline":
        if (
            provenance["training_source_tree_sha256"]
            != provenance["baseline_source_tree_sha256"]
        ):
            raise ValueError("Training and baseline K evidence used different source trees")
        training_path = project_path(ROOT, provenance["training"])
        source_compatibility["training"] = verify_scoring_source_transition(
            training_path,
            provenance["training_source_tree_sha256"],
            sources,
            role="immutable_training_output_to_candidate_scoring",
        )
        baseline_path = project_path(ROOT, provenance["baseline_score"])
        source_compatibility["baseline_oracle"] = verify_scoring_source_transition(
            baseline_path,
            provenance["baseline_source_tree_sha256"],
            sources,
            role="immutable_baseline_oracle_to_candidate_scoring",
        )
        if (
            clean_reference_receipt is None
            or clean_reference_receipt["source_tree_sha256"]
            != provenance["baseline_source_tree_sha256"]
        ):
            raise ValueError("Baseline oracle source receipt changed")
    store = Store(project_path(ROOT, prepared.case.get("store", spec["store"])))
    dataset = store.dataset(prepared.case["dataset"])
    label_key = dataset.record["loader"]["label_key"]
    named_labels = np.asarray(dataset.record["obs"][label_key], dtype=str)
    reference_codes, _ = dataset.reference_partition()
    batches = np.asarray(dataset.batch_labels(), dtype=str)
    if (
        tuple(dataset.cell_ids) != prepared.cell_ids
        or not np.array_equal(named_labels, prepared.named_labels)
        or not np.array_equal(reference_codes, prepared.reference_codes)
        or not np.array_equal(batches, prepared.batches)
        or dataset.annotation_policy != read(prepared.path / "annotation_policy.json")
    ):
        raise ValueError("Evaluation metadata/order differs from the prepared case")
    score_config = {
        "protocol_id": spec["protocol_id"],
        "prepared": _relative(prepared.path),
        "prepared_manifest": file_fingerprint(prepared.path / "run.json"),
        "case_id": prepared.case["id"],
        "artifact_id": artifact_id,
        "method": provenance["method"],
        "replicate_seed": provenance["replicate_seed"],
        "training": provenance.get("training"),
        "evaluation": config.to_dict(),
        "input": provenance,
        "clean_neighbor_reference": clean_reference_receipt,
        "source_compatibility": source_compatibility,
    }
    start = time.perf_counter()
    with RunDirectory(
        ROOT / "revision_pipeline/runs",
        kind="artifact_validation_v2_score",
        config=score_config,
        run_id=run_id,
    ) as run:
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = source_tree_sha256
        run.manifest.update(
            scientific_experiment=True,
            experiment_role=spec["role"],
            claim_boundary=spec["claim_boundary"],
            scorer_recovery_applied=any(
                item["status"] == "verified_scorer_only_recovery"
                for item in source_compatibility.values()
            ),
        )
        run.write_json("runtime_start.json", runtime())
        run.write_json("input.json", provenance)
        run.write_json("source_compatibility.json", source_compatibility)
        run.write_json("annotation_policy.json", dataset.annotation_policy)
        reference, interpretation = dataset.reference_partition()
        with threadpool_limits(limits=1):
            write_json(run.path / "progress.json", {"stage": "graph_and_grid"})
            grid = graph_and_grid(
                values,
                reference,
                artifact.cell_ids,
                config,
                run=run,
                prefix="evaluation",
                training_label_use=provenance["training_label_use"],
            )
            write_json(run.path / "progress.json", {"stage": "standard_metrics"})
            standard = metric_records(
                values, dataset, config, grid=grid, run=run, prefix="evaluation"
            )
            write_json(run.path / "progress.json", {"stage": "artifact_metrics"})
            sensitivity = counterfactual_batch_sensitivity(
                clean_output,
                counterfactual_outputs,
                values,
                prepared.batches,
                artifact.batch_levels,
                artifact.cell_ids,
                artifact.cell_ids,
            )
            if clean_reference is None:
                recovery = clean_neighbor_recovery(
                    artifact.clean,
                    values,
                    artifact.cell_ids,
                    artifact.cell_ids,
                    artifact.cell_ids,
                    k=30,
                    working_memory_mb=spec["working_memory_mb"],
                )
            else:
                recovery = clean_neighbor_recovery_from_reference(
                    clean_reference,
                    values,
                    artifact.cell_ids,
                    artifact.cell_ids,
                    k=30,
                    working_memory_mb=spec["working_memory_mb"],
                )
            rare_records = prepared.rare_groups["eligible_non_singleton_groups"]
            rare_labels = [record["supplied_label"] for record in rare_records]
            preservation = preservation_metrics_from_neighbors(
                recovery["candidate_neighbors"],
                prepared.named_labels,
                artifact.cell_ids,
                artifact.cell_ids,
                rare_groups=rare_labels,
                target_label=prepared.case["target_label"],
                k=30,
            )
        np.save(
            run.artifact_path("artifact/clean_neighbors.npy"),
            recovery["clean_neighbors"],
            allow_pickle=False,
        )
        np.save(
            run.artifact_path("artifact/candidate_neighbors.npy"),
            recovery["candidate_neighbors"],
            allow_pickle=False,
        )
        np.save(
            run.artifact_path("artifact/clean_neighbor_jaccard.npy"),
            recovery["per_cell_clean_neighbor_jaccard"],
            allow_pickle=False,
        )
        np.save(
            run.artifact_path("artifact/purity.npy"),
            preservation["purity"]["per_cell_neighbor_purity"],
            allow_pickle=False,
        )
        np.save(
            run.artifact_path("artifact/target_same_class_fraction.npy"),
            preservation["target_same_class_fraction"][
                "per_cell_target_same_class_fraction_at_k"
            ],
            allow_pickle=False,
        )
        rare_scalar = []
        for index, record in enumerate(preservation["rare_recall"]["groups"]):
            np.save(
                run.artifact_path(f"artifact/rare_recall_{index:03d}.npy"),
                record["per_cell_same_class_recall_at_k"],
                allow_pickle=False,
            )
            scalar = _without_arrays(record)
            scalar["screen_key"] = rare_records[index]["screen_key"]
            if scalar["group_repr"] != repr(rare_records[index]["supplied_label"]):
                raise ValueError("Rare-group evaluation order changed")
            rare_scalar.append(scalar)
        selected = grid["selected"]
        if (
            len(selected) != 3
            or sorted(row["leiden_seed"] for row in selected) != [0, 1, 2]
            or any(row["resolution"] != 0.5 for row in selected)
        ):
            raise ValueError("Primary fixed-resolution partition panel is incomplete")
        summary = {
            "method": provenance["method"],
            "replicate_seed": provenance["replicate_seed"],
            "fixed_resolution_ARI": {
                "values": [float(row["ARI"]) for row in selected],
                "mean": float(statistics.mean(row["ARI"] for row in selected)),
                "cluster_counts": [int(row["n_clusters"]) for row in selected],
                "Leiden_seeds": [int(row["leiden_seed"]) for row in selected],
                "resolution": 0.5,
            },
            "sensitivity": sensitivity,
            "clean_neighbor_recovery": {
                key: value
                for key, value in recovery.items()
                if not isinstance(value, np.ndarray)
            },
            "purity": _without_arrays(preservation["purity"]),
            "rare_recall": {
                "groups": rare_scalar,
                "all_rare_cells_query_full_population": True,
                "k_nonself": 30,
            },
            "target_same_class_fraction": _without_arrays(
                preservation["target_same_class_fraction"]
            ),
            "standard_metrics": _metric_lookup(standard),
            "full_resolution_grid_reported": True,
            "reference_interpretation": interpretation,
            "claim_boundary": spec["claim_boundary"],
            "clean_neighbor_reference": clean_reference_receipt,
            "source_compatibility": source_compatibility,
        }
        run.write_json("artifact/summary.json", summary)
        run.write_json(
            "timing.json",
            {
                **grid["timing"],
                "whole_score_seconds": time.perf_counter() - start,
            },
        )
        run.write_json("runtime_end.json", runtime())
        if snapshot(ROOT) != sources:
            raise RuntimeError("Scientific source changed during artifact scoring")
        write_json(run.path / "progress.json", {"stage": "completed"})
    return run.final_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--artifact", choices=("batch_simplex", "target_local_warp"), required=True)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(
        run_score(args.prepared, args.artifact, args.run_id, args.training),
        flush=True,
    )


if __name__ == "__main__":
    main()
