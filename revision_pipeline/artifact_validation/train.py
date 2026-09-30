"""Fit one GenoRefine or IDEC replicate to one prepared corrupted embedding."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

from ..data.readers import array_hash
from ..evaluate.inputs import write_refined_bundle
from ..integrity import canonical_hash, file_fingerprint, project_path
from ..pilot.common import completed, read, snapshot
from ..runs import RunDirectory, write_json
from .common import ROOT, load_prepared, specification


METHODS = ("genorefine", "idec")


def _relative(path: Path | str) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def bind_training_k(prepared, baseline_score: Path | str) -> dict:
    """Bind K only to the three fixed-resolution corrupted-baseline partitions."""
    score_path = Path(baseline_score).resolve()
    record = completed(score_path, "artifact_validation_v2_score")
    config = read(score_path / "config.json")
    prepared_manifest = file_fingerprint(prepared.path / "run.json")
    if (
        config.get("prepared") != _relative(prepared.path)
        or config.get("prepared_manifest") != prepared_manifest
        or config.get("artifact_id") not in prepared.artifacts
        or config.get("method") != "corrupted_baseline"
        or config.get("replicate_seed") is not None
    ):
        raise ValueError("K evidence is not the paired corrupted-baseline score")
    selected_path = score_path / "evaluation/selected.json"
    selected = read(selected_path)
    policy = specification()["training_K_policy"]
    from ..step4_policy import load_policy

    step4_policy = load_policy()
    expected_evaluation = read(ROOT / step4_policy["primary_evaluation_config"])
    scored_ids = tuple(read(score_path / "evaluation/cell_ids.json"))
    expected_seeds = policy["leiden_seeds"]
    if (
        config.get("evaluation") != expected_evaluation
        or scored_ids != prepared.cell_ids
        or len(selected) != len(expected_seeds)
        or sorted(row.get("leiden_seed") for row in selected) != expected_seeds
        or any(row.get("resolution") != policy["resolution"] for row in selected)
        or any(type(row.get("n_clusters")) is not int or row["n_clusters"] < 2 for row in selected)
        or any(row.get("selection_label_informed") is not False for row in selected)
    ):
        raise ValueError("Baseline score does not contain the frozen label-free K anchors")
    counts = {int(row["leiden_seed"]): int(row["n_clusters"]) for row in selected}
    n_clusters = int(statistics.median(counts.values()))
    if not 2 <= n_clusters < len(prepared.cell_ids):
        raise ValueError("Derived K is outside the valid training range")
    return {
        "n_clusters": n_clusters,
        "selection_rule": policy["rule"],
        "fixed_resolution": policy["resolution"],
        "counts_by_Leiden_seed": counts,
        "reference_labels_used": False,
        "baseline_score": _relative(score_path),
        "baseline_manifest": file_fingerprint(score_path / "run.json"),
        "baseline_source_tree_sha256": record.get("source_tree_sha256"),
        "evidence": {
            "selected": file_fingerprint(selected_path),
            "partitions": file_fingerprint(score_path / "evaluation/partitions.npy"),
            "cell_ids": file_fingerprint(score_path / "evaluation/cell_ids.json"),
            "graph": file_fingerprint(score_path / "evaluation/graph.json"),
        },
    }


def _save_inference_outputs(run, clean, counterfactuals, observed, expected_observed):
    clean = np.asarray(clean, dtype=np.float64)
    counterfactuals = np.asarray(counterfactuals, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    expected_observed = np.asarray(expected_observed, dtype=np.float64)
    if (
        clean.ndim != 2
        or counterfactuals.ndim != 3
        or counterfactuals.shape[1:] != clean.shape
        or observed.shape != clean.shape
        or expected_observed.shape != clean.shape
        or not all(
            np.isfinite(value).all()
            for value in (clean, counterfactuals, observed, expected_observed)
        )
    ):
        raise ValueError("Invalid clean/counterfactual inference outputs")
    maximum_error = float(np.max(np.abs(observed - expected_observed)))
    if not np.allclose(observed, expected_observed, rtol=1e-6, atol=1e-6):
        raise ValueError("Saved observed embedding is not reproduced by model inference")
    np.save(run.artifact_path("inference/clean.npy"), clean, allow_pickle=False)
    np.save(
        run.artifact_path("inference/counterfactuals.npy"),
        counterfactuals,
        allow_pickle=False,
    )
    run.write_json(
        "inference/checks.json",
        {
            "clean_shape": list(clean.shape),
            "counterfactual_shape": list(counterfactuals.shape),
            "observed_recovery_max_abs_error": maximum_error,
            "observed_recovered_rtol": 1e-6,
            "observed_recovered_atol": 1e-6,
            "clean_sha256": array_hash(clean),
            "counterfactuals_sha256": array_hash(counterfactuals),
            "observed_reloaded_sha256": array_hash(observed),
            "expected_observed_sha256": array_hash(expected_observed),
            "counterfactuals_used_for_training": False,
            "clean_input_used_for_training": False,
        },
    )


def _fit_genorefine(artifact, decision, seed, run):
    from ..refine.config import LayoutConfig, RefinerConfig
    from ..refine.staged import StagedGenoDR
    from ..step4_policy import check_joint_coverage, load_policy, planned_training_config

    policy = load_policy()
    config = RefinerConfig(
        training=planned_training_config(len(artifact.cell_ids), decision["n_clusters"], seed),
        layout=LayoutConfig(
            requested_side=policy["training"]["reference_map_side"],
            scaling="none",
            transport_iterations=200,
            epsilon=0.0,
        ),
    )
    x, ids, features = artifact.values, artifact.cell_ids, artifact.coordinate_names
    model = StagedGenoDR(config).fit_layout(x, cell_ids=ids, feature_ids=features)
    pretrain = model.pretrain(
        x, cell_ids=ids, feature_ids=features, directory=run.path / "pretrain"
    )
    joint = model.cluster(
        x, cell_ids=ids, feature_ids=features, directory=run.path / "joint"
    )
    model.save(run.path / "model")
    write_refined_bundle(
        run.path / "bundle",
        joint,
        ids,
        parent_reference=artifact.parent_reference(),
        training_label_use="label_free",
    )
    with np.load(run.path / "joint/visits.npz", allow_pickle=False) as saved:
        coverage = check_joint_coverage(saved["visits"], len(ids))
    # Exercise the published inference bundle rather than retaining the
    # in-memory training object for evaluation-only counterfactual transforms.
    reloaded = StagedGenoDR.load(run.path / "model")
    clean_output = reloaded.transform(
        artifact.clean, cell_ids=ids, feature_ids=features
    )
    counterfactual_outputs = np.stack(
        [
            reloaded.transform(values, cell_ids=ids, feature_ids=features)
            for values in artifact.counterfactuals
        ]
    )
    observed_reloaded = reloaded.transform(x, cell_ids=ids, feature_ids=features)
    _save_inference_outputs(
        run, clean_output, counterfactual_outputs, observed_reloaded, joint
    )
    run.write_json(
        "training_summary.json",
        {
            "method": "GenoRefine",
            "effective_config": config.to_dict(),
            "coverage": coverage,
            "pretrain_shape": list(pretrain.shape),
            "joint_shape": list(joint.shape),
            "pretrain_sha256": array_hash(pretrain),
            "joint_sha256": array_hash(joint),
            "saved_model_bundle": file_fingerprint(run.path / "model/bundle.json"),
            "counterfactual_inference_from_saved_reload": True,
            "counterfactuals_evaluation_only": True,
        },
    )
    del reloaded
    return config.to_dict()


def _fit_idec(artifact, decision, seed, run):
    from ..independent_comparator.idec_compat import (
        IDECTransformer,
        build_idec_models,
        fit_idec,
    )

    dimensions = [artifact.values.shape[1], 500, 500, 2000, 32]
    batch_size = 64
    updates = 2 * math.ceil(len(artifact.cell_ids) / batch_size)
    result = fit_idec(
        artifact.values,
        seed=seed,
        n_clusters=decision["n_clusters"],
        dimensions=dimensions,
        batch_size=batch_size,
        pretrain_epochs=100,
        joint_updates=updates,
        update_interval=50,
        gamma=0.1,
        alpha=1.0,
        learning_rate=0.001,
        kmeans_n_init=20,
        output_directory=run.artifact_path("model"),
    )
    write_refined_bundle(
        run.path / "bundle",
        result.joint_embedding,
        artifact.cell_ids,
        parent_reference=artifact.parent_reference(),
        training_label_use="label_free",
    )
    # Rebuild from the newly saved weights so counterfactual inference exercises
    # the same save/reload path used for future application.
    import tensorflow as tf

    tf.keras.backend.clear_session()
    autoencoder, encoder, joint = build_idec_models(
        dimensions, decision["n_clusters"], alpha=1.0
    )
    if (
        result.architecture["autoencoder_parameters"] != int(autoencoder.count_params())
        or result.architecture["joint_parameters"] != int(joint.count_params())
    ):
        raise RuntimeError("Reloaded IDEC architecture differs from the fitted model")
    joint.load_weights(run.path / "model/joint.weights.h5")
    transformer = IDECTransformer(
        encoder=encoder,
        run_id=run.run_id,
        case_id="artifact_validation",
        seed=seed,
        input_dim=dimensions[0],
        latent_dim=dimensions[-1],
        weights_fingerprint=file_fingerprint(run.path / "model/joint.weights.h5"),
    )
    clean_output = transformer.transform(artifact.clean, batch_size=1024)
    counterfactual_outputs = np.stack(
        [transformer.transform(values, batch_size=1024) for values in artifact.counterfactuals]
    )
    observed_reloaded = transformer.transform(artifact.values, batch_size=1024)
    _save_inference_outputs(
        run,
        clean_output,
        counterfactual_outputs,
        observed_reloaded,
        result.joint_embedding,
    )
    run.write_json("architecture.json", result.architecture)
    run.write_json("timing.json", result.timing)
    if (
        result.coverage["minimum_visits"] != 2
        or result.coverage["maximum_visits"] != 2
        or result.coverage["examples_seen"] != 2 * len(artifact.cell_ids)
        or result.coverage["empty_batches"] != 0
    ):
        raise RuntimeError("IDEC did not complete exactly two full joint passes")
    run.write_json("coverage.json", result.coverage)
    effective = {
        "method": "IDEC",
        "encoder_dimensions": dimensions,
        "latent_dim": 32,
        "batch_size": batch_size,
        "pretrain_epochs": 100,
        "joint_updates": updates,
        "joint_examples_seen": result.coverage["examples_seen"],
        "joint_pass_equivalent": result.coverage["full_pass_equivalent"],
        "update_interval": 50,
        "gamma": 0.1,
        "alpha": 1.0,
        "learning_rate": 0.001,
        "kmeans_n_init": 20,
        "joint_batch_order": "official_sequential_wraparound_no_shuffle",
        "counterfactuals_evaluation_only": True,
        "compatibility_port_not_unmodified_historical_runtime": True,
    }
    run.write_json(
        "training_summary.json",
        {
            **effective,
            "pretrain_sha256": array_hash(result.pretrain_embedding),
            "joint_sha256": array_hash(result.joint_embedding),
            "native_cluster_count": int(len(np.unique(result.native_clusters))),
            "saved_weights": transformer.weights_fingerprint,
            "counterfactual_inference_from_saved_reload": True,
        },
    )
    del autoencoder, encoder, joint, transformer
    return effective


def run_training(prepared_path, baseline_score, artifact_id, method, seed, run_id):
    spec = specification()
    if method not in METHODS or seed not in spec["replicate_seeds"]:
        raise ValueError("Method or seed is outside artifact_validation_v2")
    prepared = load_prepared(project_path(ROOT, _relative(prepared_path)))
    artifact = prepared.artifact(artifact_id)
    decision = bind_training_k(prepared, baseline_score)
    score_config = read(Path(baseline_score).resolve() / "config.json")
    if score_config["artifact_id"] != artifact_id:
        raise ValueError("Baseline K evidence belongs to another artifact")

    from ..main_benchmark.fast_runtime import configure_fast_gpu

    runtime = configure_fast_gpu()
    sources = snapshot(ROOT)
    source_tree_sha256 = canonical_hash(sources)
    if decision.get("baseline_source_tree_sha256") != source_tree_sha256:
        raise ValueError(
            "Baseline K evidence and training must use one frozen scientific source tree"
        )
    start = time.perf_counter()
    config = {
        "protocol_id": spec["protocol_id"],
        "prepared": _relative(prepared.path),
        "prepared_manifest": file_fingerprint(prepared.path / "run.json"),
        "case_id": prepared.case["id"],
        "artifact_id": artifact_id,
        "method": method,
        "replicate_seed": seed,
        "K_binding": decision,
        "artifact_parent_reference": artifact.parent_reference(),
    }
    with RunDirectory(
        ROOT / "revision_pipeline/runs",
        kind="artifact_validation_v2_training",
        config=config,
        run_id=run_id,
    ) as run:
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = source_tree_sha256
        run.manifest.update(
            scientific_experiment=True,
            experiment_role=spec["role"],
            claim_boundary=spec["claim_boundary"],
        )
        run.write_json("runtime.json", runtime)
        run.write_json("k_selection.json", decision)
        run.write_json(
            "input.json",
            {
                "parent_reference": artifact.parent_reference(),
                "cell_ids_sha256": canonical_hash(list(artifact.cell_ids)),
                "values_sha256": array_hash(artifact.values),
                "shape": list(artifact.values.shape),
                "training_label_use": "label_free",
                "counterfactuals_used_for_training": False,
                "clean_input_used_for_training": False,
            },
        )
        write_json(run.path / "progress.json", {"stage": "training", "method": method})
        effective = (
            _fit_genorefine(artifact, decision, seed, run)
            if method == "genorefine"
            else _fit_idec(artifact, decision, seed, run)
        )
        run.write_json(
            "completion.json",
            {
                "status": "completed",
                "method": method,
                "seed": seed,
                "effective_training": effective,
                "wall_seconds_inside_runner": time.perf_counter() - start,
            },
        )
        if snapshot(ROOT) != sources:
            raise RuntimeError("Scientific source changed during artifact training")
        try:
            import resource
        except ImportError:
            run.manifest.update(
                peak_memory_bytes=None,
                peak_memory_status="unavailable_on_this_platform",
            )
        else:
            multiplier = 1 if sys.platform == "darwin" else 1024
            run.manifest.update(
                peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * multiplier,
                peak_memory_status="whole_process_ru_maxrss_including_imports",
            )
        write_json(run.path / "progress.json", {"stage": "completed", "method": method})
    return run.final_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--baseline-score", type=Path, required=True)
    parser.add_argument("--artifact", choices=("batch_simplex", "target_local_warp"), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required")
    print(
        run_training(
            args.prepared,
            args.baseline_score,
            args.artifact,
            args.method,
            args.seed,
            args.run_id,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
