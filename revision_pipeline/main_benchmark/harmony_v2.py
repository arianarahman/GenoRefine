"""Objective-corrected, convergence-gated Pancreas Harmony extension.

This is a post-failure extension.  It preserves the two v1 failures and uses
only the Harmony objective to decide when to stop; annotation labels and
downstream scores are never read.
"""
import argparse
import copy
from importlib.metadata import version
from pathlib import Path
import resource
import time

import numpy as np

from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import read, snapshot
from ..runs import RunDirectory
from .common import ROOT, specification


POLICY = ROOT / "revision_pipeline/configs/pancreas_primary_harmony_policy_v2.json"


def objective_gate(objectives, epsilon, period2_comparisons):
    values = np.asarray(objectives, dtype=np.float64)
    result = {
        "policy_id": "pancreas_harmony_convergence_v2",
        "status": "failed",
        "mode": None,
        "epsilon": float(epsilon),
        "period2_comparisons": int(period2_comparisons),
        "adjacent_relative_change": None,
        "period2_relative_changes": [],
    }
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        return result
    previous = values[-2]
    if previous == 0:
        return result
    adjacent = abs((values[-1] - previous) / previous)
    result["adjacent_relative_change"] = float(adjacent)
    if adjacent < epsilon:
        result.update(status="passed", mode="adjacent_stabilization")
        return result
    needed = period2_comparisons + 2
    if len(values) < needed:
        return result
    changes = []
    for offset in range(period2_comparisons):
        current = values[-1 - offset]
        two_back = values[-3 - offset]
        if two_back == 0:
            return result
        changes.append(float(abs((current - two_back) / two_back)))
    result["period2_relative_changes"] = changes
    if max(changes) < epsilon:
        result.update(status="passed", mode="stable_period_2")
    return result


def install_objective_only_gate(policy):
    """Patch only the outer stopping test; retain native k-means stopping."""
    import harmonypy.harmony as harmony_source

    native = harmony_source.Harmony.check_convergence

    def corrected(instance, iteration_type):
        if iteration_type == 1:
            return objective_gate(
                instance.objective_harmony,
                policy["epsilon_harmony"],
                policy["period2_comparisons"],
            )["status"] == "passed"
        return native(instance, iteration_type)

    harmony_source.Harmony.check_convergence = corrected
    return native


def main():
    import anndata as ad
    import pandas as pd
    from threadpoolctl import threadpool_limits

    from ..data.pancreas_backbones import copy_parent_artifacts, fit_harmony, runtime_record
    from ..data.readers import array_hash
    from ..data.store import Store

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    policy = read(POLICY)
    if version("harmonypy") != policy["pinned_harmonypy"]:
        raise ValueError("Wrong Harmony runtime")
    spec = specification()
    sources = snapshot(ROOT)
    parent = Store(ROOT / spec["store"])
    dataset = parent.dataset("pancreas_five_study")
    pca_path = parent._artifact("preprocessing/pancreas_five_study/X_pca.npy")
    pca = np.load(pca_path, allow_pickle=False)
    cfg = read(ROOT / "revision_pipeline/configs/pancreas_backbones.json")
    for key in ("max_iter_harmony", "max_iter_kmeans", "epsilon_harmony", "epsilon_cluster"):
        cfg["harmony"][key] = policy[key]
    data = ad.AnnData(
        np.zeros((len(dataset.cell_ids), 1)),
        obs=pd.DataFrame(
            {"batch": pd.Categorical(dataset.batch_labels(), categories=cfg["batch_order"])},
            index=list(dataset.cell_ids),
        ),
    )
    data.obsm["X_pca"] = pca.copy()
    install_objective_only_gate(policy)
    with RunDirectory(
        ROOT / "revision_pipeline/runs",
        kind="step3a_data_store",
        run_id=args.run_id,
        config={
            "operation": "post_failure_pancreas_harmony_v2",
            "parameters": cfg["harmony"],
            "policy": policy,
            "policy_fingerprint": file_fingerprint(POLICY),
            "parent_manifest": file_fingerprint(parent.path / "run.json"),
            "PCA": file_fingerprint(pca_path),
            "v1_failed_builds": [
                "revision_pipeline/runs/20260918T151527Z-1cab228f119f-pan_harmony-build0",
                "revision_pipeline/runs/20260918T151527Z-1cab228f119f-pan_harmony-build1",
            ],
            "labels_or_scores_used_for_stopping": False,
        },
    ) as run:
        run.write_json("source_manifest.json", sources)
        run.write_json("runtime.json", runtime_record())
        copy_parent_artifacts(parent, run)
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            values, order, diagnostics = fit_harmony(data, cfg)
        wall = time.perf_counter() - started
        if array_hash(data.obsm["X_pca"]) != array_hash(pca):
            raise ValueError("PCA input mutated")
        gate = objective_gate(
            diagnostics["harmony_objective"],
            policy["epsilon_harmony"],
            policy["period2_comparisons"],
        )
        gate["iterations"] = diagnostics["harmony_iterations"]
        gate["max_iter_harmony"] = policy["max_iter_harmony"]
        if gate["status"] != "passed":
            raise ValueError("Pancreas Harmony v2 did not meet its objective-only convergence gate")
        stem = "embeddings/pancreas_five_study/Harmony_primary_v2"
        np.save(run.artifact_path(stem + ".npy"), values, allow_pickle=False)
        np.save(run.artifact_path(stem + ".source_rows.npy"), order, allow_pickle=False)
        old = parent.embedding("pancreas_five_study", "Harmony").metadata
        meta = dict(old, **diagnostics)
        meta.update(
            values_path=stem + ".npy",
            source_rows_path=stem + ".source_rows.npy",
            primary_convergence=gate,
            parameters=cfg["harmony"],
            origin="post-failure objective-only Harmony v2 from exact frozen canonical PCA",
            historical_comparison="not the historical ten-iteration embedding and not either failed v1 output",
            input_values_sha256=array_hash(pca),
            wall_seconds=wall,
            stopping_labels_used=False,
            stopping_downstream_scores_used=False,
            cache_key=canonical_hash(
                {
                    "parent": file_fingerprint(parent.path / "run.json"),
                    "parameters": cfg["harmony"],
                    "policy": file_fingerprint(POLICY),
                }
            ),
        )
        meta.pop("stored_values_file_sha256", None)
        run.write_json(stem + ".json", meta)
        index = copy.deepcopy(parent.index)
        index["embeddings"]["pancreas_five_study"]["Harmony"] = stem + ".json"
        run.write_json("store.json", index)
        run.write_json("convergence.json", gate)
        run.write_json(
            "validation.json",
            {
                "passed": True,
                "primary_convergence": gate,
                "original_Harmony_files_preserved": True,
                "v1_failure_evidence_preserved": True,
                "labels_or_downstream_scores_used_for_stopping": False,
            },
        )
        run.write_json("input_manifest.json", index["source_files"])
        if snapshot(ROOT) != sources:
            raise ValueError("Source changed during Harmony v2")
        run.manifest.update(
            peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            scientific_experiment=True,
            experiment_role="post_failure_objective_only_extension",
        )
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
