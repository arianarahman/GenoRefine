"""Fresh, exact-10-iteration Pancreas Harmony sensitivity build."""
import argparse
import copy
from importlib.metadata import version
import resource
import time

import numpy as np

from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import read, snapshot
from ..runs import RunDirectory
from .common import ROOT, specification


POLICY = ROOT / "revision_pipeline/configs/pancreas_harmony_fixed10_policy.json"


def disable_outer_early_stop():
    import harmonypy.harmony as harmony_source

    native = harmony_source.Harmony.check_convergence

    def fixed_budget(instance, iteration_type):
        if iteration_type == 1:
            return False
        return native(instance, iteration_type)

    harmony_source.Harmony.check_convergence = fixed_budget


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
    disable_outer_early_stop()
    with RunDirectory(
        ROOT / "revision_pipeline/runs",
        kind="step3a_data_store",
        run_id=args.run_id,
        config={
            "operation": "post_failure_pancreas_harmony_fixed10_sensitivity",
            "parameters": cfg["harmony"],
            "policy": policy,
            "policy_fingerprint": file_fingerprint(POLICY),
            "parent_manifest": file_fingerprint(parent.path / "run.json"),
            "PCA": file_fingerprint(pca_path),
            "labels_or_scores_used_for_budget": False,
        },
    ) as run:
        run.write_json("source_manifest.json", sources)
        run.write_json("runtime.json", runtime_record())
        copy_parent_artifacts(parent, run)
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            values, order, diagnostics = fit_harmony(data, cfg)
        wall = time.perf_counter() - started
        if diagnostics["harmony_iterations"] != policy["max_iter_harmony"]:
            raise ValueError("Harmony did not execute the exact declared outer-iteration budget")
        if array_hash(data.obsm["X_pca"]) != array_hash(pca):
            raise ValueError("PCA input mutated")
        completion = {
            "policy_id": policy["policy_id"],
            "status": "fixed_budget_completed",
            "iterations": diagnostics["harmony_iterations"],
            "numerical_convergence_claimed": False,
            "labels_or_downstream_scores_used": False,
        }
        stem = "embeddings/pancreas_five_study/Harmony_fixed10"
        np.save(run.artifact_path(stem + ".npy"), values, allow_pickle=False)
        np.save(run.artifact_path(stem + ".source_rows.npy"), order, allow_pickle=False)
        old = parent.embedding("pancreas_five_study", "Harmony").metadata
        meta = dict(old, **diagnostics)
        meta.update(
            values_path=stem + ".npy",
            source_rows_path=stem + ".source_rows.npy",
            primary_convergence=completion,
            parameters=cfg["harmony"],
            origin="fresh exact-10-iteration Harmony sensitivity from frozen canonical PCA",
            historical_comparison="same outer-iteration budget as historical/default Harmony; fresh canonical-order build",
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
        run.write_json("convergence.json", completion)
        run.write_json(
            "validation.json",
            {
                "passed": True,
                "primary_convergence": completion,
                "fixed_budget_sensitivity": True,
                "numerical_convergence_claimed": False,
                "original_Harmony_files_preserved": True,
                "prior_failure_evidence_preserved": True,
            },
        )
        run.write_json("input_manifest.json", index["source_files"])
        if snapshot(ROOT) != sources:
            raise ValueError("Source changed during fixed-10 Harmony")
        run.manifest.update(
            peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            scientific_experiment=True,
            experiment_role="post_failure_fixed10_descriptive_sensitivity",
        )
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
