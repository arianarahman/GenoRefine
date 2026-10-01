# Purpose: Select GraphST donor-pair K from frozen Harmony without annotations.
# Author: Ariana Rahman (Arizona State University)

"""Select GraphST donor-pair K from frozen Harmony without annotations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..data.readers import array_hash
from ..integrity import canonical_hash, file_fingerprint
from ..runs import RunDirectory
from ..step4_policy import derive_training_k
from ..spatial_multisection.k_selection import fixed_partitions
from .common import (
    RUNS, evaluation_config, evaluation_dict, harmony_path, load_fixed_harmony, source_snapshot,
    specification, validate_evaluation_runtime,
)


def execute(run_id: str) -> Path:
    spec = specification()
    values, metadata, harmony_record = load_fixed_harmony()
    config = evaluation_config()
    frozen_evaluation = evaluation_dict()
    runtime = validate_evaluation_runtime()
    sources = source_snapshot()
    context = {
        "protocol_id": spec["protocol_id"],
        "harmony_manifest": file_fingerprint(harmony_path() / "run.json"),
        "selection": spec["donor_cluster_count_selection"],
        "evaluation": frozen_evaluation,
        "runtime": runtime,
        "reference_labels_loaded": False,
        "reference_labels_used": False,
    }
    with RunDirectory(RUNS, kind="spatial_graphst_k_selection", run_id=run_id, config=context) as run:
        decisions = {}
        for donor, sections in spec["donors"].items():
            take = np.flatnonzero(metadata["donor"].astype(str).to_numpy() == donor)
            ids = metadata.iloc[take]["cell_id"].astype(str).tolist()
            if set(metadata.iloc[take]["section"].astype(str)) != set(sections):
                raise ValueError("Donor section coverage changed")
            result = fixed_partitions(values[take], ids, config)
            decision = derive_training_k(result["partitions"], ids, ids, frozen_evaluation)
            np.savez_compressed(
                run.artifact_path(f"donors/{donor}_partitions.npz"),
                **{f"seed{seed}": part for seed, part in result["partitions"].items()},
            )
            np.save(
                run.artifact_path(f"donors/{donor}_knn_indices.npy"),
                result["knn_indices"], allow_pickle=False,
            )
            decisions[donor] = {
                **decision,
                "sections": sections,
                "n_cells": len(take),
                "graph": result["provenance"],
            }
        record = {
            "protocol_id": spec["protocol_id"],
            "harmony_manifest": context["harmony_manifest"],
            "harmony_embedding_sha256": harmony_record["embedding_sha256"],
            "reference_labels_loaded": False,
            "reference_labels_used": False,
            "donors": decisions,
        }
        run.write_json("k_selection.json", record)
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.manifest.update(
            scientific_experiment=True,
            experiment_role="label_free_graphst_donor_pair_cluster_count_selection",
            training_performed=False,
            scoring_performed=False,
        )
        if source_snapshot() != sources:
            raise RuntimeError("Scientific source changed during GraphST K selection")
    return run.final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required")
    print(execute(args.run_id), flush=True)


if __name__ == "__main__":
    main()
