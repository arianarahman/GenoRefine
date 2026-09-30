"""Recompute the pooled spatial Harmony baseline for exactly ten outer iterations."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
import platform
from typing import Callable

import harmonypy
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..data.readers import array_hash
from ..integrity import canonical_hash, file_fingerprint
from ..runs import RunDirectory
from .common import RUNS, foundation_path, load_foundation_embedding, source_snapshot, specification


def validate_runtime() -> dict:
    locked = specification()["harmony_primary"]["runtime"]
    expected = locked["packages"]
    observed = {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "pandas": version("pandas"),
        "scikit-learn": version("scikit-learn"),
        "harmonypy": version("harmonypy"),
        "threadpoolctl": version("threadpoolctl"),
    }
    if observed != expected:
        raise RuntimeError(f"Fixed-Harmony runtime differs from the frozen foundation: {observed}")
    return {"packages": observed, "executable": "<PYTHON_ENV>", "validated": True}


def _orient(values: np.ndarray, expected_shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    values = np.asarray(values, dtype=np.float64)
    orientation = "cells_by_components"
    if values.shape == expected_shape[::-1] and values.shape != expected_shape:
        values = values.T
        orientation = "components_by_cells_transposed"
    if values.shape != expected_shape or not np.isfinite(values).all():
        raise ValueError(f"Harmony returned invalid coordinates: {values.shape}")
    return values.astype(np.float32), orientation


def run_fixed_harmony(
    pca: np.ndarray,
    sections: np.ndarray,
    *,
    runner: Callable | None = None,
    settings: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Run harmonypy with outer stopping disabled and require all ten iterations."""
    spec = specification()
    settings = dict(spec["harmony_primary"] if settings is None else settings)
    pca = np.asarray(pca)
    sections = np.asarray(sections, dtype="U")
    if (pca.ndim != 2 or len(pca) != len(sections) or len(set(sections)) < 2
            or not np.isfinite(pca).all()):
        raise ValueError("Invalid PCA/section inputs for fixed Harmony")
    if (settings["max_iter_harmony"] != 10 or settings["required_completed_outer_iterations"] != 10
            or settings["epsilon_harmony_runtime"] != "negative_infinity"
            or settings["thread_limit"] != 1):
        raise ValueError("Fixed-Harmony settings changed")
    runner = harmonypy.run_harmony if runner is None else runner
    meta = pd.DataFrame({settings["batch_key"]: sections})
    with threadpool_limits(limits=1):
        output = runner(
            pca,
            meta,
            settings["batch_key"],
            max_iter_harmony=10,
            epsilon_harmony=-np.inf,
            random_state=int(settings["random_state"]),
            verbose=False,
        )
    values, orientation = _orient(output.Z_corr, pca.shape)
    objectives = [float(x) for x in getattr(output, "objective_harmony", [])]
    rounds = [int(x) for x in getattr(output, "kmeans_rounds", [])]
    iterations = max(0, len(objectives) - 1)
    if iterations != 10 or len(objectives) != 11:
        raise RuntimeError(f"Harmony did not complete exactly ten outer iterations: {iterations}")
    if not np.isfinite(objectives).all() or (rounds and len(rounds) != 10):
        raise RuntimeError("Harmony objective/inner-round trace is incomplete or non-finite")
    record = {
        "implementation": "harmonypy.run_harmony",
        "batch_key": settings["batch_key"],
        "max_iter_harmony": 10,
        "epsilon_harmony_runtime": "negative_infinity",
        "outer_early_stopping_disabled": True,
        "random_state": int(settings["random_state"]),
        "thread_limit": 1,
        "iterations_completed": iterations,
        "objective_harmony": objectives,
        "kmeans_rounds": rounds,
        "returned_orientation": orientation,
        "strict_convergence_claimed": False,
        "stop_reason": "prespecified fixed ten outer iterations",
        "embedding_sha256": array_hash(values),
    }
    return values, record


def execute(run_id: str) -> Path:
    spec = specification()
    foundation = foundation_path()
    pca, metadata = load_foundation_embedding("pca50")
    sections = metadata["section"].astype(str).to_numpy(dtype="U")
    runtime = validate_runtime()
    sources = source_snapshot()
    context = {
        "protocol_id": spec["protocol_id"],
        "foundation_manifest": file_fingerprint(foundation / "run.json"),
        "settings": spec["harmony_primary"],
        "runtime": runtime,
        "repeatability_gate": "two fresh model initializations in one single-threaded process must be bitwise identical",
        "native_stop_sensitivity": {
            "embedding": file_fingerprint(foundation / "harmony50.npy"),
            "record": file_fingerprint(foundation / "preprocessing_record.json"),
            "role": "sensitivity_only",
        },
    }
    with RunDirectory(RUNS, kind="spatial_multisection_harmony_fixed", run_id=run_id, config=context) as run:
        first, record = run_fixed_harmony(pca, sections)
        second, repeat = run_fixed_harmony(pca, sections)
        if not np.array_equal(first, second) or record["objective_harmony"] != repeat["objective_harmony"]:
            raise RuntimeError("Fixed-10 Harmony repeatability gate failed")
        record.update({
            "cell_ids_sha256": canonical_hash(metadata["cell_id"].astype(str).tolist()),
            "pca_sha256": array_hash(pca),
            "section_labels_sha256": canonical_hash(sections.tolist()),
            "repeat_embedding_sha256": repeat["embedding_sha256"],
            "bitwise_repeat_passed": True,
            "native_stop_role": "sensitivity_only",
            "runtime": runtime,
        })
        np.save(run.artifact_path("harmony_fixed10.npy"), first, allow_pickle=False)
        run.write_json("harmony_record.json", record)
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        if source_snapshot() != sources:
            raise RuntimeError("Scientific source changed during fixed-Harmony execution")
        run.manifest.update(
            scientific_experiment=True,
            experiment_role="primary_six_section_fixed10_harmony_baseline",
            training_performed=False,
            scoring_performed=False,
        )
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
