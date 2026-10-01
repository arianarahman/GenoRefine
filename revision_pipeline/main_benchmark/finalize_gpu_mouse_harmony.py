# Purpose: Finalize the completed fast-GPU Mouse Harmony representation panel.
# Author: Ariana Rahman (Arizona State University)

"""Finalize the completed fast-GPU Mouse Harmony representation panel.

This is a provenance/audit step only. It does not train, score, retry, replace,
or select any scientific result.
"""
import argparse
from pathlib import Path

from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import completed, read, snapshot
from ..runs import RunDirectory
from .common import ROOT, case_spec, load_inputs, names_for
from .reporting import (
    load_results,
    summarize,
    verify_partitions,
    write_case_report,
)


RUNS = ROOT / "revision_pipeline/runs"
INPUTS = RUNS / "20260920T000704Z-94d830df9d5f-mouse_harmony-inputs"
BASELINE = RUNS / "20260920T000704Z-94d830df9d5f-mouse_harmony-baseline"
SCIENCE_SOURCE = "12a6f85667cdd4822388a1377a169bc3dd22c11093741917f9bc15c05ab7ff58"


def training_path(seed):
    if seed == 2:
        return RUNS / "20260920T053700Z-mouse_harmony-s2-fastgpu-final"
    return RUNS / f"20260920T060000Z-mouse_harmony-s{seed}-fastgpu-final"


def score_path(name):
    if name == "baseline":
        return BASELINE
    if name == "joint_2":
        return RUNS / "20260920T054100Z-mouse_harmony-joint2-fastgpu-score"
    if name.startswith("joint_"):
        seed = name.rsplit("_", 1)[1]
        return RUNS / f"20260920T061800Z-mouse_harmony-joint{seed}-fastgpu-score"
    slug = name.replace("_", "")
    return RUNS / f"20260920T063000Z-mouse_harmony-{slug}-fastgpu-score"


def verify_training(seed, path, expected_case):
    record = completed(path, "main_benchmark_training")
    config = read(path / "config.json")
    runtime = read(path / "runtime.json")
    if record.get("source_tree_sha256") != SCIENCE_SOURCE:
        raise ValueError(f"Training source mismatch for seed {seed}")
    if config.get("inputs") != str(INPUTS).replace("C:\\", "/mnt/c/").replace("\\", "/"):
        raise ValueError(f"Training input mismatch for seed {seed}")
    if config.get("case") != expected_case:
        raise ValueError(f"Training case mismatch for seed {seed}")
    if config["effective_refiner"]["training"]["replicate_seed"] != seed:
        raise ValueError(f"Training seed mismatch for seed {seed}")
    if runtime.get("runtime_profile") != "fast_gpu" or runtime.get("training_backend") != "compiled_gpu":
        raise ValueError(f"Training runtime mismatch for seed {seed}")
    return {
        "path": str(path),
        "manifest": file_fingerprint(path / "run.json"),
        "wall_seconds": record["wall_seconds"],
        "runtime_profile": runtime["runtime_profile"],
        "training_backend": runtime["training_backend"],
    }


def verify_score(name, path):
    record = completed(path, "main_benchmark_score")
    config = read(path / "config.json")
    provenance = read(path / "input.json")
    if config.get("name") != name or config.get("inputs") != str(INPUTS).replace("C:\\", "/mnt/c/").replace("\\", "/"):
        raise ValueError(f"Score binding mismatch for {name}")
    if name != "baseline" and record.get("source_tree_sha256") != SCIENCE_SOURCE:
        raise ValueError(f"Score source mismatch for {name}")
    if name in {"first32", "pca32"}:
        if provenance.get("training_label_use") != "label_free_transform_only":
            raise ValueError(f"Control provenance mismatch for {name}")
    elif name != "baseline":
        stage, seed_text = name.rsplit("_", 1)
        seed = int(seed_text)
        if provenance.get("stage") != stage or provenance.get("seed") != seed:
            raise ValueError(f"Stage/seed provenance mismatch for {name}")
        if Path(provenance["training_run"]).name != training_path(seed).name:
            raise ValueError(f"Training-run provenance mismatch for {name}")
    return {
        "path": str(path),
        "manifest": file_fingerprint(path / "run.json"),
        "wall_seconds": record["wall_seconds"],
        "source_tree_sha256": record.get("source_tree_sha256"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize-mouse-harmony-gpu-panel", action="store_true")
    args = parser.parse_args()
    if not args.finalize_mouse_harmony_gpu_panel:
        parser.error("Explicit finalization flag is required")

    case = case_spec("mouse_harmony")
    parent, dataset, context = load_inputs(INPUTS)
    if context["case"] != case:
        raise ValueError("Input case mismatch")
    expected_names = names_for(parent.values.shape[1])
    paths = {name: score_path(name) for name in expected_names}
    if set(paths) != set(expected_names):
        raise ValueError("Incomplete score path map")

    training = {str(seed): verify_training(seed, training_path(seed), case) for seed in range(5)}
    scoring = {name: verify_score(name, path) for name, path in paths.items()}
    results = load_results({name: str(path) for name, path in paths.items()})
    summary = summarize(results, parent.values.shape[1], len(set(dataset.batch_labels())))
    partitions = verify_partitions(
        {name: str(path) for name, path in paths.items()},
        dataset.reference_partition()[0],
    )
    sources = snapshot(ROOT)
    config = {
        "case": case,
        "inputs": str(INPUTS),
        "purpose": "Post-run aggregation and independent saved-partition rescore only",
        "scientific_source_tree_sha256": SCIENCE_SOURCE,
        "baseline_source_bridge": read(training_path(0) / "k_selection.json")["metadata_recovery_source_bridge"],
        "no_retraining_or_rescoring": True,
    }
    with RunDirectory(RUNS, kind="main_benchmark_fast_gpu_case", config=config) as run:
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.manifest["scientific_experiment"] = False
        run.write_json("run_index.json", {
            "inputs": str(INPUTS),
            "baseline": str(BASELINE),
            "training": training,
            "scoring": scoring,
        })
        run.write_json("summary.json", summary)
        run.write_json("partition_rescoring.json", partitions)
        write_case_report(
            run,
            case,
            summary,
            scope="Verified exploratory scores; 810 saved partitions independently rescored",
        )
        run.write_json("completion.json", {
            "status": "complete_exploratory_mouse_harmony_panel",
            "training_seeds": 5,
            "scored_representations": len(paths),
            "saved_partitions_independently_rescored": partitions["independently_rescored_saved_partitions"],
            "no_efficacy_based_selection": True,
            "independent_biological_validation_established": False,
        })
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
