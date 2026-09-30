"""Train one prespecified pooled Harmony+GenoRefine replicate."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import os
from pathlib import Path
import platform

from ..data.store import EmbeddingView
from ..integrity import canonical_hash, file_fingerprint
from ..main_benchmark.fast_runtime import configure
from ..refine.config import LayoutConfig, RefinerConfig
from ..runs import RunDirectory
from ..step4_policy import planned_training_config
from .common import RUNS, load_fixed_harmony, load_k_selection, source_snapshot, specification


def execute(harmony: Path, k_selection: Path, seed: int, run_id: str, runtime_profile: str) -> Path:
    spec = specification()
    if seed not in spec["genorefine"]["seeds"]:
        raise ValueError("Unplanned GenoRefine seed")
    values, metadata, harmony_record = load_fixed_harmony(harmony)
    decision = load_k_selection(k_selection, harmony)["pooled"]
    runtime = configure(runtime_profile)
    locked_runtime = spec["genorefine"]["runtime"]
    observed_runtime = {
        "image_id": os.environ.get("GENOREFINE_IMAGE_ID"),
        "python": platform.python_version(),
        "tensorflow_runtime": runtime["tensorflow"],
        "packages": {name: version(name) for name in locked_runtime["packages"]},
        "cuda_visible": any("GPU" in device for device in runtime["visible_devices"]),
    }
    expected_runtime = {
        "image_id": locked_runtime["image_id"],
        "python": locked_runtime["python"],
        "tensorflow_runtime": locked_runtime["tensorflow_runtime"],
        "packages": locked_runtime["packages"],
        "cuda_visible": locked_runtime["cuda_required"],
    }
    if runtime_profile != "fast_gpu" or observed_runtime != expected_runtime:
        raise RuntimeError(f"GenoRefine runtime lock mismatch: {observed_runtime}")
    runtime["spatial_panel_runtime_lock"] = {**observed_runtime, "validated": True,
                                             "image_tag": locked_runtime["image_tag"]}
    from ..step4.train import fit_paired
    sources = source_snapshot()
    ids = tuple(metadata["cell_id"].astype(str))
    features = [f"Harmony{i + 1}" for i in range(values.shape[1])]
    parent = EmbeddingView(values, ids, {
        "id": "dlpfc_six_section_harmony_fixed10",
        "dataset_fingerprint": canonical_hash({
            "foundation": spec["foundation"]["run_manifest_sha256"],
            "harmony": harmony_record["embedding_sha256"],
        }),
        "stored_values_file_sha256": file_fingerprint(Path(harmony) / "harmony_fixed10.npy")["sha256"],
        "coordinate_names": features,
        "training_label_use": "none",
    })
    training = planned_training_config(len(ids), int(decision["n_clusters"]), seed)
    config = RefinerConfig(
        training=training,
        layout=LayoutConfig(
            requested_side=spec["genorefine"]["layout_side"],
            scaling=spec["genorefine"]["layout_scaling"],
            transport_iterations=spec["genorefine"]["transport_iterations"],
            epsilon=spec["genorefine"]["transport_epsilon"],
        ),
    )
    context = {
        "protocol_id": spec["protocol_id"],
        "harmony_manifest": file_fingerprint(Path(harmony) / "run.json"),
        "k_selection_manifest": file_fingerprint(Path(k_selection) / "run.json"),
        "K_binding": decision,
        "seed": seed,
        "runtime_profile": runtime_profile,
        "effective_refiner": config.to_dict(),
        "reported_stage": "joint",
    }
    with RunDirectory(RUNS, kind="spatial_multisection_genorefine_training", run_id=run_id, config=context) as run:
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.write_json("runtime.json", runtime)
        run.write_json("k_selection.json", decision)
        run.write_json("input.json", {
            "parent_reference": parent.parent_reference(),
            "cell_ids": list(ids),
            "feature_ids": features,
            "row_order": "foundation canonical pooled order",
            "training_label_use": "none",
            "manual_layer_labels_available_only_to_later_scoring": True,
        })
        fit_paired(parent, config, run)
        if source_snapshot() != sources:
            raise RuntimeError("Scientific source changed during GenoRefine training")
        peak = None
        try:
            import resource
            peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        except (ImportError, AttributeError):
            pass
        run.manifest.update(
            scientific_experiment=True,
            experiment_role="pooled_six_section_harmony_plus_genorefine",
            peak_memory_bytes=peak,
            peak_memory_status="whole_process_Linux_RSS" if peak else "not_measured",
            training_performed=True,
            scoring_performed=False,
        )
    return run.final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmony", type=Path, required=True)
    parser.add_argument("--k-selection", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=range(5), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runtime-profile", choices=["deterministic_cpu", "fast_cpu", "fast_gpu"], default="fast_gpu")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required")
    print(execute(args.harmony, args.k_selection, args.seed, args.run_id, args.runtime_profile), flush=True)


if __name__ == "__main__":
    main()
