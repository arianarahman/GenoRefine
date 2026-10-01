# Purpose: Prepare immutable clean/corrupted inputs for one artifact-validation case.
# Author: Ariana Rahman (Arizona State University)

"""Prepare immutable clean/corrupted inputs for one artifact-validation case."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..data.readers import array_hash
from ..data.store import check_sources
from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import completed, snapshot
from ..runs import RunDirectory
from .artifacts import ArtifactBundle, batch_simplex_artifact, target_local_warp_artifact
from .common import (
    ARTIFACT_IDS,
    ROOT,
    SPEC,
    CaseInputs,
    case_spec,
    load_case_inputs,
    rare_group_registry,
    specification,
)


def _save_array(run: RunDirectory, relative: str, values: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(values)
    if array.dtype.kind == "O" or not np.issubdtype(array.dtype, np.number) and array.dtype.kind not in "US":
        raise ValueError(f"Prepared arrays must be numeric or fixed-width Unicode: {relative}")
    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
        raise ValueError(f"Prepared array contains nonfinite values: {relative}")
    path = run.artifact_path(relative)
    np.save(path, array, allow_pickle=False)
    saved = np.load(path, mmap_mode="r", allow_pickle=False)
    if saved.shape != array.shape or saved.dtype != array.dtype or not np.array_equal(saved, array):
        raise RuntimeError(f"Prepared array failed exact reload: {relative}")
    return {
        "path": relative,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "array_sha256": array_hash(array),
        "file": file_fingerprint(path),
        "reload_exact": True,
    }


def _artifact_config(spec: Mapping[str, Any], artifact_id: str) -> dict[str, Any]:
    matches = [item for item in spec["artifacts"] if item["id"] == artifact_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one fixed configuration for {artifact_id}")
    return dict(matches[0])


def generate_bundles(
    inputs: CaseInputs, spec: Mapping[str, Any]
) -> dict[str, ArtifactBundle]:
    """Generate both frozen artifacts without fitting or selecting a model."""
    global_config = _artifact_config(spec, "batch_simplex")
    local_config = _artifact_config(spec, "target_local_warp")
    common = {
        "cell_ids": inputs.cell_ids,
        "canonical_ids": inputs.cell_ids,
        "working_memory_mb": int(spec["working_memory_mb"]),
    }
    bundles = {
        "batch_simplex": batch_simplex_artifact(
            inputs.values,
            inputs.batches,
            strength=global_config["strength"],
            scale_k=global_config["scale_k_nonself"],
            **common,
        ),
        "target_local_warp": target_local_warp_artifact(
            inputs.values,
            inputs.batches,
            inputs.named_labels,
            inputs.case["target_label"],
            strength=local_config["strength"],
            scale_k=local_config["scale_k_same_class_nonself"],
            **common,
        ),
    }
    for artifact_id, bundle in bundles.items():
        expected = _artifact_config(spec, artifact_id)["implementation"]
        if (
            bundle.metadata.get("artifact") != expected
            or bundle.canonical_ids != inputs.cell_ids
            or not np.array_equal(bundle.clean, inputs.values)
            or not np.array_equal(bundle.observed, bundle.recovered_observed())
        ):
            raise RuntimeError(f"Generated artifact violates its frozen contract: {artifact_id}")
    return bundles


def _artifact_embedding_metadata(
    inputs: CaseInputs,
    artifact_id: str,
    bundle: ArtifactBundle,
    observed_record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **dict(bundle.metadata),
        "id": f"{inputs.case['backbone']}__{artifact_id}",
        "kind": "controlled_artifact_input",
        "dataset": inputs.case["dataset"],
        "dimensions": int(bundle.observed.shape[1]),
        "dataset_fingerprint": inputs.parent_reference["dataset_fingerprint"],
        "clean_parent_reference": inputs.parent_reference,
        "canonical_cell_order_sha256": canonical_hash(list(inputs.cell_ids)),
        "coordinate_names_sha256": canonical_hash(list(inputs.coordinate_names)),
        "stored_values_file_sha256": observed_record["file"]["sha256"],
        "selected_values_sha256": observed_record["array_sha256"],
        "label_use": "target_definition_and_posthoc_safeguards_only_not_training_K",
        "target_label_status": inputs.case["target_label_status"],
    }


def _persist_prepared(
    *,
    output_root: Path,
    run_id: str | None,
    spec: Mapping[str, Any],
    case: Mapping[str, Any],
    inputs: CaseInputs,
    bundles: Mapping[str, ArtifactBundle],
    sources: Mapping[str, Any],
) -> Path:
    """Persist one fully generated case.  Used directly by focused tests."""
    if set(bundles) != set(ARTIFACT_IDS) or inputs.case != dict(case):
        raise ValueError("Prepared case must contain exactly both declared artifacts")
    rare = rare_group_registry(
        inputs.named_labels,
        inputs.reference_codes,
        inputs.batches,
        fraction_max=spec["rare_fraction_max"],
    )
    run_config = {
        "protocol_id": spec["protocol_id"],
        "specification": file_fingerprint(SPEC) if Path(SPEC).exists() else None,
        "case": dict(case),
        "clean_parent_reference": inputs.parent_reference,
        "artifacts": list(ARTIFACT_IDS),
        "training_K_policy": spec["training_K_policy"],
        "training_K_computed": False,
    }
    with RunDirectory(
        output_root,
        kind="artifact_validation_v2_prepared_case",
        config=run_config,
        run_id=run_id,
    ) as run:
        run.write_json("source_manifest.json", dict(sources))
        run.manifest["source_tree_sha256"] = canonical_hash(dict(sources))
        run.write_json(
            "source_receipt.json",
            {
                "store": case.get("store", spec.get("store")),
                "store_manifest": case.get("store_manifest", spec.get("store_manifest")),
                "main_benchmark_case": case.get("case_run"),
                "main_benchmark_case_manifest": case.get("case_manifest"),
                "dataset_source_files": inputs.dataset_source_files,
                "clean_parent_reference": inputs.parent_reference,
                "all_checked_before_generation": True,
                "all_checked_after_generation": True,
            },
        )
        run.write_json("case.json", dict(case))
        run.write_json("cell_ids.json", list(inputs.cell_ids))
        run.write_json("coordinate_names.json", list(inputs.coordinate_names))
        run.write_json("annotation_policy.json", inputs.annotation_policy)
        run.write_json("rare_groups.json", rare)
        run.write_json(
            "k_policy.json",
            {
                **spec["training_K_policy"],
                "training_K_computed": False,
                "reference_codes_not_consumed_for_K": True,
            },
        )
        shared = {
            "named_labels": _save_array(run, "named_labels.npy", inputs.named_labels),
            "reference_codes": _save_array(
                run, "reference_codes.npy", inputs.reference_codes
            ),
            "batches": _save_array(run, "batches.npy", inputs.batches),
        }
        artifact_records: dict[str, Any] = {}
        for artifact_id in ARTIFACT_IDS:
            bundle = bundles[artifact_id]
            prefix = f"artifacts/{artifact_id}"
            arrays = {
                "clean": _save_array(run, f"{prefix}/clean.npy", bundle.clean),
                "observed": _save_array(run, f"{prefix}/observed.npy", bundle.observed),
                "counterfactuals": _save_array(
                    run, f"{prefix}/counterfactuals.npy", bundle.counterfactuals
                ),
                "batch_index": _save_array(
                    run, f"{prefix}/batch_index.npy", bundle.batch_index
                ),
            }
            levels_path = f"{prefix}/batch_levels.json"
            if not all(isinstance(level, str) for level in bundle.batch_levels):
                raise ValueError("Current dataset batch levels must be explicit strings")
            run.write_json(
                levels_path,
                {
                    "values": list(bundle.batch_levels),
                    "count": len(bundle.batch_levels),
                    "ordering": "deterministic_type_and_repr_sort",
                },
            )
            metadata = _artifact_embedding_metadata(
                inputs, artifact_id, bundle, arrays["observed"]
            )
            metadata_path = f"{prefix}/artifact_metadata.json"
            run.write_json(metadata_path, metadata)
            parent_reference = {
                "protocol_id": spec["protocol_id"],
                "case_id": case["id"],
                "artifact_id": artifact_id,
                "dataset_fingerprint": inputs.parent_reference["dataset_fingerprint"],
                "embedding_id": metadata["id"],
                "stored_values_file_sha256": arrays["observed"]["file"]["sha256"],
                "selected_values_sha256": arrays["observed"]["array_sha256"],
                "cell_order_sha256": canonical_hash(list(inputs.cell_ids)),
                "coordinate_names_sha256": canonical_hash(list(inputs.coordinate_names)),
                "artifact_metadata_sha256": canonical_hash(metadata),
                "clean_parent_reference": inputs.parent_reference,
                "clean_values_sha256": arrays["clean"]["array_sha256"],
                "counterfactual_values_sha256": arrays["counterfactuals"][
                    "array_sha256"
                ],
                "shape": list(bundle.observed.shape),
                "observed_reproduced_exactly_from_counterfactuals": True,
            }
            artifact_records[artifact_id] = {
                "arrays": arrays,
                "batch_levels_path": levels_path,
                "artifact_metadata_path": metadata_path,
                "parent_reference": parent_reference,
            }
        record = {
            "schema_version": 1,
            "protocol_id": spec["protocol_id"],
            "specification": file_fingerprint(SPEC),
            "case_id": case["id"],
            "case": dict(case),
            "canonical_IDs_sha256": canonical_hash(list(inputs.cell_ids)),
            "coordinate_names_sha256": canonical_hash(list(inputs.coordinate_names)),
            "shared_arrays": shared,
            "rare_group_count": len(rare["eligible_non_singleton_groups"]),
            "artifacts": artifact_records,
            "replicate_seeds_reserved_for_downstream_training": spec["replicate_seeds"],
            "training_K_policy": spec["training_K_policy"],
            "training_K_computed": False,
            "no_model_fit_or_scoring_in_preparation": True,
        }
        run.write_json("prepared_record.json", record)
        if sources:
            if snapshot(ROOT) != dict(sources):
                raise ValueError("Pipeline source changed while persisting prepared inputs")
            check_sources(ROOT, inputs.dataset_source_files)
        run.manifest.update(
            scientific_experiment=False,
            experiment_role="deterministic_input_preparation_only",
        )
    return run.final_path


def prepare_case(case_id: str, *, run_id: str | None = None) -> Path:
    spec = specification()
    case = case_spec(case_id)
    sources = snapshot(ROOT)
    case_run = Path(ROOT / case["case_run"])
    completed(case_run)
    inputs = load_case_inputs(case_id, verify_original_sources=True)
    bundles = generate_bundles(inputs, spec)
    # Recheck every original dataset source after the most expensive operation.
    check_sources(ROOT, inputs.dataset_source_files)
    if snapshot(ROOT) != sources:
        raise ValueError("Pipeline source changed during artifact preparation")
    path = _persist_prepared(
        output_root=ROOT / "revision_pipeline/runs",
        run_id=run_id,
        spec=spec,
        case=case,
        inputs=inputs,
        bundles=bundles,
        sources=sources,
    )
    if snapshot(ROOT) != sources:
        raise ValueError("Pipeline source changed before prepared input publication")
    check_sources(ROOT, inputs.dataset_source_files)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=[case["id"] for case in specification()["cases"]])
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Preparation requires explicit --execute")
    path = prepare_case(args.case, run_id=args.run_id)
    print(path, flush=True)


if __name__ == "__main__":
    main()
