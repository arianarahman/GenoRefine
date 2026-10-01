# Purpose: Frozen contracts and loaders for the real-embedding artifact panel.
# Author: Ariana Rahman (Arizona State University)

"""Frozen contracts and loaders for the real-embedding artifact panel."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from ..data.store import Store, check_sources
from ..integrity import canonical_hash, file_fingerprint, project_path, validate_cell_ids
from ..pilot.common import completed, read


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "revision_pipeline/configs/artifact_validation_v2.json"
SPEC_SHA256 = "07673e7b8b6112c7cd59041dea57e16ad086d89bd3870abf1e4b447bf49043be"
ARTIFACT_IDS = ("batch_simplex", "target_local_warp")
CASE_IDS = (
    "pan_scanorama",
    "pan_harmony",
    "hp_scanorama",
    "hp_harmony",
    "mouse_scanorama",
    "mouse_harmony",
)


def _fingerprint_matches(path: Path, expected: Mapping[str, Any], role: str) -> None:
    if file_fingerprint(path) != dict(expected):
        raise ValueError(f"Pinned {role} changed: {path}")


def specification() -> dict[str, Any]:
    """Load and validate the complete, immutable six-case protocol."""
    if file_fingerprint(SPEC)["sha256"] != SPEC_SHA256:
        raise ValueError("Artifact protocol changed; create a new version instead of refreshing pins")
    spec = read(SPEC)
    expected_keys = {
        "schema_version",
        "protocol_id",
        "role",
        "preregistered",
        "store",
        "store_manifest",
        "main_benchmark_index",
        "main_benchmark_index_fingerprint",
        "datasets",
        "backbones",
        "artifacts",
        "replicate_seeds",
        "working_memory_mb",
        "rare_fraction_max",
        "training_K_policy",
        "cases",
        "failure_policy",
        "claim_boundary",
    }
    if (
        set(spec) != expected_keys
        or spec["schema_version"] != 1
        or spec["protocol_id"] != "artifact_validation_v2"
        or spec["preregistered"] is not False
        or spec["datasets"] != ["pancreas_five_study", "hpcb", "mouse_senis"]
        or spec["backbones"] != ["Scanorama", "Harmony"]
        or spec["replicate_seeds"] != [0, 1, 2, 3, 4]
        or [item.get("id") for item in spec["artifacts"]] != list(ARTIFACT_IDS)
        or [case.get("id") for case in spec["cases"]] != list(CASE_IDS)
        or len(spec["cases"]) != 6
    ):
        raise ValueError("Undeclared artifact-validation scope")
    expected_pairs = [
        (dataset, backbone)
        for dataset in spec["datasets"]
        for backbone in spec["backbones"]
    ]
    observed_pairs = [(case["dataset"], case["backbone"]) for case in spec["cases"]]
    if observed_pairs != expected_pairs or len(set(observed_pairs)) != 6:
        raise ValueError("Cases must be the exact three-dataset by two-backbone product")
    if spec["working_memory_mb"] < 1 or not 0 < spec["rare_fraction_max"] < 1:
        raise ValueError("Invalid fixed preparation resources/policy")
    policy = spec["training_K_policy"]
    if (
        policy.get("rule") != "median_baseline_leiden_count"
        or policy.get("resolution") != 0.5
        or policy.get("leiden_seeds") != [0, 1, 2]
        or policy.get("reference_labels_used") is not False
        or policy.get("preparation_status")
        != "metadata_only_not_computed_until_training_binding"
    ):
        raise ValueError("Training K must remain label-free and uncomputed in preparation")

    store_path = project_path(ROOT, spec["store"])
    _fingerprint_matches(store_path / "run.json", spec["store_manifest"], "store manifest")
    index_path = project_path(ROOT, spec["main_benchmark_index"])
    _fingerprint_matches(
        index_path, spec["main_benchmark_index_fingerprint"], "main benchmark index"
    )
    index = read(index_path)
    for case in spec["cases"]:
        case_store = project_path(ROOT, case.get("store", spec["store"]))
        case_store_manifest = case.get("store_manifest", spec["store_manifest"])
        _fingerprint_matches(
            case_store / "run.json", case_store_manifest, f"{case['id']} store manifest"
        )
        run_path = project_path(ROOT, case["case_run"])
        _fingerprint_matches(run_path / "run.json", case["case_manifest"], "case manifest")
        run_record = read(run_path / "run.json")
        if run_record.get("status") != "succeeded":
            raise ValueError("Pinned clean-baseline case did not complete")
        display = f"{case['display_dataset']}/{case['backbone']}"
        if display not in index:
            raise ValueError(f"Main benchmark index is missing {display}")
        indexed_name = PurePosixPath(index[display]["path"]).name
        if indexed_name != run_path.name or index[display]["manifest"] != case["case_manifest"]:
            raise ValueError(f"Main benchmark index binding changed for {display}")
    return spec


def case_spec(case_id: str) -> dict[str, Any]:
    if not isinstance(case_id, str):
        raise ValueError("Case ID must be a string")
    for case in specification()["cases"]:
        if case["id"] == case_id:
            return case
    raise ValueError(f"Case outside artifact_validation_v2: {case_id!r}")


def coordinate_names(case: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        f"{case['backbone']}_coordinate_{index:03d}" for index in range(case["shape"][1])
    )


@dataclass(frozen=True)
class CaseInputs:
    case: dict[str, Any]
    values: np.ndarray
    cell_ids: tuple[str, ...]
    coordinate_names: tuple[str, ...]
    named_labels: np.ndarray
    reference_codes: np.ndarray
    batches: np.ndarray
    annotation_policy: dict[str, Any]
    parent_reference: dict[str, Any]
    dataset_source_files: dict[str, Any]


def load_case_inputs(case_id: str, *, verify_original_sources: bool = True) -> CaseInputs:
    """Load one exact clean baseline and its canonical metadata from the store."""
    spec, case = specification(), case_spec(case_id)
    store = Store(project_path(ROOT, case.get("store", spec["store"])))
    dataset = store.dataset(case["dataset"])
    if verify_original_sources:
        check_sources(ROOT, dataset.record["source_files"])
    embedding = store.embedding(case["dataset"], case["backbone"])
    if (
        embedding.parent_reference() != case["parent_reference"]
        or list(embedding.values.shape) != case["shape"]
        or embedding.cell_ids != dataset.cell_ids
    ):
        raise ValueError("Clean baseline values/order/provenance differ from the frozen case")
    label_key = dataset.record["loader"]["label_key"]
    raw_labels = np.asarray(dataset.record["obs"][label_key], dtype=str)
    reference_codes, _ = dataset.reference_partition()
    batches = np.asarray(dataset.batch_labels(), dtype=str)
    if (
        raw_labels.shape != (case["shape"][0],)
        or reference_codes.shape != raw_labels.shape
        or batches.shape != raw_labels.shape
        or case["target_label"] not in set(raw_labels.tolist())
    ):
        raise ValueError("Dataset labels/batches/target do not align to the clean baseline")
    target_mask = raw_labels == case["target_label"]
    if set(batches[target_mask].tolist()) != set(batches.tolist()):
        raise ValueError("Target-local counterfactual requires target cells in every batch")
    values = np.asarray(embedding.values, dtype=np.float64, order="C")
    values.flags.writeable = False
    raw_labels.flags.writeable = False
    reference_codes = np.asarray(reference_codes, dtype=np.int64)
    reference_codes.flags.writeable = False
    batches.flags.writeable = False
    if verify_original_sources:
        check_sources(ROOT, dataset.record["source_files"])
    return CaseInputs(
        case=case,
        values=values,
        cell_ids=tuple(embedding.cell_ids),
        coordinate_names=coordinate_names(case),
        named_labels=raw_labels,
        reference_codes=reference_codes,
        batches=batches,
        annotation_policy=dataset.annotation_policy,
        parent_reference=embedding.parent_reference(),
        dataset_source_files=dict(dataset.record["source_files"]),
    )


def rare_group_registry(
    labels: np.ndarray,
    reference_codes: np.ndarray,
    batches: np.ndarray,
    *,
    fraction_max: float,
) -> dict[str, Any]:
    labels = np.asarray(labels)
    codes = np.asarray(reference_codes)
    batch = np.asarray(batches)
    if (
        labels.ndim != 1
        or codes.shape != labels.shape
        or codes.dtype.kind not in "iu"
        or batch.shape != labels.shape
        or not 0 < fraction_max < 1
    ):
        raise ValueError("Invalid rare-group registry inputs")
    records, excluded_singletons = [], []
    for code in sorted(np.unique(codes).tolist()):
        selected = codes == code
        raw = sorted(set(labels[selected].astype(str).tolist()))
        if len(raw) != 1:
            raise ValueError("Reference code does not map to exactly one supplied label")
        cells = int(selected.sum())
        record = {
            "reference_code": int(code),
            "supplied_label": raw[0],
            "cells": cells,
            "fraction": cells / len(labels),
            "batches": int(len(np.unique(batch[selected]))),
            "screen_key": str(code),
        }
        if cells / len(labels) <= fraction_max:
            if cells > 1:
                records.append(record)
            else:
                excluded_singletons.append(record)
    return {
        "definition": "full_cohort_supplied_reference_group_fraction_at_or_below_threshold",
        "fraction_max": fraction_max,
        "eligible_non_singleton_groups": records,
        "excluded_singletons": excluded_singletons,
        "all_cells_retained_as_queries_for_each_eligible_group": True,
        "labels_used_only_for_posthoc_preservation_safeguards_not_training_or_K": True,
    }


@dataclass(frozen=True)
class PreparedArtifactView:
    artifact_id: str
    values: np.ndarray
    clean: np.ndarray
    counterfactuals: np.ndarray
    batch_index: np.ndarray
    batch_levels: tuple[str, ...]
    cell_ids: tuple[str, ...]
    coordinate_names: tuple[str, ...]
    metadata: dict[str, Any]
    _parent_reference: dict[str, Any]

    def parent_reference(self) -> dict[str, Any]:
        return dict(self._parent_reference)


@dataclass(frozen=True)
class PreparedCase:
    path: Path
    case: dict[str, Any]
    record: dict[str, Any]
    cell_ids: tuple[str, ...]
    coordinate_names: tuple[str, ...]
    named_labels: np.ndarray
    reference_codes: np.ndarray
    batches: np.ndarray
    rare_groups: dict[str, Any]
    artifacts: Mapping[str, PreparedArtifactView]

    def artifact(self, artifact_id: str) -> PreparedArtifactView:
        if artifact_id not in self.artifacts:
            raise ValueError(f"Artifact is absent from prepared case: {artifact_id!r}")
        return self.artifacts[artifact_id]


def _load_array(
    path: Path,
    relative: str,
    expected: Mapping[str, Any],
    manifest_artifacts: Mapping[str, Any],
) -> np.ndarray:
    array_path = project_path(path, relative)
    # ``completed`` has already streamed and verified every file against this
    # immutable run manifest.  Compare the embedded array receipt to that
    # verified manifest rather than reading multi-gigabyte arrays a second time.
    if manifest_artifacts.get(relative) != expected["file"]:
        raise ValueError(f"Prepared array receipt differs from run manifest: {relative}")
    values = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if list(values.shape) != expected["shape"] or values.dtype.str != expected["dtype"]:
        raise ValueError(f"Prepared array shape/dtype changed: {relative}")
    values.flags.writeable = False
    return values


def load_prepared(path: Path | str) -> PreparedCase:
    """Load one immutable prepared case for downstream training/scoring."""
    path = Path(path).resolve()
    run_manifest = completed(path, "artifact_validation_v2_prepared_case")
    record = read(path / "prepared_record.json")
    spec = specification()
    case = case_spec(record.get("case_id"))
    if (
        record.get("protocol_id") != spec["protocol_id"]
        or record.get("case") != case
        or record.get("specification") != file_fingerprint(SPEC)
        or set(record.get("artifacts", {})) != set(ARTIFACT_IDS)
        or record.get("training_K_policy") != spec["training_K_policy"]
        or record.get("training_K_computed") is not False
    ):
        raise ValueError("Prepared case record differs from frozen protocol")
    cell_ids = tuple(validate_cell_ids(read(path / "cell_ids.json")))
    if canonical_hash(list(cell_ids)) != case["parent_reference"]["cell_order_sha256"]:
        raise ValueError("Prepared canonical IDs differ from clean parent")
    names = tuple(read(path / "coordinate_names.json"))
    if names != coordinate_names(case):
        raise ValueError("Prepared coordinate names differ")
    arrays = record["shared_arrays"]
    verified_files = run_manifest["artifacts"]
    named = _load_array(
        path, arrays["named_labels"]["path"], arrays["named_labels"], verified_files
    )
    codes = _load_array(
        path,
        arrays["reference_codes"]["path"],
        arrays["reference_codes"],
        verified_files,
    )
    batches = _load_array(
        path, arrays["batches"]["path"], arrays["batches"], verified_files
    )
    if any(len(values) != len(cell_ids) for values in (named, codes, batches)):
        raise ValueError("Prepared shared arrays are not cell aligned")
    rare = read(path / "rare_groups.json")
    views: dict[str, PreparedArtifactView] = {}
    for artifact_id in ARTIFACT_IDS:
        item = record["artifacts"][artifact_id]
        clean = _load_array(
            path,
            item["arrays"]["clean"]["path"],
            item["arrays"]["clean"],
            verified_files,
        )
        observed = _load_array(
            path,
            item["arrays"]["observed"]["path"],
            item["arrays"]["observed"],
            verified_files,
        )
        counterfactuals = _load_array(
            path,
            item["arrays"]["counterfactuals"]["path"],
            item["arrays"]["counterfactuals"],
            verified_files,
        )
        batch_index = _load_array(
            path,
            item["arrays"]["batch_index"]["path"],
            item["arrays"]["batch_index"],
            verified_files,
        )
        levels_record = read(path / item["batch_levels_path"])
        levels = tuple(levels_record["values"])
        metadata = read(path / item["artifact_metadata_path"])
        if (
            observed.shape != tuple(case["shape"])
            or clean.shape != observed.shape
            or counterfactuals.shape[1:] != observed.shape
            or batch_index.shape != (len(cell_ids),)
            or len(levels) != counterfactuals.shape[0]
            or not np.array_equal(
                counterfactuals[batch_index, np.arange(len(cell_ids), dtype=np.int64)],
                observed,
            )
            or item["parent_reference"]["selected_values_sha256"]
            != item["arrays"]["observed"]["array_sha256"]
            or item["parent_reference"]["artifact_metadata_sha256"]
            != canonical_hash(metadata)
            or metadata.get("coordinate_names_sha256")
            != canonical_hash(list(names))
            or metadata.get("canonical_cell_order_sha256")
            != canonical_hash(list(cell_ids))
        ):
            raise ValueError(f"Prepared artifact integrity/alignment failed: {artifact_id}")
        views[artifact_id] = PreparedArtifactView(
            artifact_id=artifact_id,
            values=observed,
            clean=clean,
            counterfactuals=counterfactuals,
            batch_index=batch_index,
            batch_levels=levels,
            cell_ids=cell_ids,
            coordinate_names=names,
            metadata=metadata,
            _parent_reference=item["parent_reference"],
        )
    return PreparedCase(
        path=path,
        case=case,
        record=record,
        cell_ids=cell_ids,
        coordinate_names=names,
        named_labels=named,
        reference_codes=codes,
        batches=batches,
        rare_groups=rare,
        artifacts=views,
    )


__all__ = [
    "ARTIFACT_IDS",
    "CASE_IDS",
    "ROOT",
    "SPEC",
    "CaseInputs",
    "PreparedArtifactView",
    "PreparedCase",
    "case_spec",
    "coordinate_names",
    "load_case_inputs",
    "load_prepared",
    "rare_group_registry",
    "specification",
]
