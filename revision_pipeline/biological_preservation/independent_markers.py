# Purpose: Run the independent external-marker preservation panel on saved embeddings.
# Author: Ariana Rahman (Arizona State University)

"""Run the independent external-marker preservation panel on saved embeddings.

The command has two deliberately separate phases.  ``--preflight`` resolves
and verifies every input without calculating a neighbour graph.  ``--execute``
repeats that complete preflight before it publishes any scientific result.

Marker-foundation adapter
=========================
The evaluator consumes a completed run of kind
``independent_marker_foundation``.  That run must contain ``foundation.json``
with this versioned interface::

    {
      "schema": "genorefine.independent_marker_foundation.v1",
      "selection_boundary": {
        "result_independent": true,
        "evaluated_embedding_or_score_used": false
      },
      "datasets": {
        "hpcb": {
          "marker_scores_path": "datasets/hpcb/marker_scores.npy",
          "cell_ids_path": "datasets/hpcb/cell_ids.json",
          "reference_labels_path": "datasets/hpcb/reference_labels.json",
          "class_names_path": "datasets/hpcb/class_names.json",
          "metadata_path": "datasets/hpcb/metadata.json",
          "shape": [16382, 12],
          "dtype": "float64",
          "marker_scores_sha256": "...",
          "cell_order_sha256": "..."
        }
      }
    }

All named files must be registered in the foundation run manifest.  Metadata
must identify the dataset, panel content hash, source receipts, cell count and
class count.  This small adapter is isolated in :func:`load_marker_foundation`
so the preparation implementation cannot leak into representation evaluation.

Each representation is evaluated in a content-addressed immutable unit run.
Completed matching units are reused after full integrity validation, allowing
an interrupted panel to resume without recalculating finished neighbours.  A
panel run is published atomically only after all 88 expected units succeed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

import numpy as np

from ..audit import source_paths
from ..data.readers import array_hash
from ..data.store import EmbeddingView, Store, read_json
from ..evaluate.inputs import load_refined_bundle
from ..integrity import (alignment_indices, canonical_hash, file_fingerprint,
                         project_path, validate_cell_ids)
from ..runs import RunDirectory
from .marker_expression import (evaluate_external_marker_neighbors,
                                exact_nonself_neighbors)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "revision_pipeline/configs/independent_marker_preservation_v1.json"
DEFAULT_STORE = "revision_pipeline/runs/20260916T221434Z-a0884ffbd933"
FOUNDATION_SCHEMA = "genorefine.independent_marker_foundation.v1"
UNIT_KIND = "independent_marker_preservation_unit"
PANEL_KIND = "independent_marker_preservation_panel"
METHOD_ORDER = {"upstream": 0, "GenoRefine": 1, "IDEC": 2}


def _embedding_id(backbone: str) -> str:
    """Map the manuscript display spelling to the frozen Store identifier."""
    return "Online_iNMF" if backbone == "Online iNMF" else backbone


@dataclass(frozen=True)
class FoundationDataset:
    dataset_id: str
    cell_ids: tuple[str, ...]
    reference_labels: np.ndarray
    class_names: tuple[str, ...]
    marker_scores: np.ndarray
    receipt: dict


@dataclass(frozen=True)
class MarkerFoundation:
    path: Path
    datasets: Mapping[str, FoundationDataset]
    receipt: dict


@dataclass(frozen=True)
class Representation:
    unit_id: str
    case_id: str
    dataset_id: str
    dataset_display: str
    backbone: str
    method: str
    seed: int | None
    source_type: str
    source_run: Path
    bundle_path: Path | None
    receipt: dict
    cluster_score_run: Path | None = None
    cluster_receipt: dict | None = None


@dataclass(frozen=True)
class ProtocolPlan:
    root: Path
    config: dict
    config_path: Path
    config_receipt: dict
    store_path: Path
    store_receipt: dict
    main_index_path: Path
    main_index_receipt: dict
    foundation: MarkerFoundation
    representations: tuple[Representation, ...]
    source_tree_sha256: str = ""


def _read(path: Path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _portable_path(root: Path, value) -> Path:
    """Resolve an existing project path recorded by Windows or WSL."""
    if isinstance(value, Mapping):
        value = value.get("path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Recorded path must be a nonempty string")
    root = root.resolve()
    normalized = value.replace("\\", "/")
    prefix = "<PROJECT_ROOT>/"
    if normalized == "<PROJECT_ROOT>":
        return root
    relative = normalized[len(prefix):] if normalized.startswith(prefix) else normalized
    path = project_path(root, relative)
    return path


def _relative(root: Path, path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Scientific input lies outside the project: {path}") from error


def _registered_artifact(run_path: Path, manifest: Mapping, relative: str) -> Path:
    expected = manifest.get("artifacts", {}).get(relative)
    if expected is None:
        raise ValueError(f"Run does not register required artifact: {relative}")
    path = project_path(run_path, relative)
    if file_fingerprint(path) != expected:
        raise ValueError(f"Run artifact integrity failure: {relative}")
    return path


def _validate_run(run_path: Path, allowed_kinds: set[str], required: Sequence[str]) -> dict:
    run_path = Path(run_path).resolve()
    manifest_path = run_path / "run.json"
    manifest = _read(manifest_path)
    if manifest.get("status") != "succeeded" or manifest.get("kind") not in allowed_kinds:
        raise ValueError(f"Incomplete or unexpected source run: {run_path}")
    config_path = _registered_artifact(run_path, manifest, "config.json")
    if canonical_hash(_read(config_path)) != manifest.get("config_sha256"):
        raise ValueError(f"Source configuration digest mismatch: {run_path}")
    for relative in required:
        _registered_artifact(run_path, manifest, relative)
    return manifest


def _validate_protocol_config(config: Mapping) -> dict:
    required = {
        "protocol_id", "role", "scope", "datasets", "backbones",
        "representations_per_case", "replicate_seeds", "main_benchmark_index",
        "independent_idec_prefix", "normalization", "gene_matching",
        "minimum_matched_marker_genes", "minimum_class_cells_for_macro_metrics",
        "neighbor_count_nonself", "neighbor_search", "primary_endpoints",
        "secondary_endpoints", "panel_selection", "annotation_use", "inference",
        "success_policy", "expected_representation_evaluations",
    }
    if set(config) != required:
        raise ValueError("Independent-marker protocol fields differ from the versioned schema")
    if config["protocol_id"] != "independent_marker_preservation_v1":
        raise ValueError("Unsupported independent-marker protocol")
    datasets = config["datasets"]
    if not isinstance(datasets, list) or [x.get("id") for x in datasets] != ["hpcb", "mouse_senis"]:
        raise ValueError("Protocol must retain the declared HP-CB and Mouse order")
    if config["backbones"] != ["Scanorama", "Harmony", "Seurat", "Online iNMF"]:
        raise ValueError("Protocol backbones or their declared order changed")
    if config["representations_per_case"] != {"upstream": 1, "GenoRefine": 5, "IDEC": 5}:
        raise ValueError("Each case must have one upstream and five paired outputs per refiner")
    seeds = config["replicate_seeds"]
    if seeds != [0, 1, 2, 3, 4] or len(set(seeds)) != len(seeds):
        raise ValueError("The five prespecified replicate seeds are required")
    if config["expected_representation_evaluations"] != 88:
        raise ValueError("The protocol must require exactly 88 representation evaluations")
    search = config["neighbor_search"]
    expected_search = {
        "distance": "euclidean",
        "distance_precision": "float64",
        "tie_break": "distance_then_canonical_cell_index",
        "self_exclusion": "by_canonical_cell_identity",
        "row_permutation_invariance_required": True,
    }
    if search != expected_search or config["neighbor_count_nonself"] != 30:
        raise ValueError("Primary exact-neighbour policy changed")
    return dict(config)


def load_marker_foundation(path: Path, expected_dataset_ids: Sequence[str],
                           *, minimum_class_cells: int) -> MarkerFoundation:
    """Load and fully validate the isolated prepared-marker adapter."""
    path = Path(path).resolve()
    manifest = _validate_run(path, {"independent_marker_foundation"},
                             ["foundation.json"])
    document = _read(path / "foundation.json")
    if set(document) != {"schema", "selection_boundary", "datasets"}:
        raise ValueError("Unexpected marker-foundation fields")
    if document["schema"] != FOUNDATION_SCHEMA:
        raise ValueError("Unsupported marker-foundation schema")
    if document["selection_boundary"] != {
            "result_independent": True,
            "evaluated_embedding_or_score_used": False}:
        raise ValueError("Marker foundation does not assert result-independent selection")
    entries = document["datasets"]
    if not isinstance(entries, Mapping) or list(entries) != list(expected_dataset_ids):
        raise ValueError("Marker-foundation dataset coverage or order differs from protocol")
    result = {}
    dataset_receipts = {}
    required_fields = {
        "marker_scores_path", "cell_ids_path", "reference_labels_path",
        "class_names_path", "metadata_path", "shape", "dtype",
        "marker_scores_sha256", "cell_order_sha256",
    }
    for dataset_id in expected_dataset_ids:
        spec = entries[dataset_id]
        if not isinstance(spec, Mapping) or set(spec) != required_fields:
            raise ValueError(f"Malformed marker-foundation entry: {dataset_id}")
        paths = {}
        for key in ("marker_scores_path", "cell_ids_path", "reference_labels_path",
                    "class_names_path", "metadata_path"):
            relative = spec[key]
            if not isinstance(relative, str):
                raise ValueError("Foundation artifact paths must be strings")
            paths[key] = _registered_artifact(path, manifest, relative)
        ids = tuple(validate_cell_ids(_read(paths["cell_ids_path"])))
        classes = tuple(_read(paths["class_names_path"]))
        labels = np.asarray(_read(paths["reference_labels_path"]), dtype=str)
        scores = np.load(paths["marker_scores_path"], mmap_mode="r", allow_pickle=False)
        if (scores.ndim != 2 or scores.dtype.kind != "f" or not np.isfinite(scores).all()
                or tuple(scores.shape) != tuple(spec["shape"])
                or scores.dtype.name != spec["dtype"]
                or array_hash(scores) != spec["marker_scores_sha256"]):
            raise ValueError(f"Invalid marker-score matrix: {dataset_id}")
        if len(ids) != scores.shape[0] or len(labels) != len(ids):
            raise ValueError(f"Foundation rows are not aligned: {dataset_id}")
        if canonical_hash(list(ids)) != spec["cell_order_sha256"]:
            raise ValueError(f"Foundation cell-order digest differs: {dataset_id}")
        if (not classes or len(classes) != scores.shape[1]
                or len(set(classes)) != len(classes)
                or any(not isinstance(x, str) or not x.strip() for x in classes)):
            raise ValueError(f"Invalid marker classes: {dataset_id}")
        counts = Counter(map(str, labels))
        for class_name in classes:
            if counts[class_name] < 1 or counts[class_name] == len(ids):
                raise ValueError(f"Marker class lacks a valid one-vs-rest stratum: {class_name!r}")
        metadata = _read(paths["metadata_path"])
        expected_metadata = {"dataset_id", "panel_content_sha256", "source_receipts",
                             "reference_label_policy", "cells", "classes"}
        if (set(metadata) != expected_metadata or metadata["dataset_id"] != dataset_id
                or metadata["cells"] != len(ids) or metadata["classes"] != len(classes)
                or not re.fullmatch(r"[0-9a-f]{64}", metadata["panel_content_sha256"])
                or not isinstance(metadata["source_receipts"], Mapping)
                or not metadata["source_receipts"]):
            raise ValueError(f"Malformed foundation metadata: {dataset_id}")
        receipt = {
            "entry_sha256": canonical_hash(spec),
            "panel_content_sha256": metadata["panel_content_sha256"],
            "metadata": file_fingerprint(paths["metadata_path"]),
            "marker_scores": file_fingerprint(paths["marker_scores_path"]),
            "cell_ids": file_fingerprint(paths["cell_ids_path"]),
            "reference_labels": file_fingerprint(paths["reference_labels_path"]),
            "class_names": file_fingerprint(paths["class_names_path"]),
        }
        result[dataset_id] = FoundationDataset(
            dataset_id, ids, labels, classes, scores, receipt)
        dataset_receipts[dataset_id] = receipt
    return MarkerFoundation(
        path, result,
        {"run_manifest": file_fingerprint(path / "run.json"),
         "foundation": file_fingerprint(path / "foundation.json"),
         "datasets": dataset_receipts})


def deterministic_unit_id(case_id: str, method: str, seed: int | None) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", case_id):
        raise ValueError("Invalid case ID")
    if method not in METHOD_ORDER:
        raise ValueError("Unknown representation method")
    if method == "upstream":
        if seed is not None:
            raise ValueError("Upstream representation must not have a seed")
        suffix = "upstream"
    else:
        if type(seed) is not int or seed < 0:
            raise ValueError("Refined representation requires a nonnegative seed")
        suffix = ("genorefine" if method == "GenoRefine" else "idec") + f"_s{seed}"
    return f"{case_id}__{suffix}"


def enumerate_unit_records(cases: Sequence[Mapping], seeds: Sequence[int]) -> list[dict]:
    """Pure coverage enumerator used by the real preflight and mocked tests."""
    if list(seeds) != [0, 1, 2, 3, 4]:
        raise ValueError("Five declared seeds in canonical order are required")
    if len(cases) != 8:
        raise ValueError("Exactly eight primary dataset/backbone cases are required")
    seen_cases, rows = set(), []
    for case in cases:
        required = {"case_id", "dataset_id", "dataset_display", "backbone",
                    "upstream", "genorefine", "idec"}
        if set(case) != required:
            raise ValueError("Malformed resolved case")
        case_id = case["case_id"]
        if case_id in seen_cases:
            raise ValueError(f"Duplicate case: {case_id}")
        seen_cases.add(case_id)
        if set(case["genorefine"]) != set(seeds) or set(case["idec"]) != set(seeds):
            raise ValueError(f"Missing or unexpected seed in case {case_id}")
        rows.append({
            "unit_id": deterministic_unit_id(case_id, "upstream", None),
            "case_id": case_id, "dataset_id": case["dataset_id"],
            "dataset_display": case["dataset_display"], "backbone": case["backbone"],
            "method": "upstream", "seed": None, "source": case["upstream"],
        })
        for method, key in (("GenoRefine", "genorefine"), ("IDEC", "idec")):
            for seed in seeds:
                rows.append({
                    "unit_id": deterministic_unit_id(case_id, method, seed),
                    "case_id": case_id, "dataset_id": case["dataset_id"],
                    "dataset_display": case["dataset_display"], "backbone": case["backbone"],
                    "method": method, "seed": seed, "source": case[key][seed],
                })
    identifiers = [row["unit_id"] for row in rows]
    if len(rows) != 88 or len(set(identifiers)) != 88:
        raise ValueError("Resolved representation panel is not exactly 88 unique units")
    return rows


def _source_training_index(root: Path, display: str, record: Mapping) -> tuple[dict, dict, Path]:
    source = _portable_path(root, record)
    if file_fingerprint(source / "run.json") != record.get("manifest"):
        raise ValueError(f"Final-table source receipt mismatch: {display}")
    if display == "HP-CB/Scanorama":
        _validate_run(source, {"step4a_scoring_completion_verification"},
                      ["summary_recomputed.json"])
        audit_config = _read(source / "config.json")
        panel = _portable_path(root, audit_config["panel"])
        panel_manifest = _validate_run(panel, {"step4a_scoring_panel"}, ["run_index.json"])
        panel_config = _read(panel / "config.json")
        training_panel = _portable_path(root, panel_config["scoring"]["training_panel"])
        _validate_run(training_panel, {"step4a_training_panel"}, ["run_index.json"])
        training_index = _read(training_panel / "run_index.json")
        scoring_index = _read(panel / "run_index.json")
        index = {"training": training_index["training"], "scoring": scoring_index}
        case = {"id": "hp_scanorama", "dataset": "hpcb", "embedding": "Scanorama"}
        receipt = {
            "final_source": file_fingerprint(source / "run.json"),
            "scoring_panel": file_fingerprint(panel / "run.json"),
            "training_panel": file_fingerprint(training_panel / "run.json"),
            "scoring_panel_kind": panel_manifest["kind"],
        }
        return case, index, training_panel
    manifest = _validate_run(
        source, {"main_benchmark_case", "main_benchmark_fast_gpu_case"},
        ["run_index.json"])
    case = _read(source / "config.json")["case"]
    index = _read(source / "run_index.json")
    receipt = {"final_source": file_fingerprint(source / "run.json"),
               "source_kind": manifest["kind"]}
    return case, index, source


def _validate_cluster_score(path: Path, canonical_ids: Sequence[str],
                            expected_values_sha256: str) -> dict:
    """Validate the common fixed-resolution, Leiden-seed-0 partition source."""
    manifest = _validate_run(
        path,
        {"step4a_representation_scoring", "main_benchmark_score",
         "independent_idec_scoring"},
        ["evaluation/selected.json", "evaluation/partitions.npy",
         "evaluation/cell_ids.json"])
    ids = tuple(validate_cell_ids(_read(path / "evaluation/cell_ids.json")))
    if set(ids) != set(canonical_ids):
        raise ValueError(f"Cluster-score cells differ from representation: {path}")
    score_config = _read(path / "config.json")
    if score_config.get("input", {}).get("values_sha256") != expected_values_sha256:
        raise ValueError(f"Cluster partition was scored from a different representation: {path}")
    selected = _read(path / "evaluation/selected.json")
    candidates = [
        item for item in selected
        if item.get("profile") == "primary_exact_v1"
        and item.get("purpose") == "primary_evaluation"
        and float(item.get("resolution", -1)) == 0.5
        and item.get("leiden_seed") == 0
        and item.get("selection_label_informed") is False
    ]
    if len(candidates) != 1:
        raise ValueError(f"Exactly one label-free fixed-0.5 seed-0 partition is required: {path}")
    partitions = np.load(path / "evaluation/partitions.npy", mmap_mode="r", allow_pickle=False)
    partition_index = candidates[0].get("partition_index")
    if (partitions.ndim != 2 or partitions.shape[1] != len(ids)
            or type(partition_index) is not int or not 0 <= partition_index < partitions.shape[0]
            or not np.issubdtype(partitions.dtype, np.integer)):
        raise ValueError(f"Invalid cluster partition array: {path}")
    return {
        "run_manifest": file_fingerprint(path / "run.json"),
        "selected": file_fingerprint(path / "evaluation/selected.json"),
        "partitions": file_fingerprint(path / "evaluation/partitions.npy"),
        "cell_ids": file_fingerprint(path / "evaluation/cell_ids.json"),
        "partition_index": partition_index,
        "partition_policy": "primary_exact_v1_fixed_resolution_0.5_leiden_seed_0",
        "run_kind": manifest["kind"],
    }


def _validate_training_descriptor(path: Path, *, dataset_id: str, embedding_id: str,
                                  case_id: str, seed: int, method: str,
                                  expected_parent: Mapping, canonical_ids: Sequence[str]) -> tuple[dict, EmbeddingView]:
    if method == "GenoRefine":
        allowed = {"step4a_paired_training", "main_benchmark_training"}
    elif method == "IDEC":
        allowed = {"independent_idec_training"}
    else:
        raise ValueError("Training descriptor is only for refined representations")
    manifest = _validate_run(path, allowed, [
        "bundles/joint/bundle.json", "bundles/joint/cell_ids.json",
        "bundles/joint/values.npy"])
    config = _read(path / "config.json")
    if method == "IDEC":
        observed_case = config.get("case", {})
        observed_seed = config.get("seed")
    else:
        observed_case = config.get("case", config.get("panel", {}))
        observed_seed = config.get("effective_refiner", {}).get("training", {}).get("replicate_seed")
    if (observed_case.get("id", case_id) != case_id
            or observed_case.get("dataset") != dataset_id
            or observed_case.get("embedding") != embedding_id
            or observed_seed != seed):
        raise ValueError(f"Training provenance differs for {case_id}/{method}/seed {seed}")
    bundle = path / "bundles/joint"
    view = load_refined_bundle(
        bundle, expected_parent=expected_parent, output_cell_ids=canonical_ids)
    values = np.asarray(view.values)
    if values.ndim != 2 or values.shape[0] != len(canonical_ids) or not np.isfinite(values).all():
        raise ValueError("Refined representation is not a finite aligned matrix")
    receipt = {
        "run_manifest": file_fingerprint(path / "run.json"),
        "bundle_manifest": file_fingerprint(bundle / "bundle.json"),
        "values": file_fingerprint(bundle / "values.npy"),
        "cell_ids": file_fingerprint(bundle / "cell_ids.json"),
        "values_sha256": array_hash(values),
        "shape": list(values.shape),
        "training_label_use": view.metadata["training_label_use"],
        "run_kind": manifest["kind"],
    }
    return receipt, view


def build_plan(root: Path, config_path: Path, foundation_path: Path,
               *, store_path: Path | None = None) -> ProtocolPlan:
    """Resolve and verify all 88 representations before any metric is run."""
    root = Path(root).resolve()
    source_tree_sha256 = canonical_hash(_source_snapshot(root))
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_path(root, config_path.as_posix())
    config = _validate_protocol_config(_read(config_path))
    config_receipt = file_fingerprint(config_path)
    store_path = (Path(store_path).resolve() if store_path is not None
                  else project_path(root, DEFAULT_STORE))
    store = Store(store_path)
    store.verify()
    store_receipt = file_fingerprint(store_path / "run.json")
    final_path = _portable_path(root, config["main_benchmark_index"])
    final_run = final_path.parent
    final_manifest = _validate_run(final_run, {"main_benchmark_final_table"},
                                   [final_path.name])
    if final_manifest["artifacts"][final_path.name] != file_fingerprint(final_path):
        raise ValueError("Main benchmark index differs from its completed run")
    final_index = _read(final_path)
    dataset_ids = [item["id"] for item in config["datasets"]]
    foundation = load_marker_foundation(
        foundation_path, dataset_ids,
        minimum_class_cells=config["minimum_class_cells_for_macro_metrics"])
    seeds = config["replicate_seeds"]
    resolved_cases = []
    case_context = {}
    for dataset_spec in config["datasets"]:
        dataset_id, dataset_display = dataset_spec["id"], dataset_spec["display"]
        dataset = store.dataset(dataset_id)
        parent_ids = tuple(dataset.cell_ids)
        prepared = foundation.datasets[dataset_id]
        if prepared.cell_ids != parent_ids:
            raise ValueError(f"Foundation is not in canonical Store order: {dataset_id}")
        for backbone in config["backbones"]:
            embedding_id = _embedding_id(backbone)
            display = f"{dataset_display}/{backbone}"
            if display not in final_index:
                raise ValueError(f"Main benchmark is missing required case: {display}")
            case, index, provenance_path = _source_training_index(
                root, display, final_index[display])
            if case.get("dataset") != dataset_id or case.get("embedding") != embedding_id:
                raise ValueError(f"Case identity mismatch: {display}")
            case_id = case["id"]
            parent = store.embedding(dataset_id, embedding_id)
            parent_values = np.asarray(parent.values)
            if (parent.cell_ids != parent_ids or parent_values.ndim != 2
                    or not np.isfinite(parent_values).all()):
                raise ValueError(f"Invalid upstream embedding: {display}")
            parent_reference = parent.parent_reference()
            scoring = index.get("scoring")
            if not isinstance(scoring, Mapping):
                raise ValueError(f"Scoring index is absent: {display}")
            baseline_score = _portable_path(root, scoring.get("baseline"))
            upstream_hash = array_hash(parent_values)
            upstream_cluster_receipt = _validate_cluster_score(
                baseline_score, parent_ids, upstream_hash)
            upstream_receipt = {
                "run_manifest": store_receipt,
                "values_sha256": upstream_hash,
                "shape": list(parent_values.shape),
                "parent_reference": parent_reference,
            }
            training = index.get("training")
            if not isinstance(training, Mapping) or set(training) != set(map(str, seeds)):
                raise ValueError(f"GenoRefine seed coverage differs: {display}")
            gr_sources, idec_sources = {}, {}
            for seed in seeds:
                gr_path = _portable_path(root, training[str(seed)])
                gr_receipt, _ = _validate_training_descriptor(
                    gr_path, dataset_id=dataset_id, embedding_id=embedding_id,
                    case_id=case_id, seed=seed, method="GenoRefine",
                    expected_parent=parent_reference, canonical_ids=parent_ids)
                gr_sources[seed] = {"path": gr_path, "receipt": gr_receipt}
                gr_score = _portable_path(root, scoring.get(f"joint_{seed}"))
                gr_sources[seed]["cluster_score_path"] = gr_score
                gr_sources[seed]["cluster_receipt"] = _validate_cluster_score(
                    gr_score, parent_ids, gr_receipt["values_sha256"])
                idec_path = root / "revision_pipeline" / "runs" / (
                    f"{config['independent_idec_prefix']}-{case_id}-s{seed}")
                idec_receipt, _ = _validate_training_descriptor(
                    idec_path, dataset_id=dataset_id, embedding_id=embedding_id,
                    case_id=case_id, seed=seed, method="IDEC",
                    expected_parent=parent_reference, canonical_ids=parent_ids)
                idec_score = root / "revision_pipeline" / "runs" / (
                    f"{config['independent_idec_prefix']}-{case_id}-s{seed}-joint-score")
                idec_sources[seed] = {
                    "path": idec_path, "receipt": idec_receipt,
                    "cluster_score_path": idec_score,
                    "cluster_receipt": _validate_cluster_score(
                        idec_score, parent_ids, idec_receipt["values_sha256"]),
                }
            resolved_cases.append({
                "case_id": case_id, "dataset_id": dataset_id,
                "dataset_display": dataset_display, "backbone": backbone,
                "upstream": {"path": store_path, "receipt": upstream_receipt,
                             "cluster_score_path": baseline_score,
                             "cluster_receipt": upstream_cluster_receipt},
                "genorefine": gr_sources, "idec": idec_sources,
            })
            case_context[case_id] = {"parent_reference": parent_reference,
                                     "canonical_ids": parent_ids,
                                     "provenance_path": provenance_path}
    rows = enumerate_unit_records(resolved_cases, seeds)
    representations = []
    for row in rows:
        source = row.pop("source")
        source_path = Path(source["path"]).resolve()
        representations.append(Representation(
            **row,
            source_type="store_embedding" if row["method"] == "upstream" else "refined_bundle",
            source_run=source_path,
            bundle_path=None if row["method"] == "upstream" else source_path / "bundles/joint",
            receipt=source["receipt"],
            cluster_score_run=Path(source["cluster_score_path"]).resolve(),
            cluster_receipt=source["cluster_receipt"],
        ))
    if len(representations) != config["expected_representation_evaluations"]:
        raise ValueError("Resolved representation count differs from the frozen protocol")
    if canonical_hash(_source_snapshot(root)) != source_tree_sha256:
        raise RuntimeError("Pipeline source/config changed during preflight")
    return ProtocolPlan(
        root=root, config=config, config_path=config_path,
        config_receipt=config_receipt, store_path=store_path,
        store_receipt=store_receipt, main_index_path=final_path,
        main_index_receipt=file_fingerprint(final_path), foundation=foundation,
        representations=tuple(representations), source_tree_sha256=source_tree_sha256)


def preflight_summary(plan: ProtocolPlan) -> dict:
    methods = Counter(item.method for item in plan.representations)
    cases = {(item.dataset_id, item.backbone) for item in plan.representations}
    return {
        "passed": True,
        "protocol_id": plan.config["protocol_id"],
        "datasets": [item["id"] for item in plan.config["datasets"]],
        "backbones": plan.config["backbones"],
        "cases": len(cases),
        "representation_evaluations": len(plan.representations),
        "method_counts": dict(methods),
        "unique_unit_ids": len({item.unit_id for item in plan.representations}),
        "all_sources_complete_aligned_finite_and_parent_bound": True,
        "scientific_metrics_computed": False,
    }


def _load_representation(plan: ProtocolPlan, item: Representation) -> tuple[np.ndarray, tuple[str, ...]]:
    store = Store(plan.store_path)
    parent = store.embedding(item.dataset_id, _embedding_id(item.backbone))
    canonical = tuple(parent.cell_ids)
    if item.method == "upstream":
        values = np.asarray(parent.values)
    else:
        view = load_refined_bundle(
            item.bundle_path, expected_parent=parent.parent_reference(),
            output_cell_ids=canonical)
        values = np.asarray(view.values)
    if (values.ndim != 2 or values.shape[0] != len(canonical)
            or not np.isfinite(values).all()
            or array_hash(values) != item.receipt["values_sha256"]):
        raise ValueError(f"Representation changed after preflight: {item.unit_id}")
    return values, canonical


def _default_evaluator(*, marker_scores, class_names, reference_labels,
                       embedding, cell_ids, k, working_memory_mb,
                       cluster_labels=None):
    neighbors, distances = exact_nonself_neighbors(
        embedding, cell_ids, cell_ids, k, working_memory_mb=working_memory_mb)
    result = evaluate_external_marker_neighbors(
        marker_scores, class_names, reference_labels, neighbors, cell_ids,
        cluster_labels=cluster_labels)
    result["neighbor_search"] = {
        "method": "exact_blocked_euclidean",
        "distance_arithmetic": "float64_direct_with_float64_reranking",
        "self_exclusion": "cell_identity",
        "tie_break": "distance_then_canonical_cell_index",
        "working_memory_mb": int(working_memory_mb),
        "minimum_distance": float(np.min(distances)),
        "maximum_distance": float(np.max(distances)),
    }
    return result, neighbors, distances


def _unit_configuration(plan: ProtocolPlan, item: Representation,
                        *, working_memory_mb: int) -> dict:
    return {
        "schema": "genorefine.independent_marker_unit.v1",
        "source_tree_sha256": plan.source_tree_sha256,
        "protocol": plan.config_receipt,
        "main_benchmark_index": plan.main_index_receipt,
        "store_manifest": plan.store_receipt,
        "foundation": plan.foundation.receipt,
        "foundation_dataset": plan.foundation.datasets[item.dataset_id].receipt,
        "representation": {
            "unit_id": item.unit_id, "case_id": item.case_id,
            "dataset_id": item.dataset_id, "dataset_display": item.dataset_display,
            "backbone": item.backbone, "method": item.method, "seed": item.seed,
            "source_type": item.source_type,
            "source_run": _relative(plan.root, item.source_run),
            "cluster_score_run": (_relative(plan.root, item.cluster_score_run)
                                  if item.cluster_score_run is not None else None),
            "receipt": item.receipt,
            "cluster_receipt": item.cluster_receipt,
        },
        "evaluation": {
            "k_nonself": plan.config["neighbor_count_nonself"],
            "working_memory_mb": int(working_memory_mb),
            "primary_endpoints": plan.config["primary_endpoints"],
            "all_cells_no_subsampling": True,
        },
    }


def _unit_run_id(item: Representation, unit_config: Mapping) -> str:
    method = item.method.lower().replace("genorefine", "gr")
    seed = "base" if item.seed is None else f"s{item.seed}"
    return f"imarker-{item.case_id}-{method}-{seed}-{canonical_hash(unit_config)[:12]}"


def _validate_reusable_run(path: Path, kind: str, expected_config: Mapping,
                           required: Sequence[str]) -> dict:
    manifest = _validate_run(path, {kind}, required)
    observed = _read(path / "config.json")
    if observed != expected_config or manifest["config_sha256"] != canonical_hash(expected_config):
        raise ValueError(f"Existing run has a different configuration: {path}")
    return manifest


def execute_unit(plan: ProtocolPlan, item: Representation, runs_root: Path,
                 *, working_memory_mb: int = 256,
                 evaluator: Callable = _default_evaluator,
                 representation_loader: Callable | None = None) -> tuple[Path, bool]:
    """Evaluate or integrity-check and reuse one content-addressed unit."""
    if type(working_memory_mb) is not int or working_memory_mb < 1:
        raise ValueError("working_memory_mb must be a positive integer")
    unit_config = _unit_configuration(plan, item, working_memory_mb=working_memory_mb)
    run_id = _unit_run_id(item, unit_config)
    final = Path(runs_root).resolve() / run_id
    if final.exists():
        _validate_reusable_run(final, UNIT_KIND, unit_config,
                               ["result.json", "neighbors.npz", "source_receipts.json"])
        result = _read(final / "result.json")
        if result.get("unit_id") != item.unit_id:
            raise ValueError("Reusable unit result identity differs")
        return final, True
    loader = representation_loader or (lambda p, i: _load_representation(p, i))
    values, ids = loader(plan, item)
    prepared = plan.foundation.datasets[item.dataset_id]
    if tuple(ids) != prepared.cell_ids:
        order = alignment_indices(prepared.cell_ids, ids)
        values = np.ascontiguousarray(np.asarray(values)[order])
        ids = prepared.cell_ids
    cluster_labels = None
    if item.cluster_score_run is not None:
        observed_ids = tuple(validate_cell_ids(_read(
            item.cluster_score_run / "evaluation/cell_ids.json")))
        selected = _read(item.cluster_score_run / "evaluation/selected.json")
        partition_index = item.cluster_receipt["partition_index"]
        selected_match = [x for x in selected if x.get("partition_index") == partition_index]
        if len(selected_match) != 1:
            raise ValueError("Cluster partition selection changed after preflight")
        partitions = np.load(item.cluster_score_run / "evaluation/partitions.npy",
                             mmap_mode="r", allow_pickle=False)
        cluster_labels = np.asarray(partitions[partition_index])
        cluster_labels = cluster_labels[alignment_indices(ids, observed_ids)]
    result, neighbors, distances = evaluator(
        marker_scores=prepared.marker_scores, class_names=prepared.class_names,
        reference_labels=prepared.reference_labels, embedding=values,
        cell_ids=ids, k=plan.config["neighbor_count_nonself"],
        working_memory_mb=working_memory_mb, cluster_labels=cluster_labels)
    if isinstance(result.get("per_class"), Mapping):
        minimum = plan.config["minimum_class_cells_for_macro_metrics"]
        eligible = {
            name: row for name, row in result["per_class"].items()
            if int(row["positive_cells"]) >= minimum
        }
        if not eligible:
            raise ValueError("No external-marker class satisfies the macro-metric cell threshold")
        result["macro_all_retained_classes"] = result["macro"]
        metric_names = (
            "neighbor_marker_auroc", "neighbor_marker_average_precision",
            "average_precision_over_prevalence", "neighbor_marker_contrast")
        result["macro"] = {
            metric: float(np.mean([row[metric] for row in eligible.values()],
                                  dtype=np.float64))
            for metric in metric_names
        }
        result["macro_class_policy"] = {
            "minimum_positive_cells": minimum,
            "eligible_classes": sorted(eligible),
            "excluded_rare_classes": sorted(set(result["per_class"]) - set(eligible)),
            "per_class_metrics_still_reported_for_every_retained_class": True,
        }
        # Keep the evaluator's descriptive fields synchronized with the
        # protocol-level threshold used for the published macro.  The default
        # evaluator initially calculates every retained class (minimum=1) so
        # that rare-class endpoints remain available; the protocol then
        # replaces only the macro with its prespecified >=10-cell reduction.
        # Leaving the initial descriptors in place would make the metadata
        # contradict the otherwise-correct macro values.
        result["minimum_class_cells_for_macro"] = minimum
        result["macro_eligible_classes"] = sorted(eligible)
        result["macro_excluded_classes"] = sorted(
            set(result["per_class"]) - set(eligible))
    neighbors = np.asarray(neighbors)
    distances = np.asarray(distances, dtype=np.float64)
    expected_shape = (len(ids), plan.config["neighbor_count_nonself"])
    if (neighbors.shape != expected_shape or not np.issubdtype(neighbors.dtype, np.integer)
            or distances.shape != expected_shape or not np.isfinite(distances).all()):
        raise ValueError("Evaluator returned invalid neighbour artifacts")
    payload = {
        "schema": "genorefine.independent_marker_result.v1",
        "unit_id": item.unit_id, "case_id": item.case_id,
        "dataset_id": item.dataset_id, "dataset_display": item.dataset_display,
        "backbone": item.backbone, "method": item.method, "seed": item.seed,
        "metrics": result,
    }
    with RunDirectory(runs_root, kind=UNIT_KIND, config=unit_config,
                      run_id=run_id) as run:
        run.write_json("result.json", payload)
        output = run.artifact_path("neighbors.npz")
        np.savez_compressed(output, indices=neighbors.astype(np.int32, copy=False),
                            distances=distances)
        run.write_json("source_receipts.json", {
            "representation": item.receipt,
            "cluster_partition": item.cluster_receipt,
            "foundation": prepared.receipt,
            "config": plan.config_receipt,
            "main_benchmark_index": plan.main_index_receipt,
        })
        run.manifest.update(
            scientific_experiment=True,
            experiment_role="independent_external_marker_preservation_unit",
            source_tree_sha256=plan.source_tree_sha256)
    return run.final_path, False


def _summarize_units(unit_paths: Sequence[Path]) -> tuple[list[dict], list[dict]]:
    records = [_read(path / "result.json") for path in unit_paths]
    groups = defaultdict(list)
    for record in records:
        macro = record["metrics"]["macro"]
        groups[(record["dataset_display"], record["backbone"], record["method"])].append(macro)
    summaries = []
    endpoints = ("neighbor_marker_auroc", "neighbor_marker_average_precision",
                 "neighbor_marker_contrast")
    for key in sorted(groups, key=lambda x: (x[0], x[1], METHOD_ORDER[x[2]])):
        values = groups[key]
        row = {"dataset": key[0], "backbone": key[1], "method": key[2],
               "representations": len(values)}
        for endpoint in endpoints:
            scores = [float(item[endpoint]) for item in values]
            row[endpoint + "_mean"] = float(np.mean(scores, dtype=np.float64))
            row[endpoint + "_minimum"] = min(scores)
            row[endpoint + "_maximum"] = max(scores)
        summaries.append(row)
    return records, summaries


def execute_panel(plan: ProtocolPlan, runs_root: Path, run_id: str,
                  *, working_memory_mb: int = 256,
                  evaluator: Callable = _default_evaluator,
                  representation_loader: Callable | None = None) -> tuple[Path, dict]:
    """Publish all units and then atomically publish the complete panel."""
    if len(plan.representations) != 88:
        raise ValueError("Refusing to execute an incomplete representation plan")
    if (plan.source_tree_sha256
            and canonical_hash(_source_snapshot(plan.root)) != plan.source_tree_sha256):
        raise RuntimeError("Pipeline source/config changed after preflight")
    runs_root = Path(runs_root).resolve()
    unit_paths, reused = [], []
    for index, item in enumerate(plan.representations, 1):
        print(f"[{index}/88] {item.unit_id}", flush=True)
        path, was_reused = execute_unit(
            plan, item, runs_root, working_memory_mb=working_memory_mb,
            evaluator=evaluator, representation_loader=representation_loader)
        unit_paths.append(path)
        reused.append(was_reused)
    unit_index = {
        item.unit_id: {
            "path": _relative(plan.root, path),
            "manifest": file_fingerprint(path / "run.json"),
        }
        for item, path in zip(plan.representations, unit_paths)
    }
    panel_config = {
        "schema": "genorefine.independent_marker_panel.v1",
        "source_tree_sha256": plan.source_tree_sha256,
        "protocol": plan.config_receipt,
        "foundation": plan.foundation.receipt,
        "main_benchmark_index": plan.main_index_receipt,
        "unit_index_sha256": canonical_hash(unit_index),
        "expected_units": 88,
        "working_memory_mb": int(working_memory_mb),
    }
    if (plan.source_tree_sha256
            and canonical_hash(_source_snapshot(plan.root)) != plan.source_tree_sha256):
        raise RuntimeError("Pipeline source/config changed during unit evaluation")
    final = runs_root / run_id
    if final.exists():
        _validate_reusable_run(final, PANEL_KIND, panel_config,
                               ["unit_index.json", "results.json", "summary.json", "report.md"])
        if _read(final / "unit_index.json") != unit_index:
            raise ValueError("Completed panel points to a different unit set")
        return final, {"reused_panel": True, "reused_units": sum(reused),
                       "computed_units": 0}
    records, summaries = _summarize_units(unit_paths)
    with RunDirectory(runs_root, kind=PANEL_KIND, config=panel_config,
                      run_id=run_id) as run:
        run.write_json("unit_index.json", unit_index)
        run.write_json("results.json", {row["unit_id"]: row for row in records})
        run.write_json("summary.json", {
            "complete": True, "units": len(records), "cases": 8,
            "datasets": 2, "backbones": 4,
            "methods": {"upstream": 8, "GenoRefine": 40, "IDEC": 40},
            "computed_units_this_invocation": int(sum(not x for x in reused)),
            "reused_units_this_invocation": int(sum(reused)),
            "rows": summaries,
            "interpretation_boundary": (
                "External, result-independent marker sources test expression-defined "
                "neighborhood preservation. Dataset annotation names supply a "
                "prespecified cell-type crosswalk; cell-level labels are used only "
                "after marker scoring as evaluation strata."),
        })
        lines = [
            "# Independent external-marker preservation panel", "",
            "All 88 prespecified representation evaluations completed.", "",
            f"New units: {sum(not x for x in reused)}; integrity-checked reused units: {sum(reused)}.", "",
            "| Dataset | Backbone | Representation | marker AUROC | marker AP | marker contrast |",
            "|---|---|---|---:|---:|---:|",
        ]
        for row in summaries:
            lines.append(
                f"| {row['dataset']} | {row['backbone']} | {row['method']} | "
                f"{row['neighbor_marker_auroc_mean']:.4f} | "
                f"{row['neighbor_marker_average_precision_mean']:.4f} | "
                f"{row['neighbor_marker_contrast_mean']:.4f} |")
        lines.extend([
            "",
            "External sources, source filters, gene matching, minimum marker count, "
            "neighbor rule, and scoring thresholds were fixed without reading an "
            "evaluated embedding or result. Dataset annotation names supplied the "
            "prespecified crosswalk to external cell types; cell-level label "
            "assignments were used only after marker scoring as evaluation strata.",
            "",
        ])
        run.artifact_path("report.md").write_text("\n".join(lines), encoding="utf-8")
        run.manifest.update(
            scientific_experiment=True,
            experiment_role="independent_external_marker_preservation_full_panel",
            source_tree_sha256=plan.source_tree_sha256)
    return run.final_path, {"reused_panel": False, "reused_units": sum(reused),
                            "computed_units": sum(not x for x in reused)}


def _source_snapshot(root: Path) -> dict:
    return {path.relative_to(root).as_posix(): file_fingerprint(path)
            for path in source_paths(root)}


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--foundation-run", type=Path, required=True)
    parser.add_argument("--store-run", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--working-memory-mb", type=int, default=256)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute and not args.run_id:
        parser.error("--run-id is required with --execute")
    plan = build_plan(ROOT, Path(args.config), args.foundation_run,
                      store_path=args.store_run)
    if args.preflight:
        print(json.dumps(preflight_summary(plan), indent=2, sort_keys=True), flush=True)
        return
    output, status = execute_panel(
        plan, ROOT / "revision_pipeline/runs", args.run_id,
        working_memory_mb=args.working_memory_mb)
    if canonical_hash(_source_snapshot(ROOT)) != plan.source_tree_sha256:
        raise RuntimeError("Pipeline source/config changed during scientific evaluation")
    print(json.dumps({"output": str(output), **status}, indent=2), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "FoundationDataset", "MarkerFoundation", "Representation", "ProtocolPlan",
    "load_marker_foundation", "deterministic_unit_id", "enumerate_unit_records",
    "build_plan", "preflight_summary", "execute_unit", "execute_panel", "main",
]
