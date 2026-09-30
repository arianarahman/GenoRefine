"""Build the result-independent external-marker foundation for package 2."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..audit import source_paths
from ..data.readers import array_hash
from ..data.store import Store
from ..integrity import canonical_hash, file_fingerprint, project_path
from ..runs import RunDirectory
from .external_panels import (
    MarkerRecord,
    build_marker_panel,
    parse_cell_ontology,
    parse_cellmarker_mouse,
    parse_panglaodb_human,
)
from .marker_expression import prepare_external_marker_scores
from .mouse_official_labels import recover_official_mouse_labels
from .panel_io import write_panel
from .source_acquisition import load_source_locks, verify_locked_source


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "revision_pipeline/configs/independent_marker_foundation_v1.json"
FOUNDATION_SCHEMA = "genorefine.independent_marker_foundation.v1"


def _read(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _source_snapshot(root: Path) -> dict:
    return {path.relative_to(root).as_posix(): file_fingerprint(path)
            for path in source_paths(root)}


def _validate_config(value: Mapping) -> dict:
    required = {
        "protocol_id", "source_lock_config", "source_directory", "store_run",
        "minimum_exact_marker_genes", "minimum_reference_cells",
        "normalization", "datasets", "selection_boundary",
    }
    if set(value) != required or value["protocol_id"] != "independent_marker_foundation_v1":
        raise ValueError("Unsupported independent-marker foundation configuration")
    if [item.get("id") for item in value["datasets"]] != ["hpcb", "mouse_senis"]:
        raise ValueError("Foundation datasets or their canonical order changed")
    if (value["minimum_exact_marker_genes"] != 5
            or value["minimum_reference_cells"] != 10):
        raise ValueError("External-marker inclusion thresholds changed")
    expected_normalization = {
        "method": "library_size_log1p", "target_sum": 10000.0,
        "score_method": "zscore_mean", "arithmetic": "float64_sparse",
    }
    if value["normalization"] != expected_normalization:
        raise ValueError("External-marker normalization contract changed")
    if value["selection_boundary"] != {
            "result_independent": True,
            "evaluated_embedding_or_score_used": False,
            "reference_labels_used_only_for_fixed_crosswalk_and_evaluation_eligibility": True}:
        raise ValueError("Selection boundary changed")
    return dict(value)


def _verified_external_sources(root: Path, config: Mapping) -> dict[str, dict]:
    locks_path = project_path(root, config["source_lock_config"])
    directory = project_path(root, config["source_directory"])
    locks = load_source_locks(locks_path)
    records = {}
    for lock in locks:
        records[lock.source_id] = verify_locked_source(directory / lock.filename, lock)
    return records


def _load_expression(path: Path, source: str):
    import anndata as ad

    data = ad.read_h5ad(path, backed="r")
    try:
        cell_ids = list(map(str, data.obs_names))
        if source == "layers[counts]":
            values = data.layers["counts"][:]
            feature_names = list(map(str, data.var_names))
        elif source == "raw.X":
            if data.raw is None:
                raise ValueError("Mouse expression file has no raw.X")
            values = data.raw.X[:]
            feature_names = list(map(str, data.raw.var_names))
        else:
            raise ValueError(f"Unsupported expression source: {source}")
        obs = data.obs.copy()
    finally:
        data.file.close()
    return values, feature_names, cell_ids, obs


def _hpcb_inputs(root: Path, spec: Mapping, source_records: Mapping,
                 canonical_ids: Sequence[str], minimum_genes: int,
                 minimum_cells: int) -> dict:
    expression_path = project_path(root, spec["expression_path"])
    expression, features, expression_ids, obs = _load_expression(
        expression_path, spec["expression_source"])
    if expression_ids != list(canonical_ids):
        raise ValueError("HP-CB expression order differs from the frozen Store")
    raw_labels = list(map(str, obs[spec["reference_label_key"]]))
    policy = spec["label_policy"]
    labels = [policy["rename"].get(policy["pool"].get(label, label),
                                   policy["pool"].get(label, label))
              for label in raw_labels]
    source_id = spec["external_source_id"]
    records = parse_panglaodb_human(Path(source_records[source_id]["path"]))
    panel = build_marker_panel(
        records, features, spec["external_crosswalk"], dataset="hpcb",
        source_provenance={source_id: source_records[source_id]},
        minimum_genes=minimum_genes)
    counts = Counter(labels)
    excluded_by_count = {name: counts[name] for name in panel["genes_by_label"]
                         if counts[name] < minimum_cells}
    if not any(counts[name] >= minimum_cells for name in panel["genes_by_label"]):
        raise ValueError("No HP-CB marker endpoint passes the fixed cell-count rule")
    prepared = prepare_external_marker_scores(
        expression, features, expression_ids, canonical_ids, panel["genes_by_label"],
        target_sum=10000.0, min_marker_genes=minimum_genes,
        case_sensitive=True, score_method="zscore_mean")
    return {
        "panel": panel, "prepared": prepared, "reference_labels": labels,
        "source_receipts": {
            source_id: source_records[source_id],
            "expression": {"path": str(expression_path.resolve()),
                           "fingerprint": file_fingerprint(expression_path)},
        },
        "reference_label_policy": {
            "source": spec["reference_label_key"],
            "pool": policy["pool"], "rename": policy["rename"],
            "nonendpoint_labels_retained_as_negative_strata": policy["retained_as_nonendpoint"],
            "minimum_cells": minimum_cells,
            "excluded_marker_endpoints_below_minimum_cells": excluded_by_count,
            "labels_used_only_after_result_independent_marker_panel_construction": True,
        },
        "extra": {"raw_reference_class_counts": dict(sorted(Counter(raw_labels).items())),
                  "evaluation_reference_class_counts": dict(sorted(counts.items()))},
    }


def _mouse_inputs(root: Path, spec: Mapping, source_records: Mapping,
                  canonical_ids: Sequence[str], minimum_genes: int,
                  minimum_cells: int) -> dict:
    expression_path = project_path(root, spec["expression_path"])
    ontology_id = spec["ontology_source_id"]
    source_id = spec["external_source_id"]
    ontology = parse_cell_ontology(Path(source_records[ontology_id]["path"]))
    recovery = recover_official_mouse_labels(
        expression_path,
        project_path(root, spec["official_sources"]["droplet"]),
        project_path(root, spec["official_sources"]["facs"]),
        ontology=ontology)
    expression, features, expression_ids, _ = _load_expression(
        expression_path, spec["expression_source"])
    if expression_ids != list(canonical_ids):
        raise ValueError("Mouse expression order differs from the frozen Store")
    labels = list(recovery.official_labels)
    if len(labels) != len(canonical_ids):
        raise ValueError("Recovered mouse labels differ from the frozen Store")
    filter_policy = spec["external_filter"]
    parsed_records = parse_cellmarker_mouse(
        Path(source_records[source_id]["path"]), ontology,
        maximum_year=int(filter_policy["maximum_publication_year"]))
    # Match the marker database and dataset classes only through exact,
    # separately pinned Cell Ontology identities.  Local CL IDs are never used.
    unique_labels = sorted(set(labels))
    crosswalk = {label: [ontology.resolve_exact(label)] for label in unique_labels
                 if ontology.resolve_exact(label) is not None}
    ontology_records = tuple(
        MarkerRecord(record.ontology_id, record.official_gene_symbol,
                     record.ontology_id, gene_id=record.gene_id,
                     species=record.species, source=record.source,
                     year=record.year, canonical=record.canonical)
        for record in parsed_records)
    panel = build_marker_panel(
        ontology_records, features, crosswalk, dataset="mouse_senis",
        source_provenance={source_id: source_records[source_id],
                           ontology_id: source_records[ontology_id]},
        minimum_genes=minimum_genes)
    counts = Counter(labels)
    excluded_by_count = {name: counts[name] for name in panel["genes_by_label"]
                         if counts[name] < minimum_cells}
    if not any(counts[name] >= minimum_cells for name in panel["genes_by_label"]):
        raise ValueError("No Mouse marker endpoint passes the fixed cell-count rule")
    prepared = prepare_external_marker_scores(
        expression, features, expression_ids, canonical_ids, panel["genes_by_label"],
        target_sum=10000.0, min_marker_genes=minimum_genes,
        case_sensitive=True, score_method="zscore_mean")
    unresolved = sorted(label for label in unique_labels
                        if ontology.resolve_exact(label) is None)
    return {
        "panel": panel, "prepared": prepared, "reference_labels": labels,
        "source_receipts": {
            source_id: source_records[source_id],
            ontology_id: source_records[ontology_id],
            "expression": {"path": str(expression_path.resolve()),
                           "fingerprint": file_fingerprint(expression_path)},
            "official_label_sources": recovery.audit["source_files"],
        },
        "reference_label_policy": {
            "source": "exact official Tabula Muris droplet/FACS join",
            "join_key": ["cell", "method", "tissue"],
            "local_cell_ontology_id_used": False,
            "ontology_resolution": "exact non-obsolete official name",
            "minimum_cells": minimum_cells,
            "unresolved_official_class_names": unresolved,
            "excluded_marker_endpoints_below_minimum_cells": excluded_by_count,
            "labels_used_only_after_result_independent_marker_panel_construction": True,
        },
        "extra": {"official_label_recovery": recovery.document(),
                  "evaluation_reference_class_counts": dict(sorted(counts.items()))},
    }


def build_foundation(root: Path, config_path: Path, run_id: str) -> Path:
    root = Path(root).resolve()
    config_path = project_path(root, Path(config_path).as_posix())
    config = _validate_config(_read(config_path))
    before = _source_snapshot(root)
    sources = _verified_external_sources(root, config)
    store_path = project_path(root, config["store_run"])
    store = Store(store_path)
    minimum_genes = int(config["minimum_exact_marker_genes"])
    minimum_cells = int(config["minimum_reference_cells"])
    prepared_inputs = {}
    for spec in config["datasets"]:
        dataset_id = spec["id"]
        canonical_ids = store.dataset(dataset_id).cell_ids
        if dataset_id == "hpcb":
            prepared_inputs[dataset_id] = _hpcb_inputs(
                root, spec, sources, canonical_ids, minimum_genes, minimum_cells)
        elif dataset_id == "mouse_senis":
            prepared_inputs[dataset_id] = _mouse_inputs(
                root, spec, sources, canonical_ids, minimum_genes, minimum_cells)
        else:
            raise ValueError(f"Unexpected foundation dataset: {dataset_id}")

    run_config = {
        "schema": FOUNDATION_SCHEMA,
        "protocol": {"path": config_path.relative_to(root).as_posix(),
                     "fingerprint": file_fingerprint(config_path)},
        "external_source_lock": {
            "path": config["source_lock_config"],
            "fingerprint": file_fingerprint(project_path(root, config["source_lock_config"]))},
        "store": {"path": config["store_run"],
                  "manifest": file_fingerprint(store_path / "run.json")},
        "selection_boundary": config["selection_boundary"],
    }
    with RunDirectory(root / "revision_pipeline/runs",
                      kind="independent_marker_foundation",
                      config=run_config, run_id=run_id) as run:
        entries = {}
        for dataset_id in ("hpcb", "mouse_senis"):
            item = prepared_inputs[dataset_id]
            prepared = item["prepared"]
            scores = np.ascontiguousarray(prepared["scores"], dtype=np.float64)
            class_names = list(prepared["class_names"])
            cell_ids = list(prepared["cell_ids"])
            relative = f"datasets/{dataset_id}"
            scores_path = f"{relative}/marker_scores.npy"
            np.save(run.artifact_path(scores_path), scores, allow_pickle=False)
            run.write_json(f"{relative}/cell_ids.json", cell_ids)
            run.write_json(f"{relative}/reference_labels.json", item["reference_labels"])
            run.write_json(f"{relative}/class_names.json", class_names)
            panel_path = run.artifact_path(f"{relative}/external_panel.json.gz")
            write_panel(panel_path, item["panel"])
            metadata = {
                "dataset_id": dataset_id,
                "panel_content_sha256": item["panel"]["panel_content_sha256"],
                "source_receipts": item["source_receipts"],
                "reference_label_policy": item["reference_label_policy"],
                "cells": len(cell_ids), "classes": len(class_names),
            }
            metadata_path = f"{relative}/metadata.json"
            run.write_json(metadata_path, metadata)
            run.write_json(f"{relative}/preparation_report.json", {
                "marker_preparation": prepared["report"],
                "full_external_panel_classes": len(item["panel"]["genes_by_label"]),
                "evaluated_classes": class_names,
                "panel_exclusions": item["panel"]["excluded_labels"],
                "reference_audit": item["extra"],
            })
            entries[dataset_id] = {
                "marker_scores_path": scores_path,
                "cell_ids_path": f"{relative}/cell_ids.json",
                "reference_labels_path": f"{relative}/reference_labels.json",
                "class_names_path": f"{relative}/class_names.json",
                "metadata_path": metadata_path,
                "shape": list(scores.shape), "dtype": scores.dtype.name,
                "marker_scores_sha256": array_hash(scores),
                "cell_order_sha256": canonical_hash(cell_ids),
            }
        run.write_json("foundation.json", {
            "schema": FOUNDATION_SCHEMA,
            "selection_boundary": {
                "result_independent": True,
                "evaluated_embedding_or_score_used": False,
            },
            "datasets": entries,
        })
        run.write_json("source_receipts.json", {
            "external_sources": sources,
            "store_manifest": file_fingerprint(store_path / "run.json"),
            "protocol": file_fingerprint(config_path),
        })
        run.manifest.update(
            scientific_experiment=True,
            experiment_role="result_independent_external_marker_foundation")
    after = _source_snapshot(root)
    if before != after:
        raise RuntimeError("Pipeline source/config changed while building marker foundation")
    return run.final_path


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    output = build_foundation(ROOT, Path(args.config), args.run_id)
    print(json.dumps({"output": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["build_foundation", "main"]
