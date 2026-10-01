# Purpose: Consolidate rare-cell, purity, geometry and expression-signature preservation.
# Author: Ariana Rahman (Arizona State University)

"""Consolidate rare-cell, purity, geometry and expression-signature preservation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics

import numpy as np

from ..data.store import Store
from ..integrity import file_fingerprint
from ..pilot.common import completed, read
from ..runs import RunDirectory
from .metrics import centroid_geometry

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "revision_pipeline/runs/20260916T221434Z-a0884ffbd933"
FINAL = ROOT / "revision_pipeline/runs/20260920T092529Z-c79843c0c2ac"


def local_path(value):
    if isinstance(value, dict):
        value = value["path"]
    normalized = str(value).replace("\\", "/")
    prefix = "<PROJECT_ROOT>/"
    if normalized == "<PROJECT_ROOT>":
        return ROOT
    if normalized.startswith(prefix):
        return ROOT / normalized[len(prefix):]
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def mean_range(values):
    values = list(map(float, values))
    return {"mean": statistics.mean(values), "minimum": min(values), "maximum": max(values)}


def marker_purity(codes, neighbors, mask=None):
    values = np.mean(codes[:, None] == codes[neighbors], axis=1)
    return float(values.mean() if mask is None else values[mask].mean())


def resolve_case(display, record):
    source = local_path(record["path"])
    if display == "HP-CB/Scanorama":
        panel = local_path(read(source / "config.json")["panel"])
        summary = read(source / "summary_recomputed.json")
        scoring_index = read(panel / "run_index.json")
        training_panel = ROOT / read(panel / "config.json")["scoring"]["training_panel"]
        training_index = read(training_panel / "run_index.json")
        index = {"training": training_index["training"],
                 "scoring": scoring_index,
                 "baseline": scoring_index["baseline"]}
        case = {"dataset": "hpcb", "embedding": "Scanorama"}
    else:
        panel = source
        summary = read(source / "summary.json")
        index = read(source / "run_index.json")
        case = read(source / "config.json")["case"]
    return source, panel, summary, index, case


def main():
    """Consolidate cross-fitted marker, rare-cell, and geometry preservation for all eligible cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker-run", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required")
    marker_run = args.marker_run.resolve()
    completed(marker_run, "biological_marker_crossfit")
    completed(FINAL, "main_benchmark_final_table")
    final_index = read(FINAL / "run_index.json")
    store = Store(STORE)
    rows, details, source_receipts = [], {}, {}
    for display, record in final_index.items():
        if "/" not in display or display.endswith("BBKNN") or display == "Pancreas/Harmony":
            continue
        source, panel, summary, index, case = resolve_case(display, record)
        dataset_id, embedding = case["dataset"], case["embedding"]
        dataset = store.dataset(dataset_id)
        parent = store.embedding(dataset_id, embedding)
        labels = np.asarray(dataset.record["obs"][dataset.record["loader"]["label_key"]]).astype(str)
        codes = np.load(marker_run / dataset_id / "predicted_codes.npy", allow_pickle=False)
        true_codes = np.load(marker_run / dataset_id / "true_codes.npy", allow_pickle=False)
        eligible = np.load(marker_run / dataset_id / "eligible_cells.npy", allow_pickle=False)
        validated = eligible & (codes == true_codes)
        scoring = index.get("scoring", {})
        baseline_score = local_path(scoring.get("baseline", index.get("baseline")))
        baseline_neighbors = np.load(baseline_score / "geometry_neighbors.npy", allow_pickle=False)
        base_rep = summary["representations"]["baseline"]
        centroid_spearman, centroid_pearson = [], []
        marker_all, marker_validated, rare_failures = [], [], []
        purity = []
        for seed in range(5):
            training = local_path(index["training"][str(seed)])
            refined = np.load(training / "bundles/joint/values.npy", allow_pickle=False)
            geometry = centroid_geometry(parent.values, refined, labels)
            centroid_spearman.append(geometry["centroid_distance_spearman"])
            centroid_pearson.append(geometry["centroid_distance_pearson"])
            joint_score = local_path(scoring[f"joint_{seed}"])
            neighbors = np.load(joint_score / "geometry_neighbors.npy", allow_pickle=False)
            marker_all.append(marker_purity(codes, neighbors))
            marker_validated.append(marker_purity(codes, neighbors, validated))
            rep = summary["representations"][f"joint_{seed}"]
            purity.append(rep["metrics"]["reference_knn_purity"])
            base_rare = {int(x["group_code"]): x for x in base_rep["rare_groups"]}
            new_rare = {int(x["group_code"]): x for x in rep["rare_groups"]}
            rare_failures.append(sum(
                new_rare[code]["mean_same_class_recall_at_k"]
                - old["mean_same_class_recall_at_k"] < -0.05
                for code, old in base_rare.items() if old["full_cells"] > 1))
        baseline_marker_all = marker_purity(codes, baseline_neighbors)
        baseline_marker_validated = marker_purity(codes, baseline_neighbors, validated)
        item = {
            "dataset": display.split("/")[0], "backbone": display.split("/", 1)[1],
            "cells": len(labels), "classes": len(set(labels)),
            "baseline_reference_purity": base_rep["metrics"]["reference_knn_purity"],
            "joint_reference_purity_mean": statistics.mean(purity),
            "reference_purity_delta": statistics.mean(purity) - base_rep["metrics"]["reference_knn_purity"],
            "centroid_spearman_mean": statistics.mean(centroid_spearman),
            "centroid_pearson_mean": statistics.mean(centroid_pearson),
            "baseline_expression_signature_purity": baseline_marker_all,
            "joint_expression_signature_purity_mean": statistics.mean(marker_all),
            "expression_signature_purity_delta": statistics.mean(marker_all) - baseline_marker_all,
            "baseline_validated_signature_purity": baseline_marker_validated,
            "joint_validated_signature_purity_mean": statistics.mean(marker_validated),
            "rare_groups": len(base_rep["rare_groups"]),
            "rare_failures_mean": statistics.mean(rare_failures),
            "rare_safeguard_all_seeds": all(value == 0 for value in rare_failures)}
        rows.append(item)
        details[display] = {"summary": item,
                            "seed_ranges": {"reference_purity": mean_range(purity),
                                            "centroid_spearman": mean_range(centroid_spearman),
                                            "centroid_pearson": mean_range(centroid_pearson),
                                            "expression_signature_purity": mean_range(marker_all),
                                            "validated_signature_purity": mean_range(marker_validated),
                                            "rare_failure_count": mean_range(rare_failures)}}
        source_receipts[display] = {"source_run": file_fingerprint(source / "run.json"),
                                    "baseline_score": file_fingerprint(baseline_score / "run.json")}
    config = {"main_table": file_fingerprint(FINAL / "run.json"),
              "marker_run": file_fingerprint(marker_run / "run.json"),
              "source_receipts": source_receipts}
    with RunDirectory(ROOT / "revision_pipeline/runs", kind="biological_preservation_all",
                      config=config, run_id=args.run_id) as run:
        run.write_json("details.json", details)
        run.write_json("summary.json", {
            "complete_pairs": len(rows), "datasets": sorted(set(x["dataset"] for x in rows)),
            "rows": rows,
            "annotation_and_marker_boundary": {
                "hpcb": "Gene-symbol cross-fitted signatures; supplied names define signatures; not independent ground truth",
                "mouse": "Gene-symbol cross-fitted signatures; ontology IDs are incomplete/non-bijective; supplied names define signatures",
                "pancreas": "Positional cross-fitted expression signatures only; not biological marker genes because symbols are absent"},
            "claim": "Expanded preservation evidence across all 11 complete pairs; no general preservation claim when safeguards fail"})
        table = run.artifact_path("preservation_table.csv")
        with table.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        lines = ["# Biological preservation across the complete real-data panel", "",
                 "All 11 complete upstream/refinement pairs and all five seeds were retained.", "",
                 "| Dataset | Backbone | purity delta | centroid Spearman | signature-purity delta | rare safeguard |",
                 "|---|---|---:|---:|---:|---|"]
        for row in rows:
            lines.append(f"| {row['dataset']} | {row['backbone']} | {row['reference_purity_delta']:+.4f} | {row['centroid_spearman_mean']:.4f} | {row['expression_signature_purity_delta']:+.4f} | {'pass' if row['rare_safeguard_all_seeds'] else 'fail'} |")
        lines.extend(["", "Pancreas signatures use unnamed positional features and are not marker-gene validation. HP-CB and mouse signatures use gene symbols, but supplied annotations select the signatures; this is cross-fitted expression evidence rather than independent ground truth.", ""])
        run.artifact_path("report.md").write_text("\n".join(lines), encoding="utf-8")
        run.manifest.update(scientific_experiment=True,
                            experiment_role="biological_preservation_saved_output_consolidation")
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
