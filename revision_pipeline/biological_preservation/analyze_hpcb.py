# Purpose: Analyze HP-CB biological-structure preservation from saved Step 4A outputs.
# Author: Ariana Rahman (Arizona State University)

"""Analyze HP-CB biological-structure preservation from saved Step 4A outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import statistics

import numpy as np

from .metrics import centroid_geometry, neighborhood_jaccard


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "revision_pipeline/runs/20260916T221434Z-a0884ffbd933/datasets/hpcb.json"
BASELINE_VALUES = ROOT / "revision_pipeline/runs/20260916T221434Z-a0884ffbd933/embeddings/hpcb/Scanorama.npy"
SCORES = ROOT / "revision_pipeline/runs"
BASELINE_SCORE = SCORES / "20260917T203334Z-4b4cecbd7116-baseline"
TRAINING = SCORES / "20260917T181031Z-f7e46860148e"
JOINT_SCORE_TEMPLATE = "20260917T203334Z-4b4cecbd7116-joint_{seed}"
RARE_MAX = 0.01
K = 30


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_range(values):
    values = list(map(float, values))
    return {"mean": statistics.mean(values), "minimum": min(values), "maximum": max(values)}


def project_record_path(value: str) -> Path:
    normalized = str(value).replace("\\", "/")
    prefix = "<PROJECT_ROOT>/"
    if normalized == "<PROJECT_ROOT>":
        return ROOT
    if normalized.startswith(prefix):
        return ROOT / normalized[len(prefix):]
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(output: Path) -> dict:
    """Summarize seedwise neighborhood, rare-cell, centroid, and expression preservation for HP-CB."""
    output.mkdir(parents=True, exist_ok=False)
    dataset = read_json(DATASET)
    ids = dataset["cell_ids"]
    labels = np.asarray(dataset["obs"]["celltype"]).astype(str)
    if len(ids) != 16382 or len(labels) != len(ids):
        raise ValueError("Unexpected HP-CB population")

    baseline = np.load(BASELINE_VALUES, allow_pickle=False)
    base_neighbors = np.load(BASELINE_SCORE / "geometry_neighbors.npy", allow_pickle=False)
    base_rare = read_json(BASELINE_SCORE / "rare.json")
    if baseline.shape != (len(ids), 100) or base_neighbors.shape != (len(ids), K):
        raise ValueError("Unexpected baseline shape")
    if read_json(BASELINE_SCORE / "evaluation/cell_ids.json") != ids:
        raise ValueError("Baseline score order differs from the canonical order")
    if any(i in row for i, row in enumerate(base_neighbors)):
        raise ValueError("Baseline neighbor list contains self")

    index = read_json(TRAINING / "run_index.json")["training"]
    per_seed = []
    provenance = {
        "protocol": "hpcb_biological_preservation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "new_training": False,
        "analysis_sources": {
            "analyze_hpcb.py": sha256(Path(__file__).resolve()),
            "metrics.py": sha256(Path(__file__).with_name("metrics.py").resolve()),
        },
        "dataset": {"path": str(DATASET.relative_to(ROOT)), "sha256": sha256(DATASET)},
        "baseline_values": {"path": str(BASELINE_VALUES.relative_to(ROOT)), "sha256": sha256(BASELINE_VALUES)},
        "baseline_neighbors": {"path": str((BASELINE_SCORE / 'geometry_neighbors.npy').relative_to(ROOT)), "sha256": sha256(BASELINE_SCORE / "geometry_neighbors.npy")},
        "seeds": {},
    }

    for seed in range(5):
        train = project_record_path(index[str(seed)])
        values_path = train / "bundles/joint/values.npy"
        ids_path = train / "bundles/joint/cell_ids.json"
        score = SCORES / JOINT_SCORE_TEMPLATE.format(seed=seed)
        neighbors_path = score / "geometry_neighbors.npy"
        rare_path = score / "rare.json"
        if read_json(ids_path) != ids or read_json(score / "evaluation/cell_ids.json") != ids:
            raise ValueError(f"Seed {seed} order differs from the canonical order")
        refined = np.load(values_path, allow_pickle=False)
        neighbors = np.load(neighbors_path, allow_pickle=False)
        if refined.shape != (len(ids), 32) or neighbors.shape != base_neighbors.shape:
            raise ValueError(f"Unexpected seed {seed} shape")
        if any(i in row for i, row in enumerate(neighbors)):
            raise ValueError(f"Seed {seed} neighbor list contains self")
        overlap = neighborhood_jaccard(base_neighbors, neighbors)
        geometry = centroid_geometry(baseline, refined, labels)
        rare = read_json(rare_path)
        rare_by_code = {int(row["group_code"]): row for row in rare["groups"]}
        per_seed.append({
            "seed": seed,
            "overall_neighbor_jaccard": float(overlap.mean()),
            "class_neighbor_jaccard": {name: float(overlap[labels == name].mean()) for name in np.unique(labels)},
            "centroid_geometry": geometry,
            "rare": rare_by_code,
        })
        provenance["seeds"][str(seed)] = {
            "values": {"path": str(values_path.relative_to(ROOT)), "sha256": sha256(values_path)},
            "neighbors": {"path": str(neighbors_path.relative_to(ROOT)), "sha256": sha256(neighbors_path)},
            "rare": {"path": str(rare_path.relative_to(ROOT)), "sha256": sha256(rare_path)},
        }

    counts = {name: int((labels == name).sum()) for name in np.unique(labels)}
    classes = sorted(counts)
    rare_names = [name for name in classes if counts[name] / len(labels) <= RARE_MAX]
    category_order = sorted(classes)
    code_to_name = dict(enumerate(category_order))
    rare_rows = []
    base_rare_by_code = {int(row["group_code"]): row for row in base_rare["groups"]}
    for code, base_row in sorted(base_rare_by_code.items()):
        name = code_to_name[code]
        if name not in rare_names or counts[name] != base_row["full_cells"]:
            raise ValueError("Named rare-group mapping does not match the saved evaluator")
        recall = [run["rare"][code]["mean_same_class_recall_at_k"] for run in per_seed]
        purity = [run["rare"][code]["mean_neighbor_purity"] for run in per_seed]
        asw = [run["rare"][code]["mean_ASW"] for run in per_seed]
        deltas = [value - base_row["mean_same_class_recall_at_k"] for value in recall]
        rare_rows.append({
            "cell_type": name,
            "cells": counts[name],
            "batches": base_row["full_batches"],
            "baseline_ASW": base_row["mean_ASW"],
            "joint_ASW_mean": statistics.mean(asw),
            "baseline_purity_at_30": base_row["mean_neighbor_purity"],
            "joint_purity_at_30_mean": statistics.mean(purity),
            "baseline_recall_at_30": base_row["mean_same_class_recall_at_k"],
            "joint_recall_at_30_mean": statistics.mean(recall),
            "joint_recall_at_30_min": min(recall),
            "joint_recall_at_30_max": max(recall),
            "minimum_recall_delta": min(deltas),
            "all_seeds_delta_ge_minus_0_05": all(delta >= -0.05 for delta in deltas),
        })

    class_rows = []
    baseline_ratios = per_seed[0]["centroid_geometry"]["reference_separation_ratio"]
    for name in classes:
        overlaps = [run["class_neighbor_jaccard"][name] for run in per_seed]
        ratios = [run["centroid_geometry"]["refined_separation_ratio"][name] for run in per_seed]
        class_rows.append({
            "cell_type": name,
            "cells": counts[name],
            "rare_le_1pct": name in rare_names,
            "neighbor_jaccard_mean": statistics.mean(overlaps),
            "neighbor_jaccard_min": min(overlaps),
            "neighbor_jaccard_max": max(overlaps),
            "baseline_separation_ratio": baseline_ratios[name],
            "joint_separation_ratio_mean": statistics.mean(ratios),
            "joint_separation_ratio_min": min(ratios),
            "joint_separation_ratio_max": max(ratios),
            "separation_ratio_delta_mean": statistics.mean(ratios) - baseline_ratios[name],
        })

    summary = {
        "status": "completed_saved_output_analysis",
        "dataset": "HP-CB pancreas",
        "backbone": "Scanorama",
        "cells": len(ids),
        "classes": len(classes),
        "replicate_seeds": [0, 1, 2, 3, 4],
        "neighborhood": {"k_nonself": K, "overall_jaccard": mean_range(run["overall_neighbor_jaccard"] for run in per_seed)},
        "centroid_geometry": {
            "pairs": per_seed[0]["centroid_geometry"]["centroid_pairs"],
            "spearman": mean_range(run["centroid_geometry"]["centroid_distance_spearman"] for run in per_seed),
            "pearson": mean_range(run["centroid_geometry"]["centroid_distance_pearson"] for run in per_seed),
        },
        "rare_definition": "Supplied cell type frequency <=1% of the full 16,382-cell cohort",
        "rare_cell_types": rare_rows,
        "class_preservation": class_rows,
        "per_seed": per_seed,
        "rare_safeguard": {
            "rule": "Every eligible rare group must have recall@30 delta >= -0.05 in every seed",
            "passed": all(row["all_seeds_delta_ge_minus_0_05"] for row in rare_rows),
            "failed_cell_types": [row["cell_type"] for row in rare_rows if not row["all_seeds_delta_ge_minus_0_05"]],
        },
        "interpretation_limits": [
            "Cell-type names are supplied HP-CB annotations; annotation independence and exact cohort provenance remain pending.",
            "Marker-expression values are not changed by embedding refinement and therefore are not independent before/after evidence.",
            "Five algorithmic seeds are not biological replicates; no seed-based p-values were computed.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "rare_cell_table.csv", rare_rows)
    write_csv(output / "class_preservation_table.csv", class_rows)

    lines = [
        "# HP-CB rare-cell and paired-geometry preservation",
        "",
        "This post-pilot analysis reuses the frozen Scanorama input and five saved GenoRefine joint-stage embeddings. No model was retrained and no result was selected by score.",
        "",
        "## Population-level structure",
        "",
        f"Across seeds, the mean exact 30-neighbor Jaccard overlap with the baseline was {summary['neighborhood']['overall_jaccard']['mean']:.4f} (range {summary['neighborhood']['overall_jaccard']['minimum']:.4f}-{summary['neighborhood']['overall_jaccard']['maximum']:.4f}).",
        f"Across all 91 pairs of the 14 supplied cell-type centroids, baseline-versus-refined distance correlations were Spearman {summary['centroid_geometry']['spearman']['mean']:.4f} (range {summary['centroid_geometry']['spearman']['minimum']:.4f}-{summary['centroid_geometry']['spearman']['maximum']:.4f}) and Pearson {summary['centroid_geometry']['pearson']['mean']:.4f} (range {summary['centroid_geometry']['pearson']['minimum']:.4f}-{summary['centroid_geometry']['pearson']['maximum']:.4f}).",
        "",
        "## Rare cell types",
        "",
        "Rare means <=1% of all 16,382 cells. Every rare cell was evaluated against the full cohort with 30 non-self neighbors.",
        "",
        "| Cell type | n | Baseline recall@30 | Joint mean [range] | Minimum delta | Safeguard |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rare_rows:
        lines.append(f"| {row['cell_type']} | {row['cells']} | {row['baseline_recall_at_30']:.4f} | {row['joint_recall_at_30_mean']:.4f} [{row['joint_recall_at_30_min']:.4f}, {row['joint_recall_at_30_max']:.4f}] | {row['minimum_recall_delta']:+.4f} | {'pass' if row['all_seeds_delta_ge_minus_0_05'] else 'fail'} |")
    lines += [
        "",
        "The pre-specified rare-cell safeguard failed. This does not support a general rare-cell-preservation claim. The supplied names may be used descriptively, but annotation independence and exact cohort provenance remain pending.",
        "",
        "Marker-expression values are fixed upstream of the embedding transformation. They can document the supplied annotations, but unchanged marker plots cannot demonstrate that refinement preserved biology; no such claim is made here.",
        "",
        "All seed-level values, class-resolved neighborhood retention, scale-independent separation ratios, file hashes, and source paths are stored with this run.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.output.resolve())
    print(json.dumps({"output": str(args.output.resolve()), "rare_safeguard": summary["rare_safeguard"], "neighborhood": summary["neighborhood"], "centroid_geometry": summary["centroid_geometry"]}, indent=2))


if __name__ == "__main__":
    main()
