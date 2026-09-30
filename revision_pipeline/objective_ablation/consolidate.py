"""Consolidate the completed five-seed loss-objective ablation panel."""

from __future__ import annotations

import csv
from pathlib import Path
import statistics

from ..integrity import file_fingerprint
from ..pilot.common import completed, read
from ..runs import RunDirectory
from . import VARIANTS, WEIGHTS

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "revision_pipeline/runs"
STEP4 = RUNS / "20260917T203334Z-4b4cecbd7116"
SCORE_PREFIX = "20260921T062100Z-objective"
METRICS = ("ARI", "SIL_cluster", "iLISI", "purity", "SIL_reference")


def reduced(values):
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values),
            "min": min(values), "max": max(values), "values": values}


def step4_row(record):
    metrics = record["metrics"]
    sil = [value for key, value in metrics.items()
           if key.startswith("predicted_cluster_ASW_subsample_seed")]
    return {"ARI": record["mean_ARI"], "SIL_cluster": statistics.mean(sil),
            "iLISI": metrics["iLISI_scib_metrics"],
            "purity": metrics["reference_knn_purity"],
            "SIL_reference": metrics["reference_ASW_subsample"]}


def main():
    completed(STEP4, "step4a_scoring_panel")
    old = read(STEP4 / "summary.json")["representations"]
    panels = {
        "original_weight_0p1": [step4_row(old[f"joint_{seed}"]) for seed in range(5)],
        "zero_clustering_reconstruction_only": [
            step4_row(old[f"reconstruction_{seed}"]) for seed in range(5)],
    }
    inputs = {"step4": file_fingerprint(STEP4 / "run.json")}
    for variant in VARIANTS:
        rows = []
        for seed in range(5):
            score = RUNS / f"{SCORE_PREFIX}-{variant}-s{seed}-score"
            completed(score, "objective_ablation_scoring")
            saved = read(score / "summary.json")
            if saved["variant"] != variant or saved["seed"] != seed:
                raise ValueError("Variant/seed binding changed")
            rows.append({metric: saved[metric] for metric in METRICS})
            inputs[f"{variant}_{seed}"] = file_fingerprint(score / "run.json")
        panels[variant] = rows
    summary = {
        "methods": {name: {metric: reduced([row[metric] for row in rows])
                            for metric in METRICS}
                    for name, rows in panels.items()},
        "weights": {
            "zero_clustering_reconstruction_only": [1.0, 0.0],
            "cluster_weight_0p01": list(WEIGHTS["cluster_weight_0p01"]),
            "original_weight_0p1": [1.0, 0.1],
            "cluster_weight_1p0": list(WEIGHTS["cluster_weight_1p0"]),
            "no_reconstruction": list(WEIGHTS["no_reconstruction"]),
        },
        "weight_order": ["zero_clustering_reconstruction_only", "cluster_weight_0p01",
                         "original_weight_0p1", "cluster_weight_1p0",
                         "no_reconstruction"],
        "interpretation": {
            "scope": "HP-CB Scanorama; five fixed algorithmic seeds; identical seed-specific pretrained boundaries",
            "no_reconstruction": "DEC-style continuation after identical autoencoder pretraining",
            "selection": "Weights 0.01 and 1.0 requested before outcomes; original 0.1 and zero-clustering branches reused",
            "inference": "Descriptive algorithmic sensitivity; not independent biological replication",
        },
    }
    with RunDirectory(RUNS, kind="objective_ablation_consolidation",
                      config={"inputs": inputs, "metrics": list(METRICS)}) as run:
        run.write_json("summary.json", summary)
        with run.artifact_path("table.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["variant", "reconstruction_weight", "clustering_weight", *METRICS])
            for name in summary["weight_order"]:
                rw, cw = summary["weights"][name]
                writer.writerow([name, rw, cw,
                                 *[summary["methods"][name][m]["mean"] for m in METRICS]])
        lines = ["# Loss-objective ablation", "",
                 "HP-CB Scanorama; identical seed-specific pretrained boundaries; five algorithmic seeds.", "",
                 "| Variant (reconstruction, clustering) | ARI | cluster SIL | iLISI | purity | reference SIL |",
                 "|---|---:|---:|---:|---:|---:|"]
        for name in summary["weight_order"]:
            rw, cw = summary["weights"][name]
            result = summary["methods"][name]
            cells = [f"{result[m]['mean']:.4f} +/- {result[m]['sd']:.4f}" for m in METRICS]
            lines.append(f"| {name.replace('_', ' ')} ({rw:g}, {cw:g}) | " + " | ".join(cells) + " |")
        lines.extend(["", "The no-reconstruction row retains autoencoder pretraining and removes reconstruction only during continuation.", ""])
        run.artifact_path("report.md").write_text("\n".join(lines), encoding="utf-8")
        run.manifest.update(scientific_experiment=True,
                            experiment_role="loss_objective_ablation_consolidation")
    print(run.final_path)


if __name__ == "__main__":
    main()
