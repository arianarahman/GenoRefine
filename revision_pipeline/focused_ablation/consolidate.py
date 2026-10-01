# Purpose: Consolidate the complete focused-ablation panel without seed selection.
# Author: Ariana Rahman (Arizona State University)

"""Consolidate the complete focused-ablation panel without seed selection."""

from __future__ import annotations

import csv
from pathlib import Path
import statistics

from ..integrity import file_fingerprint
from ..pilot.common import completed, read
from ..runs import RunDirectory
from . import VARIANTS

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "revision_pipeline/runs"
STEP4 = RUNS / "20260917T203334Z-4b4cecbd7116"
TRAIN_PREFIX = "20260920T180000Z-ablation"
SCORE_PREFIX = "20260920T183000Z-ablation"
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
    inputs = {"step4a": file_fingerprint(STEP4 / "run.json")}
    panels = {
        "original_joint": [step4_row(old[f"joint_{seed}"]) for seed in range(5)],
        "reconstruction_only": [step4_row(old[f"reconstruction_{seed}"])
                                for seed in range(5)],
    }
    architectures = {}
    for variant in VARIANTS:
        rows = []
        for seed in range(5):
            training = RUNS / f"{TRAIN_PREFIX}-{variant}-s{seed}"
            scoring = RUNS / f"{SCORE_PREFIX}-{variant}-s{seed}-score"
            completed(training, "focused_ablation_training")
            completed(scoring, "focused_ablation_scoring")
            inputs[f"{variant}_training_{seed}"] = file_fingerprint(training / "run.json")
            inputs[f"{variant}_scoring_{seed}"] = file_fingerprint(scoring / "run.json")
            saved = read(scoring / "summary.json")
            if saved["variant"] != variant or saved["seed"] != seed:
                raise ValueError("Focused-ablation seed/variant binding changed")
            rows.append({metric: saved[metric] for metric in METRICS})
            architecture = read(training / "architecture.json")
            if seed == 0:
                architectures[variant] = architecture
            elif architecture != architectures[variant]:
                raise ValueError("Architecture metadata changed across seeds")
        panels[variant] = rows
    baseline = step4_row(old["baseline"])
    if (architectures["map12_fixed36"]["parameters"] !=
            architectures["map24_fixed36"]["parameters"]):
        raise ValueError("Controlled map variants changed network parameters")
    summary = {
        "baseline": baseline,
        "methods": {name: {metric: reduced([row[metric] for row in rows])
                            for metric in METRICS}
                    for name, rows in panels.items()},
        "directions_vs_baseline": {
            name: {metric: {"higher": sum(row[metric] > baseline[metric] for row in rows),
                            "lower": sum(row[metric] < baseline[metric] for row in rows),
                            "equal": sum(row[metric] == baseline[metric] for row in rows)}
                   for metric in METRICS}
            for name, rows in panels.items()},
        "architectures": architectures,
        "interpretation": {
            "scope": "HP-CB Scanorama; five prespecified seeds; frozen exact-kNN evaluator",
            "reconstruction_only": "Clustering-loss removal from the same pretrained boundary",
            "no_pretraining": "Randomly initialized joint objective with zero autoencoder pretraining epochs",
            "controlled_map_size": "12x12 or 24x24 learned layout, amplitude-adjusted and centered in a 36x36 input; identical downstream network capacity",
            "claim_policy": "No favorable seed or metric selection; no cross-dataset superiority inference"}}
    config = {"inputs": inputs, "variants": list(panels), "metrics": list(METRICS)}
    with RunDirectory(RUNS, kind="focused_ablation_consolidation", config=config) as run:
        run.write_json("summary.json", summary)
        with run.artifact_path("table.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["variant", *METRICS])
            writer.writerow(["Scanorama_baseline", *[baseline[m] for m in METRICS]])
            for name, result in summary["methods"].items():
                writer.writerow([name, *[result[m]["mean"] for m in METRICS]])
        lines = ["# Focused ablations", "",
                 "HP-CB Scanorama, five seeds, frozen exact-kNN evaluator. Values are mean +/- SD; no seed was selected.", "",
                 "| Variant | ARI | cluster SIL | iLISI | reference purity | reference SIL |",
                 "|---|---:|---:|---:|---:|---:|",
                 "| Scanorama baseline | " + " | ".join(f"{baseline[m]:.4f}" for m in METRICS) + " |"]
        for name, result in summary["methods"].items():
            cells = [f"{result[m]['mean']:.4f} +/- {result[m]['sd']:.4f}" for m in METRICS]
            lines.append(f"| {name.replace('_', ' ')} | " + " | ".join(cells) + " |")
        lines.extend(["", "Higher is favorable for displayed endpoints, but iLISI is interpreted only together with biological preservation.", ""])
        run.artifact_path("report.md").write_text("\n".join(lines), encoding="utf-8")
        run.manifest.update(scientific_experiment=True,
                            experiment_role="focused_ablation_consolidation")
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
