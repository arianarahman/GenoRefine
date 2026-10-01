# Purpose: Consolidate the complete independent IDEC panel without seed selection.
# Author: Ariana Rahman (Arizona State University)

"""Consolidate the complete independent IDEC panel without seed selection."""

from __future__ import annotations

import csv
from pathlib import Path
import statistics

from ..integrity import file_fingerprint
from ..pilot.common import completed, read
from ..runs import RunDirectory

ROOT = Path(__file__).resolve().parents[2]
STEP4 = ROOT / "revision_pipeline/runs/20260917T203334Z-4b4cecbd7116"
PREFIX = "20260920T162000Z-idec-hpcb"
METRICS = ("ARI", "predicted_cluster_ASW", "iLISI", "reference_knn_purity",
           "reference_ASW")


def idec_path(stage, seed):
    return ROOT / "revision_pipeline/runs" / f"{PREFIX}-{stage}-s{seed}-score"


def reduced(values):
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values),
            "min": min(values), "max": max(values), "values": values}


def idec_row(summary):
    return {"ARI": summary["mean_ARI_at_0_5"],
            "predicted_cluster_ASW": summary["predicted_cluster_ASW_subsample_mean"],
            "iLISI": summary["iLISI_scib_metrics"],
            "reference_knn_purity": summary["reference_knn_purity"],
            "reference_ASW": summary["reference_ASW_subsample"]}


def step4_row(summary):
    metrics = summary["metrics"]
    silhouettes = [value for key, value in metrics.items()
                   if key.startswith("predicted_cluster_ASW_subsample_seed")]
    return {"ARI": summary["mean_ARI"],
            "predicted_cluster_ASW": statistics.mean(silhouettes),
            "iLISI": metrics["iLISI_scib_metrics"],
            "reference_knn_purity": metrics["reference_knn_purity"],
            "reference_ASW": metrics["reference_ASW_subsample"]}


def main():
    completed(STEP4, "step4a_scoring_panel")
    step4 = read(STEP4 / "summary.json")["representations"]
    inputs = {"step4": file_fingerprint(STEP4 / "run.json")}
    panels = {"GenoRefine_joint": [step4_row(step4[f"joint_{seed}"])
                                     for seed in range(5)]}
    for stage in ("pretrain", "joint"):
        rows = []
        for seed in range(5):
            path = idec_path(stage, seed)
            completed(path, "independent_idec_scoring")
            inputs[f"IDEC_{stage}_{seed}"] = file_fingerprint(path / "run.json")
            summary = read(path / "summary.json")
            if summary["seed"] != seed or summary["stage"] != stage:
                raise ValueError("IDEC seed/stage binding changed")
            rows.append(idec_row(summary))
        panels[f"IDEC_{stage}"] = rows
    baseline = step4_row(step4["baseline"])
    summary = {"baseline": baseline,
               "methods": {method: {metric: reduced([row[metric] for row in rows])
                                     for metric in METRICS}
                           for method, rows in panels.items()}}
    summary["directions_vs_baseline"] = {
        method: {metric: {"higher": sum(row[metric] > baseline[metric] for row in rows),
                          "lower": sum(row[metric] < baseline[metric] for row in rows),
                          "equal": sum(row[metric] == baseline[metric] for row in rows)}
                 for metric in METRICS}
        for method, rows in panels.items()}
    summary["interpretation"] = {
        "scope": "HP-CB Scanorama only; descriptive five-seed independent comparator",
        "external_independence": "IDEC architecture/objective is independent of GenoRefine/GenoDR",
        "compatibility_boundary": "Current-Keras compatibility implementation of the official pinned source, not its original runtime",
        "claim_policy": "No seed selection and no general superiority inference"}
    config = {"inputs": inputs, "methods": list(panels), "metrics": list(METRICS)}
    with RunDirectory(ROOT / "revision_pipeline/runs", kind="independent_idec_consolidation",
                      config=config) as run:
        run.write_json("summary.json", summary)
        table = run.artifact_path("table.csv")
        with table.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["method", *METRICS])
            writer.writerow(["Scanorama baseline", *[baseline[m] for m in METRICS]])
            for method in ("GenoRefine_joint", "IDEC_pretrain", "IDEC_joint"):
                writer.writerow([method, *[summary["methods"][method][m]["mean"] for m in METRICS]])
        lines = ["# Independent IDEC comparator", "",
                 "Five seeds; means below use the frozen exact-kNN evaluator. No favorable seed was selected.", "",
                 "| Method | ARI | predicted-cluster SIL | iLISI | reference purity | reference SIL |",
                 "|---|---:|---:|---:|---:|---:|",
                 "| Scanorama baseline | " + " | ".join(f"{baseline[m]:.4f}" for m in METRICS) + " |"]
        for method in ("GenoRefine_joint", "IDEC_pretrain", "IDEC_joint"):
            display = method.replace("_", " ")
            cells = []
            for metric in METRICS:
                value = summary["methods"][method][metric]
                cells.append(f"{value['mean']:.4f} +/- {value['sd']:.4f}")
            lines.append(f"| {display} | " + " | ".join(cells) + " |")
        lines.extend(["", "Higher is favorable for the displayed endpoints, but iLISI must be interpreted with biological preservation rather than alone.", ""])
        run.artifact_path("report.md").write_text("\n".join(lines), encoding="utf-8")
        run.manifest.update(scientific_experiment=True,
                            experiment_role="independent_refinement_comparator_consolidation")
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()

