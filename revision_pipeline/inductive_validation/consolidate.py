# Purpose: Consolidate the complete fully held-out validation panel.
# Author: Ariana Rahman (Arizona State University)

"""Consolidate the complete fully held-out validation panel."""

import csv
import json
from pathlib import Path
import statistics

from ..integrity import file_fingerprint
from ..runs import RunDirectory
from .common import RUNS, completed, specification

PREFIX = "20260920T193000Z-inductive"
METRICS = ("ARI", "cluster_silhouette", "iLISI", "reference_knn_purity", "reference_silhouette")


def reduced(values):
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values), "values": values,
            "min": min(values), "max": max(values)}


def main():
    baseline_path = RUNS / f"{PREFIX}-baseline-score"
    completed(baseline_path, "inductive_validation_score")
    baseline = json.loads((baseline_path / "summary.json").read_text(encoding="utf-8"))
    inputs = {"baseline": file_fingerprint(baseline_path / "run.json")}; methods = {}
    for stage in ("pretrain", "reconstruction", "joint"):
        rows = []
        for seed in range(5):
            path = RUNS / f"{PREFIX}-{stage}-s{seed}-score"; completed(path, "inductive_validation_score")
            row = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            if row["seed"] != seed or row["representation"] != stage: raise ValueError("Score binding changed")
            rows.append(row); inputs[f"{stage}_{seed}"] = file_fingerprint(path / "run.json")
        methods[stage] = {metric: reduced([row[metric] for row in rows]) for metric in METRICS}
    summary = {"baseline": {m: baseline[m] for m in METRICS}, "methods": methods,
               "deltas_vs_baseline": {stage: {m: result[m]["mean"] - baseline[m] for m in METRICS}
                                       for stage, result in methods.items()},
               "claim_boundary": specification()["claim_boundary"]}
    with RunDirectory(RUNS, kind="inductive_validation_consolidation",
                      config={"protocol": specification()["protocol_id"], "inputs": inputs}) as run:
        run.write_json("summary.json", summary)
        with run.artifact_path("table.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream); writer.writerow(["representation", *METRICS])
            writer.writerow(["upstream_baseline", *[baseline[m] for m in METRICS]])
            for stage in methods: writer.writerow([stage, *[methods[stage][m]["mean"] for m in METRICS]])
        lines = ["# Fully held-out-cell validation", "", specification()["claim_boundary"], "",
                 "| Representation | ARI | cluster SIL | iLISI | purity | reference SIL |",
                 "|---|---:|---:|---:|---:|---:|",
                 "| upstream baseline | " + " | ".join(f"{baseline[m]:.4f}" for m in METRICS) + " |"]
        for stage in methods:
            lines.append(f"| {stage} | " + " | ".join(
                f"{methods[stage][m]['mean']:.4f} +/- {methods[stage][m]['sd']:.4f}" for m in METRICS) + " |")
        run.artifact_path("report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.manifest.update(scientific_experiment=True, experiment_role="fully_inductive_validation_consolidation")
    print(run.final_path, flush=True)


if __name__ == "__main__": main()
