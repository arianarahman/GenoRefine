"""Consolidate the completed standalone scVI panels and existing main-table baselines."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from ..runs import RunDirectory

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "revision_pipeline/runs"
MAIN_TABLE = RUNS / "20260920T092448Z-0805730049f2/main_table.csv"
OUTPUT_ID = "20260920T235000Z-scvi-other-datasets-report"


def score_path(dataset: str, seed: int) -> Path:
    short = "mouse" if dataset == "mouse_senis" else "pancreas"
    return RUNS / f"20260920T233000Z-scvi-{short}-s{seed}-score" / "summary.json"


def training_path(dataset: str, seed: int) -> Path:
    if dataset == "mouse_senis":
        stamp = "20260920T230000Z" if seed == 0 else "20260920T231000Z"
        short = "mouse"
    else:
        stamp, short = "20260920T232000Z", "pancreas"
    return RUNS / f"{stamp}-scvi-{short}-s{seed}" / "summary.json"


def summarize(rows: list[dict]) -> dict:
    result = {}
    for name in ("ARI", "SIL_cluster", "iLISI", "purity"):
        values = [float(row[name]) for row in rows]
        result[name] = {"mean": statistics.mean(values), "sd": statistics.stdev(values)}
    return result


def main():
    datasets = ("pancreas_five_study", "mouse_senis")
    all_rows = []
    summaries = {}
    training = {}
    for dataset in datasets:
        rows = []
        train_rows = []
        for seed in range(5):
            score = json.loads(score_path(dataset, seed).read_text(encoding="utf-8"))
            train = json.loads(training_path(dataset, seed).read_text(encoding="utf-8"))
            rows.append(score)
            train_rows.append(train)
            all_rows.append({
                "dataset": dataset,
                "seed": seed,
                "ARI": score["ARI"],
                "SIL": score["SIL_cluster"],
                "iLISI": score["iLISI"],
                "purity": score["purity"],
                "training_seconds": train["training_wall_seconds"],
                "evaluation_seconds": score["wall_seconds"],
                "epochs": train["epochs_run"],
                "evidence_status": score["evidence_status"],
            })
        summaries[dataset] = summarize(rows)
        training[dataset] = {
            "seconds_mean": statistics.mean(float(row["training_wall_seconds"]) for row in train_rows),
            "seconds_sd": statistics.stdev(float(row["training_wall_seconds"]) for row in train_rows),
            "epochs": [int(row["epochs_run"]) for row in train_rows],
            "input_audit": train_rows[0]["input_audit"],
        }

    with MAIN_TABLE.open(newline="", encoding="utf-8") as handle:
        main_rows = list(csv.DictReader(handle))
    comparisons = []
    for label, dataset in (("Pancreas", "pancreas_five_study"), ("Mouse", "mouse_senis")):
        for row in main_rows:
            if row["dataset"] == label:
                comparisons.append({
                    "dataset": label,
                    "method": row["backbone"],
                    "status": row["status"],
                    "ARI": row["upstream_ARI"],
                    "SIL": row["upstream_SIL_cluster"],
                    "iLISI": row["upstream_iLISI"],
                })
        summary = summaries[dataset]
        comparisons.append({
            "dataset": label,
            "method": "scVI",
            "status": "standalone_primary" if dataset == "mouse_senis" else "standalone_sensitivity_only",
            "ARI": summary["ARI"]["mean"],
            "SIL": summary["SIL_cluster"]["mean"],
            "iLISI": summary["iLISI"]["mean"],
        })

    config = {
        "protocol": "revision_pipeline/configs/scvi_other_datasets_v1.json",
        "score_runs": [str(score_path(dataset, seed).parent.relative_to(ROOT)) for dataset in datasets for seed in range(5)],
        "training_runs": [str(training_path(dataset, seed).parent.relative_to(ROOT)) for dataset in datasets for seed in range(5)],
        "comparison_source": str(MAIN_TABLE.relative_to(ROOT)),
    }
    with RunDirectory(RUNS, kind="scvi_other_datasets_consolidation", config=config, run_id=OUTPUT_ID) as run:
        with run.artifact_path("per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader(); writer.writerows(all_rows)
        with run.artifact_path("comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
            writer.writeheader(); writer.writerows(comparisons)
        run.write_json("summary.json", {"metrics": summaries, "training": training})

        lines = [
            "# Standalone scVI results for the remaining datasets",
            "",
            "Five prespecified training seeds were run and scored with the frozen primary evaluator. scVI is treated as a standalone integration backbone; no scVI+GenoRefine result was produced.",
            "",
            "| Dataset | ARI | SIL | iLISI | Evidence status |",
            "|---|---:|---:|---:|---|",
        ]
        for label, dataset, status in (
            ("Pancreas", "pancreas_five_study", "Sensitivity only: mixed count/transformed MAT inputs"),
            ("Mouse", "mouse_senis", "Standalone comparator: integer raw/X; droplet/FACS modeled as batches"),
        ):
            summary = summaries[dataset]
            lines.append(
                f"| {label} | {summary['ARI']['mean']:.4f} ± {summary['ARI']['sd']:.4f} | "
                f"{summary['SIL_cluster']['mean']:.4f} ± {summary['SIL_cluster']['sd']:.4f} | "
                f"{summary['iLISI']['mean']:.4f} ± {summary['iLISI']['sd']:.4f} | {status} |"
            )
        lines += [
            "",
            "## Interpretation",
            "",
            "Mouse scVI has partition agreement similar to the existing mouse upstream methods, but its iLISI is very low, indicating little droplet/FACS mixing under this protocol. Pancreas is not suitable as primary scVI evidence because the five supplied matrices are not a uniform raw-count matrix; the numerical result is retained only as a sensitivity analysis.",
            "",
            "The dimensionality (32) and all input/audit details are retained in the protocol and run records but need not appear in the main manuscript table.",
        ]
        run.artifact_path("report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.manifest.update(scientific_experiment=True, experiment_role="standalone_scvi_consolidation")
    print(run.final_path)


if __name__ == "__main__":
    main()
