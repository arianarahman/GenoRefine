"""Consolidate the full five-seed IDEC panel against upstream and GenoRefine."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics

from ..integrity import file_fingerprint
from ..pilot.common import completed, read
from ..runs import RunDirectory
from .run_idec import protocol_path


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "revision_pipeline/runs"
DEFAULT_PROTOCOL = ROOT / "revision_pipeline/configs/independent_idec_panel_v2.json"
METRICS = {
    "ARI": ("ARI", "mean_ARI_at_0_5"),
    "SIL_cluster": ("SIL_cluster", "predicted_cluster_ASW_subsample_mean"),
    "iLISI": ("iLISI", "iLISI_scib_metrics"),
    "local_label_purity": ("local_label_purity", "reference_knn_purity"),
    "SIL_reference": ("SIL_reference", "reference_ASW_subsample"),
    "D_batch": ("D_batch", "D_batch_fixed90_including_self"),
}


def reduced(values):
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def direction(delta):
    return "higher" if delta > 0 else "lower" if delta < 0 else "equal"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required")
    protocol = protocol_path(args.config)
    spec = read(protocol)
    table_path = ROOT / "revision_pipeline/runs/20260921T003801Z-c055558a1a7c/table_data.json"
    table = read(table_path)
    primary = {(row["dataset"], row["backbone"]): row for row in table["rows"]
               if row.get("GR") is not None}
    records = []
    inputs = {"protocol": file_fingerprint(protocol), "primary_table": file_fingerprint(table_path)}
    for case in spec["cases"]:
        key = (case["display_dataset"], case["embedding"].replace("Online_iNMF", "Online iNMF"))
        if key not in primary:
            raise ValueError(f"Primary benchmark row is missing: {key}")
        row = primary[key]
        if len(row["GR_seeds"]) != len(spec["replicate_seeds"]):
            raise ValueError(f"Incomplete GenoRefine seed panel: {case['id']}")
        summaries = []
        for seed in spec["replicate_seeds"]:
            path = RUNS / f"{args.prefix}-{case['id']}-s{seed}-joint-score"
            completed(path, "independent_idec_scoring")
            inputs[f"{case['id']}_seed{seed}"] = file_fingerprint(path / "run.json")
            summary = read(path / "summary.json")
            if (summary["case_id"] != case["id"] or summary["seed"] != seed
                    or summary["stage"] != "joint"):
                raise ValueError(f"Score binding changed: {case['id']} seed {seed}")
            summaries.append(summary)
        methods = {"upstream": {}, "GenoRefine": {}, "IDEC": {}}
        for metric, (main_key, idec_key) in METRICS.items():
            baseline = row["upstream"][main_key]
            gr_values = [seed_row[main_key] for seed_row in row["GR_seeds"]]
            idec_values = [summary[idec_key] for summary in summaries]
            methods["upstream"][metric] = {"value": baseline}
            methods["GenoRefine"][metric] = reduced(gr_values)
            methods["IDEC"][metric] = reduced(idec_values)
            methods["GenoRefine"][metric]["delta_vs_upstream"] = statistics.mean(gr_values) - baseline
            methods["IDEC"][metric]["delta_vs_upstream"] = statistics.mean(idec_values) - baseline
            methods["IDEC"][metric]["paired_delta_vs_GenoRefine"] = reduced(
                [idec - gr for idec, gr in zip(idec_values, gr_values)])
        records.append({
            "case_id": case["id"], "dataset": case["display_dataset"],
            "backbone": key[1], "n_clusters": case["n_clusters"],
            "joint_updates": case["joint_updates"], "methods": methods,
        })
    directions = {}
    for method in ("GenoRefine", "IDEC"):
        directions[method] = {}
        for metric in METRICS:
            labels = [direction(record["methods"][method][metric]["delta_vs_upstream"])
                      for record in records]
            directions[method][metric] = {label: labels.count(label)
                                          for label in ("higher", "lower", "equal")}
    summary = {
        "protocol_id": spec["protocol_id"],
        "coverage": {"cases": len(records), "seeds_per_case": len(spec["replicate_seeds"]),
                     "scored_IDEC_embeddings": len(records) * len(spec["replicate_seeds"]),
                     "datasets": 3, "backbones": 4},
        "records": records, "directions_vs_upstream": directions,
        "interpretation_policy": spec["success_policy"],
        "inference": spec["inference"],
    }
    config = {"protocol": file_fingerprint(protocol), "prefix": args.prefix, "inputs": inputs}
    with RunDirectory(RUNS, kind="independent_idec_full_panel_consolidation",
                      config=config, run_id=args.run_id) as run:
        run.write_json("summary.json", summary)
        with run.artifact_path("long_table.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["dataset", "backbone", "metric", "upstream", "GenoRefine_mean",
                             "GenoRefine_sd", "IDEC_mean", "IDEC_sd", "IDEC_minus_upstream",
                             "IDEC_minus_GenoRefine_paired_mean"])
            for record in records:
                for metric in METRICS:
                    methods = record["methods"]
                    writer.writerow([
                        record["dataset"], record["backbone"], metric,
                        methods["upstream"][metric]["value"],
                        methods["GenoRefine"][metric]["mean"], methods["GenoRefine"][metric]["sd"],
                        methods["IDEC"][metric]["mean"], methods["IDEC"][metric]["sd"],
                        methods["IDEC"][metric]["delta_vs_upstream"],
                        methods["IDEC"][metric]["paired_delta_vs_GenoRefine"]["mean"],
                    ])
        lines = ["# Full independent IDEC comparator panel", "",
                 "All 12 dataset-backbone pairs were evaluated with five seeds, matched 32-dimensional outputs, label-free K, 100 pretraining epochs, a dataset-matched two-pass joint budget, and the frozen primary evaluator.", "",
                 "| Dataset | Backbone | Method | ARI | SIL | iLISI | Purity |",
                 "|---|---|---|---:|---:|---:|---:|"]
        for record in records:
            for method in ("upstream", "GenoRefine", "IDEC"):
                cells = []
                for metric in ("ARI", "SIL_cluster", "iLISI", "local_label_purity"):
                    value = record["methods"][method][metric]
                    cells.append(f"{value['value']:.4f}" if method == "upstream"
                                 else f"{value['mean']:.4f} +/- {value['sd']:.4f}")
                lines.append(f"| {record['dataset']} | {record['backbone']} | {method} | "
                             + " | ".join(cells) + " |")
        lines.extend(["", "Algorithmic seeds are reported descriptively; they are not biological replicates. Higher iLISI is interpreted only with the agreement and preservation endpoints.", ""])
        run.artifact_path("report.md").write_text("\n".join(lines), encoding="utf-8")
        run.manifest.update(scientific_experiment=True,
                            experiment_role="full_independent_refinement_comparator_consolidation")
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
