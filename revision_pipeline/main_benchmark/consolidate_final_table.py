# Purpose: Assemble the audited Figure-1-aligned main benchmark tables.
# Author: Ariana Rahman (Arizona State University)

"""Assemble the audited Figure-1-aligned main benchmark tables.

This is reporting and provenance work only. It does not train, score, retry,
replace, or select any scientific result.
"""
import argparse
import csv
import io
from pathlib import Path
import statistics

from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import completed, read, snapshot
from ..runs import RunDirectory
from ..step4.scoring import metric_map
from .common import ROOT
from .reporting import main_row, verify_partitions
from .run import compare_graphs


RUNS = ROOT / "revision_pipeline/runs"
OLD_SCAN = RUNS / "20260917T205442Z-9126614c7423"
UNAVAILABLE = RUNS / "20260918T184621Z-aa6d58f411b3"
CASE_PATHS = {
    ("HP-CB", "Harmony"): RUNS / "20260918T080920Z-79dd3dd1533a",
    ("HP-CB", "Seurat"): RUNS / "20260918T104214Z-ac8d8ab46f33",
    ("HP-CB", "Online iNMF"): RUNS / "20260918T130746Z-0dfd4d139612",
    ("Pancreas", "Scanorama"): RUNS / "20260918T173250Z-e18ad182446b",
    ("Pancreas", "Seurat"): RUNS / "20260918T211240Z-c4800abea6fe",
    ("Pancreas", "Online iNMF"): RUNS / "20260918T232610Z-7e6074ad8189",
    ("Mouse", "Scanorama"): RUNS / "20260919T134701Z-2caa77e4e312",
    ("Mouse", "Harmony"): RUNS / "20260920T065519Z-c0a9a894c60f",
    ("Mouse", "Seurat"): RUNS / "20260920T082118Z-18ff0b952216",
    ("Mouse", "Online iNMF"): RUNS / "20260920T091149Z-da183535442a",
}
ORDER = [
    (dataset, backbone)
    for dataset in ("Pancreas", "HP-CB", "Mouse")
    for backbone in ("Scanorama", "Harmony", "Seurat", "Online iNMF", "BBKNN")
]
DATASET_IDS = {"Pancreas": "pancreas_five_study", "HP-CB": "hpcb", "Mouse": "mouse_senis"}


def verified_case(path):
    kind = read(path / "run.json")["kind"]
    if kind not in {"main_benchmark_case", "main_benchmark_fast_gpu_case"}:
        raise ValueError(f"Unexpected case kind at {path}")
    completed(path, kind)
    row = read(path / "main_table_row.json")
    audit = read(path / "partition_rescoring.json")
    if audit.get("passed") is not True:
        raise ValueError(f"Partition audit failed at {path}")
    return row, int(audit["independently_rescored_saved_partitions"])


def old_scanorama():
    completed(OLD_SCAN, "step4a_scoring_completion_verification")
    verification = read(OLD_SCAN / "verification.json")
    if verification.get("passed") is not True or verification.get("partitions_rescored") != 810:
        raise ValueError("HP-CB Scanorama audit is incomplete")
    row = main_row(
        {"dataset": "hpcb", "embedding": "Scanorama"},
        read(OLD_SCAN / "summary_recomputed.json"),
    )
    row["scope"] = "Verified five-seed result; 810 saved partitions independently rescored"
    return row, 810


def bbknn_metrics(path):
    selected = read(path / "evaluation/selected.json")
    values = metric_map(read(path / "coordinate_proxy_metrics/metrics.json"))
    return {
        "ARI": statistics.mean(item["ARI"] for item in selected),
        "SIL_cluster": statistics.mean(
            values[f"predicted_cluster_ASW_subsample_seed{seed}"] for seed in range(3)
        ),
        "SIL_reference": values["reference_ASW_subsample"],
        "local_label_purity": values["reference_knn_purity"],
        "iLISI": values["iLISI_scib_metrics"],
        "D_batch": values["D_batch_fixed90_including_self"],
    }


def bbknn_row(dataset):
    dataset_id = DATASET_IDS[dataset]
    first = RUNS / f"20260920T092000Z-bb-{dataset_id}-0"
    second = RUNS / f"20260920T092000Z-bb-{dataset_id}-1"
    duplicate = compare_graphs(first, second)
    from ..data.store import Store

    store = Store(ROOT / "revision_pipeline/runs/20260916T221434Z-a0884ffbd933")
    reference = store.dataset(dataset_id).reference_partition()[0]
    partition_audit = verify_partitions({"bbknn": str(first)}, reference)
    if partition_audit["independently_rescored_saved_partitions"] != 45:
        raise ValueError(f"BBKNN partition audit incomplete for {dataset}")
    return {
        "dataset": dataset,
        "backbone": "BBKNN",
        "upstream": bbknn_metrics(first),
        "GR": None,
        "GR_seeds": [],
        "GR_range": None,
        "paired_J30": None,
        "status": "baseline_only_graph_method",
        "metric_scope": "ARI from actual BBKNN graph; SIL/iLISI/purity/D_batch from labeled input-coordinate proxy",
        "scope": "Primary graph plus bitwise-identical technical duplicate; 45 primary partitions independently rescored",
    }, duplicate, partition_audit, first, second


def normalized_pair(row, dataset, backbone):
    result = dict(row)
    result["dataset"] = dataset
    result["backbone"] = backbone
    result["status"] = "complete_five_seed_pair"
    return result


def fmt(value):
    return "—" if value is None else f"{value:.4f}"


def harmony_extension_note(rows):
    row = next(item for item in rows if item["dataset"] == "Pancreas" and item["backbone"] == "Harmony")
    if row["status"] == "upstream_nonconverged":
        return "Pancreas Harmony is retained as an explicit unavailable row after two frozen-policy upstream builds failed the convergence gate. "
    policy = row.get("harmony_extension_policy")
    if policy == "pancreas_harmony_fixed10_descriptive_v3":
        return "Pancreas Harmony is a labeled fixed-10-iteration descriptive sensitivity analysis; no numerical-convergence claim is made and the earlier failure records are preserved. "
    return "Pancreas Harmony is a labeled post-failure objective-only v2 extension; the original v1 failure record is preserved. "


def main_markdown(rows):
    lines = [
        "# Updated main benchmark table",
        "",
        "Refined values are means across five algorithmic seeds. SIL is the predicted-cluster silhouette (SIL_cluster).",
        "",
        "| Dataset | Backbone | Upstream ARI | Upstream SIL | Upstream iLISI | Upstream + GR ARI | Upstream + GR SIL | Upstream + GR iLISI | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        up, gr = row.get("upstream"), row.get("GR")
        status = row["status"]
        if status == "complete_five_seed_pair":
            status = "complete; GR = five-seed mean"
        elif status == "upstream_nonconverged":
            status = "upstream non-converged; pair unavailable"
        else:
            status = "BBKNN graph only; GR N/A; SIL/iLISI are coordinate proxies"
        lines.append(
            f"| {row['dataset']} | {row['backbone']} | {fmt(up and up['ARI'])} | "
            f"{fmt(up and up['SIL_cluster'])} | {fmt(up and up['iLISI'])} | "
            f"{fmt(gr and gr['ARI'])} | {fmt(gr and gr['SIL_cluster'])} | "
            f"{fmt(gr and gr['iLISI'])} | {status} |"
        )
    lines += [
        "",
        "Higher ARI indicates stronger agreement with supplied annotations; higher SIL indicates more compact Leiden clusters; higher iLISI indicates greater batch mixing. These endpoints can disagree and must be interpreted jointly.",
        "",
        harmony_extension_note(rows) + "BBKNN has no GenoRefine arm because it returns a graph rather than a corrected coordinate embedding.",
    ]
    return "\n".join(lines) + "\n"


def latex_table(rows):
    names = {"HP-CB": "HP--CB", "Online iNMF": "Online iNMF"}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Updated primary benchmark. Refined values are means over five algorithmic seeds. SIL denotes predicted-cluster silhouette. Higher ARI, SIL, and iLISI are directionally favorable but are interpreted jointly.}",
        r"\label{tab:updated-main-benchmark}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & Backbone & \multicolumn{3}{c}{Upstream} & \multicolumn{3}{c}{Upstream + GR} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r" & & ARI & SIL & iLISI & ARI & SIL & iLISI \\",
        r"\midrule",
    ]
    previous = None
    for row in rows:
        dataset = names.get(row["dataset"], row["dataset"]) if row["dataset"] != previous else ""
        previous = row["dataset"]
        backbone = names.get(row["backbone"], row["backbone"])
        up, gr = row.get("upstream"), row.get("GR")
        if row["status"] == "upstream_nonconverged":
            values = [r"\multicolumn{6}{c}{Upstream non-converged; paired comparison unavailable}"]
            lines.append(f"{dataset} & {backbone} & {values[0]} " + r"\\")
        else:
            vals = [fmt(up and up[k]) for k in ("ARI", "SIL_cluster", "iLISI")]
            vals += ["N/A", "N/A", "N/A"] if gr is None else [fmt(gr[k]) for k in ("ARI", "SIL_cluster", "iLISI")]
            lines.append(f"{dataset} & {backbone} & " + " & ".join(vals) + r" \\")
        if row["backbone"] == "BBKNN" and row["dataset"] != "Mouse":
            lines.append(r"\midrule")
    note = harmony_extension_note(rows)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}\footnotesize " + note + r"BBKNN is graph-only, so GR is not applicable; its SIL and iLISI values are explicitly input-coordinate proxies.\end{flushleft}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def csv_text(rows, full=False):
    output = io.StringIO()
    metrics = ("ARI", "SIL_cluster", "iLISI") if not full else (
        "ARI", "SIL_cluster", "SIL_reference", "local_label_purity", "iLISI", "D_batch"
    )
    fields = ["dataset", "backbone", "status"] + [f"upstream_{m}" for m in metrics] + [f"GR_{m}" for m in metrics]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        record = {"dataset": row["dataset"], "backbone": row["backbone"], "status": row["status"]}
        for prefix, values in (("upstream", row.get("upstream")), ("GR", row.get("GR"))):
            for metric in metrics:
                record[f"{prefix}_{metric}"] = "" if values is None else values[metric]
        writer.writerow(record)
    return output.getvalue()


def main():
    """Verify completed benchmark cases and assemble the final table with provenance receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consolidate-final-table", action="store_true")
    parser.add_argument("--pancreas-harmony-case", type=Path)
    args = parser.parse_args()
    if not args.consolidate_final_table:
        parser.error("Explicit consolidation flag is required")

    rows_by_key = {}
    index = {}
    coordinate_partitions = 0
    scan_row, scan_count = old_scanorama()
    rows_by_key[("HP-CB", "Scanorama")] = normalized_pair(scan_row, "HP-CB", "Scanorama")
    coordinate_partitions += scan_count
    index["HP-CB/Scanorama"] = {"path": str(OLD_SCAN), "manifest": file_fingerprint(OLD_SCAN / "run.json")}

    for key, path in CASE_PATHS.items():
        row, count = verified_case(path)
        rows_by_key[key] = normalized_pair(row, *key)
        coordinate_partitions += count
        index["/".join(key)] = {"path": str(path), "manifest": file_fingerprint(path / "run.json")}

    if args.pancreas_harmony_case:
        harmony_path = args.pancreas_harmony_case.resolve()
        harmony_row, harmony_count = verified_case(harmony_path)
        harmony_config = read(harmony_path / "config.json")
        harmony_policy = harmony_config.get("policy_id")
        if harmony_policy not in {
            "pancreas_harmony_convergence_v2",
            "pancreas_harmony_fixed10_descriptive_v3",
        }:
            raise ValueError("Unexpected Pancreas Harmony extension policy")
        rows_by_key[("Pancreas", "Harmony")] = normalized_pair(harmony_row, "Pancreas", "Harmony")
        rows_by_key[("Pancreas", "Harmony")]["post_failure_extension"] = True
        rows_by_key[("Pancreas", "Harmony")]["harmony_extension_policy"] = harmony_policy
        coordinate_partitions += harmony_count
        index["Pancreas/Harmony"] = {
            "path": str(harmony_path),
            "manifest": file_fingerprint(harmony_path / "run.json"),
            "post_failure_extension_policy": harmony_policy,
            "original_v1_failure_record": str(UNAVAILABLE),
        }
    else:
        completed(UNAVAILABLE, "main_benchmark_harmony_failure_recovery")
        missing = read(UNAVAILABLE / "unavailable_main_table_row.json")
        missing["dataset"] = "Pancreas"
        missing["backbone"] = "Harmony"
        rows_by_key[("Pancreas", "Harmony")] = missing
        index["Pancreas/Harmony"] = {"path": str(UNAVAILABLE), "manifest": file_fingerprint(UNAVAILABLE / "run.json")}

    bb_audits = {}
    for dataset in ("Pancreas", "HP-CB", "Mouse"):
        row, duplicate, audit, first, second = bbknn_row(dataset)
        rows_by_key[(dataset, "BBKNN")] = row
        bb_audits[dataset] = {"duplicate": duplicate, "partition_audit": audit}
        index[f"{dataset}/BBKNN"] = {
            "primary": str(first),
            "primary_manifest": file_fingerprint(first / "run.json"),
            "duplicate": str(second),
            "duplicate_manifest": file_fingerprint(second / "run.json"),
        }

    rows = [rows_by_key[key] for key in ORDER]
    harmony_complete = args.pancreas_harmony_case is not None
    expected_coordinate_partitions = 9450 if harmony_complete else 8640
    if len(rows) != 15 or coordinate_partitions != expected_coordinate_partitions:
        raise ValueError("Final benchmark coverage mismatch")
    complete_pairs = [row for row in rows if row["status"] == "complete_five_seed_pair"]
    directions = {
        metric: {
            "improved": sum(row["GR"][metric] > row["upstream"][metric] for row in complete_pairs),
            "declined": sum(row["GR"][metric] < row["upstream"][metric] for row in complete_pairs),
            "tied": sum(row["GR"][metric] == row["upstream"][metric] for row in complete_pairs),
        }
        for metric in ("ARI", "SIL_cluster", "SIL_reference", "local_label_purity", "iLISI", "D_batch")
    }
    coverage = {
        "planned_rows": 15,
        "complete_five_seed_coordinate_pairs": 12 if harmony_complete else 11,
        "unavailable_coordinate_pairs": 0 if harmony_complete else 1,
        "refinement_seeds": 60 if harmony_complete else 55,
        "coordinate_representations": 210 if harmony_complete else 192,
        "coordinate_partitions_independently_rescored": coordinate_partitions,
        "bbknn_primary_rows": 3,
        "bbknn_primary_partitions_independently_rescored": 135,
        "bbknn_duplicate_partitions_bitwise_checked": 135,
        "no_efficacy_based_selection": True,
        "biological_improvement_claim_authorized": False,
    }
    sources = snapshot(ROOT)
    with RunDirectory(RUNS, kind="main_benchmark_final_table", config={
        "purpose": "Reporting-only consolidation of pinned, audited results",
        "coverage": coverage,
    }) as run:
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.manifest["scientific_experiment"] = False
        run.write_json("run_index.json", index)
        run.write_json("bbknn_repeatability.json", bb_audits)
        run.write_json("coverage.json", coverage)
        run.write_json("direction_counts.json", directions)
        run.write_json("table_data.json", {
            "rows": rows,
            "coverage": coverage,
            "direction_counts": directions,
            "main_metric_definition": {
                "ARI": "Agreement with supplied annotation partition",
                "SIL": "Predicted-cluster silhouette (SIL_cluster), matching the original table concept",
                "iLISI": "scib-metrics 0.5.8 embedding-kNN iLISI; not D_batch",
            },
        })
        run.artifact_path("main_table.md").write_text(main_markdown(rows), encoding="utf-8")
        run.artifact_path("main_table.tex").write_text(latex_table(rows), encoding="utf-8")
        run.artifact_path("main_table.csv").write_text(csv_text(rows), encoding="utf-8")
        run.artifact_path("supplement_metrics.csv").write_text(csv_text(rows, full=True), encoding="utf-8")
        run.artifact_path("report.md").write_text(
            "# Final main-table consolidation\n\n"
            f"All 15 planned rows are represented: {coverage['complete_five_seed_coordinate_pairs']} complete five-seed coordinate pairs, "
            f"{coverage['unavailable_coordinate_pairs']} unavailable coordinate pairs, and three BBKNN graph-only baselines. "
            f"All {coordinate_partitions:,} coordinate partitions were independently rescored within their case audits; "
            "all 135 BBKNN primary partitions were rescored and all three 45-partition duplicates were bitwise identical.\n\n"
            "The results do not support a universal-improvement claim. See direction_counts.json and "
            "the per-case summaries for the observed tradeoffs among annotation agreement, cluster compactness, "
            "batch mixing, local purity, and neighbourhood preservation.\n",
            encoding="utf-8",
        )
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
