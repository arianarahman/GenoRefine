"""Consolidate the complete preregistered GraphST Package 4b panel."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics

from ..integrity import canonical_hash, file_fingerprint
from ..runs import RunDirectory
from .common import (
    RUNS, load_alignment, load_k_selection, read_json, require_run, source_snapshot,
    specification, validate_evaluation_runtime,
)


SECTION_METRICS = (
    "ARI", "NMI", "predicted_cluster_silhouette", "reference_label_silhouette",
    "label_neighbor_purity", "spatial_latent_knn_jaccard",
    "physical_distance_among_latent_neighbors_mean_fullres_pixels",
    "physical_distance_ratio_to_local_spatial_knn_mean",
)
DONOR_METRICS = ("iLISI", "D_batch")
NATIVE_PARTITIONS = ("mclust", "refined")
NATIVE_METRICS = ("ARI", "NMI", "clusters")


def _path(prefix: str, suffix: str) -> Path:
    return RUNS / f"{prefix}-{suffix}"


def expected_runs(prefix: str) -> dict:
    spec = specification()
    result = {
        "alignments": {donor: _path(prefix, f"align-{donor}") for donor in spec["donors"]},
        "k_selection": _path(prefix, "k-selection"),
        "training": {}, "section_scores": {}, "donor_scores": {},
    }
    for donor, sections in spec["donors"].items():
        for seed in spec["graphst"]["seeds"]:
            result["training"][(donor, seed)] = _path(prefix, f"graphst-{donor}-s{seed}")
            result["donor_scores"][(donor, seed)] = _path(prefix, f"score-donor-{donor}-s{seed}")
            for section in sections:
                result["section_scores"][(section, donor, seed)] = _path(
                    prefix, f"score-section-{section}-s{seed}",
                )
    return result


def _stats(values: list[float]) -> dict:
    if not values or not all(isinstance(value, (int, float)) for value in values):
        raise ValueError("Cannot summarize an empty or nonnumeric endpoint")
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values), "maximum": max(values),
        "n_algorithmic_seeds": len(values),
    }


def _expected_runtime() -> dict:
    runtime = specification()["evaluation"]["runtime"]
    return {
        "python_executable": runtime["python_executable"], "python": runtime["python"],
        "packages": runtime["packages"], "wsl_distribution": runtime["wsl_distribution"],
        "validated": True,
    }


def _flatten_paths(expected: dict) -> list[Path]:
    return [
        *expected["alignments"].values(), expected["k_selection"],
        *expected["training"].values(), *expected["section_scores"].values(),
        *expected["donor_scores"].values(),
    ]


def _load_all(prefix: str) -> tuple[list[dict], list[dict], dict]:
    spec = specification()
    expected = expected_runs(prefix)
    alignments = {
        donor: load_alignment(path, donor) for donor, path in expected["alignments"].items()
    }
    decisions = load_k_selection(expected["k_selection"])
    k_manifest = file_fingerprint(expected["k_selection"] / "run.json")
    training_manifests = {}
    for (donor, seed), path in expected["training"].items():
        manifest = require_run(path, "spatial_graphst_training")
        config = read_json(path / "config.json")
        record = read_json(path / "training_record.json")
        if (config.get("protocol_id") != spec["protocol_id"]
                or config.get("donor") != donor or config.get("seed") != seed
                or config.get("alignment_manifest") != alignments[donor]["manifest"]
                or config.get("k_selection_manifest") != k_manifest
                or config.get("K_binding") != decisions["donors"][donor]
                or config.get("settings") != spec["graphst"]
                or record.get("seed") != seed or record.get("donor") != donor
                or record.get("native_raw_cluster_count") != decisions["donors"][donor]["n_clusters"]
                or record.get("reference_labels_loaded") is not False
                or record.get("reference_labels_used") is not False):
            raise ValueError(f"Training parent/content binding failed: {path.name}")
        training_manifests[(donor, seed)] = file_fingerprint(path / "run.json")
    runtime = _expected_runtime()
    section_rows, donor_rows = [], []
    for (section, donor, seed), path in expected["section_scores"].items():
        require_run(path, "spatial_graphst_section_score")
        config = read_json(path / "config.json")
        row = read_json(path / "summary.json")
        if (config.get("protocol_id") != spec["protocol_id"]
                or config.get("runtime") != runtime
                or config.get("section") != section or config.get("donor") != donor
                or config.get("algorithmic_seed") != seed
                or config.get("input", {}).get("training_manifest") != training_manifests[(donor, seed)]
                or config.get("input", {}).get("alignment_manifest") != alignments[donor]["manifest"]
                or config.get("input", {}).get("k_selection_manifest") != k_manifest
                or row.get("section") != section or row.get("donor") != donor
                or row.get("algorithmic_seed") != seed
                or set(row.get("native_graphst_partitions", {})) != set(NATIVE_PARTITIONS)):
            raise ValueError(f"Section-score parent/content binding failed: {path.name}")
        row["run"] = str(path)
        section_rows.append(row)
    for (donor, seed), path in expected["donor_scores"].items():
        require_run(path, "spatial_graphst_donor_score")
        config = read_json(path / "config.json")
        row = read_json(path / "summary.json")
        if (config.get("protocol_id") != spec["protocol_id"]
                or config.get("runtime") != runtime
                or config.get("donor") != donor or config.get("algorithmic_seed") != seed
                or config.get("input", {}).get("training_manifest") != training_manifests[(donor, seed)]
                or config.get("input", {}).get("alignment_manifest") != alignments[donor]["manifest"]
                or config.get("input", {}).get("k_selection_manifest") != k_manifest
                or row.get("donor") != donor or row.get("algorithmic_seed") != seed):
            raise ValueError(f"Donor-score parent/content binding failed: {path.name}")
        row["run"] = str(path)
        donor_rows.append(row)
    if len(section_rows) != 30 or len(donor_rows) != 15:
        raise ValueError("GraphST score coverage is incomplete")
    current_source = canonical_hash(source_snapshot())
    for path in _flatten_paths(expected):
        manifest = read_json(path / "run.json")
        if manifest.get("source_tree_sha256") != current_source:
            raise ValueError(f"Child source snapshot differs from frozen Package 4b: {path.name}")
    return section_rows, donor_rows, expected


def aggregate_sections(rows: list[dict]) -> dict:
    spec = specification()
    section_summary, donor_summary = {}, {}
    for section in spec["sections"]:
        selected = [row for row in rows if row["section"] == section]
        if sorted(row["algorithmic_seed"] for row in selected) != spec["graphst"]["seeds"]:
            raise ValueError("Incomplete/duplicate GraphST section seed panel")
        section_summary[section] = {
            metric: _stats([float(row[metric]) for row in selected]) for metric in SECTION_METRICS
        }
    macro_by_metric = {metric: [] for metric in SECTION_METRICS}
    for donor, sections in spec["donors"].items():
        donor_summary[donor] = {}
        for metric in SECTION_METRICS:
            values = []
            for seed in spec["graphst"]["seeds"]:
                per_section = [float(next(
                    row for row in rows
                    if row["section"] == section and row["donor"] == donor
                    and row["algorithmic_seed"] == seed
                )[metric]) for section in sections]
                values.append(statistics.mean(per_section))
            donor_summary[donor][metric] = _stats(values)
            macro_by_metric[metric].append(values)
    macro = {
        metric: {
            **_stats([statistics.mean(seed_values) for seed_values in zip(*donor_values)]),
            "aggregation": "two-section mean within donor, then equal three-donor macro, seed aligned",
        }
        for metric, donor_values in macro_by_metric.items()
    }
    return {"sections": section_summary, "donors": donor_summary, "macro": macro}


def aggregate_native(rows: list[dict]) -> dict:
    spec = specification()
    result = {}
    for partition in NATIVE_PARTITIONS:
        per_section, per_donor, macro = {}, {}, {}
        for section in spec["sections"]:
            selected = [row for row in rows if row["section"] == section]
            per_section[section] = {
                metric: _stats([
                    float(row["native_graphst_partitions"][partition][metric]) for row in selected
                ]) for metric in NATIVE_METRICS
            }
        for donor, sections in spec["donors"].items():
            per_donor[donor] = {}
            for metric in NATIVE_METRICS:
                values = []
                for seed in spec["graphst"]["seeds"]:
                    values.append(statistics.mean(float(next(
                        row for row in rows
                        if row["section"] == section and row["donor"] == donor
                        and row["algorithmic_seed"] == seed
                    )["native_graphst_partitions"][partition][metric]) for section in sections))
                per_donor[donor][metric] = _stats(values)
        for metric in NATIVE_METRICS:
            by_seed = []
            for seed in spec["graphst"]["seeds"]:
                donor_values = []
                for donor, sections in spec["donors"].items():
                    donor_values.append(statistics.mean(float(next(
                        row for row in rows
                        if row["section"] == section and row["donor"] == donor
                        and row["algorithmic_seed"] == seed
                    )["native_graphst_partitions"][partition][metric]) for section in sections))
                by_seed.append(statistics.mean(donor_values))
            macro[metric] = {
                **_stats(by_seed),
                "aggregation": "two-section mean within donor, then equal three-donor macro, seed aligned",
            }
        result[partition] = {
            "role": "secondary task-native GraphST partition",
            "sections": per_section, "donors": per_donor, "macro": macro,
        }
    return result


def aggregate_donors(rows: list[dict]) -> dict:
    spec = specification()
    per_donor = {}
    for donor in spec["donors"]:
        selected = [row for row in rows if row["donor"] == donor]
        if sorted(row["algorithmic_seed"] for row in selected) != spec["graphst"]["seeds"]:
            raise ValueError("Incomplete/duplicate GraphST donor score seed panel")
        per_donor[donor] = {
            metric: _stats([float(row[metric]) for row in selected]) for metric in DONOR_METRICS
        }
    macro = {}
    for metric in DONOR_METRICS:
        by_seed = [statistics.mean(float(next(
            row for row in rows if row["donor"] == donor and row["algorithmic_seed"] == seed
        )[metric]) for donor in spec["donors"]) for seed in spec["graphst"]["seeds"]]
        macro[metric] = {**_stats(by_seed), "aggregation": "equal macro-average over three donors"}
    return {"donors": per_donor, "macro": macro}


def execute(prefix: str, run_id: str) -> Path:
    spec = specification()
    runtime = validate_evaluation_runtime()
    rows, donor_rows, expected = _load_all(prefix)
    section_summary = aggregate_sections(rows)
    native_summary = aggregate_native(rows)
    donor_summary = aggregate_donors(donor_rows)
    sources = source_snapshot()
    inputs = {path.name: file_fingerprint(path / "run.json") for path in _flatten_paths(expected)}
    context = {
        "protocol_id": spec["protocol_id"], "prefix": prefix, "inputs": inputs,
        "aggregation": spec["aggregation"], "runtime": runtime,
    }
    with RunDirectory(RUNS, kind="spatial_graphst_panel", run_id=run_id, config=context) as run:
        run.write_json("section_summary.json", section_summary)
        run.write_json("native_partition_summary.json", native_summary)
        run.write_json("donor_mixing_summary.json", donor_summary)
        run.write_json("section_seed_rows.json", rows)
        run.write_json("donor_seed_rows.json", donor_rows)
        with run.artifact_path("section_seed_rows.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["section", "donor", "method", "algorithmic_seed", *SECTION_METRICS]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
        native_rows = []
        for row in rows:
            for partition in NATIVE_PARTITIONS:
                native_rows.append({
                    "section": row["section"], "donor": row["donor"],
                    "method": row["method"], "algorithmic_seed": row["algorithmic_seed"],
                    "partition": partition,
                    **{metric: row["native_graphst_partitions"][partition][metric]
                       for metric in NATIVE_METRICS},
                })
        with run.artifact_path("native_section_seed_rows.csv").open(
            "w", newline="", encoding="utf-8",
        ) as stream:
            fields = [
                "section", "donor", "method", "algorithmic_seed", "partition",
                *NATIVE_METRICS,
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(native_rows)
        lines = [
            "# GraphST Package 4b: six-section LIBD DLPFC panel", "",
            "GraphST was fit once per donor pair after label-free PASTE alignment. Five seeds are descriptive algorithmic repeats, not biological replicates; no p-values are reported.",
            "Manual cortical-layer identities/values were introduced only during Package 4b evaluation. The inherited Package 4 foundation was already restricted to spots with nonmissing labels, so the fit is label-blind conditional on that complete-case cohort.", "",
            "## Common-evaluator donor-macro endpoints", "",
            "| Endpoint | Mean | SD across five aligned algorithmic seeds |",
            "|---|---:|---:|",
        ]
        for metric in SECTION_METRICS:
            value = section_summary["macro"][metric]
            lines.append(f"| {metric} | {value['mean']:.4f} | {value['sd']:.4f} |")
        lines += ["", "## Donor-pair section mixing", "", "| Endpoint | Mean | SD |", "|---|---:|---:|"]
        for metric in DONOR_METRICS:
            value = donor_summary["macro"][metric]
            lines.append(f"| {metric} | {value['mean']:.4f} | {value['sd']:.4f} |")
        lines += [
            "", "## Secondary task-native GraphST partitions", "",
            "These native endpoints are reported separately from common-evaluator clustering.", "",
            "| Partition | Endpoint | Mean | SD |", "|---|---|---:|---:|",
        ]
        for partition in NATIVE_PARTITIONS:
            for metric in NATIVE_METRICS:
                value = native_summary[partition]["macro"][metric]
                lines.append(
                    f"| {partition} | {metric} | {value['mean']:.4f} | {value['sd']:.4f} |"
                )
        run.artifact_path("report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.write_json("completion.json", {
            "six_sections_complete": True, "three_donors_complete": True,
            "alignment_runs": 3, "training_runs": 15,
            "section_scores": len(rows), "expected_section_scores": 30,
            "donor_scores": len(donor_rows), "expected_donor_scores": 15,
            "seeds": spec["graphst"]["seeds"],
            "algorithmic_seeds_not_biological_replicates": True,
            "p_values_reported": False,
            "common_evaluator_and_native_endpoints_separated": True,
        })
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.manifest.update(
            scientific_experiment=True, experiment_role="complete_graphst_package4b_panel",
            training_performed=False, scoring_performed=False,
        )
        if source_snapshot() != sources:
            raise RuntimeError("Scientific source changed during GraphST panel consolidation")
    return run.final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required")
    print(execute(args.prefix, args.run_id), flush=True)


if __name__ == "__main__":
    main()
