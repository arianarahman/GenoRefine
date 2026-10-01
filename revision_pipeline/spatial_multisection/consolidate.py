# Purpose: Consolidate the complete six-section spatial panel with donor-first aggregation.
# Author: Ariana Rahman (Arizona State University)

"""Consolidate the complete six-section spatial panel with donor-first aggregation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics

from ..integrity import canonical_hash, file_fingerprint
from ..runs import RunDirectory
from .common import (
    RUNS, read_json, require_run, source_snapshot, specification, validate_evaluation_runtime,
)


SECTION_METRICS = (
    "ARI", "NMI", "predicted_cluster_silhouette", "reference_label_silhouette",
    "label_neighbor_purity", "spatial_latent_knn_jaccard",
    "physical_distance_among_latent_neighbors_mean_fullres_pixels",
    "physical_distance_ratio_to_local_spatial_knn_mean",
)
DONOR_METRICS = ("iLISI", "D_batch")


def _path(prefix: str, suffix: str) -> Path:
    return RUNS / f"{prefix}-{suffix}"


def validate_child_source_hashes(paths: list[Path], current_source_sha256: str) -> None:
    """Require all child runs to share the current frozen source tree."""
    for path in paths:
        manifest = read_json(path / "run.json")
        if not manifest.get("source_tree_sha256") or manifest["source_tree_sha256"] != current_source_sha256:
            raise ValueError(f"Child run source snapshot differs from the frozen current tree: {path.name}")


def expected_runs(prefix: str) -> dict:
    """Enumerate every required section, donor, method, and seed result."""
    spec = specification()
    result = {
        "harmony": _path(prefix, "harmony-fixed10"),
        "k_selection": _path(prefix, "k-selection"),
        "gr_training": [_path(prefix, f"gr-s{seed}") for seed in range(5)],
        "spagcn_training": [_path(prefix, f"spagcn-{section}-s{seed}")
                            for section in spec["sections"] for seed in range(5)],
        "section_scores": [], "donor_scores": [],
    }
    for section in spec["sections"]:
        result["section_scores"].extend([
            _path(prefix, f"score-section-{section}-harmony-fixed"),
            _path(prefix, f"score-section-{section}-harmony-native"),
        ])
        result["section_scores"].extend(_path(prefix, f"score-section-{section}-gr-s{seed}") for seed in range(5))
        result["section_scores"].extend(_path(prefix, f"score-section-{section}-spagcn-s{seed}") for seed in range(5))
    for donor in spec["donors"]:
        result["donor_scores"].extend([
            _path(prefix, f"score-donor-{donor}-harmony-fixed"),
            _path(prefix, f"score-donor-{donor}-harmony-native"),
        ])
        result["donor_scores"].extend(_path(prefix, f"score-donor-{donor}-gr-s{seed}") for seed in range(5))
    return result


def _load_all(prefix: str) -> tuple[list[dict], list[dict], dict]:
    """Load completed child results and reject missing or inconsistent coverage."""
    expected = expected_runs(prefix)
    runtime_lock = specification()["evaluation"]["runtime"]
    expected_runtime = {
        "python_executable": runtime_lock["python_executable"],
        "python": runtime_lock["python"],
        "packages": runtime_lock["packages"],
        "wsl_distribution": runtime_lock["wsl_distribution"],
        "validated": True,
    }
    require_run(expected["harmony"], "spatial_multisection_harmony_fixed")
    require_run(expected["k_selection"], "spatial_multisection_k_selection")
    for path in expected["gr_training"]:
        require_run(path, "spatial_multisection_genorefine_training")
    for path in expected["spagcn_training"]:
        require_run(path, "spatial_multisection_spagcn_training")
    section_rows, donor_rows = [], []
    for path in expected["section_scores"]:
        require_run(path, "spatial_multisection_section_score")
        if read_json(path / "config.json").get("runtime") != expected_runtime:
            raise ValueError(f"Section score runtime differs from the frozen evaluator: {path.name}")
        row = read_json(path / "summary.json"); row["run"] = str(path); section_rows.append(row)
    for path in expected["donor_scores"]:
        require_run(path, "spatial_multisection_donor_score")
        if read_json(path / "config.json").get("runtime") != expected_runtime:
            raise ValueError(f"Donor score runtime differs from the frozen evaluator: {path.name}")
        row = read_json(path / "summary.json"); row["run"] = str(path); donor_rows.append(row)
    current_source_sha256 = canonical_hash(source_snapshot())
    children = [expected["harmony"], expected["k_selection"], *expected["gr_training"],
                *expected["spagcn_training"], *expected["section_scores"], *expected["donor_scores"]]
    validate_child_source_hashes(children, current_source_sha256)
    return section_rows, donor_rows, expected


def _stats(values: list[float]) -> dict:
    if not values:
        raise ValueError("Cannot summarize an empty endpoint")
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "minimum": min(values), "maximum": max(values), "n_algorithmic_seeds": len(values)}


def aggregate_sections(rows: list[dict]) -> dict:
    """Aggregate section-level metrics across algorithmic seeds."""
    spec = specification()
    primary_methods = ("harmony_fixed", "genorefine", "spagcn")
    summary = {}
    for method in primary_methods:
        method_rows = [row for row in rows if row["method"] == method]
        wanted = 6 if method == "harmony_fixed" else 30
        if len(method_rows) != wanted:
            raise ValueError(f"Incomplete {method} section panel")
        section_summary, donor_summary = {}, {}
        if method == "harmony_fixed":
            for section in spec["sections"]:
                selected = [row for row in method_rows if row["section"] == section]
                if len(selected) != 1 or selected[0]["algorithmic_seed"] is not None:
                    raise ValueError("Fixed Harmony must have one deterministic score per section")
                section_summary[section] = {
                    metric: {"mean": float(selected[0][metric]), "n_algorithmic_seeds": 0,
                             "deterministic_single_embedding": True}
                    for metric in SECTION_METRICS
                }
            for donor, sections in spec["donors"].items():
                donor_summary[donor] = {}
                for metric in SECTION_METRICS:
                    section_values = {
                        section: float(next(row for row in method_rows if row["section"] == section)[metric])
                        for section in sections
                    }
                    donor_summary[donor][metric] = {
                        "mean": statistics.mean(section_values.values()),
                        "section_values": section_values,
                        "n_sections": 2,
                        "n_algorithmic_seeds": 0,
                    }
            macro = {}
            for metric in SECTION_METRICS:
                donor_values = {donor: donor_summary[donor][metric]["mean"] for donor in spec["donors"]}
                macro[metric] = {
                    "mean": statistics.mean(donor_values.values()),
                    "donor_values": donor_values,
                    "n_donors": 3,
                    "n_algorithmic_seeds": 0,
                    "dispersion_across_donors_not_reported_as_seed_sd": True,
                    "aggregation": "section mean within donor, then equal donor macro",
                }
            summary[method] = {"sections": section_summary, "donors": donor_summary, "macro": macro}
            continue
        seed_values = [None] if method == "harmony_fixed" else list(range(5))
        for section in spec["sections"]:
            selected = [row for row in method_rows if row["section"] == section]
            if sorted(row["algorithmic_seed"] for row in selected if row["algorithmic_seed"] is not None) != ([] if method == "harmony_fixed" else list(range(5))):
                raise ValueError("Incomplete/duplicate section seed panel")
            section_summary[section] = {metric: _stats([float(row[metric]) for row in selected]) for metric in SECTION_METRICS}
        macro_by_seed = {metric: [] for metric in SECTION_METRICS}
        for donor, sections in spec["donors"].items():
            donor_summary[donor] = {}
            for metric in SECTION_METRICS:
                values = []
                for seed in seed_values:
                    per_section = []
                    for section in sections:
                        match = [row for row in method_rows if row["section"] == section and row["algorithmic_seed"] == seed]
                        if len(match) != 1:
                            raise ValueError("Donor-first section aggregation is incomplete")
                        per_section.append(float(match[0][metric]))
                    values.append(statistics.mean(per_section))
                donor_summary[donor][metric] = _stats(values)
                macro_by_seed[metric].append(values)
        macro = {}
        for metric, donor_seed_values in macro_by_seed.items():
            by_seed = [statistics.mean(values) for values in zip(*donor_seed_values)]
            macro[metric] = {**_stats(by_seed), "aggregation": "section mean within donor, then equal donor macro"}
        summary[method] = {"sections": section_summary, "donors": donor_summary, "macro": macro}
    sensitivity_rows = [row for row in rows if row["method"] == "harmony_native_sensitivity"]
    if len(sensitivity_rows) != 6:
        raise ValueError("Native-stop Harmony sensitivity is incomplete")
    sensitivity_macro = {}
    for metric in SECTION_METRICS:
        donor_values = [statistics.mean(
            float(next(row for row in sensitivity_rows if row["section"] == section)[metric])
            for section in sections) for sections in spec["donors"].values()]
        sensitivity_macro[metric] = {
            "mean": statistics.mean(donor_values),
            "donor_values": dict(zip(spec["donors"], donor_values)),
            "n_donors": 3,
            "n_algorithmic_seeds": 0,
            "dispersion_across_donors_not_reported_as_seed_sd": True,
        }
    summary["harmony_native_sensitivity"] = {
        "role": "sensitivity_only",
        "macro": sensitivity_macro,
    }
    return summary


def aggregate_donors(rows: list[dict]) -> dict:
    """Aggregate donor-pair metrics across algorithmic seeds."""
    spec = specification()
    summary = {}
    for method in ("harmony_fixed", "genorefine", "harmony_native_sensitivity"):
        method_rows = [row for row in rows if row["method"] == method]
        wanted = 3 if method != "genorefine" else 15
        if len(method_rows) != wanted:
            raise ValueError("Incomplete donor-pair mixing panel")
        seeds = [None] if method != "genorefine" else list(range(5))
        per_donor = {}
        for donor in spec["donors"]:
            selected = [row for row in method_rows if row["donor"] == donor]
            if len(selected) != len(seeds):
                raise ValueError("Incomplete donor mixing seed panel")
            if method == "genorefine":
                per_donor[donor] = {
                    metric: _stats([float(row[metric]) for row in selected])
                    for metric in DONOR_METRICS
                }
            else:
                per_donor[donor] = {
                    metric: {
                        "value": float(selected[0][metric]),
                        "n_algorithmic_seeds": 0,
                        "deterministic_single_embedding": True,
                    }
                    for metric in DONOR_METRICS
                }
        macro = {}
        for metric in DONOR_METRICS:
            if method == "genorefine":
                by_seed = []
                for seed in seeds:
                    by_seed.append(statistics.mean(float(next(
                        row for row in method_rows if row["donor"] == donor and row["algorithmic_seed"] == seed
                    )[metric]) for donor in spec["donors"]))
                macro[metric] = {**_stats(by_seed), "aggregation": "equal macro-average over three donors"}
            else:
                donor_values = {
                    donor: float(next(row for row in method_rows if row["donor"] == donor)[metric])
                    for donor in spec["donors"]
                }
                macro[metric] = {
                    "mean": statistics.mean(donor_values.values()),
                    "donor_values": donor_values,
                    "n_donors": 3,
                    "n_algorithmic_seeds": 0,
                    "dispersion_across_donors_not_reported_as_seed_sd": True,
                    "aggregation": "equal macro-average over three donors",
                }
        summary[method] = {"donors": per_donor, "macro": macro,
                           "role": "sensitivity_only" if method == "harmony_native_sensitivity" else "primary"}
    return summary


def execute(prefix: str, run_id: str) -> Path:
    """Validate and consolidate the complete spatial multisection panel."""
    runtime = validate_evaluation_runtime()
    section_rows, donor_rows, expected = _load_all(prefix)
    section_summary = aggregate_sections(section_rows)
    donor_summary = aggregate_donors(donor_rows)
    sources = source_snapshot()
    inputs = {key: [file_fingerprint(path / "run.json") for path in value] if isinstance(value, list)
              else file_fingerprint(value / "run.json") for key, value in expected.items()}
    context = {"protocol_id": specification()["protocol_id"], "prefix": prefix, "inputs": inputs,
               "aggregation": specification()["aggregation"], "runtime": runtime}
    with RunDirectory(RUNS, kind="spatial_multisection_panel", run_id=run_id, config=context) as run:
        run.write_json("section_summary.json", section_summary)
        run.write_json("donor_mixing_summary.json", donor_summary)
        run.write_json("section_seed_rows.json", section_rows)
        run.write_json("donor_seed_rows.json", donor_rows)
        with run.artifact_path("section_seed_rows.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["section", "donor", "method", "algorithmic_seed", *SECTION_METRICS]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(section_rows)
        lines = [
            "# Six-section DLPFC spatial panel", "",
            "Primary results use the fixed-10 pooled Harmony baseline. The native-stop Harmony result is a sensitivity only.",
            "Five seeds are algorithmic repeats, not biological replicates; no p-values are reported.", "",
            "## Donor-macro section endpoints", "",
            "| Method | ARI | NMI | cluster SIL | reference SIL | purity | spatial kNN Jaccard (k=6) | physical/local ratio |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        display = (("Harmony fixed-10", "harmony_fixed"), ("Harmony + GenoRefine", "genorefine"), ("SpaGCN", "spagcn"))
        keys = ("ARI", "NMI", "predicted_cluster_silhouette", "reference_label_silhouette",
                "label_neighbor_purity", "spatial_latent_knn_jaccard", "physical_distance_ratio_to_local_spatial_knn_mean")
        for label, method in display:
            values = section_summary[method]["macro"]
            cells = [f"{values[key]['mean']:.4f}" + (f" ± {values[key]['sd']:.4f}" if values[key]["n_algorithmic_seeds"] > 1 else "") for key in keys]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines += ["", "## Donor-pair section mixing", "", "| Method | iLISI | D_batch |", "|---|---:|---:|"]
        for label, method in (("Harmony fixed-10", "harmony_fixed"), ("Harmony + GenoRefine", "genorefine")):
            values = donor_summary[method]["macro"]
            cells = [f"{values[key]['mean']:.4f}" + (f" ± {values[key]['sd']:.4f}" if values[key]["n_algorithmic_seeds"] > 1 else "") for key in DONOR_METRICS]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines += ["", "SpaGCN is trained independently per section, so cross-section mixing is not a defined endpoint for that comparator.",
                  "Manual cortical-layer labels are evaluation-only. K is selected from fixed-resolution Harmony partitions without labels."]
        run.artifact_path("report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.write_json("completion.json", {
            "six_sections_complete": True, "three_donors_complete": True,
            "section_scores": len(section_rows), "expected_section_scores": 72,
            "donor_scores": len(donor_rows), "expected_donor_scores": 21,
            "genorefine_seeds": [0, 1, 2, 3, 4], "spagcn_seeds": [0, 1, 2, 3, 4],
            "algorithmic_seeds_not_biological_replicates": True, "p_values_reported": False,
        })
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.manifest.update(scientific_experiment=True, experiment_role="complete_multi_section_spatial_panel",
                            training_performed=False, scoring_performed=False)
        if source_snapshot() != sources:
            raise RuntimeError("Scientific source changed during spatial-panel consolidation")
    return run.final_path


def main() -> None:
    """Parse panel inputs and write the consolidated spatial report."""
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
