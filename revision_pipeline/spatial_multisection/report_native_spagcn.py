"""Derive the preregistered SpaGCN task-native endpoint report from frozen Package 4 outputs.

This is a reporting-only transformation.  It neither refits a model nor rescans
an embedding.  The original Package 4 run directories remain immutable.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import completed
from ..runs import RunDirectory
from .common import ROOT, RUNS, specification


FROZEN_PANEL_SOURCE_SHA256 = "bb0f67f14ca3a460a9f708a920c2b64aa44f526994737d84b6e427b46f387b4f"
BASE_GLOBAL_PREFLIGHT_SHA256 = "4c61b60a8755122c8b5a13a49a61ca2dbbf4b1ab94d28d90b48c945de4c16304"
PANEL_ONLY_SOURCE_FILES = (
    "revision_pipeline/spatial_multisection/Dockerfile.spagcn-gpu",
    "revision_pipeline/spatial_multisection/run_panel.ps1",
)
ENDPOINTS = ("predicted", "refined")
METRICS = ("ARI", "NMI")
SEEDS = tuple(range(5))


def _read_json(path: Path) -> dict | list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stats(values: list[float], *, seeds: list[int]) -> dict:
    if len(values) != len(seeds) or not values:
        raise ValueError("A complete algorithmic-seed vector is required")
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
        "n_algorithmic_seeds": len(values),
        "seed_values": [
            {"algorithmic_seed": int(seed), "value": float(value)}
            for seed, value in zip(seeds, values)
        ],
    }


def extract_native_rows(
    section_rows: list[dict],
    *,
    sections: list[str],
    donors: dict[str, list[str]],
) -> list[dict]:
    """Return one row per section, seed and task-native SpaGCN endpoint."""
    donor_for_section = {
        section: donor for donor, donor_sections in donors.items() for section in donor_sections
    }
    if set(donor_for_section) != set(sections):
        raise ValueError("Donor-to-section mapping is incomplete")
    selected = [row for row in section_rows if row.get("method") == "spagcn"]
    if len(selected) != len(sections) * len(SEEDS):
        raise ValueError("Expected exactly five SpaGCN rows for each of six sections")
    result = []
    seen = set()
    for row in selected:
        section = str(row.get("section"))
        seed = row.get("algorithmic_seed")
        key = (section, seed)
        if section not in sections or seed not in SEEDS or key in seen:
            raise ValueError("SpaGCN section/seed coverage is invalid or duplicated")
        if row.get("donor") != donor_for_section[section]:
            raise ValueError("SpaGCN row donor does not match the frozen mapping")
        native = row.get("native_spagcn_partitions")
        if not isinstance(native, dict) or set(native) != set(ENDPOINTS):
            raise ValueError("SpaGCN task-native endpoints are missing")
        for endpoint in ENDPOINTS:
            values = native[endpoint]
            if values.get("role") != "secondary task-native SpaGCN partition":
                raise ValueError("SpaGCN native endpoint role changed")
            if any(metric not in values for metric in METRICS):
                raise ValueError("SpaGCN native ARI/NMI is incomplete")
            result.append({
                "section": section,
                "donor": donor_for_section[section],
                "algorithmic_seed": int(seed),
                "endpoint": endpoint,
                "ARI": float(values["ARI"]),
                "NMI": float(values["NMI"]),
                "clusters": int(values["clusters"]),
                "role": values["role"],
            })
        seen.add(key)
    expected = {(section, seed) for section in sections for seed in SEEDS}
    if seen != expected:
        raise ValueError("SpaGCN section/seed coverage is incomplete")
    return sorted(
        result,
        key=lambda row: (
            sections.index(row["section"]), row["algorithmic_seed"], ENDPOINTS.index(row["endpoint"]),
        ),
    )


def aggregate_native_rows(
    rows: list[dict],
    *,
    sections: list[str],
    donors: dict[str, list[str]],
) -> tuple[dict, list[dict], list[dict]]:
    """Apply seed-within-section, sections-within-donor, donor-macro aggregation."""
    lookup = {
        (row["endpoint"], row["section"], row["algorithmic_seed"]): row
        for row in rows
    }
    if len(lookup) != len(ENDPOINTS) * len(sections) * len(SEEDS):
        raise ValueError("Native endpoint rows are incomplete or duplicated")

    section_summary: dict[str, dict] = {}
    for section in sections:
        section_summary[section] = {}
        for endpoint in ENDPOINTS:
            section_summary[section][endpoint] = {}
            for metric in METRICS:
                values = [float(lookup[(endpoint, section, seed)][metric]) for seed in SEEDS]
                section_summary[section][endpoint][metric] = _stats(values, seeds=list(SEEDS))
            clusters = [int(lookup[(endpoint, section, seed)]["clusters"]) for seed in SEEDS]
            section_summary[section][endpoint]["clusters"] = {
                "values_by_seed": [
                    {"algorithmic_seed": seed, "value": value}
                    for seed, value in zip(SEEDS, clusters)
                ],
                "unique_values": sorted(set(clusters)),
            }

    donor_seed_rows: list[dict] = []
    donor_summary: dict[str, dict] = {}
    for donor, donor_sections in donors.items():
        if not donor_sections or any(section not in sections for section in donor_sections):
            raise ValueError("Invalid donor section group")
        donor_summary[donor] = {}
        for endpoint in ENDPOINTS:
            endpoint_rows = []
            for seed in SEEDS:
                row = {
                    "donor": donor,
                    "algorithmic_seed": seed,
                    "endpoint": endpoint,
                    "n_sections": len(donor_sections),
                }
                for metric in METRICS:
                    row[metric] = statistics.mean(
                        float(lookup[(endpoint, section, seed)][metric])
                        for section in donor_sections
                    )
                endpoint_rows.append(row)
                donor_seed_rows.append(row)
            donor_summary[donor][endpoint] = {
                metric: _stats(
                    [float(row[metric]) for row in endpoint_rows], seeds=list(SEEDS)
                )
                for metric in METRICS
            }
            donor_summary[donor][endpoint]["aggregation"] = (
                "equal mean over the donor's frozen sections within each algorithmic seed; "
                "descriptive statistics across the five seed-specific means"
            )

    macro_seed_rows: list[dict] = []
    macro_summary: dict[str, dict] = {}
    for endpoint in ENDPOINTS:
        endpoint_rows = []
        for seed in SEEDS:
            row = {
                "algorithmic_seed": seed,
                "endpoint": endpoint,
                "n_donors": len(donors),
            }
            for metric in METRICS:
                row[metric] = statistics.mean(
                    next(
                        donor_row[metric]
                        for donor_row in donor_seed_rows
                        if donor_row["donor"] == donor
                        and donor_row["endpoint"] == endpoint
                        and donor_row["algorithmic_seed"] == seed
                    )
                    for donor in donors
                )
            endpoint_rows.append(row)
            macro_seed_rows.append(row)
        macro_summary[endpoint] = {
            metric: _stats([float(row[metric]) for row in endpoint_rows], seeds=list(SEEDS))
            for metric in METRICS
        }
        macro_summary[endpoint]["aggregation"] = (
            "for each algorithmic seed: equal section mean within donor, then equal macro-average "
            "over three donors; descriptive statistics are then calculated over seeds"
        )

    summary = {
        "schema_version": 1,
        "method": "SpaGCN 1.2.7",
        "endpoint_role": "secondary task-native partitions; separate from common fixed-resolution evaluator",
        "aggregation_order": [
            "retain algorithmic-seed values within each section",
            "within each seed, equally average the two frozen sections for each donor",
            "within each seed, equally macro-average the three donor means",
            "report descriptive mean and sample SD across five algorithmic seeds",
        ],
        "seed_role": "descriptive algorithmic repeats, not biological replicates",
        "p_values_reported": False,
        "sections": section_summary,
        "donors": donor_summary,
        "macro": macro_summary,
    }
    return summary, donor_seed_rows, macro_seed_rows


def _validate_frozen_inputs(full_panel: Path, prefix: str) -> tuple[list[dict], dict]:
    manifest = completed(full_panel, "spatial_multisection_panel")
    if manifest.get("source_tree_sha256") != FROZEN_PANEL_SOURCE_SHA256:
        raise ValueError("Full panel does not carry the authoritative frozen Package 4 source hash")
    source_manifest_path = full_panel / "source_manifest.json"
    frozen_sources = _read_json(source_manifest_path)
    if canonical_hash(frozen_sources) != FROZEN_PANEL_SOURCE_SHA256:
        raise ValueError("Frozen Package 4 source manifest does not reproduce its authoritative hash")
    base_sources = dict(frozen_sources)
    for relative in PANEL_ONLY_SOURCE_FILES:
        if relative not in base_sources:
            raise ValueError(f"Panel-only source entry is missing: {relative}")
        del base_sources[relative]
    if canonical_hash(base_sources) != BASE_GLOBAL_PREFLIGHT_SHA256:
        raise ValueError("Panel source manifest does not reproduce the recorded base/global preflight hash")

    preflight = RUNS / f"_preflight-{prefix}" / "preflight_receipt.json"
    receipt = _read_json(preflight)
    if receipt.get("current_frozen_source_tree_sha256") != BASE_GLOBAL_PREFLIGHT_SHA256:
        raise ValueError("Original preflight receipt no longer records the base/global snapshot hash")

    rows_path = full_panel / "section_seed_rows.json"
    rows = _read_json(rows_path)
    selected = [row for row in rows if row.get("method") == "spagcn"]
    for row in selected:
        section = str(row["section"])
        seed = int(row["algorithmic_seed"])
        child = RUNS / f"{prefix}-score-section-{section}-spagcn-s{seed}"
        child_manifest = completed(child, "spatial_multisection_section_score")
        if child_manifest.get("source_tree_sha256") != FROZEN_PANEL_SOURCE_SHA256:
            raise ValueError(f"Child score source hash differs: {child.name}")
        stored = dict(row)
        run_reference = stored.pop("run")
        if Path(str(run_reference).replace("\\", "/")).name != child.name:
            raise ValueError(f"Consolidated row points to another child run: {child.name}")
        if stored != _read_json(child / "summary.json"):
            raise ValueError(f"Consolidated row differs from its immutable child summary: {child.name}")

    foundation = ROOT / specification()["foundation"]["path"]
    foundation_manifest = completed(foundation, "spatial_multisection_foundation")
    foundation_summary = _read_json(foundation / "section_summary.json")
    if (foundation_summary.get("labeled_spots_retained") != 22968
            or foundation_summary.get("missing_labels_excluded") != 113):
        raise ValueError("Frozen cohort label-availability counts changed")

    inputs = {
        "full_panel_run": file_fingerprint(full_panel / "run.json"),
        "full_panel_section_seed_rows": file_fingerprint(rows_path),
        "full_panel_source_manifest": file_fingerprint(source_manifest_path),
        "original_preflight_receipt": file_fingerprint(preflight),
        "foundation_run": file_fingerprint(foundation / "run.json"),
        "foundation_section_summary": file_fingerprint(foundation / "section_summary.json"),
        "foundation_excluded_unlabeled_spots": file_fingerprint(
            foundation / "excluded_unlabeled_spots.tsv"
        ),
        "child_spagcn_section_scores": {
            f"{row['section']}-seed{row['algorithmic_seed']}": {
                "run": file_fingerprint(
                    RUNS / f"{prefix}-score-section-{row['section']}-spagcn-s{row['algorithmic_seed']}" / "run.json"
                ),
                "summary": file_fingerprint(
                    RUNS / f"{prefix}-score-section-{row['section']}-spagcn-s{row['algorithmic_seed']}" / "summary.json"
                ),
            }
            for row in selected
        },
    }
    return rows, {
        "inputs": inputs,
        "foundation_manifest": foundation_manifest,
        "frozen_source_manifest": frozen_sources,
        "original_preflight_receipt": receipt,
    }


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def execute(*, full_panel: Path, prefix: str, run_id: str) -> Path:
    section_rows, validation = _validate_frozen_inputs(full_panel, prefix)
    spec = specification()
    native_rows = extract_native_rows(
        section_rows, sections=spec["sections"], donors=spec["donors"]
    )
    summary, donor_seed_rows, macro_seed_rows = aggregate_native_rows(
        native_rows, sections=spec["sections"], donors=spec["donors"]
    )

    clarification_path = RUNS / f"_preflight-{prefix}" / "source_hash_scope_clarification.json"
    clarification = _read_json(clarification_path)
    reporting_sources = {
        "revision_pipeline/spatial_multisection/report_native_spagcn.py": file_fingerprint(Path(__file__)),
        "revision_pipeline/spatial_multisection/tests/test_native_reporting.py": file_fingerprint(
            Path(__file__).parent / "tests/test_native_reporting.py"
        ),
    }
    provenance = {
        "schema_version": 1,
        "derivation_only": True,
        "training_performed": False,
        "embedding_scoring_performed": False,
        "original_scientific_run_artifacts_modified": False,
        "frozen_panel_source_tree_sha256": FROZEN_PANEL_SOURCE_SHA256,
        "base_global_preflight_source_tree_sha256": BASE_GLOBAL_PREFLIGHT_SHA256,
        "source_hash_scope_clarification": {
            "path": clarification_path.relative_to(ROOT).as_posix(),
            "fingerprint": file_fingerprint(clarification_path),
            "record": clarification,
        },
        "reporting_sources": reporting_sources,
        "reporting_sources_sha256": canonical_hash(reporting_sources),
        **validation["inputs"],
        "aggregation": summary["aggregation_order"],
    }
    boundaries = {
        "label_policy": {
            "cohort_freeze": (
                "Layer-label availability defined the frozen evaluation cohort: 22,968 labeled spots "
                "were retained and 113 spots without a manual cortical-layer identity were excluded and recorded."
            ),
            "fitting_and_selection": (
                "Cortical-layer identity values were withheld from preprocessing, Harmony, K selection, "
                "GenoRefine fitting, SpaGCN fitting, common graph construction, and all model selection."
            ),
            "evaluation": (
                "Manual cortical-layer identities were introduced only after fitting and graph construction "
                "to calculate evaluation endpoints."
            ),
        },
        "fixed_resolution_cluster_granularity": (
            "The common-evaluator ARI and NMI use Leiden resolution 0.5 and three Leiden seeds for every "
            "continuous representation, but the resulting number of clusters may differ by method and section. "
            "Those endpoints therefore reflect both representation geometry and fixed-resolution partition "
            "granularity. SpaGCN predicted and hex-refined partitions are reported separately as task-native "
            "secondary endpoints and are not interchangeable with the common-evaluator partitions."
        ),
        "spatial_preservation": (
            "Spatial kNN Jaccard at k=6 and physical distance among latent neighbors quantify local spatial "
            "continuity. They do not test cortical-layer boundary fidelity, histological boundary agreement, "
            "trajectory preservation, or general spatial performance beyond these six frozen sections."
        ),
        "spagcn_mixing": (
            "SpaGCN was fitted independently within each section, so donor-pair cross-section iLISI and "
            "D_batch are not defined for this comparator."
        ),
    }
    context = {
        "role": "post-completion reporting derivation",
        "prefix": prefix,
        "full_panel": full_panel.relative_to(ROOT).as_posix(),
        "frozen_panel_source_tree_sha256": FROZEN_PANEL_SOURCE_SHA256,
        "reporting_sources_sha256": provenance["reporting_sources_sha256"],
        "aggregation": summary["aggregation_order"],
    }
    with RunDirectory(
        RUNS,
        kind="spatial_multisection_native_spagcn_reporting",
        run_id=run_id,
        config=context,
    ) as run:
        run.write_json("native_spagcn_summary.json", summary)
        run.write_json("native_spagcn_section_seed_rows.json", native_rows)
        run.write_json("native_spagcn_donor_seed_rows.json", donor_seed_rows)
        run.write_json("native_spagcn_macro_seed_rows.json", macro_seed_rows)
        run.write_json("evidence_boundaries.json", boundaries)
        run.write_json("provenance.json", provenance)
        _write_csv(
            run.artifact_path("native_spagcn_section_seed_rows.csv"), native_rows,
            ["section", "donor", "algorithmic_seed", "endpoint", "ARI", "NMI", "clusters", "role"],
        )
        _write_csv(
            run.artifact_path("native_spagcn_donor_seed_rows.csv"), donor_seed_rows,
            ["donor", "algorithmic_seed", "endpoint", "n_sections", "ARI", "NMI"],
        )
        _write_csv(
            run.artifact_path("native_spagcn_macro_seed_rows.csv"), macro_seed_rows,
            ["algorithmic_seed", "endpoint", "n_donors", "ARI", "NMI"],
        )
        predicted = summary["macro"]["predicted"]
        refined = summary["macro"]["refined"]
        lines = [
            "# Package 4 SpaGCN task-native endpoints", "",
            "These values are a provenance-linked derivation from the immutable completed Package 4 scores. "
            "No model was retrained and no embedding was rescored.", "",
            "| Secondary SpaGCN endpoint | ARI | NMI |",
            "|---|---:|---:|",
            f"| Native predicted partition | {predicted['ARI']['mean']:.4f} ± {predicted['ARI']['sd']:.4f} | "
            f"{predicted['NMI']['mean']:.4f} ± {predicted['NMI']['sd']:.4f} |",
            f"| Native hex-refined partition | {refined['ARI']['mean']:.4f} ± {refined['ARI']['sd']:.4f} | "
            f"{refined['NMI']['mean']:.4f} ± {refined['NMI']['sd']:.4f} |", "",
            "Aggregation is performed within each seed: equal mean over two sections per donor, then an "
            "equal macro-average over the three donors. Mean ± sample SD is descriptive across seeds 0–4; "
            "the seeds are algorithmic repeats, not biological replicates.", "",
            "These task-native partitions remain separate from the fixed-resolution common-evaluator ARI/NMI. "
            "See evidence_boundaries.json for the label-policy, cluster-granularity, and spatial-preservation limits.",
        ]
        run.artifact_path("report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.manifest.update(
            scientific_experiment=False,
            derivation_only=True,
            training_performed=False,
            scoring_performed=False,
            source_scientific_run_id=full_panel.name,
            source_scientific_run_unchanged=True,
            frozen_panel_source_tree_sha256=FROZEN_PANEL_SOURCE_SHA256,
            reporting_sources_sha256=provenance["reporting_sources_sha256"],
        )
    return run.final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-panel", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required")
    print(execute(full_panel=args.full_panel.resolve(), prefix=args.prefix, run_id=args.run_id))


if __name__ == "__main__":
    main()
