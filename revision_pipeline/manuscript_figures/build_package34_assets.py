# Purpose: Build provenance-linked manuscript assets from frozen Packages 3, 4, and 4b.
# Author: Ariana Rahman (Arizona State University)

"""Build provenance-linked manuscript assets from frozen Packages 3, 4, and 4b.

The builder is deliberately downstream-only.  It validates every artifact recorded by
the four completed source-run manifests, recomputes displayed aggregates from the
saved seed-level rows, and writes figures, tables, and provenance into a staging
directory.  It never writes into a scientific run directory or imports manuscript
LaTeX.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "revision_pipeline" / "runs"
DEFAULT_OUT = Path(__file__).resolve().parent / "staging" / "package34_audited"

PACKAGE3 = RUNS / "20260929T192600Z-artifactv2fix2-full-panel"
PACKAGE4 = RUNS / "20260929-spatial-panel-v1-full-panel"
PACKAGE4_NATIVE = RUNS / "20260930-package4-spagcn-native-reporting-v1"
PACKAGE4B = RUNS / "20260930-graphst4b-v5-full-panel"

EXPECTED_RUNS = {
    PACKAGE3: {
        "kind": "artifact_validation_v2_panel",
        "source": "4c61b60a8755122c8b5a13a49a61ca2dbbf4b1ab94d28d90b48c945de4c16304",
    },
    PACKAGE4: {
        "kind": "spatial_multisection_panel",
        "source": "bb0f67f14ca3a460a9f708a920c2b64aa44f526994737d84b6e427b46f387b4f",
    },
    PACKAGE4_NATIVE: {
        "kind": "spatial_multisection_native_spagcn_reporting",
        "source": None,
    },
    PACKAGE4B: {
        "kind": "spatial_graphst_panel",
        "source": "4d24edecd74a6814c4e09fc13c14b47bd2798d39fcbcf93a3fe2d4c46a853c79",
    },
}

P3_FOUNDATION_HASH = "e83af7b23ae9a553d6b28e12dd1ed967f3613f713fa7f3e47e5acaaa3cab8c46"
P3_SCORER_HASH = "4c61b60a8755122c8b5a13a49a61ca2dbbf4b1ab94d28d90b48c945de4c16304"
P4_HASH = "bb0f67f14ca3a460a9f708a920c2b64aa44f526994737d84b6e427b46f387b4f"
P4B_PROTOCOL_HASH = "b20875a3c5ccb4f0ff7f164587e5836e1db715ae4e7838bf22e64040b02b11ae"
P4B_SOURCE_HASH = "4d24edecd74a6814c4e09fc13c14b47bd2798d39fcbcf93a3fe2d4c46a853c79"

DONOR_SECTIONS = {
    "Br5292": ("151507", "151508"),
    "Br5595": ("151669", "151670"),
    "Br8100": ("151673", "151674"),
}
SECTION_TO_DONOR = {
    section: donor for donor, sections in DONOR_SECTIONS.items() for section in sections
}

COMMON_METRICS = (
    "ARI",
    "NMI",
    "predicted_cluster_silhouette",
    "reference_label_silhouette",
    "label_neighbor_purity",
    "spatial_latent_knn_jaccard",
    "physical_distance_among_latent_neighbors_mean_fullres_pixels",
    "physical_distance_ratio_to_local_spatial_knn_mean",
)
MIXING_METRICS = ("iLISI", "D_batch")

DISPLAY_METHODS = ("harmony", "genorefine", "spagcn", "graphst")
METHOD_LABELS = {
    "harmony": "Harmony",
    "genorefine": "Harmony +\nGenoRefine",
    "spagcn": "SpaGCN",
    "graphst": "GraphST",
}
METHOD_COLORS = {
    "harmony": "#4B5563",
    "genorefine": "#1769AA",
    "spagcn": "#D97706",
    "graphst": "#7C3AED",
}
P3_COLORS = {"genorefine": "#1769AA", "idec": "#D97706"}

METRIC_LABELS = {
    "ARI": "ARI",
    "NMI": "NMI",
    "predicted_cluster_silhouette": "Cluster\nSilhouette",
    "reference_label_silhouette": "Reference-label\nSilhouette",
    "label_neighbor_purity": "Label-neighbor\npurity",
    "spatial_latent_knn_jaccard": "Spatial-latent\nkNN Jaccard",
    "physical_distance_among_latent_neighbors_mean_fullres_pixels": "Physical distance\namong latent neighbors",
    "physical_distance_ratio_to_local_spatial_knn_mean": "Physical/local\ndistance ratio (lower)",
    "iLISI": "iLISI",
    "D_batch": "$D_{batch}$",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    return {"sha256": sha256(path), "size_bytes": path.stat().st_size}


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def require_close(actual: float, expected: float, label: str, atol: float = 1e-12) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol):
        raise ValueError(f"Numeric mismatch for {label}: {actual!r} != {expected!r}")


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return statistics.mean(values)


def reduce_values(values: Iterable[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("Cannot reduce an empty sequence")
    return {
        "mean": mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "minimum": min(values),
        "maximum": max(values),
        "n_algorithmic_seeds": len(values) if len(values) > 1 else 0,
        "seed_values": values,
    }


def validate_run(run_dir: Path, expected: dict[str, Any]) -> dict[str, Any]:
    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    if manifest.get("status") != "succeeded" or manifest.get("kind") != expected["kind"]:
        raise ValueError(
            f"Run is not the expected completed run: {run_dir}; "
            f"status={manifest.get('status')!r}, kind={manifest.get('kind')!r}"
        )
    if expected["source"] is not None and manifest.get("source_tree_sha256") != expected["source"]:
        raise ValueError(f"Frozen source hash mismatch: {run_dir}")
    artifacts = manifest.get("artifacts", {})
    if not artifacts:
        raise ValueError(f"No recorded artifacts: {manifest_path}")
    for name, recorded in artifacts.items():
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if fingerprint(path) != recorded:
            raise ValueError(f"Artifact fingerprint mismatch: {path}")
    return manifest


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.6,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 8.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def audit_package3() -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Audit the controlled-artifact panel and return display rows, criteria, and provenance."""
    config = read_json(PACKAGE3 / "config.json")
    summary = read_json(PACKAGE3 / "summary.json")
    rows = read_csv(PACKAGE3 / "seed_level.csv")
    lineage = config.get("source_lineage", {})
    if lineage != {
        "prepared": P3_FOUNDATION_HASH,
        "baseline": P3_FOUNDATION_HASH,
        "training": P3_FOUNDATION_HASH,
        "candidate_score": P3_SCORER_HASH,
        "consolidation": P3_SCORER_HASH,
    }:
        raise ValueError("Package 3 source lineage changed")
    if summary.get("protocol_id") != "artifact_validation_v2" or len(summary.get("records", [])) != 24:
        raise ValueError("Package 3 coverage changed")
    if len(rows) != 120:
        raise ValueError(f"Expected 120 Package 3 seed rows, found {len(rows)}")

    by_key: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_key[(row["case_id"], row["artifact_id"], row["method"])].append(row)
    if any(len(group) != 5 for group in by_key.values()) or len(by_key) != 24:
        raise ValueError("Package 3 seed coverage is incomplete")

    detail: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for index, record in enumerate(summary["records"]):
        key = (record["case_id"], record["artifact_id"], record["method"])
        group = sorted(by_key[key], key=lambda row: int(row["replicate_seed"]))
        seeds = [int(row["replicate_seed"]) for row in group]
        if seeds != list(range(5)):
            raise ValueError(f"Package 3 seed set changed for {key}: {seeds}")
        sensitivity_values = [float(row["sensitivity_before"]) - float(row["sensitivity_after"]) for row in group]
        recovery_values = [float(row["clean_neighbor_jaccard_after"]) - float(row["clean_neighbor_jaccard_before"]) for row in group]
        ari_deltas = [float(row["delta_ARI"]) for row in group]
        purity_deltas = [float(row["delta_purity"]) for row in group]
        target_deltas = [float(row["delta_target_same_class_fraction"]) for row in group]

        screen = record["screen"]
        require_close(mean(sensitivity_values), screen["sensitivity"]["mean_reduction"], f"{key}/sensitivity")
        require_close(mean(recovery_values), screen["clean_neighbor_recovery"]["mean_gain"], f"{key}/recovery")
        if sum(value > 0 for value in sensitivity_values) != screen["sensitivity"]["positive_seeds"]:
            raise ValueError(f"Package 3 sensitivity positive-seed count changed for {key}")
        if sum(value > 0 for value in recovery_values) != screen["clean_neighbor_recovery"]["positive_seeds"]:
            raise ValueError(f"Package 3 recovery positive-seed count changed for {key}")
        for seed, row_record in enumerate(record["seed_rows"]):
            require_close(ari_deltas[seed], row_record["delta_ARI"], f"{key}/seed{seed}/ARI")
            require_close(purity_deltas[seed], row_record["delta_purity"], f"{key}/seed{seed}/purity")
            require_close(target_deltas[seed], row_record["delta_target_same_class_fraction"], f"{key}/seed{seed}/target")

        item = {
            "case_id": record["case_id"],
            "dataset": record["display_dataset"],
            "backbone": record["backbone"],
            "artifact": record["artifact_id"],
            "method": record["method"],
            "full_screen_passed": bool(screen["passed"]),
            "sensitivity_passed": bool(screen["sensitivity"]["passed"]),
            "sensitivity_mean_reduction": mean(sensitivity_values),
            "sensitivity_positive_seeds": sum(value > 0 for value in sensitivity_values),
            "recovery_passed": bool(screen["clean_neighbor_recovery"]["passed"]),
            "recovery_mean_gain": mean(recovery_values),
            "recovery_positive_seeds": sum(value > 0 for value in recovery_values),
            "preservation_passed": bool(screen["preservation"]["passed"]),
            "mean_delta_ARI": mean(ari_deltas),
            "mean_delta_purity": mean(purity_deltas),
            "mean_delta_target_same_class_fraction": mean(target_deltas),
        }
        detail.append(item)
        for endpoint in (
            "full_screen_passed", "sensitivity_passed", "recovery_passed", "preservation_passed",
            "sensitivity_mean_reduction", "recovery_mean_gain", "mean_delta_ARI", "mean_delta_purity",
        ):
            provenance.append({
                "asset_family": "package3_artifact_screen",
                "case_id": record["case_id"],
                "artifact": record["artifact_id"],
                "method": record["method"],
                "endpoint": endpoint,
                "display_value": item[endpoint],
                "consolidated_source": {
                    "path": rel(PACKAGE3 / "summary.json"),
                    "sha256": sha256(PACKAGE3 / "summary.json"),
                    "json_pointer": f"/records/{index}/screen",
                },
                "recomputation_source": {
                    "path": rel(PACKAGE3 / "seed_level.csv"),
                    "sha256": sha256(PACKAGE3 / "seed_level.csv"),
                    "row_key": {"case_id": key[0], "artifact_id": key[1], "method": key[2]},
                },
            })

    criteria = {
        method: {
            "conditions": 12,
            "full_screen_passed": sum(row["full_screen_passed"] for row in detail if row["method"] == method),
            "sensitivity_passed": sum(row["sensitivity_passed"] for row in detail if row["method"] == method),
            "recovery_passed": sum(row["recovery_passed"] for row in detail if row["method"] == method),
            "preservation_passed": sum(row["preservation_passed"] for row in detail if row["method"] == method),
        }
        for method in ("genorefine", "idec")
    }
    if criteria["genorefine"] != {
        "conditions": 12, "full_screen_passed": 0, "sensitivity_passed": 5,
        "recovery_passed": 0, "preservation_passed": 0,
    }:
        raise ValueError(f"Unexpected GenoRefine Package 3 outcome: {criteria['genorefine']}")
    if criteria["idec"] != {
        "conditions": 12, "full_screen_passed": 0, "sensitivity_passed": 8,
        "recovery_passed": 0, "preservation_passed": 0,
    }:
        raise ValueError(f"Unexpected IDEC Package 3 outcome: {criteria['idec']}")
    return detail, criteria, provenance


def macro_from_section_rows(
    rows: list[dict[str, Any]], method: str, metrics: Iterable[str]
) -> dict[str, dict[str, Any]]:
    selected = [row for row in rows if row["method"] == method]
    if not selected:
        raise ValueError(f"No section rows for method {method}")
    seeds = sorted({row["algorithmic_seed"] for row in selected}, key=lambda value: -1 if value is None else int(value))
    expected = [None] if seeds == [None] else list(range(5))
    if seeds != expected:
        raise ValueError(f"Unexpected seed set for {method}: {seeds}")
    result: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        seed_values = []
        for seed in seeds:
            seed_rows = [row for row in selected if row["algorithmic_seed"] == seed]
            if {row["section"] for row in seed_rows} != set(SECTION_TO_DONOR):
                raise ValueError(f"Section coverage changed for {method}/seed={seed}")
            donor_values = []
            for donor, sections in DONOR_SECTIONS.items():
                donor_rows = [row for row in seed_rows if row["section"] in sections]
                if len(donor_rows) != 2 or any(row["donor"] != donor for row in donor_rows):
                    raise ValueError(f"Donor coverage changed for {method}/{donor}/seed={seed}")
                donor_values.append(mean(float(row[metric]) for row in donor_rows))
            seed_values.append(mean(donor_values))
        result[metric] = reduce_values(seed_values)
    return result


def cluster_macro_from_section_rows(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method]
    seeds = sorted({row["algorithmic_seed"] for row in selected}, key=lambda value: -1 if value is None else int(value))
    values = []
    for seed in seeds:
        seed_rows = [row for row in selected if row["algorithmic_seed"] == seed]
        donor_values = []
        for sections in DONOR_SECTIONS.values():
            section_values = []
            for row in seed_rows:
                if row["section"] in sections:
                    clusters = row.get("clusters")
                    if not isinstance(clusters, list) or len(clusters) != 3:
                        raise ValueError(f"Missing three-Leiden-seed cluster counts for {method}")
                    section_values.append(mean(float(value) for value in clusters))
            if len(section_values) != 2:
                raise ValueError(f"Cluster-count section coverage changed for {method}")
            donor_values.append(mean(section_values))
        values.append(mean(donor_values))
    return reduce_values(values)


def macro_from_donor_rows(rows: list[dict[str, Any]], method: str) -> dict[str, dict[str, Any]]:
    selected = [row for row in rows if row["method"] == method]
    seeds = sorted({row["algorithmic_seed"] for row in selected}, key=lambda value: -1 if value is None else int(value))
    expected = [None] if seeds == [None] else list(range(5))
    if seeds != expected:
        raise ValueError(f"Unexpected donor seed set for {method}: {seeds}")
    result = {}
    for metric in MIXING_METRICS:
        seed_values = []
        for seed in seeds:
            seed_rows = [row for row in selected if row["algorithmic_seed"] == seed]
            if {row["donor"] for row in seed_rows} != set(DONOR_SECTIONS):
                raise ValueError(f"Donor coverage changed for {method}/seed={seed}")
            seed_values.append(mean(float(row[metric]) for row in seed_rows))
        result[metric] = reduce_values(seed_values)
    return result


def compare_macro(actual: dict[str, dict[str, Any]], recorded: dict[str, Any], label: str) -> None:
    for metric, reduced in actual.items():
        require_close(reduced["mean"], recorded[metric]["mean"], f"{label}/{metric}/mean")
        if reduced["sd"] is not None:
            require_close(reduced["sd"], recorded[metric]["sd"], f"{label}/{metric}/sd")
            require_close(reduced["minimum"], recorded[metric]["minimum"], f"{label}/{metric}/min")
            require_close(reduced["maximum"], recorded[metric]["maximum"], f"{label}/{metric}/max")


def native_macro_from_rows(rows: list[dict[str, Any]], partition: str) -> dict[str, dict[str, Any]]:
    selected = [row for row in rows if row["partition"] == partition]
    output = {}
    for metric in ("ARI", "NMI", "clusters"):
        by_seed = []
        for seed in range(5):
            seed_rows = [row for row in selected if int(row["algorithmic_seed"]) == seed]
            donor_values = []
            for donor, sections in DONOR_SECTIONS.items():
                donor_rows = [row for row in seed_rows if row["section"] in sections]
                if len(donor_rows) != 2 or any(row["donor"] != donor for row in donor_rows):
                    raise ValueError(f"Native coverage changed for {partition}/{donor}/seed={seed}")
                donor_values.append(mean(float(row[metric]) for row in donor_rows))
            by_seed.append(mean(donor_values))
        output[metric] = reduce_values(by_seed)
    return output


def audit_spatial() -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
    list[dict[str, Any]],
]:
    """Validate donor-balanced common, mixing, and native spatial summaries."""
    p4_section_rows = read_json(PACKAGE4 / "section_seed_rows.json")
    p4_section_summary = read_json(PACKAGE4 / "section_summary.json")
    p4_donor_rows = read_json(PACKAGE4 / "donor_seed_rows.json")
    p4_mixing_summary = read_json(PACKAGE4 / "donor_mixing_summary.json")
    p4_native = read_json(PACKAGE4_NATIVE / "native_spagcn_summary.json")
    p4_native_rows = read_csv(PACKAGE4_NATIVE / "native_spagcn_macro_seed_rows.csv")

    p4b_section_rows = read_json(PACKAGE4B / "section_seed_rows.json")
    p4b_section_summary = read_json(PACKAGE4B / "section_summary.json")
    p4b_donor_rows = read_json(PACKAGE4B / "donor_seed_rows.json")
    p4b_mixing_summary = read_json(PACKAGE4B / "donor_mixing_summary.json")
    p4b_native_rows = read_json(PACKAGE4B / "section_seed_rows.json")
    p4b_native_summary = read_json(PACKAGE4B / "native_partition_summary.json")

    if len(p4_section_rows) != 72 or len(p4b_section_rows) != 30:
        raise ValueError("Spatial section-score coverage changed")
    common = {
        "harmony": macro_from_section_rows(p4_section_rows, "harmony_fixed", COMMON_METRICS),
        "genorefine": macro_from_section_rows(p4_section_rows, "genorefine", COMMON_METRICS),
        "spagcn": macro_from_section_rows(p4_section_rows, "spagcn", COMMON_METRICS),
        "graphst": macro_from_section_rows(p4b_section_rows, "graphst", COMMON_METRICS),
    }
    compare_macro(common["harmony"], p4_section_summary["harmony_fixed"]["macro"], "Package4/Harmony")
    compare_macro(common["genorefine"], p4_section_summary["genorefine"]["macro"], "Package4/GenoRefine")
    compare_macro(common["spagcn"], p4_section_summary["spagcn"]["macro"], "Package4/SpaGCN")
    compare_macro(common["graphst"], p4b_section_summary["macro"], "Package4b/GraphST")

    clusters = {
        "harmony": cluster_macro_from_section_rows(p4_section_rows, "harmony_fixed"),
        "genorefine": cluster_macro_from_section_rows(p4_section_rows, "genorefine"),
        "spagcn": cluster_macro_from_section_rows(p4_section_rows, "spagcn"),
        "graphst": cluster_macro_from_section_rows(p4b_section_rows, "graphst"),
    }
    expected_cluster_means = {"harmony": 5.277777777777778, "genorefine": 3.2, "spagcn": 13.255555555555555, "graphst": 6.5}
    for method, expected in expected_cluster_means.items():
        require_close(clusters[method]["mean"], expected, f"common cluster profile/{method}")
        common[method]["common_cluster_count"] = clusters[method]

    mixing = {
        "harmony": macro_from_donor_rows(p4_donor_rows, "harmony_fixed"),
        "genorefine": macro_from_donor_rows(p4_donor_rows, "genorefine"),
        "graphst": macro_from_donor_rows(p4b_donor_rows, "graphst"),
    }
    compare_macro(mixing["harmony"], p4_mixing_summary["harmony_fixed"]["macro"], "Package4/Harmony mixing")
    compare_macro(mixing["genorefine"], p4_mixing_summary["genorefine"]["macro"], "Package4/GenoRefine mixing")
    compare_macro(mixing["graphst"], p4b_mixing_summary["macro"], "Package4b/GraphST mixing")

    native = {
        "spagcn_predicted": {},
        "spagcn_refined": {},
        "graphst_mclust": {},
        "graphst_refined": {},
    }
    for partition in ("predicted", "refined"):
        partition_rows = [row for row in p4_native_rows if row["endpoint"] == partition]
        for metric in ("ARI", "NMI"):
            values = [float(row[metric]) for row in sorted(partition_rows, key=lambda row: int(row["algorithmic_seed"]))]
            reduced = reduce_values(values)
            recorded = p4_native["macro"][partition][metric]
            require_close(reduced["mean"], recorded["mean"], f"SpaGCN native/{partition}/{metric}/mean")
            require_close(reduced["sd"], recorded["sd"], f"SpaGCN native/{partition}/{metric}/sd")
            native[f"spagcn_{partition}"][metric] = reduced

    graphst_native_rows = []
    for row in p4b_native_rows:
        for partition, values in row["native_graphst_partitions"].items():
            graphst_native_rows.append({
                "section": row["section"], "donor": row["donor"],
                "algorithmic_seed": row["algorithmic_seed"], "partition": partition,
                "ARI": values["ARI"], "NMI": values["NMI"], "clusters": values["clusters"],
            })
    for partition in ("mclust", "refined"):
        reduced_metrics = native_macro_from_rows(graphst_native_rows, partition)
        compare_macro(reduced_metrics, p4b_native_summary[partition]["macro"], f"GraphST native/{partition}")
        native[f"graphst_{partition}"] = reduced_metrics

    provenance: list[dict[str, Any]] = []
    common_sources = {
        "harmony": PACKAGE4 / "section_seed_rows.json",
        "genorefine": PACKAGE4 / "section_seed_rows.json",
        "spagcn": PACKAGE4 / "section_seed_rows.json",
        "graphst": PACKAGE4B / "section_seed_rows.json",
    }
    for method, metrics in common.items():
        for metric, reduced in metrics.items():
            provenance.append({
                "asset_family": "spatial_common_profile",
                "method": method,
                "endpoint": metric,
                "display_value": reduced["mean"],
                "sd": reduced["sd"],
                "minimum": reduced["minimum"],
                "maximum": reduced["maximum"],
                "seed_values": reduced["seed_values"],
                "source": {"path": rel(common_sources[method]), "sha256": sha256(common_sources[method])},
                "aggregation": "two-section mean within donor, then equal three-donor macro, seed aligned",
            })
    mixing_sources = {
        "harmony": PACKAGE4 / "donor_seed_rows.json",
        "genorefine": PACKAGE4 / "donor_seed_rows.json",
        "graphst": PACKAGE4B / "donor_seed_rows.json",
    }
    for method, metrics in mixing.items():
        for metric, reduced in metrics.items():
            provenance.append({
                "asset_family": "spatial_mixing_profile",
                "method": method,
                "endpoint": metric,
                "display_value": reduced["mean"],
                "sd": reduced["sd"],
                "minimum": reduced["minimum"],
                "maximum": reduced["maximum"],
                "seed_values": reduced["seed_values"],
                "source": {"path": rel(mixing_sources[method]), "sha256": sha256(mixing_sources[method])},
                "aggregation": "equal three-donor macro within seed",
            })
    for method, metrics in native.items():
        source = PACKAGE4_NATIVE / "native_spagcn_macro_seed_rows.csv" if method.startswith("spagcn") else PACKAGE4B / "section_seed_rows.json"
        for metric, reduced in metrics.items():
            provenance.append({
                "asset_family": "spatial_native_partition",
                "method": method,
                "endpoint": metric,
                "display_value": reduced["mean"],
                "sd": reduced["sd"],
                "minimum": reduced["minimum"],
                "maximum": reduced["maximum"],
                "seed_values": reduced["seed_values"],
                "source": {"path": rel(source), "sha256": sha256(source)},
                "aggregation": "two-section mean within donor, then equal three-donor macro, seed aligned",
            })
    return common | {"mixing": mixing}, native, provenance


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def fmt_mean_sd(reduced: dict[str, Any] | None, digits: int = 4) -> str:
    if reduced is None:
        return "NA"
    if reduced["sd"] is None:
        return fmt(reduced["mean"], digits)
    return f"{reduced['mean']:.{digits}f} ± {reduced['sd']:.{digits}f}"


def plot_package3(detail: list[dict[str, Any]], criteria: dict[str, Any], pdf: Path, png: Path) -> None:
    fig = plt.figure(figsize=(11.8, 6.7))
    gs = fig.add_gridspec(1, 2, width_ratios=(0.78, 1.45), wspace=0.30)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    criterion_keys = ["sensitivity_passed", "recovery_passed", "preservation_passed", "full_screen_passed"]
    criterion_labels = ["Sensitivity", "Clean-neighbor\nrecovery", "Preservation", "Full screen"]
    x = np.arange(len(criterion_keys))
    width = 0.34
    for offset, method in ((-width / 2, "genorefine"), (width / 2, "idec")):
        heights = [criteria[method][key] for key in criterion_keys]
        bars = ax0.bar(x + offset, heights, width=width, color=P3_COLORS[method],
                       label="GenoRefine" if method == "genorefine" else "IDEC")
        for bar, value in zip(bars, heights):
            ax0.text(bar.get_x() + bar.get_width() / 2, value + 0.25, str(value),
                     ha="center", va="bottom", fontsize=8.2, fontweight="semibold")
    ax0.set_xticks(x, criterion_labels)
    ax0.set_ylim(0, 12.8)
    ax0.set_ylabel("Conditions passing prespecified criterion (of 12)")
    ax0.set_title("a  Prespecified screen outcomes", loc="left", fontweight="bold")
    ax0.grid(axis="y", color="#D8DEE8", linewidth=0.65)
    ax0.set_axisbelow(True)
    ax0.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.99))

    case_order = [
        "hp_scanorama", "hp_harmony", "pan_scanorama", "pan_harmony",
        "mouse_scanorama", "mouse_harmony",
    ]
    artifact_order = ["batch_simplex", "target_local_warp"]
    labels = []
    values = {"genorefine": [], "idec": []}
    passed = {"genorefine": [], "idec": []}
    for case in case_order:
        for artifact in artifact_order:
            rows = [row for row in detail if row["case_id"] == case and row["artifact"] == artifact]
            identity = rows[0]
            artifact_label = "batch-simplex" if artifact == "batch_simplex" else "target-local warp"
            labels.append(f"{identity['dataset']} | {identity['backbone']} | {artifact_label}")
            for method in ("genorefine", "idec"):
                row = next(row for row in rows if row["method"] == method)
                values[method].append(row["sensitivity_mean_reduction"])
                passed[method].append(row["sensitivity_passed"])

    y = np.arange(len(labels))[::-1]
    for method, offset in (("genorefine", 0.12), ("idec", -0.12)):
        ax1.scatter(values[method], y + offset, s=38, marker="o" if method == "genorefine" else "s",
                    color=P3_COLORS[method], edgecolor="white", linewidth=0.6, zorder=3,
                    label="GenoRefine" if method == "genorefine" else "IDEC")
        for value, ypos, is_pass in zip(values[method], y + offset, passed[method]):
            if is_pass:
                ax1.scatter([value], [ypos], s=72, facecolor="none", edgecolor="#111827",
                            linewidth=0.9, zorder=4)
    ax1.axvline(0, color="#6B7280", linewidth=1.0)
    ax1.set_yticks(y, labels)
    ax1.set_xlabel("Mean artifact-score reduction across five seeds  (positive = lower artifact score)")
    ax1.set_title("b  Sensitivity by case-artifact condition", loc="left", fontweight="bold")
    ax1.grid(axis="x", color="#D8DEE8", linewidth=0.65)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False, loc="lower left", ncol=2)

    fig.suptitle("Expanded injected-artifact validation: partial sensitivity without full correction",
                 fontsize=13.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.012,
             "Outlined symbols pass the prespecified sensitivity criterion (≥4/5 positive seeds); "
             "no condition passed clean-neighbor recovery, preservation, or the joint screen. "
             "Algorithmic seeds are descriptive repeats, not biological replicates.",
             ha="center", va="bottom", fontsize=8.0, color="#374151")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.13)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_spatial_common(spatial: dict[str, Any], pdf: Path, png: Path) -> None:
    panels = (
        "ARI", "NMI", "predicted_cluster_silhouette", "reference_label_silhouette",
        "label_neighbor_purity", "spatial_latent_knn_jaccard",
        "physical_distance_ratio_to_local_spatial_knn_mean", "common_cluster_count",
        "iLISI", "D_batch",
    )
    fig, axes = plt.subplots(2, 5, figsize=(14.8, 6.7))
    axes = axes.ravel()
    for ax, metric in zip(axes, panels):
        source = spatial["mixing"] if metric in MIXING_METRICS else spatial
        methods = [method for method in DISPLAY_METHODS if method in source and metric in source[method]]
        x = np.arange(len(methods))
        for index, method in enumerate(methods):
            reduced = source[method][metric]
            ax.scatter(index, reduced["mean"], s=48, color=METHOD_COLORS[method],
                       edgecolor="white", linewidth=0.65, zorder=3)
            if reduced["sd"] is not None:
                ax.errorbar(index, reduced["mean"], yerr=reduced["sd"], fmt="none",
                            ecolor=METHOD_COLORS[method], elinewidth=1.3, capsize=2.5, zorder=2)
        ax.set_xticks(x, [METHOD_LABELS[method] for method in methods], rotation=27, ha="right")
        ax.set_title(METRIC_LABELS.get(metric, metric), fontweight="semibold", pad=8)
        ax.grid(axis="y", color="#D8DEE8", linewidth=0.65)
        ax.set_axisbelow(True)
        if metric in {"ARI", "NMI", "label_neighbor_purity", "iLISI"}:
            ax.set_ylim(bottom=0)
        if metric == "spatial_latent_knn_jaccard":
            ax.set_ylim(bottom=0)
        if metric == "common_cluster_count":
            ax.set_title("Common-evaluator\ncluster count", fontweight="semibold", pad=8)
        if metric in MIXING_METRICS:
            ax.text(0.5, -0.35, "SpaGCN: NA (per-section fit)", transform=ax.transAxes,
                    ha="center", va="top", fontsize=7.0, color="#6B7280")
    fig.suptitle("Six-section DLPFC spatial evaluation: common fixed-resolution profiles and donor-pair mixing",
                 fontsize=13.5, fontweight="bold", y=0.985)
    fig.legend(
        handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=METHOD_COLORS[m],
                        markeredgecolor="white", markersize=7, label=METHOD_LABELS[m].replace("\n", " "))
                 for m in DISPLAY_METHODS],
        loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=4, frameon=False,
    )
    fig.text(0.5, 0.012,
             "Points are donor-macro means; bars are sample SD across five algorithmic seeds (Harmony is one fixed embedding). "
             "ARI/NMI are fixed-resolution profiles, not matched-K domain recovery. GraphST uses donor-pair PASTE-aligned models; "
             "SpaGCN is fit per section, so the tasks are asymmetric.",
             ha="center", va="bottom", fontsize=7.7, color="#374151")
    fig.subplots_adjust(left=0.045, right=0.99, top=0.83, bottom=0.18, wspace=0.34, hspace=0.52)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_native(native: dict[str, Any], pdf: Path, png: Path) -> None:
    order = ("spagcn_predicted", "spagcn_refined", "graphst_mclust", "graphst_refined")
    labels = ("SpaGCN\npredicted", "SpaGCN\nhex-refined", "GraphST\nmclust", "GraphST\nspatial-refined")
    colors = (METHOD_COLORS["spagcn"], "#F59E0B", METHOD_COLORS["graphst"], "#A78BFA")
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 4.8))
    for ax, metric in zip(axes, ("ARI", "NMI")):
        for index, (method, color) in enumerate(zip(order, colors)):
            reduced = native[method][metric]
            ax.scatter(index, reduced["mean"], s=62, color=color, edgecolor="white", linewidth=0.7, zorder=3)
            ax.errorbar(index, reduced["mean"], yerr=reduced["sd"], fmt="none", ecolor=color,
                        elinewidth=1.5, capsize=3.0, zorder=2)
        ax.set_xticks(range(4), labels)
        ax.set_ylim(0, 0.72)
        ax.set_ylabel(metric)
        ax.set_title(f"Task-native {metric}", fontweight="semibold")
        ax.grid(axis="y", color="#D8DEE8", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.axvline(1.5, color="#9CA3AF", linestyle="--", linewidth=0.9)
    fig.suptitle("Secondary task-native spatial partitions", fontsize=13.0, fontweight="bold", y=0.98)
    fig.text(0.5, 0.018,
             "Mean ± sample SD across five descriptive algorithmic seeds. These method-native partitions are reported separately "
             "from the common fixed-resolution evaluator and should not be conflated with it.",
             ha="center", va="bottom", fontsize=8.0, color="#374151")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.84, bottom=0.21, wspace=0.26)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def add_table_page(
    pdf: PdfPages,
    title: str,
    subtitle: str,
    columns: list[str],
    rows: list[list[str]],
    footnote: str,
    col_widths: list[float] | None = None,
    fontsize: float = 7.4,
    figsize: tuple[float, float] = (13.3, 7.4),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    fig.suptitle(title, x=0.5, y=0.973, fontsize=13.5, fontweight="bold")
    ax.text(0.5, 0.935, subtitle, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.4, color="#374151")
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", colLoc="center",
                     colWidths=col_widths, bbox=[0.012, 0.16, 0.976, 0.70])
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.42)
        cell.set_edgecolor("#B8C0CC")
        if row == 0:
            cell.set_facecolor("#DCEAF7")
            cell.set_text_props(weight="bold", color="#152033")
            cell.set_height(cell.get_height() * 1.25)
        else:
            cell.set_facecolor("#F7F9FC" if row % 2 else "#FFFFFF")
            if col == 0:
                cell.set_text_props(ha="left")
    ax.text(0.012, 0.042, footnote, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.7, color="#374151", wrap=True)
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_package3_table(detail: list[dict[str, Any]], criteria: dict[str, Any], path: Path) -> None:
    with PdfPages(path) as pdf:
        rows = []
        for method in ("genorefine", "idec"):
            label = "GenoRefine" if method == "genorefine" else "IDEC"
            values = criteria[method]
            rows.append([label, f"{values['sensitivity_passed']}/12", f"{values['recovery_passed']}/12",
                         f"{values['preservation_passed']}/12", f"{values['full_screen_passed']}/12"])
        add_table_page(
            pdf,
            "Package 3 prespecified correction-and-preservation screen",
            "Two injected artifact families across six dataset-backbone cases; five algorithmic seeds per method and condition.",
            ["Method", "Sensitivity", "Clean-neighbor recovery", "Preservation", "Joint screen"],
            rows,
            "Passing the joint screen required all prespecified components. No joint screen passed (0/24 overall); this does not support a claim of general artifact correction.",
            [0.20, 0.19, 0.22, 0.19, 0.19],
            fontsize=8.7,
            figsize=(10.5, 4.5),
        )


def write_spatial_table(spatial: dict[str, Any], native: dict[str, Any], path: Path) -> None:
    with PdfPages(path) as pdf:
        layer_rows = []
        spatial_rows = []
        for method in DISPLAY_METHODS:
            label = METHOD_LABELS[method].replace("\n", " ")
            layer_rows.append([
                label,
                fmt_mean_sd(spatial[method]["ARI"]),
                fmt_mean_sd(spatial[method]["NMI"]),
                fmt_mean_sd(spatial[method]["predicted_cluster_silhouette"]),
                fmt_mean_sd(spatial[method]["reference_label_silhouette"]),
                fmt_mean_sd(spatial[method]["label_neighbor_purity"]),
            ])
            spatial_rows.append([
                label,
                fmt_mean_sd(spatial[method]["spatial_latent_knn_jaccard"]),
                fmt_mean_sd(spatial[method]["physical_distance_among_latent_neighbors_mean_fullres_pixels"]),
                fmt_mean_sd(spatial[method]["physical_distance_ratio_to_local_spatial_knn_mean"]),
                fmt_mean_sd(spatial[method]["common_cluster_count"], digits=2),
                fmt_mean_sd(spatial["mixing"].get(method, {}).get("iLISI")),
                fmt_mean_sd(spatial["mixing"].get(method, {}).get("D_batch")),
            ])
        add_table_page(
            pdf,
            "Six-section DLPFC spatial panel: layer and partition profiles (1 of 3)",
            "Donor-macro profiles after section-within-donor aggregation; mean ± sample SD across five algorithmic seeds where applicable.",
            ["Representation", "ARI", "NMI", "Cluster SIL", "Reference SIL", "Purity"],
            layer_rows,
            "Harmony is a single fixed embedding after 10 outer iterations. ARI/NMI use a shared fixed Leiden resolution (0.5), but realized cluster counts differ; interpret them as fixed-resolution profiles.",
            [0.20] + [0.15] * 5,
            fontsize=8.0,
            figsize=(11.5, 6.2),
        )
        add_table_page(
            pdf,
            "Six-section DLPFC spatial panel: local spatial structure and mixing (2 of 3)",
            "Donor-macro profiles after section-within-donor aggregation; mean ± sample SD across five algorithmic seeds where applicable.",
            ["Representation", "Spatial J", "Physical px", "Physical/local", "Clusters", "iLISI", "$D_{batch}$"],
            spatial_rows,
            "Lower physical/local ratio is favorable. SpaGCN mixing is NA because it is fit per section. GraphST uses donor-pair PASTE alignment; common tasks are therefore intentionally compared with explicit asymmetry.",
            [0.20] + [0.13] * 6,
            fontsize=7.7,
            figsize=(11.8, 6.2),
        )
        native_rows = []
        labels = {
            "spagcn_predicted": ("SpaGCN", "Predicted"),
            "spagcn_refined": ("SpaGCN", "Hex-refined"),
            "graphst_mclust": ("GraphST", "mclust"),
            "graphst_refined": ("GraphST", "Spatial-refined"),
        }
        for key, (method, partition) in labels.items():
            native_rows.append([
                method, partition, fmt_mean_sd(native[key]["ARI"]), fmt_mean_sd(native[key]["NMI"]),
                fmt_mean_sd(native[key].get("clusters"), digits=2),
            ])
        add_table_page(
            pdf,
            "Six-section DLPFC spatial panel: secondary native partitions (3 of 3)",
            "Task-native endpoints are kept separate from the common fixed-resolution evaluator.",
            ["Method", "Native partition", "ARI", "NMI", "Clusters"],
            native_rows,
            "Values are mean ± sample SD across five descriptive algorithmic seeds after section-to-donor-to-macro aggregation. SpaGCN cluster counts are fixed by its section-specific label-free K and are omitted here; GraphST donor-pair K values are 6, 7, and 6.",
            [0.18, 0.24, 0.18, 0.18, 0.16],
            fontsize=8.5,
            figsize=(10.6, 5.0),
        )


def write_readme(path: Path) -> None:
    text = """# Audited Package 3/4/4b manuscript assets

This directory is a downstream-only reporting bundle built from four completed, frozen runs. The builder validates every artifact fingerprint recorded by those run manifests, recomputes displayed seed-to-section-to-donor aggregates, and fails closed on coverage or numeric mismatch. It does not modify any scientific run artifact, configuration, model, or manuscript source.

## Package 3: injected-artifact validation

- Source: `revision_pipeline/runs/20260929T192600Z-artifactv2fix2-full-panel`.
- Preparation, baselines, and 120 training runs use frozen foundation hash `e83af7b23ae9a553d6b28e12dd1ed967f3613f713fa7f3e47e5acaaa3cab8c46`.
- Corrected scoring and consolidation use hash `4c61b60a8755122c8b5a13a49a61ca2dbbf4b1ab94d28d90b48c945de4c16304` under the recorded fail-closed expected/observed float64-hash bridge. Native embedding precision was unchanged.
- Outcome: 0/24 prespecified joint correction-and-preservation screens passed. GenoRefine met sensitivity in 5/12 conditions and IDEC in 8/12; neither method met clean-neighbor recovery or preservation in any condition.
- Boundary: this panel does not support a claim of general artifact correction. Algorithmic seeds are descriptive repeats, not biological replicates; no p-values are reported.

## Packages 4 and 4b: six-section LIBD DLPFC spatial panel

- Package 4 source: `revision_pipeline/runs/20260929-spatial-panel-v1-full-panel`, authoritative Package-4-specific hash `bb0f67f14ca3a460a9f708a920c2b64aa44f526994737d84b6e427b46f387b4f`.
- Package 4b source: `revision_pipeline/runs/20260930-graphst4b-v5-full-panel`, protocol hash `b20875a3c5ccb4f0ff7f164587e5836e1db715ae4e7838bf22e64040b02b11ae`, source hash `4d24edecd74a6814c4e09fc13c14b47bd2798d39fcbcf93a3fe2d4c46a853c79`.
- Six sections (two per donor) contain 22,968 retained spots. Layer identities were withheld from fitting and model selection; however, 113 spots lacking a reference layer were excluded when the benchmark cohort was frozen. Fits are therefore label-blind conditional on label availability.
- The common evaluator uses fixed Leiden resolution 0.5 and three Leiden seeds. Because realized common-evaluator cluster counts differ (Harmony 5.28, GenoRefine 3.20, SpaGCN 13.26, GraphST 6.50 donor-macro), common ARI/NMI are fixed-resolution profiles, not matched-K domain recovery.
- Task asymmetry remains explicit: Harmony and GenoRefine use a pooled six-section representation, SpaGCN is fit per section with histology, and GraphST uses label-free PASTE alignment followed by three donor-pair models. SpaGCN therefore has no donor-pair mixing endpoint.
- Spatial-neighborhood preservation is weak in absolute terms for Harmony, GenoRefine, and SpaGCN. GenoRefine modestly improves section mixing but does not improve the layer/local-spatial endpoint profile. Purpose-built GraphST performs substantially better on those common endpoints.
- SpaGCN and GraphST method-native partitions are reported separately from common-evaluator partitions. They must not be conflated.
- Five seeds are descriptive algorithmic repeats, not biological replicates. No p-values are reported.

## Files

- `package3_artifact_screen.{pdf,png}`: prespecified criterion counts and condition-level sensitivity.
- `package3_artifact_screen_table.{pdf,csv}`: compact summary and full condition-level machine-readable table.
- `spatial_common_profiles.{pdf,png}`: common fixed-resolution endpoints, realized cluster granularity, and donor-pair mixing.
- `spatial_native_partitions.{pdf,png}`: separately labeled SpaGCN and GraphST task-native ARI/NMI.
- `spatial_profiles_table.{pdf,csv}`: common-evaluator, mixing, cluster-granularity, and native-partition tables.
- `number_provenance.json`: one provenance entry per displayed number, with source fingerprints and aggregation contracts.
- `asset_manifest.json`: output fingerprints and audit coverage.

Regenerate with `python revision_pipeline/manuscript_figures/build_package34_assets.py`. The default destination is this directory.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    """Audit Packages 3, 4, and 4b and render provenance-linked manuscript assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    setup_style()
    source_manifests = []
    for run_dir, expected in EXPECTED_RUNS.items():
        manifest = validate_run(run_dir, expected)
        source_manifests.append({
            "run_id": manifest["run_id"],
            "kind": manifest["kind"],
            "source_tree_sha256": manifest.get("source_tree_sha256"),
            "run_manifest": rel(run_dir / "run.json"),
            "run_manifest_sha256": sha256(run_dir / "run.json"),
            "recorded_artifacts_verified": len(manifest["artifacts"]),
        })

    p3_detail, p3_criteria, p3_provenance = audit_package3()
    spatial, native, spatial_provenance = audit_spatial()

    p3_fig_pdf = out / "package3_artifact_screen.pdf"
    p3_fig_png = out / "package3_artifact_screen.png"
    p3_table_pdf = out / "package3_artifact_screen_table.pdf"
    p3_table_csv = out / "package3_artifact_screen_table.csv"
    spatial_fig_pdf = out / "spatial_common_profiles.pdf"
    spatial_fig_png = out / "spatial_common_profiles.png"
    native_fig_pdf = out / "spatial_native_partitions.pdf"
    native_fig_png = out / "spatial_native_partitions.png"
    spatial_table_pdf = out / "spatial_profiles_table.pdf"
    spatial_table_csv = out / "spatial_profiles_table.csv"
    provenance_path = out / "number_provenance.json"
    readme_path = out / "README.md"

    plot_package3(p3_detail, p3_criteria, p3_fig_pdf, p3_fig_png)
    write_package3_table(p3_detail, p3_criteria, p3_table_pdf)
    write_csv(p3_table_csv, p3_detail, list(p3_detail[0]))
    plot_spatial_common(spatial, spatial_fig_pdf, spatial_fig_png)
    plot_native(native, native_fig_pdf, native_fig_png)
    write_spatial_table(spatial, native, spatial_table_pdf)

    spatial_csv_rows = []
    for method in DISPLAY_METHODS:
        row = {"method": method, "task": {
            "harmony": "pooled six-section fixed embedding",
            "genorefine": "pooled six-section refinement",
            "spagcn": "per-section spatial model with histology",
            "graphst": "three donor-pair PASTE-aligned spatial models",
        }[method]}
        for metric in COMMON_METRICS:
            reduced = spatial[method][metric]
            row[f"{metric}_mean"] = reduced["mean"]
            row[f"{metric}_sd"] = reduced["sd"]
        reduced = spatial[method]["common_cluster_count"]
        row["common_cluster_count_mean"] = reduced["mean"]
        row["common_cluster_count_sd"] = reduced["sd"]
        for metric in MIXING_METRICS:
            reduced = spatial["mixing"].get(method, {}).get(metric)
            row[f"{metric}_mean"] = None if reduced is None else reduced["mean"]
            row[f"{metric}_sd"] = None if reduced is None else reduced["sd"]
        spatial_csv_rows.append(row)
    native_labels = {
        "spagcn_predicted": ("SpaGCN", "predicted"),
        "spagcn_refined": ("SpaGCN", "hex_refined"),
        "graphst_mclust": ("GraphST", "mclust"),
        "graphst_refined": ("GraphST", "spatial_refined"),
    }
    for key, (method, partition) in native_labels.items():
        row = {"method": method, "task": f"native_partition:{partition}"}
        for metric in ("ARI", "NMI", "clusters"):
            reduced = native[key].get(metric)
            row[f"{metric}_mean"] = None if reduced is None else reduced["mean"]
            row[f"{metric}_sd"] = None if reduced is None else reduced["sd"]
        spatial_csv_rows.append(row)
    fields = ["method", "task"]
    for metric in COMMON_METRICS:
        fields.extend([f"{metric}_mean", f"{metric}_sd"])
    fields.extend(["common_cluster_count_mean", "common_cluster_count_sd",
                   "iLISI_mean", "iLISI_sd", "D_batch_mean", "D_batch_sd",
                   "clusters_mean", "clusters_sd"])
    write_csv(spatial_table_csv, spatial_csv_rows, fields)
    write_readme(readme_path)

    provenance = {
        "schema": "genorefine.package34_manuscript_asset_provenance.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder": rel(Path(__file__)),
        "builder_sha256": sha256(Path(__file__)),
        "scope": "Frozen completed Packages 3, 4, Package-4 SpaGCN native reporting, and 4b only; no scientific recomputation beyond aggregation.",
        "source_runs": source_manifests,
        "source_lineages": {
            "package3_foundation_training": P3_FOUNDATION_HASH,
            "package3_scoring_consolidation": P3_SCORER_HASH,
            "package4_authoritative": P4_HASH,
            "package4b_protocol": P4B_PROTOCOL_HASH,
            "package4b_source": P4B_SOURCE_HASH,
        },
        "aggregation_contracts": {
            "package3": "five descriptive algorithmic seeds per method and case-artifact condition",
            "spatial_common": "within seed: two-section mean within donor, then equal three-donor macro; then descriptive mean/sample SD across five seeds",
            "spatial_mixing": "within seed: equal three-donor macro; then descriptive mean/sample SD across five seeds",
            "harmony": "single fixed embedding; no algorithmic-seed SD",
        },
        "evidence_boundaries": {
            "package3": "0/24 joint screens passed; no general artifact-correction claim",
            "label_policy": "Layer identities withheld from fitting/model selection; 113 unlabeled spots excluded at frozen-cohort construction",
            "fixed_resolution": "Common ARI/NMI use resolution 0.5 but realized cluster counts differ; treat as fixed-resolution profiles",
            "task_asymmetry": "Pooled Harmony/GenoRefine, per-section SpaGCN, and donor-pair PASTE-aligned GraphST are not identical tasks",
            "native_partitions": "Method-native partitions are secondary and separate from the common evaluator",
            "uncertainty": "SD describes algorithmic-seed spread, not biological replication or confidence intervals",
        },
        "values": p3_provenance + spatial_provenance,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    outputs = [
        p3_fig_pdf, p3_fig_png, p3_table_pdf, p3_table_csv,
        spatial_fig_pdf, spatial_fig_png, native_fig_pdf, native_fig_png,
        spatial_table_pdf, spatial_table_csv, provenance_path, readme_path,
    ]
    manifest = {
        "schema": "genorefine.package34_manuscript_asset_manifest.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder": {"path": rel(Path(__file__)), **fingerprint(Path(__file__))},
        "source_runs": source_manifests,
        "outputs": {path.name: {"path": rel(path), **fingerprint(path)} for path in outputs},
        "audits": {
            "all_recorded_source_artifacts_verified": True,
            "package3_seed_rows_verified": 120,
            "package3_conditions_verified": 24,
            "package4_section_rows_verified": 72,
            "package4b_section_rows_verified": 30,
            "spatial_seed_section_donor_aggregates_recomputed": True,
            "native_partition_aggregates_recomputed": True,
            "displayed_value_provenance_entries": len(provenance["values"]),
        },
    }
    manifest_path = out / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(out),
        "outputs": list(manifest["outputs"]) + [manifest_path.name],
        "provenance_entries": len(provenance["values"]),
        "source_runs": [record["run_id"] for record in source_manifests],
    }, indent=2))


if __name__ == "__main__":
    main()
