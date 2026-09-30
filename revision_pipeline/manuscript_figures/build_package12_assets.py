"""Build audited manuscript assets from completed Package 1 and Package 2 runs.

This module is deliberately read-only with respect to scientific run directories.  It
validates the recorded artifacts, recomputes every displayed aggregate from seed-level
records, and writes figures, tables, and machine-readable provenance to a staging
directory.  It does not import or modify manuscript LaTeX.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import statistics
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "revision_pipeline" / "runs"
DEFAULT_OUT = Path(__file__).resolve().parent / "staging" / "package12_audited"

PACKAGE1 = RUNS / "20260929T004000Z-idec2-full-panel"
PACKAGE1_PROTOCOL = ROOT / "revision_pipeline" / "configs" / "independent_idec_panel_v2.json"
PRIMARY_TABLE = RUNS / "20260921T003801Z-c055558a1a7c" / "table_data.json"
PACKAGE2 = RUNS / "20260929T094500Z-imarker-panel-v1"
PACKAGE2_FOUNDATION = RUNS / "20260929T093000Z-imarker-foundation-v1"

CASE_ORDER = [
    "hp_scanorama", "hp_harmony", "hp_seurat", "hp_inmf",
    "pan_scanorama", "pan_harmony", "pan_seurat", "pan_inmf",
    "mouse_scanorama", "mouse_harmony", "mouse_seurat", "mouse_inmf",
]
MARKER_CASE_ORDER = [
    "hp_scanorama", "hp_harmony", "hp_seurat", "hp_inmf",
    "mouse_scanorama", "mouse_harmony", "mouse_seurat", "mouse_inmf",
]
METHODS = ("upstream", "GenoRefine", "IDEC")
MAIN_METRICS = (
    ("ARI", "Adjusted Rand index (ARI)"),
    ("SIL_cluster", "Predicted-cluster Silhouette (SIL)"),
    ("iLISI", "Local inverse Simpson's index (iLISI)"),
    ("local_label_purity", "Local label purity"),
)
MARKER_METRICS = (
    ("neighbor_marker_auroc", "Marker-neighborhood AUROC"),
    ("neighbor_marker_average_precision", "Marker-neighborhood AP"),
    ("neighbor_marker_contrast", "Marker-neighborhood contrast"),
)
PLOT_TITLES = {
    "ARI": "Adjusted Rand index\n(ARI)",
    "SIL_cluster": "Predicted-cluster\nSilhouette (SIL)",
    "iLISI": "Local inverse Simpson's\nindex (iLISI)",
    "local_label_purity": "Local label purity",
    "neighbor_marker_auroc": "Marker-neighborhood\nAUROC",
    "neighbor_marker_average_precision": "Marker-neighborhood\naverage precision (AP)",
    "neighbor_marker_contrast": "Marker-neighborhood\ncontrast",
}
TABLE_TITLES = {
    "ARI": "ARI",
    "SIL_cluster": "Silhouette (SIL)",
    "iLISI": "iLISI",
    "local_label_purity": "Local label purity",
    "neighbor_marker_auroc": "AUROC",
    "neighbor_marker_average_precision": "Average precision (AP)",
    "neighbor_marker_contrast": "Contrast",
}
MAIN_SOURCE_KEYS = {
    "ARI": ("ARI", "mean_ARI_at_0_5"),
    "SIL_cluster": ("SIL_cluster", "predicted_cluster_ASW_subsample_mean"),
    "iLISI": ("iLISI", "iLISI_scib_metrics"),
    "local_label_purity": ("local_label_purity", "reference_knn_purity"),
}
DISPLAY_DATASET = {"hp": "HP-CB", "pan": "Pancreas", "mouse": "Mouse"}

COLORS = {"upstream": "#4D4D4D", "GenoRefine": "#1769AA", "IDEC": "#D97706"}
MARKERS = {"upstream": "D", "GenoRefine": "o", "IDEC": "s"}
OFFSETS = {"upstream": 0.22, "GenoRefine": 0.0, "IDEC": -0.22}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    return {"sha256": sha256(path), "size_bytes": path.stat().st_size}


def assert_fp(path: Path, expected: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = fingerprint(path)
    if actual != expected:
        raise ValueError(f"Fingerprint mismatch for {label}: {path}; {actual} != {expected}")


def validate_run(run_dir: Path, expected_kind: str, artifacts: Iterable[str] | None = None) -> dict[str, Any]:
    manifest_path = run_dir / "run.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "succeeded" or manifest.get("kind") != expected_kind:
        raise ValueError(
            f"Run is not a completed {expected_kind}: {run_dir} "
            f"(status={manifest.get('status')}, kind={manifest.get('kind')})"
        )
    names = list(artifacts) if artifacts is not None else list(manifest.get("artifacts", {}))
    for name in names:
        if name not in manifest.get("artifacts", {}):
            raise ValueError(f"Artifact {name} is absent from manifest: {manifest_path}")
        assert_fp(run_dir / name, manifest["artifacts"][name], f"{run_dir.name}/{name}")
    return manifest


def close(a: float, b: float, *, atol: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)


def require_close(a: float, b: float, label: str) -> None:
    if not close(a, b):
        raise ValueError(f"Numeric mismatch for {label}: {a!r} != {b!r}")


def case_label(record: dict[str, Any]) -> str:
    return f"{record['dataset']}  |  {record['backbone']}"


def reduce_values(values: list[float]) -> dict[str, Any]:
    if len(values) < 2:
        return {
            "value": values[0], "mean": values[0], "sd": None,
            "min": values[0], "max": values[0], "values": values,
        }
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def audit_package1() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    panel_manifest = validate_run(
        PACKAGE1,
        "independent_idec_full_panel_consolidation",
        ("config.json", "long_table.csv", "report.md", "summary.json"),
    )
    panel_config = read_json(PACKAGE1 / "config.json")
    panel_summary = read_json(PACKAGE1 / "summary.json")
    if panel_summary.get("protocol_id") != "independent_idec_panel_v2":
        raise ValueError("Unexpected Package 1 protocol")
    assert_fp(PACKAGE1_PROTOCOL, panel_config["inputs"]["protocol"], "Package 1 protocol")
    assert_fp(PRIMARY_TABLE, panel_config["inputs"]["primary_table"], "Package 1 primary table")

    primary = read_json(PRIMARY_TABLE)
    primary_rows = {(row["dataset"], row["backbone"]): row for row in primary["rows"]}
    records_by_id = {row["case_id"]: row for row in panel_summary["records"]}
    protocol = read_json(PACKAGE1_PROTOCOL)
    case_specs = {row["id"]: row for row in protocol["cases"]}
    if set(records_by_id) != set(CASE_ORDER):
        raise ValueError(f"Package 1 case coverage changed: {sorted(records_by_id)}")

    provenance: list[dict[str, Any]] = []
    ordered: list[dict[str, Any]] = []
    for case_id in CASE_ORDER:
        record = records_by_id[case_id]
        spec = case_specs[case_id]
        backbone = spec["embedding"].replace("Online_iNMF", "Online iNMF")
        source_row = primary_rows[(spec["display_dataset"], backbone)]
        if record["dataset"] != spec["display_dataset"] or record["backbone"] != backbone:
            raise ValueError(f"Case identity mismatch: {case_id}")
        if len(source_row["GR_seeds"]) != 5:
            raise ValueError(f"Expected five GenoRefine seeds: {case_id}")

        score_summaries: list[dict[str, Any]] = []
        score_sources: list[dict[str, Any]] = []
        for seed in range(5):
            run_dir = RUNS / f"20260929T004000Z-idec2-{case_id}-s{seed}-joint-score"
            manifest = validate_run(run_dir, "independent_idec_scoring", ("summary.json",))
            input_key = f"{case_id}_seed{seed}"
            assert_fp(run_dir / "run.json", panel_config["inputs"][input_key], input_key)
            score = read_json(run_dir / "summary.json")
            if score.get("case_id") != case_id or score.get("seed") != seed or score.get("stage") != "joint":
                raise ValueError(f"Package 1 score binding mismatch: {case_id} seed {seed}")
            score_summaries.append(score)
            score_sources.append({
                "seed": seed,
                "run_id": manifest["run_id"],
                "run_manifest": str((run_dir / "run.json").relative_to(ROOT)).replace("\\", "/"),
                "run_manifest_sha256": sha256(run_dir / "run.json"),
                "artifact": str((run_dir / "summary.json").relative_to(ROOT)).replace("\\", "/"),
                "artifact_sha256": sha256(run_dir / "summary.json"),
            })

        audited_methods: dict[str, Any] = {method: {} for method in METHODS}
        for metric, _ in MAIN_METRICS:
            primary_key, idec_key = MAIN_SOURCE_KEYS[metric]
            upstream = float(source_row["upstream"][primary_key])
            gr_values = [float(seed_row[primary_key]) for seed_row in source_row["GR_seeds"]]
            idec_values = [float(score[idec_key]) for score in score_summaries]

            reduced = {
                "upstream": reduce_values([upstream]),
                "GenoRefine": reduce_values(gr_values),
                "IDEC": reduce_values(idec_values),
            }
            for method in METHODS:
                published = record["methods"][method][metric]
                key = "value" if method == "upstream" else "mean"
                require_close(reduced[method][key], published[key], f"{case_id}/{method}/{metric}/{key}")
                if method != "upstream":
                    for stat in ("sd", "min", "max"):
                        require_close(reduced[method][stat], published[stat], f"{case_id}/{method}/{metric}/{stat}")
                    if reduced[method]["values"] != published["values"]:
                        raise ValueError(f"Seed values changed for {case_id}/{method}/{metric}")
                audited_methods[method][metric] = reduced[method]

                if method == "upstream":
                    raw_sources = [{
                        "artifact": str(PRIMARY_TABLE.relative_to(ROOT)).replace("\\", "/"),
                        "artifact_sha256": sha256(PRIMARY_TABLE),
                        "json_pointer": f"/rows/{primary['rows'].index(source_row)}/upstream/{primary_key}",
                    }]
                elif method == "GenoRefine":
                    raw_sources = [{
                        "seed": seed,
                        "artifact": str(PRIMARY_TABLE.relative_to(ROOT)).replace("\\", "/"),
                        "artifact_sha256": sha256(PRIMARY_TABLE),
                        "json_pointer": f"/rows/{primary['rows'].index(source_row)}/GR_seeds/{seed}/{primary_key}",
                    } for seed in range(5)]
                else:
                    raw_sources = [dict(source, json_pointer=f"/{idec_key}")
                                   for source in score_sources]

                provenance.append({
                    "asset_family": "main_independent_comparator",
                    "case_id": case_id,
                    "dataset": record["dataset"],
                    "backbone": record["backbone"],
                    "metric": metric,
                    "method": method,
                    "statistic": "fixed_value" if method == "upstream" else "mean_across_five_algorithmic_seeds",
                    "display_value": reduced[method][key],
                    "sd": reduced[method].get("sd"),
                    "minimum": reduced[method]["min"],
                    "maximum": reduced[method]["max"],
                    "seed_values": reduced[method]["values"],
                    "consolidated_source": {
                        "artifact": str((PACKAGE1 / "summary.json").relative_to(ROOT)).replace("\\", "/"),
                        "artifact_sha256": sha256(PACKAGE1 / "summary.json"),
                        "json_pointer": f"/records/{panel_summary['records'].index(record)}/methods/{method}/{metric}",
                    },
                    "raw_sources": raw_sources,
                })
        ordered.append({**record, "methods": audited_methods})

    source_record = {
        "package": "Package 1",
        "run_id": panel_manifest["run_id"],
        "kind": panel_manifest["kind"],
        "run_manifest": str((PACKAGE1 / "run.json").relative_to(ROOT)).replace("\\", "/"),
        "run_manifest_sha256": sha256(PACKAGE1 / "run.json"),
        "summary": str((PACKAGE1 / "summary.json").relative_to(ROOT)).replace("\\", "/"),
        "summary_sha256": sha256(PACKAGE1 / "summary.json"),
        "protocol": str(PACKAGE1_PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256(PACKAGE1_PROTOCOL),
        "primary_table": str(PRIMARY_TABLE.relative_to(ROOT)).replace("\\", "/"),
        "primary_table_sha256": sha256(PRIMARY_TABLE),
        "coverage": panel_summary["coverage"],
    }
    return ordered, provenance, source_record


def audit_package2() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    foundation_manifest = validate_run(
        PACKAGE2_FOUNDATION,
        "independent_marker_foundation",
        ("config.json", "foundation.json"),
    )
    panel_manifest = validate_run(
        PACKAGE2,
        "independent_marker_preservation_panel",
        ("config.json", "results.json", "summary.json", "unit_index.json"),
    )
    config = read_json(PACKAGE2 / "config.json")
    assert_fp(PACKAGE2_FOUNDATION / "run.json", config["foundation"]["run_manifest"], "Package 2 foundation manifest")
    summary = read_json(PACKAGE2 / "summary.json")
    results = read_json(PACKAGE2 / "results.json")
    unit_index = read_json(PACKAGE2 / "unit_index.json")
    if not summary.get("complete") or summary.get("units") != 88 or len(results) != 88 or len(unit_index) != 88:
        raise ValueError("Package 2 is not the completed 88-unit panel")

    summary_rows = {(row["dataset"], row["backbone"], row["method"]): row for row in summary["rows"]}
    provenance: list[dict[str, Any]] = []
    ordered: list[dict[str, Any]] = []
    for case_id in MARKER_CASE_ORDER:
        units = [value for value in results.values() if value["case_id"] == case_id]
        if len(units) != 11:
            raise ValueError(f"Expected 11 Package 2 units for {case_id}, found {len(units)}")
        identity = units[0]
        methods: dict[str, Any] = {}
        for display_method, result_method in (("upstream", "upstream"), ("GenoRefine", "GenoRefine"), ("IDEC", "IDEC")):
            method_units = [unit for unit in units if unit["method"] == result_method]
            method_units.sort(key=lambda item: -1 if item["seed"] is None else int(item["seed"]))
            expected_n = 1 if display_method == "upstream" else 5
            if len(method_units) != expected_n:
                raise ValueError(f"Package 2 coverage mismatch for {case_id}/{display_method}")
            methods[display_method] = {}
            source_units = []
            for unit in method_units:
                unit_id = unit["unit_id"]
                entry = unit_index[unit_id]
                run_dir = ROOT / entry["path"]
                assert_fp(run_dir / "run.json", entry["manifest"], f"Package 2 unit manifest {unit_id}")
                manifest = validate_run(run_dir, "independent_marker_preservation_unit", ("result.json",))
                disk_result = read_json(run_dir / "result.json")
                if disk_result != unit:
                    raise ValueError(f"Panel copy differs from unit result: {unit_id}")
                source_units.append({
                    "unit_id": unit_id,
                    "seed": unit["seed"],
                    "run_id": manifest["run_id"],
                    "run_manifest": str((run_dir / "run.json").relative_to(ROOT)).replace("\\", "/"),
                    "run_manifest_sha256": sha256(run_dir / "run.json"),
                    "artifact": str((run_dir / "result.json").relative_to(ROOT)).replace("\\", "/"),
                    "artifact_sha256": sha256(run_dir / "result.json"),
                })

            row = summary_rows[(identity["dataset_display"], identity["backbone"], result_method)]
            for metric, _ in MARKER_METRICS:
                values = [float(unit["metrics"]["macro"][metric]) for unit in method_units]
                reduced = reduce_values(values)
                require_close(reduced["mean"], row[f"{metric}_mean"], f"{case_id}/{display_method}/{metric}/mean")
                require_close(reduced["min"], row[f"{metric}_minimum"], f"{case_id}/{display_method}/{metric}/min")
                require_close(reduced["max"], row[f"{metric}_maximum"], f"{case_id}/{display_method}/{metric}/max")
                methods[display_method][metric] = reduced
                raw_sources = [dict(source, json_pointer="/metrics/macro/" + metric)
                               for source in source_units]
                provenance.append({
                    "asset_family": "supplemental_external_marker",
                    "case_id": case_id,
                    "dataset": identity["dataset_display"],
                    "backbone": identity["backbone"],
                    "metric": metric,
                    "method": display_method,
                    "statistic": "fixed_value" if display_method == "upstream" else "mean_across_five_algorithmic_seeds",
                    "display_value": reduced["mean"],
                    "sd": reduced.get("sd"),
                    "minimum": reduced["min"],
                    "maximum": reduced["max"],
                    "seed_values": reduced["values"],
                    "consolidated_source": {
                        "artifact": str((PACKAGE2 / "summary.json").relative_to(ROOT)).replace("\\", "/"),
                        "artifact_sha256": sha256(PACKAGE2 / "summary.json"),
                        "summary_key": {
                            "dataset": identity["dataset_display"],
                            "backbone": identity["backbone"],
                            "method": result_method,
                            "metric": metric,
                        },
                    },
                    "raw_sources": raw_sources,
                })
        ordered.append({
            "case_id": case_id,
            "dataset": identity["dataset_display"],
            "backbone": identity["backbone"],
            "methods": methods,
        })

    source_record = {
        "package": "Package 2",
        "run_id": panel_manifest["run_id"],
        "kind": panel_manifest["kind"],
        "run_manifest": str((PACKAGE2 / "run.json").relative_to(ROOT)).replace("\\", "/"),
        "run_manifest_sha256": sha256(PACKAGE2 / "run.json"),
        "summary": str((PACKAGE2 / "summary.json").relative_to(ROOT)).replace("\\", "/"),
        "summary_sha256": sha256(PACKAGE2 / "summary.json"),
        "results": str((PACKAGE2 / "results.json").relative_to(ROOT)).replace("\\", "/"),
        "results_sha256": sha256(PACKAGE2 / "results.json"),
        "unit_index": str((PACKAGE2 / "unit_index.json").relative_to(ROOT)).replace("\\", "/"),
        "unit_index_sha256": sha256(PACKAGE2 / "unit_index.json"),
        "foundation_run_id": foundation_manifest["run_id"],
        "foundation_manifest_sha256": sha256(PACKAGE2_FOUNDATION / "run.json"),
        "coverage": {key: summary[key] for key in ("units", "cases", "datasets", "backbones", "methods")},
    }
    return ordered, provenance, source_record


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.4,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def metric_bounds(records: list[dict[str, Any]], metric: str) -> tuple[float, float]:
    values = []
    for record in records:
        for method in METHODS:
            reduced = record["methods"][method][metric]
            values.extend((reduced["min"], reduced["max"]))
    lo, hi = min(values), max(values)
    span = max(hi - lo, 0.02)
    padding = span * 0.08
    if metric in {"ARI", "SIL_cluster", "iLISI", "local_label_purity",
                  "neighbor_marker_auroc", "neighbor_marker_average_precision"}:
        return max(0.0, lo - padding), min(1.0, hi + padding)
    return lo - padding, hi + padding


def plot_comparison(
    records: list[dict[str, Any]],
    metrics: tuple[tuple[str, str], ...],
    out_pdf: Path,
    out_png: Path,
    title: str,
    note: str,
) -> None:
    nrows = len(records)
    width = 13.4 if len(metrics) == 4 else 11.6
    fig, axes = plt.subplots(1, len(metrics), figsize=(width, 7.2 if nrows == 12 else 5.6), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(nrows)[::-1]
    for col, ((metric, metric_title), ax) in enumerate(zip(metrics, axes)):
        for method in METHODS:
            xs = []
            low = []
            high = []
            ys = []
            for ypos, record in zip(y, records):
                reduced = record["methods"][method][metric]
                mean = reduced["mean"]
                xs.append(mean)
                low.append(mean - reduced["min"])
                high.append(reduced["max"] - mean)
                ys.append(ypos + OFFSETS[method])
            if method == "upstream":
                ax.scatter(xs, ys, s=34, marker=MARKERS[method], color=COLORS[method],
                           edgecolor="white", linewidth=0.5, zorder=4)
            else:
                ax.errorbar(xs, ys, xerr=np.array([low, high]), fmt=MARKERS[method],
                            color=COLORS[method], ecolor=COLORS[method], markersize=5.2,
                            elinewidth=1.25, capsize=2.2, markeredgecolor="white",
                            markeredgewidth=0.45, zorder=3)
        ax.set_xlim(*metric_bounds(records, metric))
        ax.set_title(PLOT_TITLES.get(metric, metric_title), pad=9, fontweight="semibold")
        ax.grid(axis="x", color="#D9DEE7", linewidth=0.65, alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        for boundary in (3.5, 7.5) if nrows == 12 else (3.5,):
            ax.axhline(nrows - 1 - boundary, color="#8A94A6", linewidth=0.9)
        if col == 0:
            ax.set_yticks(y)
            ax.set_yticklabels([case_label(record) for record in records])
        else:
            ax.tick_params(labelleft=False)

    handles = [Line2D([0], [0], marker=MARKERS[m], color=COLORS[m], linestyle="None",
                      markeredgecolor="white", markersize=7, label=m)
               for m in METHODS]
    fig.legend(handles=handles, labels=["Upstream", "GenoRefine", "IDEC"],
               loc="upper center", bbox_to_anchor=(0.59, 0.925), ncol=3, frameon=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.52, y=0.995)
    fig.text(0.5, 0.018, note, ha="center", va="bottom", fontsize=8.0, color="#374151")
    fig.subplots_adjust(left=0.205 if nrows == 12 else 0.235, right=0.985, top=0.815,
                        bottom=0.12 if nrows == 12 else 0.145, wspace=0.24)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fmt_fixed(value: float) -> str:
    return f"{value:.4f}"


def fmt_mean_sd(reduced: dict[str, Any]) -> str:
    return f"{reduced['mean']:.4f} ± {reduced['sd']:.4f}"


def fmt_mean_range(reduced: dict[str, Any]) -> str:
    return f"{reduced['mean']:.4f} [{reduced['min']:.4f}, {reduced['max']:.4f}]"


def add_table_page(
    pdf: PdfPages,
    records: list[dict[str, Any]],
    metrics: tuple[tuple[str, str], ...],
    title: str,
    subtitle: str,
    cell_formatter,
    footnote: str,
    figsize: tuple[float, float] = (13.6, 7.5),
) -> None:
    columns = ["Dataset", "Backbone"]
    for metric, _ in metrics:
        short = TABLE_TITLES[metric]
        columns.extend([f"{short}\nUpstream", f"{short}\nGenoRefine", f"{short}\nIDEC"])
    data = []
    for record in records:
        row = [record["dataset"], record["backbone"]]
        for metric, _ in metrics:
            row.extend([
                fmt_fixed(record["methods"]["upstream"][metric]["mean"]),
                cell_formatter(record["methods"]["GenoRefine"][metric]),
                cell_formatter(record["methods"]["IDEC"][metric]),
            ])
        data.append(row)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    fig.suptitle(title, x=0.5, y=0.975, fontsize=14, fontweight="bold")
    ax.text(0.5, 0.935, subtitle, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.7, color="#374151")
    numeric_count = len(columns) - 2
    col_widths = [0.085, 0.105] + [(0.81 / numeric_count)] * numeric_count
    table = ax.table(cellText=data, colLabels=columns, cellLoc="center", colLoc="center",
                     colWidths=col_widths, bbox=[0.012, 0.085, 0.976, 0.80])
    table.auto_set_font_size(False)
    table.set_fontsize(7.5 if len(metrics) == 2 else 7.0)
    table.scale(1.0, 1.25)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor("#B8C0CC")
        if row == 0:
            cell.set_facecolor("#DCEAF7")
            cell.set_text_props(weight="bold", color="#152033")
            cell.set_height(cell.get_height() * 1.35)
        else:
            group = (row - 1) // 4
            cell.set_facecolor("#F7F9FC" if group % 2 == 0 else "#FFFFFF")
            if col in (0, 1):
                cell.set_text_props(ha="left")
            if col >= 2 and (col - 2) % 3 == 1:
                cell.set_text_props(color=COLORS["GenoRefine"])
            elif col >= 2 and (col - 2) % 3 == 2:
                cell.set_text_props(color=COLORS["IDEC"])
    ax.text(0.012, 0.038, footnote,
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, color="#374151")
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_main_table(records: list[dict[str, Any]], out_pdf: Path) -> None:
    with PdfPages(out_pdf) as pdf:
        add_table_page(
            pdf, records, MAIN_METRICS[:2],
            "Independent embedding-refinement comparison (1 of 2)",
            "Adjusted Rand index and predicted-cluster Silhouette across all 12 dataset-backbone cases.",
            fmt_mean_sd,
            "Upstream is a fixed embedding. GenoRefine and IDEC are mean ± sample SD across five algorithmic seeds; seeds are not biological replicates.",
        )
        add_table_page(
            pdf, records, MAIN_METRICS[2:],
            "Independent embedding-refinement comparison (2 of 2)",
            "Batch mixing (iLISI) and local label purity across all 12 dataset-backbone cases.",
            fmt_mean_sd,
            "Upstream is a fixed embedding. GenoRefine and IDEC are mean ± sample SD across five algorithmic seeds; seeds are not biological replicates.",
        )


def write_marker_table(records: list[dict[str, Any]], out_pdf: Path) -> None:
    with PdfPages(out_pdf) as pdf:
        columns = ["Dataset", "Backbone", "Representation", "AUROC", "Average precision (AP)", "Contrast"]
        data = []
        row_methods = []
        for record in records:
            for index, method in enumerate(METHODS):
                values = []
                for metric, _ in MARKER_METRICS:
                    reduced = record["methods"][method][metric]
                    values.append(fmt_fixed(reduced["mean"]) if method == "upstream" else fmt_mean_range(reduced))
                data.append([
                    record["dataset"] if index == 0 else "",
                    record["backbone"] if index == 0 else "",
                    "Upstream" if method == "upstream" else method,
                    *values,
                ])
                row_methods.append(method)

        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        fig.suptitle("Independent external-marker neighborhood preservation",
                     x=0.5, y=0.978, fontsize=14, fontweight="bold")
        ax.text(0.5, 0.94,
                "Fixed external marker panels; all eligible annotation strata; marker selection did not use an evaluated embedding.",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.6, color="#374151")
        table = ax.table(
            cellText=data,
            colLabels=columns,
            cellLoc="center",
            colLoc="center",
            colWidths=[0.09, 0.13, 0.10, 0.20, 0.24, 0.20],
            bbox=[0.02, 0.075, 0.96, 0.82],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.3)
        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0.42)
            cell.set_edgecolor("#B8C0CC")
            if row == 0:
                cell.set_facecolor("#DCEAF7")
                cell.set_text_props(weight="bold", color="#152033")
                cell.set_height(cell.get_height() * 1.18)
                continue
            method = row_methods[row - 1]
            cell.set_facecolor({
                "upstream": "#F3F4F6",
                "GenoRefine": "#F2F7FC",
                "IDEC": "#FFF7ED",
            }[method])
            if col in (0, 1, 2):
                cell.set_text_props(ha="left")
            if method == "GenoRefine" and col >= 2:
                cell.set_text_props(color=COLORS["GenoRefine"], ha="left" if col == 2 else "center")
            elif method == "IDEC" and col >= 2:
                cell.set_text_props(color=COLORS["IDEC"], ha="left" if col == 2 else "center")
        ax.text(0.02, 0.035,
                "Upstream is a fixed embedding. GenoRefine and IDEC are mean [minimum, maximum] across five algorithmic seeds; seeds are not biological replicates.",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7.8, color="#374151")
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)


def write_main_csv(records: list[dict[str, Any]], path: Path) -> None:
    fields = ["case_id", "dataset", "backbone"]
    for metric, _ in MAIN_METRICS:
        fields.extend([
            f"{metric}_upstream", f"{metric}_GenoRefine_mean", f"{metric}_GenoRefine_sd",
            f"{metric}_GenoRefine_min", f"{metric}_GenoRefine_max",
            f"{metric}_IDEC_mean", f"{metric}_IDEC_sd", f"{metric}_IDEC_min", f"{metric}_IDEC_max",
        ])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in ("case_id", "dataset", "backbone")}
            for metric, _ in MAIN_METRICS:
                row[f"{metric}_upstream"] = record["methods"]["upstream"][metric]["mean"]
                for method in ("GenoRefine", "IDEC"):
                    reduced = record["methods"][method][metric]
                    for stat in ("mean", "sd", "min", "max"):
                        row[f"{metric}_{method}_{stat}"] = reduced[stat]
            writer.writerow(row)


def write_marker_csv(records: list[dict[str, Any]], path: Path) -> None:
    fields = ["case_id", "dataset", "backbone"]
    for metric, _ in MARKER_METRICS:
        fields.extend([
            f"{metric}_upstream", f"{metric}_GenoRefine_mean", f"{metric}_GenoRefine_min",
            f"{metric}_GenoRefine_max", f"{metric}_IDEC_mean", f"{metric}_IDEC_min", f"{metric}_IDEC_max",
        ])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in ("case_id", "dataset", "backbone")}
            for metric, _ in MARKER_METRICS:
                row[f"{metric}_upstream"] = record["methods"]["upstream"][metric]["mean"]
                for method in ("GenoRefine", "IDEC"):
                    reduced = record["methods"][method][metric]
                    for stat in ("mean", "min", "max"):
                        row[f"{metric}_{method}_{stat}"] = reduced[stat]
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    setup_style()
    main_records, main_provenance, package1_source = audit_package1()
    marker_records, marker_provenance, package2_source = audit_package2()

    main_figure_pdf = out / "main_independent_comparator_12case.pdf"
    main_figure_png = out / "main_independent_comparator_12case.png"
    main_table_pdf = out / "main_independent_comparator_12case_table.pdf"
    main_table_csv = out / "main_independent_comparator_12case_table.csv"
    marker_figure_pdf = out / "supp_external_marker_8case.pdf"
    marker_figure_png = out / "supp_external_marker_8case.png"
    marker_table_pdf = out / "supp_external_marker_8case_table.pdf"
    marker_table_csv = out / "supp_external_marker_8case_table.csv"

    plot_comparison(
        main_records, MAIN_METRICS, main_figure_pdf, main_figure_png,
        "Upstream, GenoRefine, and independent IDEC comparison",
        "Points show fixed upstream values or five-seed means; horizontal spans show the seed range. "
        "Seeds are algorithmic replicates. Higher iLISI is interpreted together with agreement and preservation endpoints.",
    )
    write_main_table(main_records, main_table_pdf)
    write_main_csv(main_records, main_table_csv)

    plot_comparison(
        marker_records, MARKER_METRICS, marker_figure_pdf, marker_figure_png,
        "Independent external-marker neighborhood preservation",
        "External marker panels were fixed before embedding evaluation. Points show fixed upstream values or five-seed means; "
        "horizontal spans show the seed range. Pancreas is excluded because a matched expression foundation was unavailable.",
    )
    write_marker_table(marker_records, marker_table_pdf)
    write_marker_csv(marker_records, marker_table_csv)

    provenance = {
        "schema": "genorefine.manuscript_asset_provenance.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
        "scope": "Completed Package 1 and Package 2 runs only; no scientific recomputation.",
        "source_runs": [package1_source, package2_source],
        "display_contract": {
            "upstream": "one fixed evaluated embedding",
            "GenoRefine": "mean and dispersion/range across five prespecified algorithmic seeds",
            "IDEC": "mean and dispersion/range across five prespecified algorithmic seeds",
            "uncertainty_interpretation": "algorithmic-seed variability, not biological replication or confidence intervals",
        },
        "values": main_provenance + marker_provenance,
    }
    provenance_path = out / "number_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    outputs = [
        main_figure_pdf, main_figure_png, main_table_pdf, main_table_csv,
        marker_figure_pdf, marker_figure_png, marker_table_pdf, marker_table_csv,
        provenance_path,
    ]
    manifest = {
        "schema": "genorefine.manuscript_asset_manifest.v1",
        "builder_sha256": sha256(Path(__file__).resolve()),
        "sources": [package1_source, package2_source],
        "outputs": {
            path.name: {**fingerprint(path), "relative_path": str(path.relative_to(ROOT)).replace("\\", "/")}
            for path in outputs
        },
        "audits": {
            "package1_values": len(main_provenance),
            "package2_values": len(marker_provenance),
            "package1_expected": 12 * 4 * 3,
            "package2_expected": 8 * 3 * 3,
            "source_artifact_fingerprints_verified": True,
            "seed_level_aggregates_recomputed": True,
        },
    }
    if manifest["audits"]["package1_values"] != manifest["audits"]["package1_expected"]:
        raise ValueError("Package 1 provenance coverage is incomplete")
    if manifest["audits"]["package2_values"] != manifest["audits"]["package2_expected"]:
        raise ValueError("Package 2 provenance coverage is incomplete")
    manifest_path = out / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "output_dir": str(out),
        "files": [path.name for path in outputs] + [manifest_path.name],
        "audited_values": len(main_provenance) + len(marker_provenance),
    }, indent=2))


if __name__ == "__main__":
    main()
