"""Audit the recorded PBMC parameter-sensitivity table.

The output is descriptive.  Ordinary one-way ANOVA is reproduced only as an
arithmetic audit because the three runs per setting are algorithmic executions,
network-wide seed control was not established, and clustering used the stored
labels to target the number of clusters.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def omnibus(groups: list[np.ndarray]) -> dict:
    result = stats.f_oneway(*groups)
    values = np.concatenate(groups)
    grand = float(values.mean())
    ss_between = float(sum(len(group) * (float(group.mean()) - grand) ** 2 for group in groups))
    ss_within = float(sum(((group - float(group.mean())) ** 2).sum() for group in groups))
    ss_total = ss_between + ss_within
    k = len(groups)
    n = len(values)
    ms_within = ss_within / (n - k)
    eta_squared = ss_between / ss_total
    omega_squared = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
    return {
        "F": float(result.statistic),
        "df_between": k - 1,
        "df_within": n - k,
        "p": float(result.pvalue),
        "eta_squared": float(eta_squared),
        "omega_squared": float(omega_squared),
        "interpretation": "arithmetic_audit_not_validated_inference",
    }


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    source = (
        project / "PBMCSingeBatchDataset" / "Sensitivity_Results"
        / "Sensitivity_Structural_Results.csv"
    )
    frame = pd.read_csv(source)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = project / "revision_pipeline" / "runs" / f"{stamp}-pbmc-sensitivity-audit"
    output.mkdir(parents=True, exist_ok=False)

    summaries = []
    tests = {}
    for sweep, subset in frame.groupby("Sweep_Type", sort=False):
        tests[sweep] = {}
        for metric in ("ARI", "Silhouette"):
            groups = []
            for value, group in subset.groupby("Parameter_Value", sort=True):
                array = group.sort_values("Seed")[metric].to_numpy(dtype=float)
                groups.append(array)
                summaries.append(
                    {
                        "sweep": sweep,
                        "parameter_value": int(value),
                        "metric": metric,
                        "n": len(array),
                        "mean": float(array.mean()),
                        "sample_sd": float(array.std(ddof=1)),
                        "minimum": float(array.min()),
                        "maximum": float(array.max()),
                    }
                )
            tests[sweep][metric] = omnibus(groups)

    dim_shared = frame[
        (frame["Sweep_Type"] == "Embedding_Dim")
        & (frame["Parameter_Value"] == 32)
    ].sort_values("Seed")
    map_shared = frame[
        (frame["Sweep_Type"] == "Map_Resolution")
        & (frame["Parameter_Value"] == 40)
    ].sort_values("Seed")
    if dim_shared["Seed"].tolist() != map_shared["Seed"].tolist():
        raise ValueError("Shared nominal-setting seed lists do not match")
    shared = {"seeds": dim_shared["Seed"].astype(int).tolist()}
    for metric in ("ARI", "Silhouette"):
        differences = map_shared[metric].to_numpy() - dim_shared[metric].to_numpy()
        shared[metric] = {
            "map_sweep_minus_dimension_sweep": differences.tolist(),
            "mean_difference": float(differences.mean()),
            "sample_sd_difference": float(differences.std(ddof=1)),
        }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_frame = pd.DataFrame(summaries)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    column_specs = [
        ("Embedding_Dim", "Output dimension"),
        ("Map_Resolution", "Requested map side"),
    ]
    row_specs = [("ARI", "ARI"), ("Silhouette", "Predicted-cluster silhouette")]
    for row_index, (metric, y_label) in enumerate(row_specs):
        for column_index, (sweep, x_label) in enumerate(column_specs):
            axis = axes[row_index, column_index]
            view = summary_frame[
                (summary_frame["sweep"] == sweep)
                & (summary_frame["metric"] == metric)
            ].sort_values("parameter_value")
            axis.errorbar(
                view["parameter_value"], view["mean"], yerr=view["sample_sd"],
                marker="o", linewidth=1.8, capsize=4, color="#1f77b4",
            )
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            axis.set_xticks(view["parameter_value"])
            axis.grid(True, linestyle="--", alpha=0.35)
            axis.set_title(f"{y_label} by {x_label.lower()}", fontsize=10)
    fig.suptitle("PBMC parameter sensitivity (mean +/- sample SD; n=3 runs)",
                 fontsize=12, fontweight="bold")
    figure_path = (
        project / "Overleaf files" / "supplementary_figures"
        / "pbmc_parameter_sensitivity.png"
    )
    fig.savefig(figure_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    result = {
        "status": "complete",
        "source": str(source),
        "source_sha256": sha256(source),
        "scope": "recorded algorithmic runs; not biological replicates",
        "limitations": [
            "three runs per setting",
            "network-wide random-state control was not established",
            "Leiden resolution was label-informed through target class count",
            "ordinary ANOVA assumes independent residuals and is retained only as an arithmetic audit",
        ],
        "summaries": summaries,
        "ordinary_one_way_anova_audit": tests,
        "shared_nominal_setting": shared,
        "figure": {"path": str(figure_path), "sha256": sha256(figure_path)},
    }
    (output / "statistics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)

    lines = [
        "# PBMC parameter-sensitivity audit",
        "",
        "The recorded table contains three algorithmic executions per setting. "
        "The ordinary one-way ANOVA values below reproduce the requested arithmetic "
        "but are not treated as confirmatory inference.",
        "",
    ]
    for sweep, metrics in tests.items():
        for metric, test in metrics.items():
            lines.append(
                f"- {sweep}, {metric}: F({test['df_between']},{test['df_within']})="
                f"{test['F']:.6f}, p={test['p']:.8g}, eta-squared={test['eta_squared']:.4f}, "
                f"omega-squared={test['omega_squared']:.4f}."
            )
    lines.extend(
        [
            "",
            "The nominal d'=32, map-side-40 condition was executed separately in "
            "the two sweeps and is therefore not pooled.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
