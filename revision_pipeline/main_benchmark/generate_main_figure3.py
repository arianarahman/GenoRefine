# Purpose: Generate main Figure 3: Scanorama versus Scanorama + GenoRefine.
# Author: Ariana Rahman (Arizona State University)

"""Generate main Figure 3: Scanorama versus Scanorama + GenoRefine.

The script reuses audited seed-0 UMAP coordinates from the complete gallery run
for pancreas, HP-CB, and mouse. It performs no model fitting, UMAP fitting, or
score-based panel selection.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np


UMAP_RUN = "20260921T054834Z-umap-galleries"
CANONICAL_STORE = "20260916T221434Z-a0884ffbd933"
DATASETS = (
    {
        "title": "Pancreas",
        "stem": "pancreas_five_study",
        "label_field": "class_code",
        "point_size": 1.25,
        "alpha": 0.72,
        "legend_mode": "numeric_codes",
    },
    {
        "title": "HP-CB",
        "stem": "hpcb",
        "label_field": "celltype",
        "point_size": 1.10,
        "alpha": 0.70,
        "legend_mode": "all",
    },
    {
        "title": "Mouse",
        "stem": "mouse_senis",
        "label_field": "cell_ontology_class",
        "point_size": 0.38,
        "alpha": 0.54,
        "legend_mode": "top",
        "legend_top_n": 15,
    },
)
CONDITIONS = (
    ("Scanorama", "baseline"),
    ("Scanorama + GenoRefine", "refined_seed0"),
)

MOUSE_LABEL_ALIASES = {
    "basal cell of epidermis": "Epidermal basal cell",
    "mesenchymal stem cell of adipose": "Adipose MSC",
    "fibroblast of cardiac tissue": "Cardiac fibroblast",
    "endothelial cell of coronary artery": "Coronary endothelial cell",
    "bladder urothelial cell": "Bladder urothelial cell",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _natural_level_key(value: str):
    try:
        return (0, int(value))
    except ValueError:
        return (1, value.casefold(), value)


def _display_label(dataset_title: str, value: str) -> str:
    if dataset_title == "Pancreas":
        suffix = " (unmapped)" if value == "15" else ""
        return f"Class {value}{suffix}"
    if dataset_title == "HP-CB":
        return value.replace("_", " ")
    return MOUSE_LABEL_ALIASES.get(value, value)


def _legend_spec(dataset_config: dict, labels: np.ndarray) -> dict:
    reference_levels = sorted(set(labels), key=_natural_level_key)
    counts = Counter(labels.tolist())
    legend_mode = dataset_config["legend_mode"]

    if legend_mode == "top":
        ranked = sorted(
            reference_levels,
            key=lambda value: (-counts[value], _natural_level_key(value)),
        )
        selected_levels = ranked[: int(dataset_config["legend_top_n"])]
        visual_levels = [*selected_levels, "__other__"]
        lookup = {value: index for index, value in enumerate(selected_levels)}
        other_code = len(selected_levels)
        label_codes = np.asarray(
            [lookup.get(value, other_code) for value in labels],
            dtype=np.int16,
        )
    else:
        visual_levels = list(reference_levels)
        lookup = {value: index for index, value in enumerate(visual_levels)}
        label_codes = np.asarray([lookup[value] for value in labels], dtype=np.int16)

    palette = list(plt.get_cmap("tab20").colors)
    colors = [palette[index % len(palette)] for index in range(len(visual_levels))]
    if visual_levels[-1] == "__other__":
        colors[-1] = (0.72, 0.72, 0.72)

    entries = []
    for index, value in enumerate(visual_levels):
        if value == "__other__":
            n_cells = int(sum(counts[item] for item in reference_levels if item not in lookup))
            display = f"Other ({len(reference_levels) - len(lookup)} groups)"
            source_value = None
        else:
            n_cells = int(counts[value])
            display = _display_label(dataset_config["title"], value)
            source_value = value
        entries.append(
            {
                "source_value": source_value,
                "display_label": display,
                "n_cells": n_cells,
                "color_rgb": [float(channel) for channel in colors[index][:3]],
            }
        )

    return {
        "reference_levels": reference_levels,
        "visual_levels": visual_levels,
        "label_codes": label_codes,
        "colors": colors,
        "entries": entries,
    }


def main() -> int:
    """Assemble Figure 3 from audited seed-0 coordinates without fitting or selecting models."""
    project = Path(__file__).resolve().parents[2]
    runs = project / "revision_pipeline" / "runs"
    coordinates = runs / UMAP_RUN / "coordinates"
    dataset_dir = runs / CANONICAL_STORE / "datasets"

    loaded = []
    for dataset_config in DATASETS:
        dataset_path = dataset_dir / f"{dataset_config['stem']}.json"
        with dataset_path.open("r", encoding="utf-8") as handle:
            dataset = json.load(handle)

        labels = np.asarray(
            [str(value) for value in dataset["obs"][dataset_config["label_field"]]]
        )
        legend = _legend_spec(dataset_config, labels)

        condition_paths = {
            condition_title: coordinates / f"{dataset_config['stem']}_scanorama_{suffix}.npy"
            for condition_title, suffix in CONDITIONS
        }
        condition_arrays = {}
        expected_shape = (len(labels), 2)
        for condition_title, path in condition_paths.items():
            values = np.asarray(np.load(path), dtype=np.float32)
            if values.shape != expected_shape:
                raise ValueError(f"Unexpected coordinate shape for {path}: {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite UMAP coordinates in {path}")
            condition_arrays[condition_title] = values

        loaded.append(
            {
                **dataset_config,
                "dataset_path": dataset_path,
                "labels": labels,
                "levels": legend["reference_levels"],
                "visual_levels": legend["visual_levels"],
                "label_codes": legend["label_codes"],
                "legend_colors": legend["colors"],
                "legend_entries": legend["entries"],
                "condition_paths": condition_paths,
                "condition_arrays": condition_arrays,
            }
        )

    fig = plt.figure(figsize=(8.05, 5.85))
    grid = fig.add_gridspec(
        3,
        3,
        width_ratios=(1.0, 1.0, 0.70),
        left=0.067,
        right=0.995,
        top=0.945,
        bottom=0.015,
        wspace=0.025,
        hspace=0.055,
    )
    axes = np.empty((3, 2), dtype=object)
    panel_index = 0
    for row_index, dataset in enumerate(loaded):
        cmap = ListedColormap(dataset["legend_colors"])
        for column_index, (condition_title, _) in enumerate(CONDITIONS):
            ax = fig.add_subplot(grid[row_index, column_index])
            axes[row_index, column_index] = ax
            xy = dataset["condition_arrays"][condition_title]
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                c=dataset["label_codes"],
                cmap=cmap,
                s=dataset["point_size"],
                alpha=dataset["alpha"],
                linewidths=0,
                rasterized=True,
                vmin=-0.5,
                vmax=len(dataset["visual_levels"]) - 0.5,
            )
            ax.text(
                0.015,
                0.965,
                f"({chr(97 + panel_index)})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.2,
                fontweight="bold",
                color="black",
            )
            ax.text(
                0.985,
                0.025,
                f"{len(dataset['labels']):,} cells; {len(dataset['levels'])} groups",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.1,
                color="black",
                bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            panel_index += 1

        legend_ax = fig.add_subplot(grid[row_index, 2])
        legend_ax.axis("off")
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=4.4,
                label=entry["display_label"],
            )
            for color, entry in zip(dataset["legend_colors"], dataset["legend_entries"])
        ]
        legend_title = {
            "Pancreas": "Reference codes",
            "HP-CB": "Cell types",
            "Mouse": "15 largest groups",
        }[dataset["title"]]
        legend_ax.legend(
            handles=handles, title=legend_title, loc="center left",
            frameon=False, borderaxespad=0, handletextpad=0.45,
            labelspacing=0.10, fontsize=6.15, title_fontsize=6.7,
        )

        axes[row_index, 0].text(
            -0.035,
            0.5,
            dataset["title"],
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=8.6,
            fontweight="bold",
        )

    for column_index, (condition_title, _) in enumerate(CONDITIONS):
        axes[0, column_index].set_title(
            condition_title,
            fontsize=9.2,
            fontweight="bold",
            pad=5,
        )

    output_dir = project / "Overleaf files" / "main_new_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "verified_scanorama_cross_dataset_umap.pdf"
    output_png = output_dir / "verified_scanorama_cross_dataset_umap.png"
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    manifest = {
        "role": "descriptive main-manuscript Scanorama versus Scanorama + GenoRefine UMAP comparison",
        "selection": "predeclared seed-0 refined output and audited upstream baseline for every dataset",
        "model_fitting": False,
        "umap_fitting": False,
        "coordinate_run": UMAP_RUN,
        "conditions": [title for title, _ in CONDITIONS],
        "datasets": [],
        "coordinate_sources": {},
    }
    for dataset in loaded:
        manifest["datasets"].append(
            {
                "title": dataset["title"],
                "dataset_path": str(dataset["dataset_path"]),
                "dataset_sha256": _sha256(dataset["dataset_path"]),
                "n_cells": len(dataset["labels"]),
                "reference_group_field": dataset["label_field"],
                "n_reference_groups": len(dataset["levels"]),
                "visual_label_policy": (
                    "all reference codes"
                    if dataset["legend_mode"] == "numeric_codes"
                    else "all named reference groups"
                    if dataset["legend_mode"] == "all"
                    else (
                        f"{dataset['legend_top_n']} most abundant named groups; "
                        "remaining groups combined as Other for visualization only"
                    )
                ),
                "legend_entries": dataset["legend_entries"],
            }
        )
        manifest["coordinate_sources"][dataset["title"]] = {
            condition_title: {"path": str(path), "sha256": _sha256(path)}
            for condition_title, path in dataset["condition_paths"].items()
        }
    manifest.update(
        {
            "output_pdf": str(output_pdf),
            "output_pdf_sha256": _sha256(output_pdf),
            "output_png": str(output_png),
            "output_png_sha256": _sha256(output_png),
        }
    )
    manifest_path = output_dir / "verified_scanorama_cross_dataset_umap.manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(output_pdf)
    print(output_png)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
