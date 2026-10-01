# Purpose: Generate manuscript UMAP galleries from the final main-benchmark artifacts.
# Author: Ariana Rahman (Arizona State University)

"""Generate manuscript UMAP galleries from the final main-benchmark artifacts.

The gallery is descriptive.  It uses the predeclared seed-0 joint embedding for
every refined arm; no representation is selected by its score.  Each UMAP is
fitted separately with the same settings and canonical cell order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DATASETS = {
    "Pancreas": "pancreas_five_study",
    "HP-CB": "hpcb",
    "Mouse": "mouse_senis",
}
METHODS = ("Scanorama", "Harmony", "Seurat", "Online iNMF")
GALLERY_PAGES = (
    (("Scanorama", "Harmony"), "Scanorama and Harmony"),
    (("Seurat", "Online iNMF"), "Seurat and Online iNMF"),
)
FINAL_CONSOLIDATION = "20260921T003801Z-c055558a1a7c"
CANONICAL_STORE = "20260916T221434Z-a0884ffbd933"
UMAP_SETTINGS = {
    "n_neighbors": 15,
    "min_dist": 0.30,
    "metric": "euclidean",
    "random_state": 42,
    "transform_seed": 42,
    "low_memory": True,
}


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_path(value: str) -> Path:
    if value.startswith("/mnt/c/"):
        return Path("C:/" + value[len("/mnt/c/") :])
    return Path(value)


def _entry_path(value) -> Path:
    return _local_path(value["path"] if isinstance(value, dict) else value)


def _method_file(method: str) -> str:
    return method.replace(" ", "_")


def _load_case(
    project: Path,
    final_index: dict,
    dataset_name: str,
    dataset_id: str,
    method: str,
) -> dict:
    case_name = f"{dataset_name}/{method}"
    case_path = _entry_path(final_index[case_name])
    if (case_path / "run_index.json").exists():
        case_index = _read_json(case_path / "run_index.json")
        training_path = _entry_path(case_index["training"]["0"])
        inputs_path = _entry_path(case_index["inputs"])
        context = _read_json(inputs_path / "context.json")
        store = _local_path(context["store"])
    elif case_name == "HP-CB/Scanorama":
        # The original Step 4A audit predates the main-benchmark run-index
        # format.  Its verified seed-0 training artifact is pinned here.
        training_path = (
            project / "revision_pipeline" / "runs"
            / "20260917T181031Z-f7e46860148e-seed_0_initial"
        )
        store = (
            project / "revision_pipeline" / "runs" / CANONICAL_STORE
        )
    else:
        raise FileNotFoundError(f"No run index for {case_name}: {case_path}")
    refined_path = training_path / "bundles" / "joint" / "values.npy"
    refined_ids_path = training_path / "bundles" / "joint" / "cell_ids.json"
    stem = _method_file(method)
    baseline_path = store / "embeddings" / dataset_id / f"{stem}.npy"
    if not baseline_path.exists() and method == "Harmony":
        baseline_path = store / "embeddings" / dataset_id / "Harmony_fixed10.npy"

    dataset_path = (
        project
        / "revision_pipeline"
        / "runs"
        / CANONICAL_STORE
        / "datasets"
        / f"{dataset_id}.json"
    )
    dataset = _read_json(dataset_path)
    canonical_ids = dataset["cell_ids"]
    refined_ids = _read_json(refined_ids_path)
    if isinstance(refined_ids, dict):
        refined_ids = refined_ids.get("cell_ids", refined_ids.get("ids"))
    if refined_ids != canonical_ids:
        raise ValueError(f"Canonical cell order mismatch for {case_name}")

    baseline = np.load(baseline_path, mmap_mode="r")
    refined = np.load(refined_path, mmap_mode="r")
    if baseline.shape[0] != len(canonical_ids) or refined.shape[0] != len(canonical_ids):
        raise ValueError(f"Row-count mismatch for {case_name}")
    if not np.isfinite(baseline).all() or not np.isfinite(refined).all():
        raise ValueError(f"Non-finite values for {case_name}")

    return {
        "case": case_name,
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "method": method,
        "dataset_path": dataset_path,
        "baseline_path": baseline_path,
        "refined_path": refined_path,
        "n_cells": len(canonical_ids),
        "baseline_shape": list(baseline.shape),
        "refined_shape": list(refined.shape),
        "baseline_sha256": _sha256(baseline_path),
        "refined_sha256": _sha256(refined_path),
    }


def _fit_one(task: tuple[str, str, str]) -> tuple[str, str]:
    key, source, target = task
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    import umap

    values = np.asarray(np.load(source, mmap_mode="r"), dtype=np.float32)
    coordinates = umap.UMAP(**UMAP_SETTINGS).fit_transform(values)
    np.save(target, np.asarray(coordinates, dtype=np.float32))
    return key, target


def _codes(values):
    labels = [str(value) for value in values]
    levels = sorted(set(labels))
    lookup = {value: index for index, value in enumerate(levels)}
    return np.asarray([lookup[value] for value in labels]), levels


def _plot_gallery(
    dataset_name: str,
    dataset: dict,
    records: list[dict],
    output: Path,
    methods: tuple[str, ...] = METHODS,
    page_label: str | None = None,
):
    """Render matched cell-type and batch panels for upstream and seed-0 refined coordinates."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if dataset_name == "Pancreas":
        label_key, batch_key = "class_code", "batch"
    elif dataset_name == "HP-CB":
        label_key, batch_key = "celltype", "tech"
    else:
        label_key, batch_key = "cell_ontology_class", "method"

    labels, label_levels = _codes(dataset["obs"][label_key])
    batches, batch_levels = _codes(dataset["obs"][batch_key])
    label_cmap = plt.get_cmap("turbo", max(len(label_levels), 2))
    batch_cmap = plt.get_cmap("tab10", max(len(batch_levels), 2))
    point_size = 1.4 if len(labels) <= 17_000 else 0.65
    point_alpha = 0.72 if len(labels) <= 17_000 else 0.55

    ordered = []
    by_method = {record["method"]: record for record in records}
    for method in methods:
        record = by_method[method]
        ordered.extend(
            [
                (method, record["baseline_umap"]),
                (f"{method} + GenoRefine", record["refined_umap"]),
            ]
        )

    if len(ordered) != 4:
        raise ValueError("Each gallery page must contain two upstream/refined method pairs")

    fig, axes = plt.subplots(2, 4, figsize=(13.4, 7.4), constrained_layout=True)
    for tile, (title, coord_path) in enumerate(ordered):
        tile_col = tile
        xy = np.load(coord_path)
        ax_batch = axes[0, tile_col]
        ax_label = axes[1, tile_col]
        ax_batch.scatter(
            xy[:, 0], xy[:, 1], c=batches, cmap=batch_cmap,
            s=point_size, alpha=point_alpha, linewidths=0, rasterized=True,
        )
        ax_label.scatter(
            xy[:, 0], xy[:, 1], c=labels, cmap=label_cmap,
            s=point_size, alpha=point_alpha, linewidths=0, rasterized=True,
        )
        ax_batch.set_title(title, fontsize=9.2, fontweight="bold", pad=3)
        ax_batch.text(0.01, 0.98, "Batch", transform=ax_batch.transAxes,
                      ha="left", va="top", fontsize=7.2,
                      bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=1.5))
        ax_label.text(0.01, 0.98, "Reference group", transform=ax_label.transAxes,
                      ha="left", va="top", fontsize=7.2,
                      bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=1.5))
        for axis in (ax_batch, ax_label):
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)

    batch_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=4,
               markerfacecolor=batch_cmap(index), markeredgecolor="none", label=level)
        for index, level in enumerate(batch_levels)
    ]
    fig.legend(handles=batch_handles, title="Batch", loc="lower center",
               ncol=min(len(batch_levels), 9), frameon=False,
               fontsize=7.2, title_fontsize=7.5, bbox_to_anchor=(0.5, -0.005))
    title = f"{dataset_name}: fixed seed-0 UMAP gallery ({len(labels):,} cells)"
    if page_label:
        title = f"{title} - {page_label}"
    fig.suptitle(
        title,
        fontsize=12, fontweight="bold",
    )
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "label_key": label_key,
        "batch_key": batch_key,
        "label_count": len(label_levels),
        "batch_count": len(batch_levels),
        "label_legend": "omitted_from_gallery_to_preserve_panel_legibility",
        "batch_levels": batch_levels,
    }


def main() -> int:
    """Generate or reuse audited UMAP coordinates, render galleries, and write their manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[2]
    runs = project / "revision_pipeline" / "runs"
    final_dir = runs / FINAL_CONSOLIDATION
    final_index = _read_json(final_dir / "run_index.json")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs / f"{stamp}-umap-galleries"
    coords_dir = run_dir / "coordinates"
    figures_dir = project / "Overleaf files" / "supplementary_figures"
    coords_dir.mkdir(parents=True, exist_ok=False)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    tasks = []
    for dataset_name, dataset_id in DATASETS.items():
        for method in METHODS:
            record = _load_case(project, final_index, dataset_name, dataset_id, method)
            safe = f"{dataset_id}_{_method_file(method).lower()}"
            baseline_umap = coords_dir / f"{safe}_baseline.npy"
            refined_umap = coords_dir / f"{safe}_refined_seed0.npy"
            record["baseline_umap"] = str(baseline_umap)
            record["refined_umap"] = str(refined_umap)
            cases.append(record)
            if args.force or not baseline_umap.exists():
                tasks.append((f"{record['case']}/baseline", str(record["baseline_path"]), str(baseline_umap)))
            if args.force or not refined_umap.exists():
                tasks.append((f"{record['case']}/refined", str(record["refined_path"]), str(refined_umap)))

    completed = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(_fit_one, task) for task in tasks]
        for future in as_completed(futures):
            key, path = future.result()
            completed.append({"key": key, "path": path, "sha256": _sha256(Path(path))})
            print(f"completed {key}", flush=True)

    gallery_records = {}
    for dataset_name, dataset_id in DATASETS.items():
        dataset_path = (
            project / "revision_pipeline" / "runs" / CANONICAL_STORE
            / "datasets" / f"{dataset_id}.json"
        )
        dataset = _read_json(dataset_path)
        subset = [record for record in cases if record["dataset_name"] == dataset_name]
        pages = []
        for page_number, (methods, page_label) in enumerate(GALLERY_PAGES, start=1):
            output = figures_dir / f"{dataset_id}_primary_umap_gallery_part{page_number}.png"
            policy = _plot_gallery(
                dataset_name,
                dataset,
                subset,
                output,
                methods=methods,
                page_label=page_label,
            )
            pages.append(
                {
                    "page": page_number,
                    "methods": list(methods),
                    "output": str(output),
                    "sha256": _sha256(output),
                    **policy,
                }
            )
        gallery_records[dataset_name] = {"pages": pages}

    manifest = {
        "status": "complete",
        "role": "descriptive_visualization_not_metric_input",
        "selection": "baseline plus joint refined seed 0 for every case; no score-based selection",
        "cell_order": "canonical dataset order, verified against every refined bundle",
        "umap_settings": UMAP_SETTINGS,
        "umap_version": __import__("umap").__version__,
        "cases": [
            {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in record.items()
            }
            for record in cases
        ],
        "coordinates": sorted(completed, key=lambda item: item["key"]),
        "galleries": gallery_records,
    }
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
