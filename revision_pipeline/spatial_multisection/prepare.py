"""Prepare the locked six-section LIBD DLPFC Harmony foundation.

This module stops at source-aligned counts, metadata, PCA and Harmony.  It does
not train GenoRefine or SpaGCN and it does not calculate outcome metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
from typing import Callable

import anndata as ad
import harmonypy
import numpy as np
import pandas as pd
from PIL import Image
import scanpy as sc
from scipy import sparse
from threadpoolctl import threadpool_limits

from ..data.readers import array_hash
from ..integrity import canonical_hash, file_fingerprint
from ..runs import RunDirectory
from .source_acquisition import (
    DEFAULT_CONFIG,
    ROOT,
    load_spec,
    source_index,
    source_root,
    verify_sources,
)


RUNS = ROOT / "revision_pipeline/runs"
_POSITION_COLUMNS = [
    "barcode", "in_tissue", "array_row", "array_col",
    "pxl_row_in_fullres", "pxl_col_in_fullres",
]


@dataclass
class SpatialBundle:
    counts: sparse.csr_matrix
    gene_ids: np.ndarray
    gene_symbols: np.ndarray
    metadata: pd.DataFrame
    excluded: pd.DataFrame
    section_summary: list[dict]


@dataclass
class PreparedFoundation:
    counts: sparse.csr_matrix
    gene_metadata: pd.DataFrame
    cell_metadata: pd.DataFrame
    excluded: pd.DataFrame
    pca: np.ndarray
    harmony: np.ndarray
    record: dict


def _text(values) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype="U")


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def dependency_versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": _package_version("numpy"),
        "scipy": _package_version("scipy"),
        "pandas": _package_version("pandas"),
        "anndata": _package_version("anndata"),
        "scanpy": _package_version("scanpy"),
        "scikit-learn": _package_version("scikit-learn"),
        "harmonypy": _package_version("harmonypy"),
        "Pillow": _package_version("Pillow"),
        "threadpoolctl": _package_version("threadpoolctl"),
    }


def csr_hash(matrix: sparse.spmatrix) -> str:
    matrix = sparse.csr_matrix(matrix)
    matrix.sort_indices()
    digest = hashlib.sha256()
    digest.update(canonical_hash({"shape": list(matrix.shape), "dtype": matrix.dtype.str}).encode())
    for array in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def verify_manifest_urls(spec: dict, root: Path) -> dict:
    """Confirm that the commit-pinned study manifest names each frozen H5 URL."""
    index = source_index(spec)
    manifest_item = index[("source_manifest", None)]
    frame = pd.read_csv(root / manifest_item["relative_path"], sep="\t", dtype=str)
    required = {"SampleID", "h5_filtered", "image_hi"}
    if not required.issubset(frame.columns) or frame["SampleID"].duplicated().any():
        raise ValueError("Pinned AWS source manifest has an unexpected schema")
    rows = frame.set_index("SampleID")
    verified = {}
    verified_images = {}
    for section in spec["dataset"]["sections"]:
        section_id = section["id"]
        if section_id not in rows.index:
            raise ValueError(f"Section {section_id} is absent from the source manifest")
        declared = index[("counts", section_id)]["url"]
        manifest_url = str(rows.loc[section_id, "h5_filtered"])
        if declared != manifest_url:
            raise ValueError(f"H5 URL for {section_id} does not match the pinned source manifest")
        verified[section_id] = manifest_url
        declared_image = index[("histology_hires", section_id)]["url"]
        manifest_image = str(rows.loc[section_id, "image_hi"])
        if declared_image != manifest_image:
            raise ValueError(f"High-resolution image URL for {section_id} does not match the pinned source manifest")
        verified_images[section_id] = manifest_image
    return {
        "manifest_relative_path": manifest_item["relative_path"],
        "repository_commit": spec["dataset"]["repository_commit"],
        "verified_h5_urls": verified,
        "verified_image_hi_urls": verified_images,
    }


def read_labels(path: Path, spec: dict) -> pd.DataFrame:
    labels = pd.read_csv(
        path, sep="\t", header=None, names=["barcode", "section", "label"], dtype=str,
    )
    if labels.isna().any().any() or labels.duplicated(["section", "barcode"]).any():
        raise ValueError("Layer map contains missing values or duplicate section/barcode pairs")
    section_ids = {row["id"] for row in spec["dataset"]["sections"]}
    labels = labels[labels["section"].isin(section_ids)].copy()
    unexpected = sorted(set(labels["label"]) - set(spec["dataset"]["label_classes"]))
    if unexpected:
        raise ValueError(f"Unexpected layer labels: {unexpected}")
    return labels


def read_positions(path: Path) -> pd.DataFrame:
    positions = pd.read_csv(path, header=None, names=_POSITION_COLUMNS)
    if positions["barcode"].isna().any() or positions["barcode"].duplicated().any():
        raise ValueError("Position file has a missing or duplicate barcode")
    positions["barcode"] = positions["barcode"].astype(str)
    numeric = _POSITION_COLUMNS[1:]
    for column in numeric:
        positions[column] = pd.to_numeric(positions[column], errors="raise")
    if not set(positions["in_tissue"].unique()).issubset({0, 1}):
        raise ValueError("in_tissue must contain only zero and one")
    if not np.isfinite(positions[numeric].to_numpy(dtype=np.float64)).all():
        raise ValueError("Position file contains a non-finite coordinate")
    return positions


def read_scalefactors(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        values = json.load(stream)
    required = {
        "spot_diameter_fullres", "tissue_hires_scalef",
        "fiducial_diameter_fullres", "tissue_lowres_scalef",
    }
    if set(values) != required:
        raise ValueError(f"Unexpected 10x scalefactor fields in {path.name}")
    result = {}
    for key in sorted(required):
        value = float(values[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {key} in {path.name}")
        result[key] = value
    return result


def read_hires_image(path: Path) -> dict:
    with Image.open(path) as image:
        width, height = image.size
        image_format = image.format
        image.verify()
    if image_format != "PNG" or width <= 0 or height <= 0:
        raise ValueError(f"Invalid high-resolution tissue image: {path.name}")
    return {"width_pixels": int(width), "height_pixels": int(height), "format": image_format}


def _read_counts(path: Path) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    data = sc.read_10x_h5(path, gex_only=True)
    barcodes = _text(data.obs_names)
    symbols = _text(data.var_names)
    if "gene_ids" not in data.var:
        raise ValueError("10x H5 does not contain gene_ids")
    gene_ids = _text(data.var["gene_ids"])
    if len(set(barcodes)) != len(barcodes) or len(set(gene_ids)) != len(gene_ids):
        raise ValueError("10x H5 has duplicate barcodes or gene IDs")
    counts = sparse.csr_matrix(data.X)
    if counts.data.size and (
        not np.isfinite(counts.data).all() or np.any(counts.data < 0)
        or np.any(counts.data != np.floor(counts.data))
    ):
        raise ValueError("10x source is not a finite nonnegative integer count matrix")
    counts = counts.astype(np.int32)
    counts.sort_indices()
    return counts, barcodes, gene_ids, symbols


def load_section(
    section: dict,
    root: Path,
    index: dict[tuple[str, str | None], dict],
    labels: pd.DataFrame,
) -> SpatialBundle:
    section_id = section["id"]
    counts, barcodes, gene_ids, gene_symbols = _read_counts(
        root / index[("counts", section_id)]["relative_path"]
    )
    if len(barcodes) != section["expected_spots"]:
        raise ValueError(f"Unexpected spot count for {section_id}: {len(barcodes)}")
    positions = read_positions(root / index[("positions", section_id)]["relative_path"])
    scale_item = index[("scalefactors", section_id)]
    image_item = index[("histology_hires", section_id)]
    scalefactors = read_scalefactors(root / scale_item["relative_path"])
    image_info = read_hires_image(root / image_item["relative_path"])
    in_tissue = positions.loc[positions["in_tissue"].eq(1)].copy()
    if set(in_tissue["barcode"]) != set(barcodes):
        raise ValueError(f"10x and in-tissue position barcodes do not agree for {section_id}")
    positions_by_barcode = in_tissue.set_index("barcode")
    hires_scale = scalefactors["tissue_hires_scalef"]
    hires_x = in_tissue["pxl_col_in_fullres"].to_numpy(dtype=np.float64) * hires_scale
    hires_y = in_tissue["pxl_row_in_fullres"].to_numpy(dtype=np.float64) * hires_scale
    if (
        np.any(hires_x < 0) or np.any(hires_x >= image_info["width_pixels"])
        or np.any(hires_y < 0) or np.any(hires_y >= image_info["height_pixels"])
    ):
        raise ValueError(f"Scaled spot centers fall outside the high-resolution image for {section_id}")
    section_labels = labels.loc[labels["section"].eq(section_id), ["barcode", "label"]]
    if not set(section_labels["barcode"]).issubset(set(barcodes)):
        raise ValueError(f"Layer map contains non-H5 barcodes for {section_id}")
    labels_by_barcode = section_labels.set_index("barcode")["label"]
    observed_labels = labels_by_barcode.reindex(barcodes)
    keep = observed_labels.notna().to_numpy()
    missing = ~keep
    if int(keep.sum()) != section["expected_labeled_spots"] or int(missing.sum()) != section["expected_missing_labels"]:
        raise ValueError(f"Label availability changed for {section_id}")
    kept_barcodes = barcodes[keep]
    aligned_positions = positions_by_barcode.loc[kept_barcodes]
    metadata = pd.DataFrame({
        "cell_id": [f"{section_id}::{barcode}" for barcode in kept_barcodes],
        "barcode": kept_barcodes,
        "section": section_id,
        "donor": section["donor"],
        "label": observed_labels.loc[keep].to_numpy(dtype=str),
        "array_row": aligned_positions["array_row"].to_numpy(dtype=np.int64),
        "array_col": aligned_positions["array_col"].to_numpy(dtype=np.int64),
        "pxl_row_in_fullres": aligned_positions["pxl_row_in_fullres"].to_numpy(dtype=np.int64),
        "pxl_col_in_fullres": aligned_positions["pxl_col_in_fullres"].to_numpy(dtype=np.int64),
        "hires_pxl_row": aligned_positions["pxl_row_in_fullres"].to_numpy(dtype=np.float64) * hires_scale,
        "hires_pxl_col": aligned_positions["pxl_col_in_fullres"].to_numpy(dtype=np.float64) * hires_scale,
    })
    missing_barcodes = barcodes[missing]
    missing_positions = positions_by_barcode.loc[missing_barcodes]
    excluded = pd.DataFrame({
        "cell_id": [f"{section_id}::{barcode}" for barcode in missing_barcodes],
        "barcode": missing_barcodes,
        "section": section_id,
        "donor": section["donor"],
        "exclusion_reason": "manual_layer_label_missing",
        "array_row": missing_positions["array_row"].to_numpy(dtype=np.int64),
        "array_col": missing_positions["array_col"].to_numpy(dtype=np.int64),
        "pxl_row_in_fullres": missing_positions["pxl_row_in_fullres"].to_numpy(dtype=np.int64),
        "pxl_col_in_fullres": missing_positions["pxl_col_in_fullres"].to_numpy(dtype=np.int64),
        "hires_pxl_row": missing_positions["pxl_row_in_fullres"].to_numpy(dtype=np.float64) * hires_scale,
        "hires_pxl_col": missing_positions["pxl_col_in_fullres"].to_numpy(dtype=np.float64) * hires_scale,
    })
    summary = [{
        "section": section_id,
        "donor": section["donor"],
        "spots_in_filtered_h5": int(len(barcodes)),
        "labeled_spots_retained": int(keep.sum()),
        "missing_labels_excluded": int(missing.sum()),
        "histology_hires": {
            "image_relative_path": image_item["relative_path"],
            "scalefactors_relative_path": scale_item["relative_path"],
            "image": image_info,
            "scalefactors": scalefactors,
            "coordinate_transform": "hires x/y = full-resolution pixel column/row multiplied by tissue_hires_scalef",
        },
        "label_counts": {
            str(key): int(value)
            for key, value in metadata["label"].value_counts().sort_index().items()
        },
    }]
    return SpatialBundle(
        counts=counts[keep].tocsr(), gene_ids=gene_ids, gene_symbols=gene_symbols,
        metadata=metadata, excluded=excluded, section_summary=summary,
    )


def load_spatial_bundle(spec: dict, root: Path) -> SpatialBundle:
    index = source_index(spec)
    labels = read_labels(root / index[("labels", None)]["relative_path"], spec)
    pieces = [load_section(section, root, index, labels) for section in spec["dataset"]["sections"]]
    gene_ids = pieces[0].gene_ids
    gene_symbols = pieces[0].gene_symbols
    for piece in pieces[1:]:
        if not np.array_equal(piece.gene_ids, gene_ids) or not np.array_equal(piece.gene_symbols, gene_symbols):
            raise ValueError("Gene identity/order differs among sections")
    counts = sparse.vstack([piece.counts for piece in pieces], format="csr", dtype=np.int32)
    metadata = pd.concat([piece.metadata for piece in pieces], ignore_index=True)
    excluded = pd.concat([piece.excluded for piece in pieces], ignore_index=True)
    summary = [row for piece in pieces for row in piece.section_summary]
    expected = spec["dataset"]
    if len(metadata) != expected["expected_labeled_spots"] or len(excluded) != expected["expected_missing_labels"]:
        raise ValueError("Pooled retained/excluded spot counts do not match the frozen contract")
    if metadata["cell_id"].duplicated().any() or excluded["cell_id"].duplicated().any():
        raise ValueError("Global section-prefixed IDs must be unique")
    if set(metadata["cell_id"]) & set(excluded["cell_id"]):
        raise ValueError("A spot cannot be both retained and excluded")
    return SpatialBundle(counts, gene_ids, gene_symbols, metadata, excluded, summary)


def _harmony_result(
    pca: np.ndarray,
    metadata: pd.DataFrame,
    config: dict,
    runner: Callable | None,
) -> tuple[np.ndarray, dict]:
    runner = harmonypy.run_harmony if runner is None else runner
    frame = pd.DataFrame({config["batch_key"]: metadata[config["batch_key"]].astype(str).to_numpy()})
    with threadpool_limits(limits=config["thread_limit"]):
        output = runner(
            pca,
            frame,
            config["batch_key"],
            max_iter_harmony=config["max_iter_harmony"],
            epsilon_harmony=config["epsilon_harmony"],
            random_state=config["random_state"],
            verbose=False,
        )
    values = np.asarray(output.Z_corr, dtype=np.float64)
    orientation = "cells_by_components"
    if values.shape == pca.T.shape and values.shape != pca.shape:
        values = values.T
        orientation = "components_by_cells_transposed"
    if values.shape != pca.shape or not np.isfinite(values).all():
        raise ValueError(f"Harmony returned an invalid shape or non-finite values: {values.shape}")
    objectives = [float(value) for value in getattr(output, "objective_harmony", [])]
    iterations = max(0, len(objectives) - 1)
    relative_change = None
    library_stop_criterion_met = False
    strict_converged = False
    if len(objectives) >= 2 and objectives[-2] != 0:
        relative_change = (objectives[-2] - objectives[-1]) / abs(objectives[-2])
        library_stop_criterion_met = relative_change < config["epsilon_harmony"]
        strict_converged = 0.0 <= relative_change < config["epsilon_harmony"]
    if library_stop_criterion_met and not strict_converged:
        stop_interpretation = "harmonypy stop criterion was met after the objective increased; not classified as strict convergence"
    elif strict_converged:
        stop_interpretation = "strict nonnegative relative-change convergence"
    elif iterations >= config["max_iter_harmony"]:
        stop_interpretation = "maximum Harmony iterations reached"
    else:
        stop_interpretation = "Harmony returned before a recorded convergence decision"
    record = {
        "batch_key": config["batch_key"],
        "max_iter_harmony": int(config["max_iter_harmony"]),
        "epsilon_harmony": float(config["epsilon_harmony"]),
        "random_state": int(config["random_state"]),
        "thread_limit": int(config["thread_limit"]),
        "iterations_completed": int(iterations),
        "harmonypy_stop_criterion_met": bool(library_stop_criterion_met),
        "strict_converged": bool(strict_converged),
        "final_objective_increased": bool(relative_change is not None and relative_change < 0),
        "final_relative_objective_change": None if relative_change is None else float(relative_change),
        "stop_interpretation": stop_interpretation,
        "objective_harmony": objectives,
        "returned_orientation": orientation,
    }
    return values.astype(np.float32), record


def preprocess_bundle(
    bundle: SpatialBundle,
    preprocessing: dict,
    *,
    harmony_runner: Callable | None = None,
) -> PreparedFoundation:
    counts = sparse.csr_matrix(bundle.counts)
    detected = np.asarray((counts > 0).sum(axis=0)).ravel()
    gene_keep = detected >= preprocessing["pooled_gene_min_cells"]
    if int(gene_keep.sum()) <= preprocessing["hvg"]["n_top_genes"]:
        raise ValueError("Too few pooled genes remain for the requested HVG panel")
    counts = counts[:, gene_keep].astype(np.int32).tocsr()
    gene_ids = bundle.gene_ids[gene_keep]
    gene_symbols = bundle.gene_symbols[gene_keep]
    obs = bundle.metadata[["section", "donor", "label"]].copy()
    obs.index = bundle.metadata["cell_id"].astype(str)
    var = pd.DataFrame({"gene_id": gene_ids, "gene_symbol": gene_symbols}, index=gene_ids)
    data = ad.AnnData(X=counts.astype(np.float32), obs=obs, var=var)
    row_sums = np.asarray(data.X.sum(axis=1)).ravel()
    if np.any(row_sums <= 0):
        raise ValueError("A retained spot has zero pooled counts")
    sc.pp.normalize_total(data, target_sum=preprocessing["normalize_total_target_sum"])
    if not preprocessing["log1p"]:
        raise ValueError("The frozen protocol requires log1p")
    sc.pp.log1p(data)
    hvg = preprocessing["hvg"]
    sc.pp.highly_variable_genes(
        data, n_top_genes=hvg["n_top_genes"], flavor=hvg["flavor"],
        batch_key=hvg["batch_key"], subset=False, inplace=True,
    )
    hvg_mask = data.var["highly_variable"].to_numpy(dtype=bool)
    if int(hvg_mask.sum()) != hvg["n_top_genes"]:
        raise ValueError(f"Expected {hvg['n_top_genes']} HVGs, observed {int(hvg_mask.sum())}")
    reduced = data[:, hvg_mask].copy()
    scale = preprocessing["scale"]
    sc.pp.scale(reduced, zero_center=scale["zero_center"], max_value=scale["max_value"])
    pca_config = preprocessing["pca"]
    sc.tl.pca(
        reduced, n_comps=pca_config["n_components"], zero_center=scale["zero_center"],
        svd_solver=pca_config["svd_solver"], random_state=pca_config["random_state"],
        dtype="float32",
    )
    pca = np.asarray(reduced.obsm["X_pca"], dtype=np.float32)
    if pca.shape != (counts.shape[0], pca_config["n_components"]) or not np.isfinite(pca).all():
        raise ValueError("PCA output has an invalid shape or non-finite value")
    harmony, harmony_record = _harmony_result(
        pca, bundle.metadata, preprocessing["harmony"], harmony_runner,
    )
    hvg_nbatches = data.var["highly_variable_nbatches"].to_numpy(dtype=np.int64)
    hvg_dispersion = data.var["dispersions_norm"].to_numpy(dtype=np.float64)
    hvg_rank = np.full(len(gene_ids), np.nan, dtype=np.float64)
    selected = np.flatnonzero(hvg_mask)
    ranked = selected[np.lexsort((gene_ids[selected], -hvg_dispersion[selected], -hvg_nbatches[selected]))]
    hvg_rank[ranked] = np.arange(1, len(ranked) + 1, dtype=np.float64)
    gene_metadata = pd.DataFrame({
        "gene_id": gene_ids,
        "gene_symbol": gene_symbols,
        "detected_spots": detected[gene_keep].astype(np.int64),
        "highly_variable": hvg_mask,
        "highly_variable_nbatches": hvg_nbatches,
        "hvg_selection_rank": hvg_rank,
        "mean_log_expression": data.var["means"].to_numpy(dtype=np.float64),
        "dispersion": data.var["dispersions"].to_numpy(dtype=np.float64),
        "normalized_dispersion": data.var["dispersions_norm"].to_numpy(dtype=np.float64),
    })
    cell_metadata = bundle.metadata.copy()
    record = {
        "preprocessing": preprocessing,
        "versions": dependency_versions(),
        "n_spots_retained": int(counts.shape[0]),
        "n_spots_excluded_missing_label": int(len(bundle.excluded)),
        "n_genes_source": int(len(bundle.gene_ids)),
        "n_genes_after_pooled_min_cells": int(counts.shape[1]),
        "n_highly_variable_genes": int(hvg_mask.sum()),
        "pca_shape": list(pca.shape),
        "harmony_shape": list(harmony.shape),
        "harmony_convergence": harmony_record,
        "label_use": "Layer availability determines the retained evaluation cohort; layer class values are not used by gene filtering, normalization, HVG selection, scaling, PCA, or Harmony.",
        "logical_hashes": {
            "counts_csr": csr_hash(counts),
            "cell_ids": array_hash(_text(cell_metadata["cell_id"])),
            "labels": array_hash(_text(cell_metadata["label"])),
            "sections": array_hash(_text(cell_metadata["section"])),
            "donors": array_hash(_text(cell_metadata["donor"])),
            "spatial_pixel_xy": array_hash(cell_metadata[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=np.float64)),
            "spatial_array_col_row": array_hash(cell_metadata[["array_col", "array_row"]].to_numpy(dtype=np.float64)),
            "spatial_hires_pixel_xy": array_hash(cell_metadata[["hires_pxl_col", "hires_pxl_row"]].to_numpy(dtype=np.float64)),
            "hvg_gene_ids": array_hash(_text(gene_metadata.loc[gene_metadata["highly_variable"], "gene_id"])),
            "pca": array_hash(pca),
            "harmony": array_hash(harmony),
        },
    }
    return PreparedFoundation(counts, gene_metadata, cell_metadata, bundle.excluded, pca, harmony, record)


def prepare_foundation(
    spec: dict,
    project_root: Path | str = ROOT,
    runs_root: Path | str = RUNS,
    *,
    run_id: str | None = None,
    harmony_runner: Callable | None = None,
) -> Path:
    project_root = Path(project_root).resolve()
    root = source_root(spec, project_root)
    inventory = verify_sources(spec, project_root)
    manifest_validation = verify_manifest_urls(spec, root)
    run_config = {
        "protocol": spec,
        "source_inventory": inventory,
        "source_manifest_validation": manifest_validation,
        "implementation": {
            "revision_pipeline/spatial_multisection/source_acquisition.py": file_fingerprint(
                project_root / "revision_pipeline/spatial_multisection/source_acquisition.py"
            ),
            "revision_pipeline/spatial_multisection/prepare.py": file_fingerprint(
                project_root / "revision_pipeline/spatial_multisection/prepare.py"
            ),
            "revision_pipeline/configs/spatial_multisection_v1.json": file_fingerprint(
                project_root / "revision_pipeline/configs/spatial_multisection_v1.json"
            ),
        },
    }
    with RunDirectory(
        runs_root, kind="spatial_multisection_foundation", config=run_config, run_id=run_id,
    ) as run:
        bundle = load_spatial_bundle(spec, root)
        prepared = preprocess_bundle(bundle, spec["preprocessing"], harmony_runner=harmony_runner)
        sparse.save_npz(run.artifact_path("counts_pooled_min3.npz"), prepared.counts, compressed=True)
        prepared.cell_metadata.to_csv(run.artifact_path("cell_metadata.tsv"), sep="\t", index=False)
        prepared.excluded.to_csv(run.artifact_path("excluded_unlabeled_spots.tsv"), sep="\t", index=False)
        prepared.gene_metadata.to_csv(run.artifact_path("gene_metadata.tsv"), sep="\t", index=False)
        np.save(run.artifact_path("pca50.npy"), prepared.pca, allow_pickle=False)
        np.save(run.artifact_path("harmony50.npy"), prepared.harmony, allow_pickle=False)
        run.write_json("source_inventory.json", inventory)
        run.write_json("section_summary.json", {
            "sections": bundle.section_summary,
            "total_spots_in_filtered_h5": int(sum(row["spots_in_filtered_h5"] for row in bundle.section_summary)),
            "labeled_spots_retained": int(len(prepared.cell_metadata)),
            "missing_labels_excluded": int(len(prepared.excluded)),
            "donors": sorted(prepared.cell_metadata["donor"].unique().tolist()),
            "label_counts": {
                str(key): int(value)
                for key, value in prepared.cell_metadata["label"].value_counts().sort_index().items()
            },
        })
        run.write_json("spatial_assets.json", {
            "source_root": spec["dataset"]["source_root"],
            "sections": {
                row["section"]: row["histology_hires"] for row in bundle.section_summary
            },
            "downstream_use": "Locked high-resolution histology and scale factors for a later SpaGCN stage; no SpaGCN computation is performed here.",
        })
        run.write_json("preprocessing_record.json", prepared.record)
        run.write_json("embedding_contract.json", {
            "row_alignment": "Rows of pca50.npy and harmony50.npy match cell_metadata.tsv exactly.",
            "cell_id": "section::10x_barcode",
            "counts": "counts_pooled_min3.npz is a CSR matrix in the same row order and gene_metadata.tsv column order.",
            "spatial_pixel_xy": ["pxl_col_in_fullres", "pxl_row_in_fullres"],
            "spatial_array_col_row": ["array_col", "array_row"],
            "spatial_hires_pixel_xy": ["hires_pxl_col", "hires_pxl_row"],
            "histology_assets": "See spatial_assets.json; image/scalefactor files are source-locked in source_inventory.json.",
            "pca_dimensions": 50,
            "harmony_dimensions": 50,
            "harmony_batch_key": spec["preprocessing"]["harmony"]["batch_key"],
            "logical_hashes": prepared.record["logical_hashes"],
            "scope_boundary": spec["scope"],
        })
        run.manifest.update(
            scientific_experiment=True,
            experiment_role="six_section_dlpfc_locked_preprocessing_foundation",
            training_performed=False,
            scoring_performed=False,
        )
    return run.final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--runs-root", type=Path, default=RUNS)
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required")
    spec = load_spec(args.config)
    result = prepare_foundation(
        spec, args.project_root, args.runs_root, run_id=args.run_id,
    )
    print(json.dumps({"run": str(result)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
