# Purpose: Train one histology-aware SpaGCN replicate for one locked DLPFC section.
# Author: Ariana Rahman (Arizona State University)

"""Train one histology-aware SpaGCN replicate for one locked DLPFC section."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
from PIL import Image
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA
import torch

from ..data.readers import array_hash
from ..integrity import canonical_hash, file_fingerprint
from ..runs import RunDirectory
from .common import (
    ROOT, RUNS, foundation_path, load_k_selection, load_metadata, read_json, source_snapshot,
    specification,
)
from .spagcn_compat import fit_device_compatible, load_official_spagcn, seed_all
from .spagcn_runtime import validate_runtime


def _load_section_inputs(section: str) -> tuple[ad.AnnData, object, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spec = specification()
    if section not in spec["sections"]:
        raise ValueError("Unplanned DLPFC section")
    foundation = foundation_path()
    metadata = load_metadata()
    take = np.flatnonzero(metadata["section"].astype(str).to_numpy() == section)
    section_meta = metadata.iloc[take].reset_index(drop=True).drop(columns=["label"])
    counts = sparse.load_npz(foundation / "counts_pooled_min3.npz").tocsr()[take]
    genes = np.genfromtxt(
        foundation / "gene_metadata.tsv", delimiter="\t", names=True, dtype=None,
        encoding="utf-8", usecols=(0, 1),
    )
    gene_ids = np.asarray(genes["gene_id"], dtype="U")
    gene_symbols = np.asarray(genes["gene_symbol"], dtype="U")
    if counts.shape != (len(section_meta), len(gene_ids)) or len(section_meta) < 100:
        raise ValueError("Section count/metadata alignment failed")
    data = ad.AnnData(X=counts.astype(np.float32))
    data.obs_names = section_meta["cell_id"].astype(str).tolist()
    data.var_names = gene_symbols.tolist()
    data.var["gene_id"] = gene_ids
    data.var_names_make_unique()
    hires = section_meta[["hires_pxl_row", "hires_pxl_col"]].to_numpy(dtype=np.float64)
    array_grid = section_meta[["array_row", "array_col"]].to_numpy(dtype=np.float64)
    fullres = section_meta[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=np.float64)
    return data, section_meta, take, array_grid, hires, fullres


def _preprocess(data: ad.AnnData, spg, settings: dict) -> tuple[ad.AnnData, np.ndarray]:
    data = data.copy()
    spg.prefilter_genes(data, min_cells=settings["preprocessing"]["min_cells"])
    spg.prefilter_specialgenes(
        data,
        Gene1Pattern=settings["preprocessing"]["excluded_gene_prefixes"][0],
        Gene2Pattern=settings["preprocessing"]["excluded_gene_prefixes"][1],
    )
    if data.n_vars <= settings["num_pcs"]:
        raise ValueError("Too few genes after official SpaGCN filtering")
    sc.pp.normalize_total(data, target_sum=settings["preprocessing"]["normalize_total"])
    sc.pp.log1p(data)
    dense = data.X.toarray() if sparse.issparse(data.X) else np.asarray(data.X)
    pca = PCA(n_components=settings["num_pcs"], svd_solver=settings["pca_solver"])
    # Match SpaGCN 1.2.7 exactly: its public train method calls fit and then
    # transform rather than the convenience fit_transform path.
    pca.fit(dense)
    embedding = pca.transform(dense)
    if embedding.shape != (data.n_obs, settings["num_pcs"]) or not np.isfinite(embedding).all():
        raise ValueError("SpaGCN PCA failed")
    return data, np.asarray(embedding, dtype=np.float32)


def _histology_adjacency(
    section: str, array_grid: np.ndarray, hires: np.ndarray, spg, settings: dict,
) -> tuple[np.ndarray, dict]:
    assets = read_json(foundation_path() / "spatial_assets.json")
    source_root = ROOT / assets["source_root"]
    item = assets["sections"][section]
    image_path = source_root / item["image_relative_path"]
    scalefactors_path = source_root / item["scalefactors_relative_path"]
    inventory = read_json(foundation_path() / "source_inventory.json")["verified_files"]
    locked = {}
    for role, relative, path in (
        ("histology_hires", item["image_relative_path"], image_path),
        ("scalefactors", item["scalefactors_relative_path"], scalefactors_path),
    ):
        matches = [entry for entry in inventory
                   if entry.get("role") == role and str(entry.get("section")) == section
                   and entry.get("relative_path") == relative]
        if len(matches) != 1:
            raise ValueError(f"Frozen foundation lacks one locked {role} record for {section}")
        observed = file_fingerprint(path)
        expected = {"sha256": matches[0]["sha256"], "size_bytes": matches[0]["size_bytes"]}
        if observed != expected:
            raise ValueError(f"Locked external spatial asset changed: {relative}")
        locked[role] = observed
    with Image.open(image_path) as opened:
        image = np.asarray(opened.convert("RGB"))
    pixels = np.rint(hires).astype(np.int64)
    if (pixels[:, 0].min() < 0 or pixels[:, 1].min() < 0
            or pixels[:, 0].max() >= image.shape[0] or pixels[:, 1].max() >= image.shape[1]):
        raise ValueError("Section spot coordinate lies outside the locked high-resolution image")
    adjacency = spg.calculate_adj_matrix(
        x=array_grid[:, 0].tolist(), y=array_grid[:, 1].tolist(),
        x_pixel=pixels[:, 0].tolist(), y_pixel=pixels[:, 1].tolist(),
        image=image, beta=settings["histology"]["beta"], alpha=settings["histology"]["alpha"],
        histology=True,
    )
    adjacency = np.asarray(adjacency, dtype=np.float32)
    if adjacency.shape != (len(hires), len(hires)) or not np.isfinite(adjacency).all():
        raise ValueError("SpaGCN histology adjacency is invalid")
    record = {
        "image": file_fingerprint(image_path),
        "locked_image_source": locked["histology_hires"],
        "locked_scalefactors_source": locked["scalefactors"],
        "image_relative_path": item["image_relative_path"],
        "scalefactors_relative_path": item["scalefactors_relative_path"],
        "spatial_geometry_coordinate_source": ["array_row", "array_col"],
        "histology_sampling_coordinate_source": ["rounded hires_pxl_row", "rounded hires_pxl_col"],
        "beta": settings["histology"]["beta"],
        "alpha": settings["histology"]["alpha"],
        "adjacency_sha256": array_hash(adjacency),
    }
    return adjacency, record


def execute(section: str, seed: int, harmony: Path, k_selection: Path, run_id: str, device: str) -> Path:
    """Train one histology-aware SpaGCN seed and persist latent and native partitions."""
    spec = specification()
    settings = spec["spagcn"]
    if seed not in settings["seeds"]:
        raise ValueError("Unplanned SpaGCN seed")
    runtime = validate_runtime(require_image_id=True)
    decision = load_k_selection(k_selection, harmony)["sections"][section]
    sources = source_snapshot()
    seed_all(seed)
    spg = load_official_spagcn()
    data, metadata, _, array_grid, hires, fullres = _load_section_inputs(section)
    processed, expression_pca = _preprocess(data, spg, settings)
    adjacency, adjacency_record = _histology_adjacency(section, array_grid, hires, spg, settings)
    search = settings["length_scale_search"]
    length_scale = spg.search_l(
        p=settings["p"], adj=adjacency, start=search["start"], end=search["end"],
        tol=search["tolerance"], max_run=search["max_runs"],
    )
    if length_scale is None or not np.isfinite(length_scale) or length_scale <= 0:
        raise RuntimeError("SpaGCN length-scale search failed")
    adjacency_exp = np.exp(-(adjacency ** 2) / (2 * float(length_scale) ** 2)).astype(np.float32)
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if resolved_device != "cuda":
        raise RuntimeError("Scientific SpaGCN panel runs require the locked CUDA runtime")
    fitted = fit_device_compatible(
        expression_pca, adjacency_exp, n_clusters=int(decision["n_clusters"]), seed=seed,
        learning_rate=settings["learning_rate"], max_epochs=settings["max_epochs"],
        tolerance=settings["tolerance"], weight_decay=settings["weight_decay"],
        update_interval=settings["update_interval"], kmeans_n_init=settings["kmeans_n_init"],
        model_alpha=settings["model_alpha"], device=resolved_device, reseed=False,
    )
    refined = np.asarray(spg.refine(
        sample_id=metadata["cell_id"].astype(str).tolist(), pred=fitted["predicted"].tolist(),
        dis=adjacency, shape="hexagon",
    ), dtype=np.int64)
    if refined.shape != (len(metadata),):
        raise ValueError("SpaGCN native label refinement returned an invalid shape")
    context = {
        "protocol_id": spec["protocol_id"],
        "section": section,
        "donor": str(metadata["donor"].iloc[0]),
        "seed": seed,
        "harmony_manifest": file_fingerprint(Path(harmony) / "run.json"),
        "k_selection_manifest": file_fingerprint(Path(k_selection) / "run.json"),
        "K_binding": decision,
        "settings": settings,
        "device": resolved_device,
        "runtime": runtime,
        "implementation": "API-compatible device port of vendored official SpaGCN 1.2.7 simple_GC_DEC.fit",
    }
    with RunDirectory(RUNS, kind="spatial_multisection_spagcn_training", run_id=run_id, config=context) as run:
        np.savez_compressed(
            run.artifact_path("outputs.npz"),
            ids=metadata["cell_id"].astype(str).to_numpy(dtype="U"),
            latent=np.asarray(fitted["latent"], dtype=np.float32),
            probabilities=np.asarray(fitted["probabilities"], dtype=np.float32),
            predicted=np.asarray(fitted["predicted"], dtype=np.int64),
            refined=refined,
            spatial_fullres_xy=fullres,
        )
        run.write_json("training_record.json", {
            "section": section,
            "donor": str(metadata["donor"].iloc[0]),
            "seed": seed,
            "n_spots": len(metadata),
            "n_genes_after_filter": processed.n_vars,
            "K": int(decision["n_clusters"]),
            "length_scale": float(length_scale),
            "epochs_completed": int(fitted["epochs_completed"]),
            "losses": fitted["losses"],
            "label_change_fraction": fitted["label_change_fraction"],
            "device": fitted["device"],
            "latent_sha256": array_hash(fitted["latent"]),
            "probabilities_sha256": array_hash(fitted["probabilities"]),
            "predicted_sha256": array_hash(fitted["predicted"]),
            "refined_sha256": array_hash(refined),
            "cell_ids_sha256": canonical_hash(metadata["cell_id"].astype(str).tolist()),
            "reference_labels_used": False,
            "native_prediction_role": "secondary task-native spatial-domain endpoint",
            "official_logic_equivalence": "covered by focused small-fixture test",
            "runtime": runtime,
            "histology": adjacency_record,
        })
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        if source_snapshot() != sources:
            raise RuntimeError("Scientific source changed during SpaGCN training")
        run.manifest.update(
            scientific_experiment=True,
            experiment_role="six_section_histology_aware_spagcn_comparator",
            training_performed=True,
            scoring_performed=False,
        )
    return run.final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=specification()["sections"], required=True)
    parser.add_argument("--seed", type=int, choices=range(5), required=True)
    parser.add_argument("--harmony", type=Path, required=True)
    parser.add_argument("--k-selection", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required")
    print(execute(args.section, args.seed, args.harmony, args.k_selection, args.run_id, args.device), flush=True)


if __name__ == "__main__":
    main()
