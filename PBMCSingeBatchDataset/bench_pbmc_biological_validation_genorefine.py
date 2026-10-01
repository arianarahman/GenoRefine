# Purpose: Biological validation (single-batch PBMC control) for GenoRefine.
# Author: Ariana Rahman (Arizona State University)

"""
=============================================================================
Script Name:    bench_pbmc_biological_validation_genorefine.py
Date:           February 2026
Description:
    Biological validation (single-batch PBMC control) for GenoRefine.

    CLEANUP (GenoRefine manuscript version):
      - Renames all manuscript-facing GenoIntig labels to GenoRefine.
      - Uses GenoRefine-consistent cluster labels: obs['genorefine_clusters'].
      - Uses GenoRefine-consistent embedding key: obsm['X_genorefine_pca'].
      - Preserves a full-gene, log-normalized expression matrix in adata.raw
        BEFORE HVG subsetting and scaling.
      - Marker plots and differential-expression checks use adata.raw so
        canonical marker genes are not lost if they are not selected as HVGs.
      - Keeps PBMC as a single-batch biological signal preservation control.
        This script does not evaluate batch mixing or iLISI.

    GOAL
      Verify that GenoRefine, implemented here as GenoDR refinement on top of
      PCA, preserves canonical PBMC marker signals in a single-batch dataset
      where no batch integration is required.

    OUTPUTS
      - Figure_7a_GenoRefine_Dotplot_Clusters.png
      - Figure_7b_GenoRefine_Violin_CellTypes.png
      - Table_IV_GenoRefine_DE_Consistency.csv
      - PBMC_GenoRefine_Marker_Audit.csv

=============================================================================
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse

# --- GenoDR import (function API) ---
try:
    import genomap.genoDR as gp  # gp.genoDR(...)
except Exception:
    gp = None


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class BioConfig:
    data_file: str = "./Dataset/pbmcs_ctrl_labeled.h5ad"
    out_folder: str = "./Biological_Validation_Results_PBMC_SingleBatch"
    seed: int = 0

    # Keys
    batch_key: str = "batch"
    class_key_candidates: Tuple[str, ...] = ("class", "celltype", "cell_type", "labels")

    # Preprocessing
    hvg: int = 3000
    hvg_flavor_counts: str = "seurat_v3"       # only when raw counts are available
    hvg_flavor_fallback: str = "cell_ranger"   # safe fallback for non-count inputs
    n_pcs: int = 50
    normalize_target_sum: float = 1e4
    max_scale_value: float = 10.0

    # Graph + clustering
    n_neighbors: int = 15
    leiden_res: float = 0.8

    # GenoRefine / GenoDR backend
    genorefine_embed_key: str = "X_genorefine_pca"
    cluster_key: str = "genorefine_clusters"
    genodr_dim: int = 64
    genodr_col: int = 50
    genodr_row: int = 50

    # Counts-layer integer-like heuristic
    integer_like_threshold: float = 0.95
    integer_like_sample: int = 5000


CFG = BioConfig()

# Canonical PBMC markers. Edit if your PBMC labels use different conventions.
MARKER_DICT: Dict[str, List[str]] = {
    "T cells": ["CD3D", "CD3E"],
    "B cells": ["MS4A1"],
    "Monocytes": ["LST1", "S100A8"],
}


# -----------------------------
# Utilities
# -----------------------------
def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _copy_matrix(X: Any) -> Any:
    if sparse.issparse(X):
        return X.copy()
    return np.array(X, copy=True)


def _pick_class_key(adata: ad.AnnData, candidates: Tuple[str, ...]) -> Optional[str]:
    for k in candidates:
        if k in adata.obs.columns:
            return k
    return None


def _is_already_log_like(X: Any) -> bool:
    """
    Heuristic: if matrix contains many non-integers and values are mostly small,
    it is likely already log-normalized.
    """
    try:
        data = X.data if sparse.issparse(X) else np.asarray(X).ravel()
        if data.size == 0:
            return False
        nonint = np.mean(~np.isclose(data, np.round(data)))
        small = np.percentile(data, 99) < 50
        return bool((nonint > 0.5) and small)
    except Exception:
        return False


def _nan_guard_matrix(X: Any, name: str) -> Any:
    """Replace NaN/Inf with zero in dense or sparse matrix-like inputs."""
    if sparse.issparse(X):
        X = X.copy()
        if X.data.size and not np.isfinite(X.data).all():
            logging.warning("%s: found NaN/Inf in sparse data; replacing with 0.", name)
            X.data = np.nan_to_num(X.data, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    X_arr = np.asarray(X)
    if not np.isfinite(X_arr).all():
        logging.warning("%s: found NaN/Inf; replacing with 0.", name)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return X_arr


def _nan_guard_array(X: np.ndarray, name: str) -> np.ndarray:
    X = np.asarray(X)
    if not np.isfinite(X).all():
        logging.warning("%s: found NaN/Inf; replacing with 0.", name)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _layer_is_integer_like(X_layer: Any, threshold: float = 0.95, sample_n: int = 5000) -> bool:
    """Return True if the provided matrix looks like raw counts."""
    try:
        data = X_layer.data if sparse.issparse(X_layer) else np.asarray(X_layer).ravel()
        if data.size == 0:
            return False
        take = min(int(sample_n), int(data.size))
        sample = data[:take]
        frac_int = float(np.mean(np.isclose(sample, np.round(sample))))
        return frac_int >= float(threshold)
    except Exception:
        return False


def _gene_universe_for_markers(adata: ad.AnnData) -> pd.Index:
    if adata.raw is not None:
        return pd.Index(adata.raw.var_names.astype(str))
    return pd.Index(adata.var_names.astype(str))


def _valid_marker_dict(adata: ad.AnnData, marker_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Filter markers against adata.raw.var_names when available."""
    genes = set(_gene_universe_for_markers(adata))
    valid = {k: [g for g in v if g in genes] for k, v in marker_dict.items()}
    return {k: v for k, v in valid.items() if len(v) > 0}


def _save_scanpy_plot(plot_obj: Any, out_png: str) -> None:
    """Save Scanpy plot objects robustly across Scanpy versions."""
    if hasattr(plot_obj, "savefig"):
        plot_obj.savefig(out_png)
    else:
        import matplotlib.pyplot as plt
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()


def _write_marker_audit(adata: ad.AnnData, cfg: BioConfig) -> None:
    raw_genes = set(adata.raw.var_names.astype(str)) if adata.raw is not None else set()
    hvg_genes = set(adata.var_names.astype(str))
    rows: List[Dict[str, Any]] = []
    for group, markers in MARKER_DICT.items():
        for gene in markers:
            rows.append({
                "Cell Type Group": group,
                "Marker Gene": gene,
                "Present in full log-normalized raw": bool(gene in raw_genes),
                "Present in HVG subset": bool(gene in hvg_genes),
            })
    out_csv = os.path.join(cfg.out_folder, "PBMC_GenoRefine_Marker_Audit.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logging.info("Saved marker audit: %s", out_csv)


# -----------------------------
# Data loading and preprocessing
# -----------------------------
def load_pbmc_singlebatch(cfg: BioConfig) -> Tuple[ad.AnnData, str]:
    """
    Load PBMC, standardize metadata, preserve full-gene log-normalized raw,
    select HVGs, scale, and compute PCA.

    Result:
      - adata.raw stores full-gene log-normalized expression for marker plots and DE.
      - adata.X is HVG-subset, scaled analysis matrix for PCA and GenoRefine.
      - adata.layers['lognorm_hvg'] stores HVG-subset log-normalized expression before scaling.
      - adata.obsm['X_pca'] exists.
    """
    logging.info("Loading PBMC data: %s", cfg.data_file)
    if not os.path.exists(cfg.data_file):
        raise FileNotFoundError(f"Could not find PBMC data file: {cfg.data_file}")

    adata = ad.read_h5ad(cfg.data_file)
    adata.var_names_make_unique()

    # Standardize batch metadata. PBMC is expected to be single-batch, but this
    # keeps downstream code compatible with multi-batch AnnData objects.
    if cfg.batch_key in adata.obs.columns:
        adata.obs["batch"] = adata.obs[cfg.batch_key].astype(str).astype("category")
    else:
        adata.obs["batch"] = pd.Categorical(["batch_0"] * adata.n_obs)

    # Standardize class metadata.
    class_key = _pick_class_key(adata, cfg.class_key_candidates)
    if class_key is not None:
        adata.obs["class"] = adata.obs[class_key].astype(str).astype("category")
        class_key_used = "class"
        logging.info("Using class labels from obs['%s'] -> obs['class'].", class_key)
    else:
        class_key_used = ""
        logging.warning(
            "No class labels found in obs. Tried: %s. Will proceed with cluster-based plots only.",
            cfg.class_key_candidates,
        )

    # Choose expression source and preserve raw counts separately for seurat_v3 HVG selection.
    temp_counts_layer = "_raw_counts_for_hvg"
    use_counts_hvg = False

    if "counts" in adata.layers and _layer_is_integer_like(
        adata.layers["counts"], cfg.integer_like_threshold, cfg.integer_like_sample
    ):
        logging.info("Found integer-like layers['counts']; using it for analysis and seurat_v3 HVGs.")
        raw_counts = _copy_matrix(adata.layers["counts"])
        adata.X = _copy_matrix(raw_counts)
        adata.layers[temp_counts_layer] = _copy_matrix(raw_counts)
        use_counts_hvg = True

        adata.X = _nan_guard_matrix(adata.X, "raw_counts_X")
        sc.pp.normalize_total(adata, target_sum=float(cfg.normalize_target_sum))
        sc.pp.log1p(adata)
        hvg_flavor = cfg.hvg_flavor_counts
        hvg_layer = temp_counts_layer
    else:
        if "counts" in adata.layers:
            logging.warning(
                "layers['counts'] exists but does not look raw-count-like. "
                "Using it as expression input and falling back to '%s' HVGs.",
                cfg.hvg_flavor_fallback,
            )
            adata.X = _copy_matrix(adata.layers["counts"])
        else:
            logging.info("No counts layer found; using adata.X as expression input.")
            adata.X = _copy_matrix(adata.X)

        adata.X = _nan_guard_matrix(adata.X, "expression_X")
        if _is_already_log_like(adata.X):
            logging.info("Expression matrix appears log-normalized; skipping normalize_total/log1p.")
        else:
            logging.info("Normalizing and log1p-transforming expression matrix.")
            sc.pp.normalize_total(adata, target_sum=float(cfg.normalize_target_sum))
            sc.pp.log1p(adata)
        hvg_flavor = cfg.hvg_flavor_fallback
        hvg_layer = None

    # Preserve full-gene log-normalized expression BEFORE HVG subsetting and scaling.
    # Marker plots and rank_genes_groups use this raw object so canonical markers are
    # retained even if they are absent from the HVG subset.
    adata.raw = adata.copy()
    logging.info("Stored full-gene log-normalized expression in adata.raw: shape=%s.", adata.raw.shape)

    # Select HVGs. If seurat_v3 on raw counts fails, fall back to cell_ranger on log-normalized X.
    n_top = min(int(cfg.hvg), int(adata.n_vars))
    try:
        logging.info("Selecting HVGs: flavor='%s', layer=%s, n_top=%d.", hvg_flavor, hvg_layer, n_top)
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top,
            flavor=hvg_flavor,
            layer=hvg_layer,
            subset=True,
        )
    except Exception as e:
        if use_counts_hvg:
            logging.warning(
                "Primary seurat_v3 HVG selection failed (%s). Falling back to flavor='%s' on log-normalized X.",
                str(e),
                cfg.hvg_flavor_fallback,
            )
            # Restore full log-normalized X from raw, then select fallback HVGs.
            adata = adata.raw.to_adata()
            adata.raw = adata.copy()
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=n_top,
                flavor=cfg.hvg_flavor_fallback,
                subset=True,
            )
        else:
            raise

    # Remove temporary raw-count layer after HVG selection, if present.
    if temp_counts_layer in adata.layers:
        del adata.layers[temp_counts_layer]

    # Preserve HVG-subset log-normalized expression before scaling.
    adata.layers["lognorm_hvg"] = _copy_matrix(adata.X)

    _write_marker_audit(adata, cfg)

    # Scale for PCA. Keep raw/lognorm layers untouched for biological validation.
    if sparse.issparse(adata.X):
        sc.pp.scale(adata, max_value=float(cfg.max_scale_value), zero_center=False)
    else:
        sc.pp.scale(adata, max_value=float(cfg.max_scale_value))

    sc.tl.pca(adata, n_comps=int(cfg.n_pcs), random_state=int(cfg.seed))
    adata.obsm["X_pca"] = _nan_guard_array(np.asarray(adata.obsm["X_pca"]), "X_pca")

    return adata, class_key_used


# -----------------------------
# GenoRefine single-batch refinement
# -----------------------------
def run_genorefine_singlebatch(adata: ad.AnnData, cfg: BioConfig, class_key_used: str) -> ad.AnnData:
    """
    Run GenoRefine for the single-batch PBMC control:
      - GenoDR refinement on PCA embedding.
      - Leiden clustering on cfg.genorefine_embed_key -> cfg.cluster_key.
    """
    if gp is None:
        raise ImportError("genomap.genoDR could not be imported. Install or verify genomap.")

    logging.info("Running GenoRefine single-batch control: GenoDR(PCA) ...")

    X_pca = _nan_guard_array(np.asarray(adata.obsm["X_pca"]), "X_pca")

    if class_key_used:
        n_clusters = max(2, int(adata.obs[class_key_used].nunique()))
    else:
        n_clusters = 8

    Z = gp.genoDR(
        X_pca,
        n_dim=int(cfg.genodr_dim),
        n_clusters=int(n_clusters),
        colNum=int(cfg.genodr_col),
        rowNum=int(cfg.genodr_row),
    )
    adata.obsm[cfg.genorefine_embed_key] = _nan_guard_array(np.asarray(Z), cfg.genorefine_embed_key)

    logging.info("Clustering on %s ...", cfg.genorefine_embed_key)
    sc.pp.neighbors(adata, use_rep=cfg.genorefine_embed_key, n_neighbors=int(cfg.n_neighbors))
    sc.tl.leiden(
        adata,
        resolution=float(cfg.leiden_res),
        key_added=cfg.cluster_key,
        random_state=int(cfg.seed),
    )

    return adata


# Backward-compatible alias for older calls.
run_genointig_singlebatch = run_genorefine_singlebatch


# -----------------------------
# Biological plots and DE consistency
# -----------------------------
def generate_biological_plots(adata: ad.AnnData, cfg: BioConfig, class_key_used: str) -> None:
    """Save marker dotplot and stacked violin using full-gene adata.raw."""
    logging.info("Generating marker expression plots using adata.raw=%s ...", adata.raw is not None)

    valid_markers = _valid_marker_dict(adata, MARKER_DICT)
    if len(valid_markers) == 0:
        logging.error("None of the requested marker genes were found in adata.raw or adata.var_names.")
        return

    use_raw = adata.raw is not None

    # DotPlot across GenoRefine clusters.
    out1 = os.path.join(cfg.out_folder, "Figure_7a_GenoRefine_Dotplot_Clusters.png")
    dp = sc.pl.dotplot(
        adata,
        valid_markers,
        groupby=cfg.cluster_key,
        standard_scale="var",
        use_raw=use_raw,
        show=False,
        return_fig=True,
        title="Marker expression in GenoRefine clusters (single-batch PBMC)",
    )
    _save_scanpy_plot(dp, out1)
    logging.info("Saved: %s", out1)

    # Stacked violin grouped by cell type labels if available, else by clusters.
    groupby_key = "class" if class_key_used else cfg.cluster_key
    out2 = os.path.join(cfg.out_folder, "Figure_7b_GenoRefine_Violin_CellTypes.png")
    sv = sc.pl.stacked_violin(
        adata,
        valid_markers,
        groupby=groupby_key,
        use_raw=use_raw,
        show=False,
        return_fig=True,
        title=f"Marker preservation by {groupby_key}",
    )
    _save_scanpy_plot(sv, out2)
    logging.info("Saved: %s", out2)


def check_de_consistency(adata: ad.AnnData, cfg: BioConfig) -> None:
    """
    Generate Table IV: checks whether canonical markers are top-ranked within
    the GenoRefine cluster where they score highest.

    Marker DE uses adata.raw when available, preserving markers outside the HVG subset.
    Scanpy 'scores' are ranking statistics, not guaranteed Z-scores.
    """
    logging.info("Running DE on %s using adata.raw=%s ...", cfg.cluster_key, adata.raw is not None)

    if cfg.cluster_key not in adata.obs.columns:
        logging.error("Cluster key '%s' not found in adata.obs.", cfg.cluster_key)
        return

    try:
        sc.tl.rank_genes_groups(
            adata,
            cfg.cluster_key,
            method="t-test",
            use_raw=(adata.raw is not None),
        )
    except Exception as e:
        logging.error("DE analysis failed: %s", e)
        return

    result_df = sc.get.rank_genes_groups_df(adata, group=None)
    available_genes = set(result_df["names"].astype(str))
    consistency_rows: List[Dict[str, Any]] = []

    for cell_group, markers in MARKER_DICT.items():
        for gene in markers:
            if gene not in available_genes:
                consistency_rows.append({
                    "Cell Type Group": cell_group,
                    "Marker Gene": gene,
                    "Best Mapping Cluster": "NA",
                    "Rank in Cluster": np.nan,
                    "Score (t-test)": np.nan,
                    "Status": "Not detected",
                })
                continue

            gene_rows = result_df[result_df["names"] == gene].sort_values("scores", ascending=False)
            best = gene_rows.iloc[0]
            cluster_id = best["group"]

            cluster_genes = result_df[result_df["group"] == cluster_id]["names"].astype(str).tolist()
            rank = cluster_genes.index(gene) + 1 if gene in cluster_genes else 999

            consistency_rows.append({
                "Cell Type Group": cell_group,
                "Marker Gene": gene,
                "Best Mapping Cluster": str(cluster_id),
                "Rank in Cluster": int(rank),
                "Score (t-test)": float(best["scores"]),
                "Status": (
                    "Preserved" if rank <= 2 else
                    "High" if rank <= 10 else
                    "Moderate" if rank <= 100 else
                    "Lost"
                ),
            })

    df = pd.DataFrame(consistency_rows).sort_values(["Cell Type Group", "Marker Gene"])
    csv_path = os.path.join(cfg.out_folder, "Table_IV_GenoRefine_DE_Consistency.csv")
    df.to_csv(csv_path, index=False)
    logging.info("Saved: %s", csv_path)

    print("\n--- Table IV: Marker Gene Consistency after GenoRefine (single-batch PBMC) ---")
    print(df.to_string(index=False))


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    sc.settings.verbosity = 0

    cfg = CFG
    ensure_out_dir(cfg.out_folder)
    np.random.seed(int(cfg.seed))

    adata, class_key_used = load_pbmc_singlebatch(cfg)

    n_batches = int(adata.obs["batch"].nunique())
    logging.info(
        "PBMC loaded: n_obs=%d, n_vars=%d HVGs, raw_n_vars=%s, batches=%d",
        adata.n_obs,
        adata.n_vars,
        adata.raw.n_vars if adata.raw is not None else "NA",
        n_batches,
    )

    if n_batches < 2:
        logging.info("Single-batch detected: running GenoRefine biological validation without batch integration.")
    else:
        logging.warning(
            "Multiple batches detected, but this script is intended for single-batch biological controls. "
            "Use the multi-batch benchmark scripts for integration validation."
        )

    adata = run_genorefine_singlebatch(adata, cfg, class_key_used)

    generate_biological_plots(adata, cfg, class_key_used)
    check_de_consistency(adata, cfg)

    logging.info("Done. Results written to: %s", cfg.out_folder)


if __name__ == "__main__":
    main()
