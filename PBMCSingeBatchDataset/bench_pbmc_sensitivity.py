"""
=============================================================================
Script Name:    bench_pbmc_sensitivity.py
Author:         Ariana Rahman
Affiliation:    Arizona State University / Stanford University
Date:           February 2026

Description:
    Parameter sensitivity analysis for the GenoIntig (Scanorama -> genoDR)
    framework on PBMC, aligned with the PBMC master benchmark format
    (bench_pbmc_master_v4.py) and the pancreas v7 philosophy.

    This script is designed to work with PBMC files produced by your pipeline:
      1) convert_h5_to_h5ad(): writes raw counts into adata.layers["counts"]
         and writes a log-normalized matrix to adata.X / adata.raw.
      2) find_labels_v3(): adds adata.obs["celltype"] while preserving layers["counts"].

    Alignment / correctness fixes (relative to older sensitivity drafts):
      1) Use labeled file with preserved counts:
         - DATA_FILE points to pbmcs_ctrl_labeled.h5ad (not annotated).
      2) No “double log”:
         - Build analysis adata.X from layers["counts"] (raw counts) then
           normalize_total + log1p ONCE on X.
         - Never modify layers["counts"] in-place.
      3) seurat_v3 HVG correctness:
         - HVGs are selected using seurat_v3 on layer="counts" (raw counts).
      4) No “double HVG” in Scanorama:
         - We subset to HVGs BEFORE Scanorama.
         - We do NOT pass hvg=... to scanorama.correct_scanpy.
      5) PBMC-master-consistent clustering policy:
         - Tune Leiden resolution to match the number of classes (target_k),
           rather than using a fixed resolution.
      6) Single-batch safety:
         - If only one batch exists, Scanorama is skipped and we use PCA as
           the base embedding for genoDR (still runs genoDR sensitivity).

Sweeps:
  1) Embedding Dimension: n_dim in [16, 32, 64, 128] with map_size fixed.
  2) Map Resolution:      map_size in [30, 40, 50, 60] with n_dim fixed.

For each parameter value, runs multiple seeds and reports ARI + Silhouette.
Outputs:
  - ./Sensitivity_Results/Sensitivity_Structural_Results.csv
  - ./Sensitivity_Results/Parameter_Sensitivity_Plot.png

Usage:
  python bench_pbmc_sensitivity.py
=============================================================================
"""

import os
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import scanorama
import anndata as ad
import scipy.sparse as sp

from sklearn.metrics import adjusted_rand_score, silhouette_score

# GenoDR import (function API)
try:
    import genomap.genoDR as gp  # gp.genoDR(...)
except Exception:
    gp = None


# -----------------------------
# CONFIGURATION
# -----------------------------
OUTPUT_FOLDER = "./Sensitivity_Results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# IMPORTANT: align with bench_pbmc_master_v4.py input assumptions
DATA_FILE = "./Dataset/pbmcs_ctrl_labeled.h5ad"

BATCH_KEY = "batch"
CELLTYPE_KEY = "celltype"

# preprocessing / canonical-ish parameters
HVG_N = 2000
HVG_FLAVOR = "seurat_v3"   # only valid when layer="counts" exists
N_PCS = 50
N_NEIGHBORS = 15

# Leiden tuning grid (keep similar to PBMC master v4)
LEIDEN_RES_GRID = list(np.round(np.linspace(0.3, 1.6, 14), 2))


# -----------------------------
# HELPERS
# -----------------------------
def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)


def _sanitize_nans_in_X(adata: ad.AnnData) -> None:
    """Replace NaNs with 0 safely for sparse/dense matrices."""
    if sp.issparse(adata.X):
        if adata.X.data.size and np.isnan(adata.X.data).any():
            adata.X.data[np.isnan(adata.X.data)] = 0
    else:
        if np.issubdtype(adata.X.dtype, np.floating) and np.isnan(adata.X).any():
            adata.X[np.isnan(adata.X)] = 0


def tune_leiden_resolution_to_target(
    adata: ad.AnnData,
    use_rep: Optional[str],
    target_k: int,
    n_neighbors: int,
    res_grid: List[float],
    key_added: str,
    random_state: int,
) -> float:
    """
    Build neighbor graph from use_rep and tune Leiden resolution to reach target_k.
    This matches the PBMC master v4 clustering philosophy.
    """
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)

    best_res = float(res_grid[0])
    best_diff = 10**9

    for r in res_grid:
        sc.tl.leiden(adata, resolution=float(r), key_added=key_added, random_state=random_state)
        k_found = int(adata.obs[key_added].nunique())
        diff = abs(k_found - int(target_k))
        if diff < best_diff:
            best_diff = diff
            best_res = float(r)
        if diff == 0:
            break

    sc.tl.leiden(adata, resolution=best_res, key_added=key_added, random_state=random_state)
    return float(best_res)


# -----------------------------
# DATA LOADING / PREP
# -----------------------------
def load_and_prepare_data() -> Optional[ad.AnnData]:
    """
    Loads PBMC data and prepares an analysis matrix aligned with bench_pbmc_master_v4.py:

      - If layers["counts"] exists: adata.X <- raw counts
      - normalize_total + log1p ONCE on X (never modify counts layer)
      - HVG selection using seurat_v3 on layer="counts", subset=True
      - sparse-safe scaling
      - PCA computed (used for single-batch fallback and optional diagnostics)

    Then produces X_scanorama:
      - If multi-batch: run Scanorama on the HVG-subset, log-normalized X
        without passing hvg=...
      - If single-batch: skip Scanorama and use X_pca as X_scanorama
    """
    logging.info(f"Loading data from {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        logging.error(f"Data file not found at {DATA_FILE}")
        return None

    adata = sc.read(DATA_FILE)

    if BATCH_KEY not in adata.obs:
        logging.error(f"Batch key '{BATCH_KEY}' not found in adata.obs.")
        return None
    if CELLTYPE_KEY not in adata.obs:
        logging.error(f"Celltype key '{CELLTYPE_KEY}' not found in adata.obs.")
        return None

    # Canonical columns (match PBMC master style)
    adata.obs["batch"] = adata.obs[BATCH_KEY].astype("category")
    adata.obs["class"] = adata.obs[CELLTYPE_KEY].astype("category")

    # Build analysis X from raw counts if present (no double log)
    if "counts" in adata.layers:
        logging.info("Found layers['counts']; using it as raw counts source for analysis X.")
        adata.X = adata.layers["counts"].copy()
    else:
        logging.warning(
            "No layers['counts'] found. Falling back to current adata.X. "
            "If adata.X is already log-normalized, this may degrade seurat_v3 HVG correctness."
        )

    # Minimal filtering (safe)
    sc.pp.filter_cells(adata, min_genes=1)

    # Normalize + log1p ONCE on X (never modify counts layer)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    _sanitize_nans_in_X(adata)

    # HVG selection (prefer seurat_v3 on raw counts layer)
    if "counts" in adata.layers:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=HVG_N,
            flavor=HVG_FLAVOR,
            layer="counts",
            subset=True,
        )
    else:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=HVG_N,
            flavor="cell_ranger",
            subset=True,
        )

    # Sparse-safe scaling
    if sp.issparse(adata.X):
        sc.pp.scale(adata, max_value=10, zero_center=False)
    else:
        sc.pp.scale(adata, max_value=10)

    # PCA (helps single-batch fallback; also stabilizes downstream)
    sc.tl.pca(adata, n_comps=N_PCS, random_state=0)

    # Build X_scanorama base embedding
    n_batches = int(adata.obs["batch"].nunique())
    if n_batches < 2:
        logging.warning(
            "Single-batch PBMC detected -> skipping Scanorama and using PCA as base embedding "
            "(X_scanorama := X_pca) for genoDR sensitivity."
        )
        adata.obsm["X_scanorama"] = np.asarray(adata.obsm["X_pca"])
        return adata

    # Multi-batch: Scanorama on HVG-subset log-normalized data
    logging.info("Running Scanorama (no hvg=... passed; HVGs already subset)...")
    adatas_list = [adata[adata.obs["batch"] == b].copy() for b in adata.obs["batch"].cat.categories]

    # IMPORTANT: do NOT pass hvg=HVG_N (avoid “double HVG”)
    adatas_corrected = scanorama.correct_scanpy(adatas_list, return_dimred=True)
    adata_full = ad.concat(adatas_corrected, join="outer")
    adata_full.obs_names_make_unique()

    # Ensure class/batch are properly typed after concat
    adata_full.obs["batch"] = adata_full.obs["batch"].astype("category")
    if "class" not in adata_full.obs:
        # In case scanorama concat dropped it (unlikely, but safe)
        # We can recover from original adata if obs_names remained stable.
        if adata_full.obs_names.isin(adata.obs_names).all():
            adata_full.obs["class"] = adata.obs.loc[adata_full.obs_names, "class"].astype("category")
        else:
            raise RuntimeError("Class labels could not be preserved through Scanorama concat.")
    else:
        adata_full.obs["class"] = adata_full.obs["class"].astype("category")

    return adata_full


# -----------------------------
# EXPERIMENT
# -----------------------------
def run_experiment(adata: ad.AnnData, params: Dict[str, int], seed: int) -> Tuple[float, float]:
    """
    Runs genoDR with supported parameters (n_dim, map_size), clusters with Leiden
    tuned to target_k (#classes), then returns (ARI, Silhouette).
    """
    if gp is None:
        logging.error("genomap.genoDR not available. Install/verify genomap.")
        return np.nan, np.nan

    set_all_seeds(seed)

    n_dim = int(params.get("n_dim", 32))
    map_size = int(params.get("map_size", 40))

    # GenoDR expects n_clusters; use #classes
    n_clusters = int(adata.obs["class"].nunique())
    n_clusters = max(2, n_clusters)

    try:
        X_base = np.asarray(adata.obsm["X_scanorama"])
        X_base = np.nan_to_num(X_base)

        Z = gp.genoDR(
            X_base,
            n_dim=n_dim,
            n_clusters=n_clusters,
            colNum=map_size,
            rowNum=map_size,
        )

        adata_run = adata.copy()
        adata_run.obsm["X_genointig"] = Z

        # Cluster: tune Leiden to target #classes (PBMC master v4 policy)
        target_k = int(adata_run.obs["class"].nunique())
        used_res = tune_leiden_resolution_to_target(
            adata=adata_run,
            use_rep="X_genointig",
            target_k=target_k,
            n_neighbors=N_NEIGHBORS,
            res_grid=LEIDEN_RES_GRID,
            key_added="leiden",
            random_state=seed,
        )

        y_true = adata_run.obs["class"].astype(str).to_numpy()
        y_pred = adata_run.obs["leiden"].astype(str).to_numpy()

        ari = float(adjusted_rand_score(y_true, y_pred))
        try:
            sil = float(silhouette_score(adata_run.obsm["X_genointig"], y_pred))
        except Exception:
            sil = float("nan")

        # (used_res is intentionally not returned; but easy to log if needed)
        _ = used_res
        return ari, sil

    except Exception as e:
        logging.error(f"Run failed for params={params}, seed={seed}: {e}")
        return np.nan, np.nan


# -----------------------------
# MAIN SWEEP LOGIC
# -----------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    adata = load_and_prepare_data()
    if adata is None:
        return

    # DEFINE SWEEPS (Structural Parameters)
    sweeps = {
        "Embedding_Dim": {
            "param": "n_dim",
            "values": [16, 32, 64, 128],
            "defaults": {"map_size": 40},
        },
        "Map_Resolution": {
            "param": "map_size",
            "values": [30, 40, 50, 60],
            "defaults": {"n_dim": 32},
        },
    }

    seeds = [42, 123, 2024]
    all_results = []

    for sweep_name, cfg in sweeps.items():
        logging.info(f"--- Starting Sweep: {sweep_name} ---")
        param_name = cfg["param"]

        for val in cfg["values"]:
            for seed in seeds:
                current_params = cfg["defaults"].copy()
                current_params[param_name] = int(val)

                logging.info(f"Running {sweep_name}: {param_name}={val}, Seed={seed}")
                ari, sil = run_experiment(adata, current_params, seed)

                all_results.append(
                    {
                        "Sweep_Type": sweep_name,
                        "Parameter_Name": param_name,
                        "Parameter_Value": int(val),
                        "n_dim": int(current_params.get("n_dim", 32)),
                        "map_size": int(current_params.get("map_size", 40)),
                        "Seed": int(seed),
                        "ARI": float(ari),
                        "Silhouette": float(sil),
                    }
                )

    df = pd.DataFrame(all_results)
    out_csv = os.path.join(OUTPUT_FOLDER, "Sensitivity_Structural_Results.csv")
    df.to_csv(out_csv, index=False)
    logging.info(f"Saved raw results to: {out_csv}")

    if df.empty:
        logging.error("No results generated.")
        return

    # Plot: 2-panel ARI mean±std vs parameter (keeps your original plot style)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: n_dim sweep
    left_name = "Embedding_Dim"
    left = df[df["Sweep_Type"] == left_name].copy()
    left_summary = left.groupby("Parameter_Value")["ARI"].agg(["mean", "std"]).reset_index()

    axes[0].errorbar(
        left_summary["Parameter_Value"],
        left_summary["mean"],
        yerr=left_summary["std"],
        fmt="-o",
        capsize=5,
        label="ARI",
    )
    axes[0].set_title("Sensitivity to Embedding Dimension")
    axes[0].set_xlabel("n_dim")
    axes[0].set_ylabel("ARI Score")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].set_xticks(left_summary["Parameter_Value"])

    # Right: map_size sweep
    right_name = "Map_Resolution"
    right = df[df["Sweep_Type"] == right_name].copy()
    right_summary = right.groupby("Parameter_Value")["ARI"].agg(["mean", "std"]).reset_index()

    axes[1].errorbar(
        right_summary["Parameter_Value"],
        right_summary["mean"],
        yerr=right_summary["std"],
        fmt="-o",
        capsize=5,
        label="ARI",
    )
    axes[1].set_title("Sensitivity to Map Resolution")
    axes[1].set_xlabel("map_size")
    axes[1].set_ylabel("ARI Score")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].set_xticks(right_summary["Parameter_Value"])

    plt.tight_layout()
    out_png = os.path.join(OUTPUT_FOLDER, "Parameter_Sensitivity_Plot.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    logging.info(f"Saved plot to: {out_png}")
    logging.info("Sensitivity analysis complete.")


if __name__ == "__main__":
    main()
