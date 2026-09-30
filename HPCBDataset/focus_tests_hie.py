"""
FILE: focus_tests_hie_canonical_v6.py
-------------------------------------------------------------------------------
Author: Ariana Rahman
Affiliation: Arizona State University / Stanford University
Date: April 2026 (canonical v6)
-------------------------------------------------------------------------------

PURPOSE
    Mechanistic / focus tests for the HIE benchmark using the same frozen
    canonical-HVG preprocessing adopted by bench_hie_master_canonical_v7.py.

CANONICAL CONSISTENCY
    This file is intended to be run after:
        1) make_hie_canonical_hvg.py
        2) seurat_integration_hie_canonical_v5.R
        3) rliger_online_inmf_hie_canonical_v8.R
        4) bench_hie_master_canonical_v7.py or *_fixed.py

    It uses:
        ./Benchmark_Hie_Out/hie_hvg_canonical.txt
        ./Benchmark_Hie_Out/hie_hvg_canonical.md5

    Clean preprocessing order:
        read raw HIE .h5ad
        -> resolve batch/class keys
        -> subset to frozen canonical HVGs
        -> normalize/log
        -> scale
        -> PCA
        -> load exported method embeddings
        -> run topology, kNN stability, compactness, and silhouette focus tests

OUTPUTS
    Written to:
        ./Benchmark_Hie_Out/Focus_Tests_Hie

    - focus_preprocess_fingerprint.csv
    - focus_topology_preservation.csv
    - focus_knn_stability_purity_summary.csv
    - focus_compactness_per_class.csv
    - focus_silhouette_samples.csv
    - focus_hie_metrics.csv
    - focus_hie_pairwise_refinement.csv
    - Optional plots:
        sup_topology_preservation.png
        sup_knn_stability_delta_purity.png
        sup_compactness_per_class.png
        sup_silhouette_violin.png
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)

import scanpy as sc
import anndata as ad

from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr, pearsonr


# -----------------------------
# Config
# -----------------------------
@dataclass
class HieFocusConfig:
    data_file: str = "./Dataset/human_pancreas_norm_complexBatch.h5ad"
    benchmark_folder: str = "./Benchmark_Hie_Out"
    out_folder: str = "./Benchmark_Hie_Out/Focus_Tests_Hie"
    seed: int = 0

    # fixed HIE metadata keys, matching make_hie_canonical_hvg.py and HIE master
    batch_key: str = "tech"
    label_key: str = "celltype"

    # Frozen canonical HVG artifacts from make_hie_canonical_hvg.py
    use_canonical_hvg_file: bool = True
    canonical_hvg_file: str = "hie_hvg_canonical.txt"
    canonical_hvg_md5_file: str = "hie_hvg_canonical.md5"
    canonical_hvg_metadata_file: str = "hie_hvg_canonical_metadata.json"
    check_canonical_hvg_md5_file: bool = True
    require_exact_canonical_hvg_match: bool = True
    fallback_to_dynamic_hvgs_if_missing: bool = False

    # External embeddings from R, aligned by obs_names
    seurat_csv_name: str = "X_seurat_hie.csv"
    online_inmf_csv_name: str = "X_online_inmf_hie.csv"

    # Exported embeddings from bench_hie_master_canonical_v7.py
    embeddings_dir_name: str = "Embeddings"

    # Dynamic HVG fallback only used if canonical file is missing and fallback is enabled
    do_hvg: bool = True
    hvg: int = 2000
    hvg_flavor_counts: str = "seurat_v3"
    hvg_flavor_fallback: str = "cell_ranger"

    # preprocessing after canonical subsetting
    do_normalize_log1p: bool = True
    do_scale: bool = True
    max_scale_value: float = 10.0
    n_pcs: int = 50

    # UMAP / neighbor graph for plots only
    n_neighbors: int = 15
    umap_min_dist: float = 0.3

    # Focus tests
    knn_k: int = 30
    max_classes_for_topology: int = 200


CFG = HieFocusConfig()


# -----------------------------
# Utilities
# -----------------------------
def set_all_seeds(seed: int) -> None:
    np.random.seed(int(seed))


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def md5_master_style_trailing_newline(items: List[str]) -> str:
    h = hashlib.md5()
    for s in items:
        h.update(str(s).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def md5_no_trailing_newline(items: List[str]) -> str:
    payload = "\n".join(map(str, items)).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def _nan_guard(X: np.ndarray, name: str) -> np.ndarray:
    X = np.asarray(X)
    if not np.isfinite(X).all():
        logging.warning("%s: NaN/Inf found; replacing with 0.", name)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _as_str(arr) -> np.ndarray:
    return np.asarray(arr).astype(str)


def _normalize_obs_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"-1$", "", s)
    s = re.sub(r"\.1$", "", s)
    s = s.replace(".", "-")
    return s


def _read_canonical_hvg_file(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ser = pd.read_csv(path, header=None).iloc[:, 0].astype(str)
    hvgs: List[str] = []
    seen = set()
    for x in ser.tolist():
        g = str(x).strip()
        if g == "" or g.lower() == "nan":
            continue
        if g not in seen:
            hvgs.append(g)
            seen.add(g)

    if len(hvgs) == 0:
        raise ValueError(f"Canonical HVG file is empty: {path}")
    return hvgs


def _subset_with_frozen_canonical_hvgs(a: ad.AnnData, cfg: HieFocusConfig) -> ad.AnnData:
    hvg_path = os.path.join(cfg.benchmark_folder, cfg.canonical_hvg_file)
    requested_hvgs = _read_canonical_hvg_file(hvg_path)

    present = set(a.var_names.astype(str).tolist())
    matched_hvgs = [g for g in requested_hvgs if g in present]
    missing_count = len(requested_hvgs) - len(matched_hvgs)

    if len(matched_hvgs) == 0:
        raise ValueError(f"No canonical HVGs matched adata.var_names. File={hvg_path}")

    if cfg.require_exact_canonical_hvg_match and missing_count > 0:
        missing_examples = [g for g in requested_hvgs if g not in present][:10]
        raise ValueError(
            f"Canonical HVG mismatch. requested={len(requested_hvgs)} matched={len(matched_hvgs)} "
            f"missing={missing_count}. Examples missing={missing_examples}"
        )

    md5_master = md5_master_style_trailing_newline(matched_hvgs)
    md5_r_style = md5_no_trailing_newline(matched_hvgs)

    logging.info(
        "Using frozen HIE canonical HVG file: %s | requested=%d matched=%d missing=%d md5_master=%s md5_r_style=%s",
        hvg_path, len(requested_hvgs), len(matched_hvgs), missing_count, md5_master, md5_r_style,
    )

    md5_path = os.path.join(cfg.benchmark_folder, cfg.canonical_hvg_md5_file)
    if cfg.check_canonical_hvg_md5_file and os.path.exists(md5_path):
        expected = open(md5_path, "r", encoding="utf-8").read().strip().splitlines()[0].strip()
        if expected and expected not in (md5_master, md5_r_style):
            raise ValueError(
                f"Canonical HIE HVG MD5 mismatch. expected={expected} observed_master={md5_master} observed_r_style={md5_r_style}"
            )
        logging.info("Canonical HIE HVG MD5 verified: %s", expected)
    elif cfg.check_canonical_hvg_md5_file:
        logging.warning("Canonical HIE HVG MD5 file not found: %s", md5_path)

    meta_path = os.path.join(cfg.benchmark_folder, cfg.canonical_hvg_metadata_file)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            logging.info(
                "Canonical HIE HVG metadata: batch_key=%s label_key=%s n_top=%s",
                meta.get("batch_key"), meta.get("label_key"), meta.get("n_top_genes"),
            )
        except Exception as e:
            logging.warning("Could not parse canonical HIE HVG metadata file %s: %s", meta_path, e)

    out = a[:, matched_hvgs].copy()
    out.uns["canonical_hvg_file"] = hvg_path
    out.uns["canonical_hvg_md5"] = md5_master
    out.uns["canonical_hvg_md5_r_style"] = md5_r_style
    out.uns["canonical_hvg_n_genes"] = len(matched_hvgs)
    return out


def load_embedding_csv_align(adata: ad.AnnData, csv_path: str, key: str) -> None:
    """Load external embedding CSV and align rows to adata.obs_names.

    Attempt 1: direct alignment.
    Attempt 2: normalized-name alignment, useful for '-1' suffix or '.' vs '-'.
    """
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing embedding CSV: {csv_path}")

    df = pd.read_csv(csv_path, index_col=0)
    df.index = df.index.astype(str)

    # Attempt 1: direct match
    df1 = df.reindex(adata.obs_names.astype(str))
    match_rate_1 = float(df1.notna().all(axis=1).mean())
    if match_rate_1 >= 0.999:
        X = _nan_guard(df1.to_numpy(dtype=np.float32, copy=True), name=f"CSV[{key}]")
        adata.obsm[key] = X
        logging.info("Loaded embedding '%s' from %s shape=%s direct_match=%.3f", key, csv_path, X.shape, match_rate_1)
        return

    # Attempt 2: normalized match
    obs_norm = pd.Index([_normalize_obs_name(x) for x in adata.obs_names.astype(str)])
    df_norm = pd.Index([_normalize_obs_name(x) for x in df.index.astype(str)])
    norm_to_orig: Dict[str, str] = {}
    dup = set()
    for orig, norm in zip(df.index.astype(str), df_norm):
        if norm in norm_to_orig:
            dup.add(norm)
        else:
            norm_to_orig[norm] = orig
    for norm in dup:
        norm_to_orig.pop(norm, None)

    aligned_rows = [norm_to_orig.get(n, None) for n in obs_norm]
    ok = [r is not None for r in aligned_rows]
    match_rate_2 = float(sum(ok) / max(1, len(ok)))
    if match_rate_2 >= 0.999:
        df2 = df.loc[aligned_rows].copy()
        df2.index = adata.obs_names.astype(str)
        X = _nan_guard(df2.to_numpy(dtype=np.float32, copy=True), name=f"CSV[{key}]")
        adata.obsm[key] = X
        logging.info("Loaded embedding '%s' from %s shape=%s normalized_match=%.3f", key, csv_path, X.shape, match_rate_2)
        return

    missing_ex = [str(adata.obs_names[i]) for i, r in enumerate(aligned_rows) if r is None][:10]
    raise ValueError(
        f"Embedding CSV alignment failed for {csv_path}. "
        f"direct_match={match_rate_1*100:.2f}%, normalized_match={match_rate_2*100:.2f}%. "
        f"Missing examples={missing_ex}"
    )


def _try_load_exported_embedding(adata: ad.AnnData, embeddings_dir: str, method_tag: str, out_key: str) -> bool:
    if not embeddings_dir:
        return False

    # Try common names from sanitize_method_name(method)
    candidates = [
        os.path.join(embeddings_dir, f"X_{method_tag}.csv"),
        os.path.join(embeddings_dir, f"{method_tag}.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            load_embedding_csv_align(adata, path, key=out_key)
            return True
    return False


def write_preprocess_fingerprint(adata: ad.AnnData, cfg: HieFocusConfig) -> None:
    rows = [{
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "batch_key": str(cfg.batch_key),
        "label_key": str(cfg.label_key),
        "n_batches": int(adata.obs["batch"].nunique()),
        "n_classes": int(adata.obs["class"].nunique()),
        "canonical_hvg_file": str(adata.uns.get("canonical_hvg_file", "NA")),
        "canonical_hvg_md5": str(adata.uns.get("canonical_hvg_md5", "NA")),
        "canonical_hvg_md5_r_style": str(adata.uns.get("canonical_hvg_md5_r_style", "NA")),
        "var_md5_master": md5_master_style_trailing_newline([str(v) for v in adata.var_names.tolist()]),
        "obs_md5_master": md5_master_style_trailing_newline([str(o) for o in adata.obs_names.tolist()]),
        "X_pca_shape": str(np.asarray(adata.obsm["X_pca"]).shape) if "X_pca" in adata.obsm else "NA",
        "X_pca_var_pc1": float(np.var(np.asarray(adata.obsm["X_pca"])[:, 0])) if "X_pca" in adata.obsm else np.nan,
    }]
    out_csv = os.path.join(cfg.out_folder, "focus_preprocess_fingerprint.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logging.info("Saved: %s", out_csv)


# -----------------------------
# Loading + canonical preprocessing
# -----------------------------
def load_and_preprocess(cfg: HieFocusConfig) -> ad.AnnData:
    """Load HIE and apply canonical-HVG preprocessing."""
    if not os.path.exists(cfg.data_file):
        raise FileNotFoundError(f"Could not find HIE data file: {cfg.data_file}")

    a = sc.read(cfg.data_file)
    a.var_names_make_unique()

    if cfg.batch_key not in a.obs.columns:
        raise KeyError(f"Missing obs['{cfg.batch_key}'] in {cfg.data_file}. Available={list(a.obs.columns)[:40]}")
    if cfg.label_key not in a.obs.columns:
        raise KeyError(f"Missing obs['{cfg.label_key}'] in {cfg.data_file}. Available={list(a.obs.columns)[:40]}")

    a.obs["batch"] = pd.Categorical(a.obs[cfg.batch_key].astype(str))
    a.obs["class"] = pd.Categorical(a.obs[cfg.label_key].astype(str))

    used_frozen_hvgs = False
    if cfg.use_canonical_hvg_file:
        try:
            a = _subset_with_frozen_canonical_hvgs(a, cfg)
            used_frozen_hvgs = True
        except FileNotFoundError as e:
            if not cfg.fallback_to_dynamic_hvgs_if_missing:
                raise
            logging.warning("%s -- falling back to dynamic HVG selection.", e)

    counts_layer_name: Optional[str] = "counts" if "counts" in a.layers.keys() else None

    if cfg.do_normalize_log1p:
        if counts_layer_name is not None:
            logging.info("Preprocess: normalize_total/log1p on layer='counts' after canonical feature subsetting.")
            sc.pp.normalize_total(a, target_sum=1e4, layer=counts_layer_name)
            sc.pp.log1p(a, layer=counts_layer_name)
            a.X = a.layers[counts_layer_name]
        else:
            logging.info("Preprocess: normalize_total/log1p on a.X after canonical feature subsetting.")
            sc.pp.normalize_total(a, target_sum=1e4)
            sc.pp.log1p(a)

    if cfg.do_hvg and not used_frozen_hvgs:
        flavor = str(cfg.hvg_flavor_counts)
        layer_for_hvg = counts_layer_name if counts_layer_name is not None else None
        if flavor == "seurat_v3" and layer_for_hvg is None:
            logging.warning("HVG seurat_v3 requested but no counts layer -> fallback to %s.", cfg.hvg_flavor_fallback)
            flavor = str(cfg.hvg_flavor_fallback)
        sc.pp.highly_variable_genes(
            a,
            n_top_genes=int(cfg.hvg),
            flavor=flavor,
            subset=True,
            layer=layer_for_hvg,
        )
        if counts_layer_name is not None:
            a.X = a.layers[counts_layer_name]

    if cfg.do_scale:
        sc.pp.scale(a, max_value=float(cfg.max_scale_value))

    sc.tl.pca(a, n_comps=int(cfg.n_pcs), random_state=int(cfg.seed))
    a.obsm["X_pca"] = _nan_guard(np.asarray(a.obsm["X_pca"]), "X_pca")

    sc.pp.neighbors(a, use_rep="X_pca", n_neighbors=int(cfg.n_neighbors))
    sc.tl.umap(a, min_dist=float(cfg.umap_min_dist), random_state=int(cfg.seed))

    # External R embeddings: Seurat and Online iNMF
    seurat_csv = os.path.join(cfg.benchmark_folder, cfg.seurat_csv_name)
    online_csv = os.path.join(cfg.benchmark_folder, cfg.online_inmf_csv_name)
    try:
        load_embedding_csv_align(a, seurat_csv, key="X_seurat")
    except Exception as e:
        logging.warning("Seurat CSV not loaded: %s", e)

    try:
        load_embedding_csv_align(a, online_csv, key="X_online_inmf")
    except Exception as e:
        logging.warning("Online iNMF CSV not loaded: %s", e)

    # Exported embeddings from HIE master benchmark
    embeddings_dir = os.path.join(cfg.benchmark_folder, cfg.embeddings_dir_name)
    exported_map = [
        ("Scanorama", "X_scanorama"),
        ("Harmony", "X_harmony"),
        ("NMF", "X_nmf"),
        ("DiffusionMap", "X_diffmap"),
        ("PHATE", "X_phate"),
        ("GenoDR_Scanorama", "X_genodr_scanorama"),
        ("GenoDR_Harmony", "X_genodr_harmony"),
        ("GenoDR_Seurat", "X_genodr_seurat"),
        ("GenoDR_Online_iNMF", "X_genodr_online_inmf"),
        ("GenoDR_DiffusionMap", "X_genodr_diffmap"),
        ("GenoDR_PHATE", "X_genodr_phate"),
        ("Seurat", "X_seurat"),
        ("Online_iNMF", "X_online_inmf"),
    ]
    for tag, out_key in exported_map:
        try:
            loaded = _try_load_exported_embedding(a, embeddings_dir, method_tag=tag, out_key=out_key)
            if loaded:
                logging.info("Loaded exported embedding: %s -> obsm['%s']", tag, out_key)
        except Exception as e:
            logging.warning("Failed to load exported embedding %s: %s", tag, e)

    return a


# -----------------------------
# Embedding getters
# -----------------------------
def get_embeddings(adata: ad.AnnData) -> Dict[str, Tuple[str, np.ndarray]]:
    emb: Dict[str, Tuple[str, np.ndarray]] = {}
    emb["PCA"] = ("X_pca", _nan_guard(np.asarray(adata.obsm["X_pca"]), "X_pca"))

    key_to_method = [
        ("X_scanorama", "Scanorama"),
        ("X_harmony", "Harmony"),
        ("X_pca", "BBKNN"),
        ("X_seurat", "Seurat"),
        ("X_online_inmf", "Online_iNMF"),
        ("X_nmf", "NMF"),
        ("X_diffmap", "DiffusionMap"),
        ("X_phate", "PHATE"),
        ("X_genodr_scanorama", "GenoDR(Scanorama)"),
        ("X_genodr_harmony", "GenoDR(Harmony)"),
        ("X_genodr_seurat", "GenoDR(Seurat)"),
        ("X_genodr_online_inmf", "GenoDR(Online_iNMF)"),
        ("X_genodr_diffmap", "GenoDR(DiffusionMap)"),
        ("X_genodr_phate", "GenoDR(PHATE)"),
    ]
    for k, method in key_to_method:
        if k in adata.obsm:
            emb[method] = (k, _nan_guard(np.asarray(adata.obsm[k]), k))
    return emb


# -----------------------------
# Focus test computations
# -----------------------------
def topology_preservation_centroids(
    X_ref: np.ndarray,
    X_new: np.ndarray,
    labels: np.ndarray,
    max_classes: int,
    seed: int,
) -> Tuple[float, float, int]:
    labels = _as_str(labels)
    classes = np.unique(labels)
    if len(classes) < 3:
        return np.nan, np.nan, int(len(classes))

    if len(classes) > int(max_classes):
        rng = np.random.default_rng(int(seed))
        classes = rng.choice(classes, size=int(max_classes), replace=False)

    cent_ref = []
    cent_new = []
    for c in classes:
        mask = labels == c
        cent_ref.append(np.mean(X_ref[mask], axis=0))
        cent_new.append(np.mean(X_new[mask], axis=0))

    D_ref = pairwise_distances(np.vstack(cent_ref))
    D_new = pairwise_distances(np.vstack(cent_new))

    a = D_ref.flatten()
    b = D_new.flatten()

    sp = spearmanr(a, b, nan_policy="omit").correlation
    pr = pearsonr(a, b)[0]
    return float(sp), float(pr), int(len(classes))


def knn_stability_and_purity_delta(X_before: np.ndarray, X_after: np.ndarray, labels: np.ndarray, k: int) -> Tuple[float, float]:
    labels = _as_str(labels)
    k_eff = int(k) + 1
    nn1 = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(X_before)
    nn2 = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(X_after)

    idx1 = nn1.kneighbors(return_distance=False)[:, 1:]
    idx2 = nn2.kneighbors(return_distance=False)[:, 1:]

    stability = []
    purity_delta = []
    for i in range(idx1.shape[0]):
        set1, set2 = set(idx1[i]), set(idx2[i])
        inter = len(set1 & set2)
        union = len(set1 | set2)
        stability.append(inter / union if union > 0 else 0.0)

        purity_before = float(np.mean(labels[list(set1)] == labels[i])) if len(set1) else 0.0
        purity_after = float(np.mean(labels[list(set2)] == labels[i])) if len(set2) else 0.0
        purity_delta.append(purity_after - purity_before)

    return float(np.mean(stability)), float(np.mean(purity_delta))


def compactness_per_class(X: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    labels = _as_str(labels)
    out_rows = []
    for c in np.unique(labels):
        mask = labels == c
        Xc = X[mask]
        n = Xc.shape[0]
        if n < 3:
            continue
        D = pairwise_distances(Xc)
        tri = D[np.triu_indices(n, k=1)]
        out_rows.append({
            "Class": str(c),
            "Compactness_MeanPairwiseDist": float(np.mean(tri)),
            "Compactness_MedianPairwiseDist": float(np.median(tri)),
            "NCells": int(n),
        })
    return pd.DataFrame(out_rows)


# -----------------------------
# Plot helpers
# -----------------------------
def _maybe_plot_topology(df: pd.DataFrame, out_png: str) -> None:
    import matplotlib.pyplot as plt
    d = df.dropna(subset=["TopoPreserve_Spearman_Centroids"]).copy()
    if d.empty:
        return
    d = d.sort_values("TopoPreserve_Spearman_Centroids", ascending=False)
    plt.figure(figsize=(10, 4))
    plt.bar(d["Method"].astype(str), d["TopoPreserve_Spearman_Centroids"].values)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Spearman (centroid distances)")
    plt.title("Topology preservation (centroid-distance correlation)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close("all")


def _maybe_plot_knn(df: pd.DataFrame, out_png: str) -> None:
    import matplotlib.pyplot as plt
    if df.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.scatter(df["Stability_Mean"].values, df["Delta_Purity_Mean"].values)
    for _, r in df.iterrows():
        plt.text(r["Stability_Mean"], r["Delta_Purity_Mean"], str(r["Pair"]), fontsize=8)
    plt.xlabel("kNN stability (Jaccard)")
    plt.ylabel("Delta purity (after - before)")
    plt.title("Neighborhood stability vs purity gain")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close("all")


def _maybe_plot_compactness(df: pd.DataFrame, out_png: str) -> None:
    import matplotlib.pyplot as plt
    if df.empty:
        return
    d = df.groupby("Method", as_index=False)["Compactness_MedianPairwiseDist"].median()
    d = d.sort_values("Compactness_MedianPairwiseDist", ascending=True)
    plt.figure(figsize=(10, 4))
    plt.bar(d["Method"].astype(str), d["Compactness_MedianPairwiseDist"].values)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Median within-class distance")
    plt.title("Within-class compactness (lower = tighter)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close("all")


def _maybe_plot_silhouette(df: pd.DataFrame, out_png: str) -> None:
    import matplotlib.pyplot as plt
    if df.empty:
        return
    methods = list(df["Method"].unique())
    data = [df.loc[df["Method"] == m, "Silhouette"].values for m in methods]
    plt.figure(figsize=(10, 4))
    plt.boxplot(data, tick_labels=methods, showfliers=False)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Silhouette width (per cell, class labels)")
    plt.title("Silhouette distribution by method")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close("all")


# -----------------------------
# Main
# -----------------------------
def main(cfg: HieFocusConfig = CFG) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    set_all_seeds(int(cfg.seed))
    ensure_out_dir(cfg.out_folder)

    logging.info("Loading + preprocessing HIE using frozen canonical HVGs...")
    adata = load_and_preprocess(cfg)
    write_preprocess_fingerprint(adata, cfg)

    labels = adata.obs["class"].astype(str).to_numpy()
    X_ref = _nan_guard(np.asarray(adata.obsm["X_pca"]), "X_pca_ref")

    if len(np.unique(labels)) < 2:
        raise RuntimeError("Focus tests require at least 2 classes in obs['class'].")

    logging.info("Collecting embeddings...")
    emb = get_embeddings(adata)
    logging.info("Embeddings available: %s", list(emb.keys()))

    if len(emb) <= 2:
        logging.warning(
            "Only PCA/BBKNN available. This usually means exported embeddings or R CSVs are missing/misaligned. "
            "Check benchmark_folder=%s and embeddings_dir=%s",
            cfg.benchmark_folder,
            os.path.join(cfg.benchmark_folder, cfg.embeddings_dir_name),
        )

    # 1) Topology preservation
    topo_rows = []
    for method, (embed_key, X) in emb.items():
        sp, pr, ncls = topology_preservation_centroids(
            X_ref=X_ref,
            X_new=_nan_guard(np.asarray(X), f"topo_in[{embed_key}]"),
            labels=labels,
            max_classes=int(cfg.max_classes_for_topology),
            seed=int(cfg.seed),
        )
        topo_rows.append({
            "Method": method,
            "EmbedKey": embed_key,
            "TopoPreserve_Spearman_Centroids": sp,
            "TopoPreserve_Pearson_Centroids": pr,
            "NClasses_Used": int(ncls),
        })
    df_topo = pd.DataFrame(topo_rows)
    df_topo.to_csv(os.path.join(cfg.out_folder, "focus_topology_preservation.csv"), index=False)

    # 2) kNN stability + purity for refinement pairs
    pairs = []
    for base in ["Scanorama", "Harmony", "Seurat", "Online_iNMF", "DiffusionMap", "PHATE"]:
        refined = f"GenoDR({base})"
        if base in emb and refined in emb:
            pairs.append((base, refined))

    knn_rows = []
    for base, refined in pairs:
        _, Xb = emb[base]
        _, Xa = emb[refined]
        stab, dp = knn_stability_and_purity_delta(
            X_before=_nan_guard(np.asarray(Xb), f"knn_before[{base}]"),
            X_after=_nan_guard(np.asarray(Xa), f"knn_after[{refined}]"),
            labels=labels,
            k=int(cfg.knn_k),
        )
        knn_rows.append({
            "Pair": f"{base} -> {refined}",
            "Group": "Refinement",
            "Stability_Mean": float(stab),
            "Delta_Purity_Mean": float(dp),
            "NCells": int(adata.n_obs),
        })

    df_knn = pd.DataFrame(knn_rows)
    df_knn.to_csv(os.path.join(cfg.out_folder, "focus_knn_stability_purity_summary.csv"), index=False)
    df_knn.to_csv(os.path.join(cfg.out_folder, "focus_hie_pairwise_refinement.csv"), index=False)

    # 3) Compactness per class
    comp_parts = []
    for method, (embed_key, X) in emb.items():
        dfc = compactness_per_class(_nan_guard(np.asarray(X), f"comp_in[{embed_key}]"), labels)
        if dfc.empty:
            continue
        dfc.insert(0, "EmbedKey", embed_key)
        dfc.insert(0, "Method", method)
        comp_parts.append(dfc)
    df_comp = pd.concat(comp_parts, ignore_index=True) if comp_parts else pd.DataFrame()
    df_comp.to_csv(os.path.join(cfg.out_folder, "focus_compactness_per_class.csv"), index=False)

    # 4) Silhouette samples, using class labels for biological compactness validation
    sil_rows = []
    cell_ids = adata.obs_names.values.astype(str)
    for method, (embed_key, X) in emb.items():
        Xv = _nan_guard(np.asarray(X), f"sil_in[{embed_key}]")
        try:
            s = silhouette_samples(Xv, labels, metric="euclidean")
        except Exception as e:
            logging.warning("Silhouette failed for %s (%s); skipping.", method, e)
            continue
        for cid, lab, val in zip(cell_ids, labels, s):
            sil_rows.append({
                "Method": method,
                "EmbedKey": embed_key,
                "CellID": str(cid),
                "Class": str(lab),
                "Silhouette": float(val),
            })
    df_sil = pd.DataFrame(sil_rows)
    df_sil.to_csv(os.path.join(cfg.out_folder, "focus_silhouette_samples.csv"), index=False)

    # 5) Simple method-level metrics summary for focus tests
    metrics_rows = []
    nbatches = int(len(np.unique(adata.obs["batch"].astype(str).to_numpy())))
    nclasses = int(len(np.unique(labels)))
    for method, (embed_key, X) in emb.items():
        s_mean = np.nan
        try:
            s_mean = float(np.mean(silhouette_samples(_nan_guard(np.asarray(X), "sil_mean_in"), labels)))
        except Exception:
            pass
        metrics_rows.append({
            "Method": method,
            "EmbedKey": embed_key,
            "Silhouette_Mean_ClassLabels": float(s_mean) if np.isfinite(s_mean) else np.nan,
            "NCells": int(adata.n_obs),
            "NClasses": nclasses,
            "NBatches": nbatches,
        })
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(os.path.join(cfg.out_folder, "focus_hie_metrics.csv"), index=False)

    _maybe_plot_topology(df_topo, os.path.join(cfg.out_folder, "sup_topology_preservation.png"))
    _maybe_plot_knn(df_knn, os.path.join(cfg.out_folder, "sup_knn_stability_delta_purity.png"))
    _maybe_plot_compactness(df_comp, os.path.join(cfg.out_folder, "sup_compactness_per_class.png"))
    _maybe_plot_silhouette(df_sil, os.path.join(cfg.out_folder, "sup_silhouette_violin.png"))

    logging.info("Done. Outputs written to: %s", cfg.out_folder)


if __name__ == "__main__":
    main()
