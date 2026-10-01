# Purpose: Stress-testing and robustness analysis for single-cell integration methods using the
#          5-dataset Pancreas benchmark.
# Author: Ariana Rahman (Arizona State University)

"""
FILE: bench_pancreas_multi_robustness_canonical_strict_v15.py
-------------------------------------------------------------------------------
Date: April 2026 (v16 GenoRefine core-method robustness)
-------------------------------------------------------------------------------

PURPOSE
    Stress-testing and robustness analysis for single-cell integration methods
    using the 5-dataset Pancreas benchmark.

    v16 is aligned to bench_pancreas_multi_master_v13.py:
      - Human-readable batch names: Baron, Muraro, Segerstolpe, Wang, Xin
      - obs_names: Cell-<i>-Batch-<BatchName>
      - Backward-compatible alignment for R-exported embeddings (legacy rownames)
      - Restricts robustness evaluation to Scanorama, Harmony, Seurat, Online iNMF,
        and their corresponding GenoRefine-refined embeddings.

EXPERIMENTS
    1) Dataset Imbalance:
       Downsample one batch to {100%, 50%, 25%, 10%} using the frozen canonical HVG file from make_pancreas_canonical_hvg.py.

    2) Non-Overlapping Cell-Type Distributions (Negative Control):
       Remove a specific cell type from one batch entirely.

v6 features retained
    - Cluster purity + cluster entropy summaries (per cluster and weighted mean).
    - Outputs:
        robustness_imbalance.csv
        robustness_nonoverlap.csv
        purity_entropy_nonoverlap_[Method].csv
        purity_entropy_nonoverlap_summary.csv
        purity_entropy_imbalance_summary.csv

v16 additions
    - Uses Benchmark_Out/pancreas_hvg_canonical.txt instead of recomputing HVGs.
    - Applies clean canonical-HVG workflow: load raw -> subset canonical HVGs -> normalize/log -> scale/PCA.
    - Uses perturbation-specific Seurat and Online iNMF embeddings rerun in R for each imbalance/nonoverlap condition.
    - Reads/writes Python cell manifests so R and Python use exactly the same perturbed cells.
    - Keeps nonoverlap/imbalance UMAPs and sanitized filenames.
-------------------------------------------------------------------------------
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")  # use non-interactive backend on Windows/headless runs

import time
import logging
import hashlib
import gc
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import scipy.io as sio

import matplotlib
matplotlib.use("Agg", force=True)

import anndata as ad
import scanpy as sc
import scanorama

from sklearn.metrics import adjusted_rand_score, rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

# Optional deps
try:
    import bbknn  # type: ignore
except Exception:
    bbknn = None

try:
    import harmonypy as hm  # type: ignore
except Exception:
    hm = None

try:
    from sklearn.decomposition import NMF  # type: ignore
except Exception:
    NMF = None

try:
    import genomap.genoDR as gp  # type: ignore
except Exception:
    gp = None

try:
    import phate  # type: ignore
except Exception:
    phate = None


# -----------------------------
# Config
# -----------------------------
@dataclass
class BenchConfig:
    data_folder: str = "./Dataset"
    out_folder: str = "./Benchmark_Out"
    seed: int = 0

    data_files: Tuple[str, ...] = (
        "dataBaronX.mat",
        "dataMuraroX.mat",
        "dataScapleX.mat",
        "dataWangX.mat",
        "dataXinX.mat",
    )
    class_label_file: str = "classLabel.mat"

    # Batch token -> display name (match master_v9)
    batch_token_to_name: Dict[str, str] = None  # filled in __post_init__

    # Preprocess (canonical alignment)
    hvg: int = 2000
    n_pcs: int = 50
    n_neighbors: int = 15
    hvg_flavor: str = "seurat_v3"

    # Frozen canonical HVG file produced by make_pancreas_canonical_hvg.py.
    # This mirrors bench_pancreas_multi_master_v13.py.
    use_canonical_hvg_file: bool = True
    canonical_hvg_file: str = "pancreas_hvg_canonical.txt"
    require_exact_canonical_hvg_match: bool = True

    # Neighbor policy (match master_v13 behavior)
    neighbor_metric: str = "cosine"
    neighbor_n_dims: int = 30

    # Clustering (canonical alignment)
    leiden_target_n_clusters: bool = True
    leiden_resolution_default: float = 0.5
    leiden_res_grid: Tuple[float, ...] = tuple(np.round(np.linspace(0.2, 1.6, 15), 2))

    # Metrics
    lisi_k: int = 90

    # NMF (optional baseline)
    nmf_components: int = 30
    nmf_max_iter: int = 500

    # DiffusionMap params
    diffmap_n_comps: int = 30
    diffmap_embed_key: str = "X_diffmap"

    # PHATE params
    phate_n_components: int = 30
    phate_knn: int = 15
    phate_decay: int = 40
    phate_t: str = "auto"
    phate_embed_key: str = "X_phate"

    # GenoRefine backend parameters
    genorefine_dim: int = 32
    genorefine_col: int = 33
    genorefine_row: int = 33

    # CSV exports from R (v9 supports both new and legacy rownames)
    seurat_csv_name: str = "X_seurat_pancreas.csv"
    online_inmf_csv_name: str = "X_online_inmf_pancreas.csv"

    # Strict robustness R reruns for Seurat / Online iNMF
    # The R scripts write perturbation-specific embeddings here.
    r_robustness_folder: str = "Robustness_R_Embeddings"
    require_strict_r_rerun_embeddings: bool = True
    generate_cell_manifests: bool = True

    # Robustness settings
    imbalance_batch_to_downsample: str = "Baron"
    imbalance_fracs: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.10)

    # Non-overlap negative control
    nonoverlap_batch: str = "Baron"
    nonoverlap_celltype: str = "beta"

    # v6: purity/entropy outputs
    save_per_method_purity_entropy_tables: bool = True
    entropy_eps: float = 1e-12

    # v7: optional visuals
    save_umap_nonoverlap: bool = False
    save_umap_imbalance_for_fracs: Tuple[float, ...] = (1.0, 0.10)

    def __post_init__(self):
        if self.batch_token_to_name is None:
            self.batch_token_to_name = {
                "Baron": "Baron",
                "Muraro": "Muraro",
                "Scaple": "Segerstolpe",
                "Segerstolpe": "Segerstolpe",
                "Wang": "Wang",
                "Xin": "Xin",
            }


CFG = BenchConfig()


# -----------------------------
# Utilities
# -----------------------------
def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def md5_of_list(items: List[str]) -> str:
    h = hashlib.md5()
    for s in items:
        h.update(str(s).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def sanitize_method_name(name: str) -> str:
    s = name.replace(" ", "_")
    s = s.replace("(", "_").replace(")", "")
    s = s.replace("/", "_").replace("__", "_")
    return s


def _nan_guard(X: np.ndarray, name: str) -> np.ndarray:
    X = np.asarray(X)
    if not np.isfinite(X).all():
        logging.warning(f"{name}: NaN/Inf found; replacing with 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _rep_for_neighbors(adata: ad.AnnData, rep_key: Optional[str], n_dims: int) -> Optional[str]:
    """Return an obsm key safe for neighbors, slicing to n_dims when needed."""
    if rep_key is None:
        return None
    if rep_key not in adata.obsm:
        raise KeyError(f"Missing embedding '{rep_key}' in adata.obsm")
    X = np.asarray(adata.obsm[rep_key])
    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite values found in embedding '{rep_key}'")
    if n_dims is None or int(n_dims) <= 0 or X.shape[1] <= int(n_dims):
        return rep_key
    key2 = f"{rep_key}__nd{int(n_dims)}"
    if key2 not in adata.obsm or adata.obsm[key2].shape[0] != X.shape[0]:
        adata.obsm[key2] = X[:, : int(n_dims)].copy()
    return key2


def log_preprocess_fingerprint(adata: ad.AnnData, cfg: BenchConfig, tag: str) -> None:
    var_names = [str(v) for v in adata.var_names.tolist()]
    obs_names = [str(o) for o in adata.obs_names.tolist()]

    msg = [
        f"[{tag}] fingerprint: n_obs={adata.n_obs}, n_vars={adata.n_vars}",
        f"[{tag}] HVG source=frozen_file:{cfg.canonical_hvg_file}, flavor={cfg.hvg_flavor}, HVG n={adata.n_vars}",
        f"[{tag}] batches={adata.obs['batch'].nunique()}, classes={adata.obs['class'].nunique()}",
        f"[{tag}] var_md5={md5_of_list(var_names)}",
        f"[{tag}] obs_md5={md5_of_list(obs_names)}",
    ]

    if "X_pca" in adata.obsm:
        x0 = np.asarray(adata.obsm["X_pca"])
        msg.append(f"[{tag}] X_pca shape={x0.shape}, var(PC1)={float(np.var(x0[:, 0])):.6f}")

    logging.info("\n".join(msg))


def compute_lisi_python(X: np.ndarray, labels: np.ndarray, k: int = 90) -> float:
    X = np.asarray(X)
    labels = np.asarray(labels)
    n = X.shape[0]
    k = min(int(k), n - 1) if n > 1 else 1

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    _, idx = nn.kneighbors(X)

    uniq, enc = np.unique(labels, return_inverse=True)
    n_classes = len(uniq)

    lisi_scores: List[float] = []
    for i in range(n):
        neigh = enc[idx[i]]
        counts = np.bincount(neigh, minlength=n_classes)
        probs = counts / k
        simpson = float(np.sum(probs**2))
        lisi = 1.0 / simpson if simpson > 0 else 1.0
        lisi_scores.append(lisi)

    return float(np.mean(lisi_scores))


def tune_leiden_resolution_to_target(
    adata: ad.AnnData,
    use_rep: Optional[str],
    target_k: int,
    n_neighbors: int,
    res_grid: List[float],
    key_added: str,
    random_state: int,
) -> float:
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)

    best_res = float(res_grid[0])
    best_diff = 10**9

    for r in res_grid:
        sc.tl.leiden(adata, resolution=float(r), key_added=key_added, random_state=random_state)
        k = int(adata.obs[key_added].nunique())
        diff = abs(k - int(target_k))
        if diff < best_diff:
            best_diff = diff
            best_res = float(r)
        if diff == 0:
            break

    sc.tl.leiden(adata, resolution=best_res, key_added=key_added, random_state=random_state)
    return best_res


def tune_leiden_resolution_existing_graph(
    adata: ad.AnnData,
    target_k: int,
    res_grid: List[float],
    key_added: str,
    random_state: int,
) -> float:
    best_res = float(res_grid[0])
    best_diff = 10**9

    for r in res_grid:
        sc.tl.leiden(adata, resolution=float(r), key_added=key_added, random_state=random_state)
        k = int(adata.obs[key_added].nunique())
        diff = abs(k - int(target_k))
        if diff < best_diff:
            best_diff = diff
            best_res = float(r)
        if diff == 0:
            break

    sc.tl.leiden(adata, resolution=best_res, key_added=key_added, random_state=random_state)
    return best_res


def cluster_on_rep(
    adata: ad.AnnData,
    rep_key: Optional[str],
    cfg: BenchConfig,
    cluster_key: str = "leiden",
) -> float:
    n_classes = int(adata.obs["class"].nunique())
    res_grid = list(cfg.leiden_res_grid)
    safe_rep = _rep_for_neighbors(adata, rep_key, int(cfg.neighbor_n_dims))

    if cfg.leiden_target_n_clusters:
        used_res = tune_leiden_resolution_to_target(
            adata=adata,
            use_rep=safe_rep,
            target_k=n_classes,
            n_neighbors=cfg.n_neighbors,
            res_grid=res_grid,
            key_added=cluster_key,
            random_state=cfg.seed,
        )
    else:
        sc.pp.neighbors(adata, use_rep=safe_rep, n_neighbors=cfg.n_neighbors, metric=str(cfg.neighbor_metric))
        sc.tl.leiden(adata, resolution=cfg.leiden_resolution_default, key_added=cluster_key, random_state=cfg.seed)
        used_res = cfg.leiden_resolution_default

    return float(used_res)


def compute_metrics(
    adata: ad.AnnData,
    embed_key_for_metrics: str,
    cfg: BenchConfig,
    cluster_key: str = "leiden",
) -> Dict[str, float]:
    y_true = adata.obs["class"].astype(str).to_numpy()
    y_pred = adata.obs[cluster_key].astype(str).to_numpy()
    X = _nan_guard(np.asarray(adata.obsm[embed_key_for_metrics]), f"metrics_in[{embed_key_for_metrics}]")

    ari = adjusted_rand_score(y_true, y_pred)
    ri = rand_score(y_true, y_pred)

    try:
        sil = float(silhouette_score(X, y_pred))
    except Exception:
        sil = float("nan")

    ilisi = compute_lisi_python(X, adata.obs["batch"].astype(str).to_numpy(), k=cfg.lisi_k)
    return {"ARI": float(ari), "RI": float(ri), "Silhouette": float(sil), "iLISI": float(ilisi)}


def confusion_matrix_table(
    adata: ad.AnnData,
    class_key: str = "class",
    cluster_key: str = "leiden",
    normalize: str = "index",
) -> pd.DataFrame:
    return pd.crosstab(
        adata.obs[class_key].astype(str),
        adata.obs[cluster_key].astype(str),
        normalize=normalize,
    )


def save_umap_png(adata: ad.AnnData, color_key: str, out_png: str, title: str) -> None:
    """Save UMAP using a non-interactive matplotlib backend.

    This avoids Tkinter/Tcl crashes on Windows batch runs (e.g.,
    ``RuntimeError: main thread is not in main loop`` /
    ``Tcl_AsyncDelete: async handler deleted by the wrong thread``).
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    sc.pl.umap(adata, color=color_key, show=False, title=title)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close("all")
    gc.collect()


# -----------------------------
# v6: Cluster purity + entropy
# -----------------------------
def cluster_purity_entropy(
    adata: ad.AnnData,
    class_key: str = "class",
    cluster_key: str = "leiden",
    eps: float = 1e-12,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    y = adata.obs[class_key].astype(str)
    z = adata.obs[cluster_key].astype(str)

    tab = pd.crosstab(z, y)  # rows=cluster, cols=class; counts
    if tab.empty:
        summary = {"Purity_MeanWeighted": float("nan"), "Entropy_MeanWeighted": float("nan")}
        return summary, pd.DataFrame(columns=["Cluster", "N", "DominantClass", "Purity", "Entropy"])

    counts = tab.to_numpy(dtype=float)
    n_per_cluster = counts.sum(axis=1)
    n_total = float(n_per_cluster.sum()) if n_per_cluster.size else 0.0

    probs = counts / np.maximum(n_per_cluster[:, None], eps)

    purity = probs.max(axis=1)
    dom_idx = probs.argmax(axis=1)
    dom_class = tab.columns.to_numpy()[dom_idx]

    probs_clip = np.clip(probs, eps, 1.0)
    entropy = -np.sum(probs_clip * np.log2(probs_clip), axis=1)

    df = pd.DataFrame({
        "Cluster": tab.index.astype(str),
        "N": n_per_cluster.astype(int),
        "DominantClass": dom_class.astype(str),
        "Purity": purity.astype(float),
        "Entropy": entropy.astype(float),
    }).sort_values("N", ascending=False)

    if n_total <= 0:
        summary = {"Purity_MeanWeighted": float("nan"), "Entropy_MeanWeighted": float("nan")}
    else:
        w = n_per_cluster / n_total
        summary = {
            "Purity_MeanWeighted": float(np.sum(w * purity)),
            "Entropy_MeanWeighted": float(np.sum(w * entropy)),
        }

    return summary, df


# -----------------------------
# Data loading (match master_v9 naming)
# -----------------------------
def _find_mat_data_key(d: Dict) -> str:
    keys = [k for k in d.keys() if not k.startswith("__")]
    starts = [k for k in keys if k.lower().startswith("data")]
    if starts:
        return starts[0]
    contains = [k for k in keys if "data" in k.lower()]
    if contains:
        return contains[0]
    return keys[0]


def _extract_batch_token_from_filename(fn: str) -> str:
    base = os.path.splitext(os.path.basename(fn))[0]  # dataBaronX
    if base.lower().startswith("data") and base.endswith("X"):
        return base[4:-1]
    if base.lower().startswith("data"):
        return base[4:]
    return base


def load_base_data(cfg: BenchConfig) -> ad.AnnData:
    """Load raw pancreas batches without normalization.

    This intentionally mirrors bench_pancreas_multi_master_v13.py:
      raw .mat matrices -> canonical obs names -> class labels -> batch labels.
    Canonical HVG subsetting and normalize/log are performed later in
    preprocess_with_frozen_hvgs(), after perturbations are applied.
    """
    adatas = []
    batch_names_in_order: List[str] = []

    for i, fn in enumerate(cfg.data_files):
        path = os.path.join(cfg.data_folder, fn)
        d = sio.loadmat(path)
        key = _find_mat_data_key(d)
        X = np.asarray(d[key])
        a = ad.AnnData(X)

        token = _extract_batch_token_from_filename(fn)
        batch_name = cfg.batch_token_to_name.get(token, token)
        batch_names_in_order.append(batch_name)

        a.obs_names = [f"Cell-{j+1}-Batch-{batch_name}" for j in range(a.n_obs)]
        a.obs["batch"] = batch_name
        a.obs["batch_id"] = int(i + 1)  # legacy 1..5 for CSV fallback
        adatas.append(a)

    adata = ad.concat(adatas, join="outer")
    adata.obs_names_make_unique()
    adata.var_names = pd.Index(adata.var_names.astype(str))

    cls_path = os.path.join(cfg.data_folder, cfg.class_label_file)
    cls = sio.loadmat(cls_path)["classLabel"].squeeze()

    class_name_map = {
        1: "MHC class II", 2: "acinar", 3: "ductal", 4: "gamma", 5: "macrophage",
        6: "alpha", 7: "beta", 8: "endothelial", 9: "epsilon", 10: "mast",
        11: "mesenchymal", 12: "stellate", 13: "delta", 14: "schwann"
    }
    mapped = np.array([class_name_map.get(int(x), f"Unknown-{int(x)}") for x in cls])

    if len(mapped) != adata.n_obs:
        raise ValueError(f"classLabel length {len(mapped)} != n_obs {adata.n_obs}")

    adata.obs["class"] = pd.Categorical(mapped.astype(str))
    adata.obs["batch"] = pd.Categorical(
        adata.obs["batch"].astype(str),
        categories=batch_names_in_order,
        ordered=True,
    )
    adata.obs["batch_id"] = adata.obs["batch_id"].astype(int)
    return adata


# -----------------------------
# Frozen canonical HVG preprocessing
# -----------------------------
def _read_canonical_hvg_file(txt_path: str) -> List[str]:
    if not os.path.exists(txt_path):
        raise FileNotFoundError(txt_path)
    ser = pd.read_csv(txt_path, header=None).iloc[:, 0].astype(str)
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
        raise ValueError(f"Canonical HVG file is empty: {txt_path}")
    return hvgs


def load_canonical_hvgs(adata_full_raw: ad.AnnData, cfg: BenchConfig) -> List[str]:
    """Read the canonical HVG output from make_pancreas_canonical_hvg.py."""
    txt_path = os.path.join(cfg.out_folder, cfg.canonical_hvg_file)
    requested = _read_canonical_hvg_file(txt_path)
    present = set(adata_full_raw.var_names.astype(str).tolist())
    matched = [g for g in requested if g in present]
    missing_count = int(len(requested) - len(matched))

    if cfg.require_exact_canonical_hvg_match and len(matched) != len(requested):
        raise ValueError(
            f"Canonical HVG file match is incomplete. File={txt_path} | "
            f"requested={len(requested)} matched={len(matched)} missing={missing_count}"
        )
    if len(matched) < 100:
        raise ValueError(f"Too few canonical HVGs matched: {len(matched)}")

    logging.info(
        "Using frozen canonical HVG file: %s | requested=%d matched=%d missing=%d md5=%s",
        txt_path, len(requested), len(matched), missing_count, md5_of_list(matched)
    )
    return matched


def preprocess_with_frozen_hvgs(adata_in: ad.AnnData, hvgs: List[str], cfg: BenchConfig) -> ad.AnnData:
    """Clean canonical-HVG workflow: subset raw features first, then normalize/log.

    This matches bench_pancreas_multi_master_v13.py and avoids recomputing HVGs
    inside robustness perturbations.
    """
    present = set(adata_in.var_names.astype(str).tolist())
    matched = [g for g in hvgs if g in present]
    if cfg.require_exact_canonical_hvg_match and len(matched) != len(hvgs):
        raise ValueError(
            f"Perturbed dataset is missing canonical HVGs: requested={len(hvgs)} matched={len(matched)}"
        )

    a = adata_in[:, matched].copy()
    logging.info(
        "Applying clean canonical-HVG workflow: subset raw feature matrix first, then normalize/log. n_vars=%d md5=%s",
        a.n_vars, md5_of_list([str(x) for x in a.var_names.tolist()])
    )
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=cfg.n_pcs, random_state=cfg.seed)
    return a


# -----------------------------
# CSV loader (v9-style: new OR legacy names)
# -----------------------------
def _build_legacy_obs_names_from_current(adata_in: ad.AnnData) -> pd.Index:
    # Current: Cell-<i>-Batch-<Name> ; Legacy: Cell-<i>-Batch-<1..5>
    cell_nums = []
    for name in adata_in.obs_names.astype(str):
        try:
            token = name.split("-Batch-")[0]  # Cell-123
            n = int(token.replace("Cell-", ""))
        except Exception:
            n = None
        cell_nums.append(n)

    batch_ids = adata_in.obs["batch_id"].astype(int).to_numpy()

    legacy = []
    for idx, (cn, bid) in enumerate(zip(cell_nums, batch_ids)):
        if cn is None:
            cn = idx + 1
        legacy.append(f"Cell-{cn}-Batch-{bid}")
    return pd.Index(legacy)


def _load_embedding_csv(adata_in: ad.AnnData, csv_path: str, obsm_key: str) -> Tuple[ad.AnnData, str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, index_col=0)

    # Attempt 1: direct v9 obs_names
    df1 = df.reindex(adata_in.obs_names)
    match_rate_1 = float(df1.notna().all(axis=1).mean())
    if match_rate_1 >= 0.999:
        adata = adata_in.copy()
        adata.obsm[obsm_key] = _nan_guard(df1.to_numpy(), f"CSV[{obsm_key}]")
        return adata, obsm_key

    # Attempt 2: legacy numeric batch id names
    legacy_names = _build_legacy_obs_names_from_current(adata_in)
    df2 = df.reindex(legacy_names)
    match_rate_2 = float(df2.notna().all(axis=1).mean())
    if match_rate_2 >= 0.999:
        adata = adata_in.copy()
        adata.obsm[obsm_key] = _nan_guard(df2.to_numpy(), f"CSV[{obsm_key}]")
        return adata, obsm_key

    missing1 = df1.index[df1.isna().any(axis=1)].tolist()[:10]
    missing2 = df2.index[df2.isna().any(axis=1)].tolist()[:10]
    raise ValueError(
        f"Embedding CSV alignment failed.\n"
        f"CSV: {csv_path}\n"
        f"Attempt1 (v9 obs_names) match={match_rate_1*100:.2f}% missing ex={missing1}\n"
        f"Attempt2 (legacy obs_names) match={match_rate_2*100:.2f}% missing ex={missing2}\n"
        f"Fix: ensure CSV rownames match either:\n"
        f"  v9:     Cell-<i>-Batch-<Baron|Muraro|Segerstolpe|Wang|Xin>\n"
        f"  legacy: Cell-<i>-Batch-<1..5>\n"
    )


def _frac_tag(frac: float) -> str:
    return f"frac{float(frac):.2f}".replace(".", "_")


def _context_tag(cfg: BenchConfig, test_tag: str, frac: Optional[float] = None) -> str:
    if test_tag == "imbalance":
        if frac is None:
            raise ValueError("frac is required for imbalance context")
        return f"imbalance_{cfg.imbalance_batch_to_downsample}_{_frac_tag(float(frac))}"
    if test_tag == "nonoverlap":
        return f"nonoverlap_{cfg.nonoverlap_batch}_{str(cfg.nonoverlap_celltype).replace(' ', '_')}"
    return "full"


def _r_robustness_dir(cfg: BenchConfig) -> str:
    return os.path.join(cfg.out_folder, cfg.r_robustness_folder)


def _strict_r_embedding_csv_path(cfg: BenchConfig, method: str, test_tag: str, frac: Optional[float]) -> str:
    prefix_map = {
        "Seurat": "X_seurat_pancreas",
        "Online_iNMF": "X_online_inmf_pancreas",
    }
    if method not in prefix_map:
        raise ValueError(f"No strict R embedding prefix for method={method}")
    ctx = _context_tag(cfg, test_tag, frac)
    return os.path.join(_r_robustness_dir(cfg), f"{prefix_map[method]}_{ctx}.csv")


def _cell_manifest_path(cfg: BenchConfig, test_tag: str, frac: Optional[float]) -> str:
    ctx = _context_tag(cfg, test_tag, frac)
    return os.path.join(_r_robustness_dir(cfg), f"cells_{ctx}.csv")


def _write_cell_manifest(adata_in: ad.AnnData, cfg: BenchConfig, test_tag: str, frac: Optional[float]) -> None:
    ensure_out_dir(_r_robustness_dir(cfg))
    out_path = _cell_manifest_path(cfg, test_tag, frac)
    df = pd.DataFrame({
        "cell_id": adata_in.obs_names.astype(str),
        "batch": adata_in.obs["batch"].astype(str).to_numpy(),
        "class": adata_in.obs["class"].astype(str).to_numpy(),
        "batch_id": adata_in.obs["batch_id"].astype(int).to_numpy() if "batch_id" in adata_in.obs.columns else -1,
    })
    df.to_csv(out_path, index=False)


def _read_cell_manifest(path: str) -> List[str]:
    df = pd.read_csv(path)
    if "cell_id" in df.columns:
        cells = df["cell_id"].astype(str).tolist()
    else:
        cells = df.iloc[:, 0].astype(str).tolist()
    cells = [c for c in cells if c and c.lower() != "nan"]
    if not cells:
        raise ValueError(f"Cell manifest is empty: {path}")
    return cells


def subset_by_manifest_or_default(
    adata_full_raw: ad.AnnData,
    cfg: BenchConfig,
    test_tag: str,
    frac: Optional[float],
    default_builder,
) -> ad.AnnData:
    """Use an existing R/Python cell manifest when present; otherwise build default perturbation and write it.

    This makes the strict R reruns and Python robustness experiment use the same
    cell subset. Run make_pancreas_robustness_manifests.py first when using R reruns.
    """
    manifest = _cell_manifest_path(cfg, test_tag, frac)
    if os.path.exists(manifest):
        cells = _read_cell_manifest(manifest)
        missing = [c for c in cells if c not in set(adata_full_raw.obs_names.astype(str))]
        if missing:
            raise ValueError(f"Cell manifest has cells not found in raw AnnData: {manifest}; example={missing[:5]}")
        out = adata_full_raw[cells].copy()
        out.obs["batch"] = out.obs["batch"].astype("category")
        out.obs["class"] = out.obs["class"].astype("category")
        if "batch_id" in out.obs.columns:
            out.obs["batch_id"] = out.obs["batch_id"].astype(int)
        return out

    out = default_builder()
    if cfg.generate_cell_manifests:
        _write_cell_manifest(out, cfg, test_tag, frac)
        logging.info("Wrote cell manifest for R rerun: %s", manifest)
    return out


# -----------------------------
# Method runners
# -----------------------------
def run_scanorama(adata_in: ad.AnnData, cfg: BenchConfig) -> Tuple[ad.AnnData, str]:
    # adata_in already HVG-subset; do NOT do Scanorama HVG selection again.
    batches = []
    for b in list(adata_in.obs["batch"].cat.categories):
        batches.append(adata_in[adata_in.obs["batch"] == b].copy())

    corrected = scanorama.correct_scanpy(batches, return_dimred=True)
    adata = ad.concat(corrected, join="outer")
    adata.obs_names_make_unique()
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    adata.obs["class"] = adata.obs["class"].astype("category")
    adata.obsm["X_scanorama"] = _nan_guard(np.asarray(adata.obsm["X_scanorama"]), "X_scanorama")
    return adata, "X_scanorama"


def _coerce_embedding_to_nobs_by_dims(Z, n_obs: int, context: str) -> np.ndarray:
    """Coerce external embedding output to shape (n_obs, n_dims).

    Handles old/new harmonypy orientation differences and dataframe/matrix outputs.
    """
    if hasattr(Z, "to_numpy"):
        X = Z.to_numpy()
    else:
        X = np.asarray(Z)
    X = np.asarray(X)
    if X.ndim > 2:
        X = np.squeeze(X)
    if X.ndim == 1:
        if X.shape[0] == int(n_obs):
            X = X[:, None]
        else:
            raise ValueError(f"{context}: unexpected 1D embedding shape {X.shape}; expected n_obs={n_obs}.")
    if X.ndim != 2:
        raise ValueError(f"{context}: unexpected embedding ndim={X.ndim}, shape={getattr(X, 'shape', None)}")
    if X.shape[0] == int(n_obs):
        return X
    if X.shape[1] == int(n_obs):
        logging.warning(
            "%s: transposing external embedding from shape %s to (%d, %d)",
            context, tuple(X.shape), int(n_obs), int(X.shape[0]),
        )
        return X.T
    raise ValueError(f"{context}: could not align embedding to n_obs={n_obs}. Got shape={X.shape}.")


def run_harmony(adata_in: ad.AnnData, cfg: BenchConfig) -> Tuple[ad.AnnData, str]:
    if hm is None:
        raise RuntimeError("harmonypy not installed")
    adata = adata_in.copy()
    Xp = _nan_guard(np.asarray(adata.obsm["X_pca"]), "X_pca_for_harmony")
    ho = hm.run_harmony(Xp, adata.obs, ["batch"])
    X_harmony = _coerce_embedding_to_nobs_by_dims(ho.Z_corr, adata.n_obs, "X_harmony")
    adata.obsm["X_harmony"] = _nan_guard(X_harmony, "X_harmony")
    return adata, "X_harmony"


def run_bbknn(adata_in: ad.AnnData, cfg: BenchConfig) -> Tuple[ad.AnnData, str]:
    if bbknn is None:
        raise RuntimeError("bbknn not installed")
    adata = adata_in.copy()
    bbknn.bbknn(adata, batch_key="batch")  # builds neighbors graph using X_pca
    return adata, "X_pca"


def run_nmf(adata_in: ad.AnnData, cfg: BenchConfig) -> Tuple[ad.AnnData, str]:
    if NMF is None:
        raise RuntimeError("sklearn NMF not available")
    adata = adata_in.copy()
    X = adata.X
    X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    min_val = float(np.min(X_dense))
    if min_val < 0:
        X_dense = X_dense - min_val
    model = NMF(
        n_components=cfg.nmf_components,
        init="nndsvda",
        max_iter=cfg.nmf_max_iter,
        random_state=cfg.seed,
    )
    adata.obsm["X_nmf"] = _nan_guard(model.fit_transform(X_dense), "X_nmf")
    return adata, "X_nmf"


def run_seurat_csv(
    adata_in: ad.AnnData,
    cfg: BenchConfig,
    test_tag: str = "full",
    frac: Optional[float] = None,
) -> Tuple[ad.AnnData, str]:
    if test_tag in ("imbalance", "nonoverlap"):
        csv_path = _strict_r_embedding_csv_path(cfg, "Seurat", test_tag, frac)
        if cfg.require_strict_r_rerun_embeddings and not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Strict Seurat R-rerun embedding is missing: {csv_path}. "
                f"Run seurat_integration_pancreas_robustness_canonical_v1.R after generating manifests."
            )
        if os.path.exists(csv_path):
            return _load_embedding_csv(adata_in, csv_path, "X_seurat")
    csv_path = os.path.join(cfg.out_folder, cfg.seurat_csv_name)
    return _load_embedding_csv(adata_in, csv_path, "X_seurat")


def run_online_inmf_csv(
    adata_in: ad.AnnData,
    cfg: BenchConfig,
    test_tag: str = "full",
    frac: Optional[float] = None,
) -> Tuple[ad.AnnData, str]:
    if test_tag in ("imbalance", "nonoverlap"):
        csv_path = _strict_r_embedding_csv_path(cfg, "Online_iNMF", test_tag, frac)
        if cfg.require_strict_r_rerun_embeddings and not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Strict Online iNMF R-rerun embedding is missing: {csv_path}. "
                f"Run rliger_online_inmf_pancreas_robustness_canonical_v1.R after generating manifests."
            )
        if os.path.exists(csv_path):
            return _load_embedding_csv(adata_in, csv_path, "X_online_inmf")
    csv_path = os.path.join(cfg.out_folder, cfg.online_inmf_csv_name)
    return _load_embedding_csv(adata_in, csv_path, "X_online_inmf")


def run_diffmap(adata_in: ad.AnnData, cfg: BenchConfig) -> Tuple[ad.AnnData, str]:
    adata = adata_in.copy()
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=int(cfg.n_neighbors))
    sc.tl.diffmap(adata, n_comps=int(cfg.diffmap_n_comps))
    if cfg.diffmap_embed_key not in adata.obsm:
        raise KeyError("Expected diffusion map embedding missing at obsm['X_diffmap'].")
    adata.obsm[cfg.diffmap_embed_key] = _nan_guard(np.asarray(adata.obsm[cfg.diffmap_embed_key]), "X_diffmap")
    return adata, cfg.diffmap_embed_key


def run_phate(adata_in: ad.AnnData, cfg: BenchConfig) -> Tuple[ad.AnnData, str]:
    if phate is None:
        raise RuntimeError("phate not installed (pip install phate)")
    adata = adata_in.copy()
    X = _nan_guard(np.asarray(adata.obsm["X_pca"]), "X_pca_for_PHATE")

    ph = phate.PHATE(
        n_components=int(cfg.phate_n_components),
        knn=int(cfg.phate_knn),
        decay=int(cfg.phate_decay),
        t=cfg.phate_t,
        random_state=int(cfg.seed),
        verbose=False,
    )
    Z = ph.fit_transform(X)
    adata.obsm[cfg.phate_embed_key] = _nan_guard(np.asarray(Z), "X_phate")
    return adata, cfg.phate_embed_key


def run_genorefine_on_existing_key(
    adata_in: ad.AnnData,
    cfg: BenchConfig,
    in_key: str,
    out_key: str,
) -> Tuple[ad.AnnData, str]:
    if gp is None:
        raise RuntimeError("GenoRefine backend not available (install genomap)")
    adata = adata_in.copy()
    if in_key not in adata.obsm:
        raise KeyError(f"Missing embedding '{in_key}' in adata.obsm")
    X = _nan_guard(np.asarray(adata.obsm[in_key]), f"genorefine_in[{in_key}]")
    n_clusters = max(2, int(adata.obs["class"].nunique()))
    Z = gp.genoDR(
        X,
        n_dim=int(cfg.genorefine_dim),
        n_clusters=int(n_clusters),
        colNum=int(cfg.genorefine_col),
        rowNum=int(cfg.genorefine_row),
    )
    adata.obsm[out_key] = _nan_guard(np.asarray(Z), f"genorefine_out[{out_key}]")
    return adata, out_key


# -----------------------------
# Robustness perturbations
# -----------------------------
def make_imbalance_dataset(adata: ad.AnnData, batch_to_downsample: str, frac: float, seed: int) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    mask_other = adata.obs["batch"].astype(str) != batch_to_downsample
    mask_target = ~mask_other

    ad_other = adata[mask_other].copy()
    ad_target = adata[mask_target].copy()

    n_keep = int(np.floor(ad_target.n_obs * frac))
    if n_keep < 2:
        raise ValueError(f"Too few cells left in {batch_to_downsample} at frac={frac}: n_keep={n_keep}")

    keep_idx = rng.choice(ad_target.n_obs, size=n_keep, replace=False)
    ad_target = ad_target[keep_idx].copy()

    out = ad.concat([ad_other, ad_target], join="outer")
    out.obs_names_make_unique()
    out.obs["batch"] = out.obs["batch"].astype("category")
    out.obs["class"] = out.obs["class"].astype("category")
    # Preserve batch_id if present
    if "batch_id" in adata.obs.columns:
        out.obs["batch_id"] = out.obs.get("batch_id", pd.Series(index=out.obs_names)).astype(int)
    return out


def make_nonoverlap_dataset(adata: ad.AnnData, batch: str, celltype: str) -> ad.AnnData:
    b = adata.obs["batch"].astype(str)
    c = adata.obs["class"].astype(str)
    drop_mask = (b == batch) & (c == celltype)
    out = adata[~drop_mask].copy()
    out.obs_names_make_unique()
    out.obs["batch"] = out.obs["batch"].astype("category")
    out.obs["class"] = out.obs["class"].astype("category")
    if "batch_id" in out.obs.columns:
        out.obs["batch_id"] = out.obs["batch_id"].astype(int)
    return out


# -----------------------------
# Runner for one method on a dataset
# -----------------------------
def run_method_and_score(
    method: str,
    adata_base: ad.AnnData,
    cfg: BenchConfig,
    test_tag: str,
    frac: Optional[float] = None,
) -> Tuple[Dict[str, object], ad.AnnData, Dict[str, float], pd.DataFrame]:
    t0 = time.time()

    # ---- base methods ----
    if method == "Scanorama":
        adata, embed_key = run_scanorama(adata_base, cfg)
        rep_for_neighbors = embed_key

    elif method == "Harmony":
        adata, embed_key = run_harmony(adata_base, cfg)
        rep_for_neighbors = embed_key

    elif method == "BBKNN":
        adata, embed_key = run_bbknn(adata_base, cfg)
        rep_for_neighbors = None

    elif method == "NMF":
        adata, embed_key = run_nmf(adata_base, cfg)
        rep_for_neighbors = embed_key

    elif method == "Seurat":
        adata, embed_key = run_seurat_csv(adata_base, cfg, test_tag=test_tag, frac=frac)
        rep_for_neighbors = embed_key

    elif method in ("Online_iNMF", "Online iNMF"):
        adata, embed_key = run_online_inmf_csv(adata_base, cfg, test_tag=test_tag, frac=frac)
        rep_for_neighbors = embed_key

    elif method == "DiffusionMap":
        adata, embed_key = run_diffmap(adata_base, cfg)
        rep_for_neighbors = embed_key

    elif method == "PHATE":
        adata, embed_key = run_phate(adata_base, cfg)
        rep_for_neighbors = embed_key

    # ---- GenoRefine refinements ----
    elif method == "GenoRefine(Scanorama)":
        ad_scan, scan_key = run_scanorama(adata_base, cfg)
        adata, embed_key = run_genorefine_on_existing_key(ad_scan, cfg, scan_key, "X_genorefine_scanorama")
        rep_for_neighbors = embed_key

    elif method == "GenoRefine(Harmony)":
        ad_har, har_key = run_harmony(adata_base, cfg)
        adata, embed_key = run_genorefine_on_existing_key(ad_har, cfg, har_key, "X_genorefine_harmony")
        rep_for_neighbors = embed_key

    elif method == "GenoRefine(Seurat)":
        ad_seu, seu_key = run_seurat_csv(adata_base, cfg, test_tag=test_tag, frac=frac)
        adata, embed_key = run_genorefine_on_existing_key(ad_seu, cfg, seu_key, "X_genorefine_seurat")
        rep_for_neighbors = embed_key

    elif method == "GenoRefine(Online iNMF)":
        ad_inmf, inmf_key = run_online_inmf_csv(adata_base, cfg, test_tag=test_tag, frac=frac)
        adata, embed_key = run_genorefine_on_existing_key(ad_inmf, cfg, inmf_key, "X_genorefine_online_inmf")
        rep_for_neighbors = embed_key

    elif method == "GenoRefine(DiffusionMap)":
        ad_dm, dm_key = run_diffmap(adata_base, cfg)
        adata, embed_key = run_genorefine_on_existing_key(ad_dm, cfg, dm_key, "X_genorefine_diffmap")
        rep_for_neighbors = embed_key

    elif method == "GenoRefine(PHATE)":
        ad_ph, ph_key = run_phate(adata_base, cfg)
        adata, embed_key = run_genorefine_on_existing_key(ad_ph, cfg, ph_key, "X_genorefine_phate")
        rep_for_neighbors = embed_key

    else:
        raise ValueError(method)

    # ---- clustering policy ----
    cluster_key = "leiden"
    if method == "BBKNN":
        if cfg.leiden_target_n_clusters:
            target_k = int(adata.obs["class"].nunique())
            used_res = tune_leiden_resolution_existing_graph(
                adata=adata,
                target_k=target_k,
                res_grid=list(cfg.leiden_res_grid),
                key_added=cluster_key,
                random_state=cfg.seed,
            )
        else:
            used_res = float(cfg.leiden_resolution_default)
            sc.tl.leiden(adata, resolution=used_res, key_added=cluster_key, random_state=cfg.seed)
    else:
        used_res = cluster_on_rep(adata, rep_key=rep_for_neighbors, cfg=cfg, cluster_key=cluster_key)

    # ---- metrics ----
    metrics = compute_metrics(adata, embed_key_for_metrics=embed_key, cfg=cfg, cluster_key=cluster_key)
    runtime_s = float(time.time() - t0)

    # ---- purity/entropy ----
    pe_summary, pe_table = cluster_purity_entropy(
        adata,
        class_key="class",
        cluster_key=cluster_key,
        eps=cfg.entropy_eps,
    )

    out = {
        "Method": method,
        "EmbedKey": embed_key,
        "LeidenRes": float(used_res),
        "Runtime_s": float(runtime_s),
        **metrics,
        "Purity_MeanWeighted": float(pe_summary.get("Purity_MeanWeighted", np.nan)),
        "Entropy_MeanWeighted": float(pe_summary.get("Entropy_MeanWeighted", np.nan)),
        "Test": str(test_tag),
    }
    if frac is not None:
        out["Frac"] = float(frac)

    pe_table = pe_table.copy()
    pe_table["Method"] = method
    pe_table["Test"] = str(test_tag)
    if frac is not None:
        pe_table["Frac"] = float(frac)

    return out, adata, pe_summary, pe_table


# -----------------------------
# Main robustness experiments
# -----------------------------
def _filter_methods_for_deps(methods_all: List[str], cfg: BenchConfig) -> List[str]:
    filtered: List[str] = []
    for m in methods_all:
        if m in ("Harmony", "GenoRefine(Harmony)") and hm is None:
            continue
        if m == "BBKNN" and bbknn is None:
            continue
        if m in ("PHATE", "GenoRefine(PHATE)") and phate is None:
            continue
        if m.startswith("GenoRefine(") and gp is None:
            continue

        # External R embeddings are checked at run time because strict robustness
        # uses perturbation-specific CSVs for Seurat and Online iNMF.

        filtered.append(m)
    return filtered


def run_imbalance_experiment(
    adata_full_raw: ad.AnnData,
    frozen_hvgs: List[str],
    cfg: BenchConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    methods_all = [
        "Scanorama",
        "Harmony",
        "Seurat",
        "Online iNMF",
        "GenoRefine(Scanorama)",
        "GenoRefine(Harmony)",
        "GenoRefine(Seurat)",
        "GenoRefine(Online iNMF)",
    ]
    methods = _filter_methods_for_deps(methods_all, cfg)
    logging.info(f"[Imbalance] Methods to run: {methods}")

    rows: List[Dict[str, object]] = []
    pe_tables_all: List[pd.DataFrame] = []

    for frac in cfg.imbalance_fracs:
        logging.info(f"[Imbalance] downsample {cfg.imbalance_batch_to_downsample} to {frac*100:.0f}%")

        ad_pert_raw = subset_by_manifest_or_default(
            adata_full_raw,
            cfg,
            test_tag="imbalance",
            frac=float(frac),
            default_builder=lambda frac=frac: make_imbalance_dataset(
                adata_full_raw, cfg.imbalance_batch_to_downsample, frac, seed=cfg.seed
            ),
        )
        ad_pert = preprocess_with_frozen_hvgs(ad_pert_raw, frozen_hvgs, cfg)

        if float(frac) == 1.0:
            log_preprocess_fingerprint(ad_pert, cfg, tag="IMBALANCE_FRAC_1.0")

        for m in methods:
            logging.info(f"  method={m}")
            try:
                res, adata_method, _, pe_tbl = run_method_and_score(m, ad_pert, cfg, test_tag="imbalance", frac=float(frac))
                res.update({"DownsampleBatch": cfg.imbalance_batch_to_downsample})
                rows.append(res)
                pe_tables_all.append(pe_tbl)

                if cfg.save_per_method_purity_entropy_tables:
                    out_tbl = os.path.join(
                        cfg.out_folder, f"purity_entropy_imbalance_{sanitize_method_name(m)}_frac{frac:.2f}.csv"
                    )
                    pe_tbl.to_csv(out_tbl, index=False)

                # Optional: a couple of UMAP snapshots to keep outputs manageable
                if float(frac) in set(cfg.save_umap_imbalance_for_fracs):
                    sc.tl.umap(adata_method, random_state=cfg.seed)
                    safe_m = sanitize_method_name(m)
                    save_umap_png(
                        adata_method, "class",
                        os.path.join(cfg.out_folder, f"UMAP_imbalance_frac{frac:.2f}_{safe_m}_class.png"),
                        title=f"Imbalance frac={frac:.2f} {m} (class)",
                    )
                    save_umap_png(
                        adata_method, "batch",
                        os.path.join(cfg.out_folder, f"UMAP_imbalance_frac{frac:.2f}_{safe_m}_batch.png"),
                        title=f"Imbalance frac={frac:.2f} {m} (batch)",
                    )

            except Exception as e:
                logging.exception(f"FAILED imbalance method {m} frac={frac}: {e}")
                rows.append({
                    "Test": "imbalance",
                    "DownsampleBatch": cfg.imbalance_batch_to_downsample,
                    "Frac": float(frac),
                    "Method": m,
                    "Error": str(e),
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.out_folder, "robustness_imbalance.csv"), index=False)

    # Extreme imbalance summary (100% vs 10%)
    df_ok = df.dropna(subset=["ARI"]).copy()
    df_ext = df_ok[df_ok["Frac"].isin([1.0, 0.10])].copy()
    df_ext.to_csv(os.path.join(cfg.out_folder, "table_extreme_imbalance.csv"), index=False)

    # Aggregate purity/entropy tables
    df_pe = pd.concat(pe_tables_all, ignore_index=True) if pe_tables_all else pd.DataFrame()
    if len(df_pe):
        df_pe.to_csv(os.path.join(cfg.out_folder, "purity_entropy_imbalance_allclusters.csv"), index=False)
        df_sum = df_ok.groupby(["Method", "Frac"])[["Purity_MeanWeighted", "Entropy_MeanWeighted"]].mean().reset_index()
        df_sum.to_csv(os.path.join(cfg.out_folder, "purity_entropy_imbalance_summary.csv"), index=False)

    # Plot metric curves
    try:
        import matplotlib.pyplot as plt
        df_plot = df.dropna(subset=["ARI", "Silhouette", "iLISI"]).copy()

        for metric in ["ARI", "Silhouette", "iLISI"]:
            plt.figure(figsize=(9, 6))
            for m in sorted(df_plot["Method"].dropna().unique()):
                sub = df_plot[df_plot["Method"] == m].sort_values("Frac")
                plt.plot(sub["Frac"], sub[metric], marker="o", label=m)
            plt.xlabel("Fraction of cells retained in downsampled batch")
            plt.ylabel(metric)
            plt.title(f"Robustness to dataset imbalance: {metric}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.out_folder, f"fig_imbalance_curves_{metric}.png"), dpi=300)
            plt.close()
    except Exception as e:
        logging.warning(f"Could not plot curves: {e}")

    return df, df_pe


def run_nonoverlap_experiment(
    adata_full_raw: ad.AnnData,
    frozen_hvgs: List[str],
    cfg: BenchConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark methods after removing one batch-cell-type combination, using frozen HVGs."""
    methods_all = [
        "Scanorama",
        "Harmony",
        "Seurat",
        "Online iNMF",
        "GenoRefine(Scanorama)",
        "GenoRefine(Harmony)",
        "GenoRefine(Seurat)",
        "GenoRefine(Online iNMF)",
    ]
    methods = _filter_methods_for_deps(methods_all, cfg)
    logging.info(f"[Non-overlap] Methods to run: {methods}")

    logging.info(f"[Non-overlap] remove celltype='{cfg.nonoverlap_celltype}' from batch='{cfg.nonoverlap_batch}'")

    ad_pert_raw = subset_by_manifest_or_default(
        adata_full_raw,
        cfg,
        test_tag="nonoverlap",
        frac=None,
        default_builder=lambda: make_nonoverlap_dataset(adata_full_raw, cfg.nonoverlap_batch, cfg.nonoverlap_celltype),
    )
    ad_pert = preprocess_with_frozen_hvgs(ad_pert_raw, frozen_hvgs, cfg)
    log_preprocess_fingerprint(ad_pert, cfg, tag="NONOVERLAP_PREPROCESSED")

    rows: List[Dict[str, object]] = []
    pe_tables_all: List[pd.DataFrame] = []

    for m in methods:
        logging.info(f"  method={m}")
        try:
            res, adata_method, _, pe_tbl = run_method_and_score(m, ad_pert, cfg, test_tag="nonoverlap", frac=None)
            res.update({"RemovedBatch": cfg.nonoverlap_batch, "RemovedCelltype": cfg.nonoverlap_celltype})
            rows.append(res)

            safe_m = sanitize_method_name(m)

            cm = confusion_matrix_table(adata_method, class_key="class", cluster_key="leiden", normalize="index")
            cm.to_csv(os.path.join(cfg.out_folder, f"confusion_nonoverlap_{safe_m}.csv"))

            pe_tables_all.append(pe_tbl)
            if cfg.save_per_method_purity_entropy_tables:
                out_tbl = os.path.join(cfg.out_folder, f"purity_entropy_nonoverlap_{safe_m}.csv")
                pe_tbl.to_csv(out_tbl, index=False)

            if cfg.save_umap_nonoverlap:
                sc.tl.umap(adata_method, random_state=cfg.seed)
                save_umap_png(
                    adata_method, "class",
                    os.path.join(cfg.out_folder, f"UMAP_nonoverlap_{safe_m}_class.png"),
                    title=f"Nonoverlap {m} (class)",
                )
                save_umap_png(
                    adata_method, "batch",
                    os.path.join(cfg.out_folder, f"UMAP_nonoverlap_{safe_m}_batch.png"),
                    title=f"Nonoverlap {m} (batch)",
                )

        except Exception as e:
            logging.exception(f"FAILED nonoverlap method {m}: {e}")
            rows.append({
                "Test": "nonoverlap",
                "RemovedBatch": cfg.nonoverlap_batch,
                "RemovedCelltype": cfg.nonoverlap_celltype,
                "Method": m,
                "Error": str(e),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.out_folder, "robustness_nonoverlap.csv"), index=False)

    df_pe = pd.concat(pe_tables_all, ignore_index=True) if pe_tables_all else pd.DataFrame()
    if len(df_pe):
        df_pe.to_csv(os.path.join(cfg.out_folder, "purity_entropy_nonoverlap_allclusters.csv"), index=False)

        df_ok = df.dropna(subset=["ARI"]).copy()
        df_sum = df_ok.groupby(["Method"])[["Purity_MeanWeighted", "Entropy_MeanWeighted"]].mean().reset_index()
        df_sum.to_csv(os.path.join(cfg.out_folder, "purity_entropy_nonoverlap_summary.csv"), index=False)

    return df, df_pe


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    set_all_seeds(CFG.seed)
    ensure_out_dir(CFG.out_folder)

    # dependency warnings
    if hm is None:
        logging.warning("harmonypy not installed -> Harmony and GenoRefine(Harmony) will be skipped.")
    if gp is None:
        logging.warning("GenoRefine backend not available -> GenoRefine refinements will be skipped.")

    logging.info("Loading full multi-pancreas raw dataset (v13 canonical naming; no normalize/log yet)...")
    adata_full_raw = load_base_data(CFG)

    logging.info("Loading frozen canonical HVGs from make_pancreas_canonical_hvg.py output...")
    frozen_hvgs = load_canonical_hvgs(adata_full_raw, CFG)
    logging.info(f"Canonical HVGs: {len(frozen_hvgs)} genes")

    # Optional baseline preprocess for fingerprint/debug (do NOT feed into perturbations)
    adata_full_pp = preprocess_with_frozen_hvgs(adata_full_raw, frozen_hvgs, CFG)
    log_preprocess_fingerprint(adata_full_pp, CFG, tag="FULL_PREPROCESSED")

    # Imbalance robustness
    run_imbalance_experiment(adata_full_raw, frozen_hvgs, CFG)

    # Non-overlap negative control
    run_nonoverlap_experiment(adata_full_raw, frozen_hvgs, CFG)

    logging.info("Robustness experiments complete (canonical strict v16 GenoRefine core methods).")


if __name__ == "__main__":
    main()
