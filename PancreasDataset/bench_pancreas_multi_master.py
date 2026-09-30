"""
FILE: bench_pancreas_multi_master_v13.py
-------------------------------------------------------------------------------
Author: Ariana Rahman
Affiliation: Arizona State University / Stanford University
Date: April 2026 (v13)
-------------------------------------------------------------------------------

PURPOSE
    Canonical end-to-end benchmark for the 5-dataset Pancreas integration suite.
    Compares baseline integration methods:
        Scanorama, Harmony, BBKNN, Seurat, Online_iNMF, DiffusionMap, PHATE
    and evaluates GenoDR as a post-integration embedding refinement on:
        Scanorama, Harmony, Seurat, Online_iNMF, DiffusionMap, PHATE

v13 CHANGES (clean canonical-HVG preprocessing order)
    1) Keeps the v12 Harmony compatibility fix and method list.
    2) Uses the frozen canonical HVG file in a cleaner order:
         subset raw features first, then normalize/log for downstream PCA/integration.
    3) Preserves the v10 dynamic per-batch HVG logic as an optional fallback path.
    4) Preserves canonical-file order when subsetting features.
    5) Keeps outputs and dependency-aware filtering unchanged.

INPUTS
    - Expression matrices (.mat) in CFG.data_folder:
        dataBaronX.mat, dataMuraroX.mat, dataScapleX.mat, dataWangX.mat, dataXinX.mat
    - Ground-truth labels (.mat) in CFG.data_folder:
        classLabel.mat (vector aligned to concatenated cells across batches)
    - Frozen canonical HVG file (used by default, in CFG.out_folder):
        pancreas_hvg_canonical.txt
    - External embeddings exported from R (optional, in CFG.out_folder):
        X_seurat_pancreas.csv
        X_online_inmf_pancreas.csv

OUTPUTS (written to CFG.out_folder)
    - benchmark_results_pancreas.csv
    - benchmark_ranked_pancreas.csv
    - table_pancreas_main.csv
    - table_pancreas_main.tex
    - fig_pancreas_bars_main_metrics.png
    - UMAP_{Method}_class.png, UMAP_{Method}_batch.png
    - Confusion_{Method}.csv
    - Heatmap_Confusion_{Method}.png
-------------------------------------------------------------------------------
"""

import os
import time
import logging
import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd
import scipy.io as sio

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
    # Pancreas benchmark .mat inputs
    data_folder: str = "./Dataset"
    out_folder: str = "./Benchmark_Out"
    seed: int = 0

    data_files: Tuple[str, ...] = (
        "dataBaronX.mat",
        "dataMuraroX.mat",
        "dataScapleX.mat",  # commonly intended to be Segerstolpe
        "dataWangX.mat",
        "dataXinX.mat",
    )
    class_label_file: str = "classLabel.mat"  # contains classLabel vector

    # Batch name normalization (file token -> display name)
    batch_token_to_name: Dict[str, str] = None  # filled in __post_init__

    # Preprocess
    hvg: int = 2000
    n_pcs: int = 50
    n_neighbors: int = 15
    neighbor_metric: str = "cosine"
    neighbor_n_dims: int = 30
    strict_use_rep: bool = True

    # HVG flavor (canonical preprocessing spec)
    hvg_flavor: str = "seurat_v3"
    hvg_union_per_batch: bool = True
    hvg_per_batch: int = 2000
    hvg_union_mode: str = "union"  # union or intersection

    # Frozen canonical HVG file (v11 default path)
    use_canonical_hvg_file: bool = True
    canonical_hvg_file: str = "pancreas_hvg_canonical.txt"
    fallback_to_dynamic_hvgs_if_missing: bool = False
    require_exact_canonical_hvg_match: bool = True

    # Clustering
    leiden_target_n_clusters: bool = True
    leiden_resolution_default: float = 0.5
    leiden_res_grid: Tuple[float, ...] = tuple(np.round(np.linspace(0.2, 1.6, 15), 2))

    # Metrics
    lisi_k: int = 90

    # DiffusionMap parameters
    diffmap_n_comps: int = 30
    diffmap_embed_key: str = "X_diffmap"

    # PHATE parameters
    phate_n_components: int = 30
    phate_knn: int = 15
    phate_decay: int = 40
    phate_t: str = "auto"
    phate_embed_key: str = "X_phate"

    # GenoDR params
    genodr_dim: int = 32
    genodr_col: int = 33
    genodr_row: int = 33

    # CSV exports from R
    seurat_csv_name: str = "X_seurat_pancreas.csv"
    online_inmf_csv_name: str = "X_online_inmf_pancreas.csv"

    def __post_init__(self):
        if self.batch_token_to_name is None:
            self.batch_token_to_name = {
                "Baron": "Baron",
                "Muraro": "Muraro",
                "Scaple": "Segerstolpe",  # normalize the common typo/token
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
        h.update(s.encode("utf-8"))
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


def _rep_for_neighbors(adata: ad.AnnData, rep_key: str, n_dims: int) -> str:
    """Return an obsm key safe for neighbors (slice to n_dims if needed)."""
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


def _normalize_obs_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"-1$", "", s)
    s = re.sub(r"\.1$", "", s)
    s = s.replace(".", "-")
    return s


def _select_hvgs_per_batch_union(
    adata: ad.AnnData,
    batch_key: str,
    n_top: int,
    flavor: str,
    layer: Optional[str],
    mode: str = "union",
) -> List[str]:
    """Compute HVGs per batch and return union/intersection gene list."""

    def _matrix_from_layer(aobj: ad.AnnData, lyr: Optional[str]):
        return aobj.layers[lyr] if lyr is not None and lyr in aobj.layers else aobj.X

    def _nonzero_gene_mask(aobj: ad.AnnData, lyr: Optional[str]) -> np.ndarray:
        X = _matrix_from_layer(aobj, lyr)
        sums = np.asarray(X.sum(axis=0)).ravel()
        return sums > 0

    def _run_hvg_safe(aobj: ad.AnnData, n_top_local: int, flavor_local: str, lyr: Optional[str], context: str):
        try:
            sc.pp.highly_variable_genes(
                aobj,
                n_top_genes=int(n_top_local),
                flavor=str(flavor_local),
                subset=False,
                layer=lyr,
            )
            return str(flavor_local)
        except ValueError as e:
            msg = str(e)
            if "Bin edges must be unique" not in msg:
                raise
            fallback = "seurat" if str(flavor_local) != "seurat" else "cell_ranger"
            logging.warning(
                "HVG bin-edge issue in %s with flavor=%s; retrying with flavor=%s after removing all-zero genes.",
                context, flavor_local, fallback,
            )
            keep = _nonzero_gene_mask(aobj, lyr)
            if keep.sum() == 0:
                raise ValueError(f"All genes are zero in {context}; cannot compute HVGs.") from e
            trimmed = aobj[:, keep].copy()
            sc.pp.highly_variable_genes(
                trimmed,
                n_top_genes=min(int(n_top_local), trimmed.n_vars),
                flavor=fallback,
                subset=False,
                layer=lyr,
            )
            hv = np.zeros(aobj.n_vars, dtype=bool)
            hv_idx = np.where(keep)[0][trimmed.var["highly_variable"].to_numpy()]
            hv[hv_idx] = True
            aobj.var["highly_variable"] = hv
            return fallback

    if batch_key not in adata.obs:
        raise KeyError(f"Missing batch_key '{batch_key}' in adata.obs")
    batches = adata.obs[batch_key].astype(str).to_numpy()
    uniq = pd.unique(batches)
    if len(uniq) < 2:
        _run_hvg_safe(adata, int(n_top), str(flavor), layer, f"all cells ({batch_key})")
        return list(adata.var_names[adata.var["highly_variable"]])

    sets: List[set] = []
    for b in uniq:
        sub = adata[batches == b].copy()
        _run_hvg_safe(sub, int(n_top), str(flavor), layer, f"batch={b}")
        sets.append(set(sub.var_names[sub.var["highly_variable"]].tolist()))

    if str(mode).lower().startswith("inter"):
        hvgs = set.intersection(*sets) if sets else set()
        if len(hvgs) < min(500, int(n_top)):
            hvgs = set.union(*sets)
    else:
        hvgs = set.union(*sets)

    return list(hvgs)


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


def _subset_with_frozen_canonical_hvgs(adata: ad.AnnData, cfg: BenchConfig) -> ad.AnnData:
    txt_path = os.path.join(cfg.out_folder, cfg.canonical_hvg_file)
    requested_hvgs = _read_canonical_hvg_file(txt_path)

    present = set(adata.var_names.astype(str).tolist())
    matched_hvgs = [g for g in requested_hvgs if g in present]
    missing_count = int(len(requested_hvgs) - len(matched_hvgs))

    if len(matched_hvgs) == 0:
        raise ValueError(
            f"No canonical HVGs from file matched adata.var_names. File={txt_path}"
        )

    if cfg.require_exact_canonical_hvg_match and len(matched_hvgs) != len(requested_hvgs):
        raise ValueError(
            f"Canonical HVG file match is incomplete. File={txt_path} | "
            f"requested={len(requested_hvgs)} matched={len(matched_hvgs)} missing={missing_count}"
        )

    logging.info(
        "Using frozen canonical HVG file: %s | requested=%d matched=%d missing=%d md5=%s",
        txt_path,
        len(requested_hvgs),
        len(matched_hvgs),
        missing_count,
        md5_of_list(matched_hvgs),
    )
    return adata[:, matched_hvgs].copy()


def log_preprocess_fingerprint(adata: ad.AnnData, cfg: BenchConfig, tag: str = "BASE") -> None:
    var_names = [str(v) for v in adata.var_names.tolist()]
    obs_names = [str(o) for o in adata.obs_names.tolist()]

    hvg_source = (
        f"frozen_file:{cfg.canonical_hvg_file}"
        if getattr(cfg, "use_canonical_hvg_file", False)
        else f"dynamic:{cfg.hvg_union_mode}"
    )
    msg = [
        f"[{tag}] fingerprint: n_obs={adata.n_obs}, n_vars={adata.n_vars}",
        f"[{tag}] HVG source={hvg_source}, flavor={cfg.hvg_flavor}, HVG n={cfg.hvg}",
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

    if cfg.leiden_target_n_clusters:
        used_res = tune_leiden_resolution_to_target(
            adata=adata,
            use_rep=_rep_for_neighbors(adata, rep_key, int(cfg.neighbor_n_dims)),
            target_k=n_classes,
            n_neighbors=cfg.n_neighbors,
            res_grid=res_grid,
            key_added=cluster_key,
            random_state=cfg.seed,
        )
    else:
        sc.pp.neighbors(
            adata,
            use_rep=_rep_for_neighbors(adata, rep_key, int(cfg.neighbor_n_dims)),
            n_neighbors=cfg.n_neighbors,
            metric=str(cfg.neighbor_metric),
        )
        sc.tl.leiden(
            adata,
            resolution=cfg.leiden_resolution_default,
            key_added=cluster_key,
            random_state=cfg.seed,
        )
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
    sc.pl.umap(adata, color=color_key, show=False, title=title)
    import matplotlib.pyplot as plt
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def save_confusion_heatmap(cm: pd.DataFrame, out_png: str, title: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, cmap="YlGnBu")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_grouped_bar_main_metrics(df_results: pd.DataFrame, out_png: str) -> None:
    import matplotlib.pyplot as plt

    df = df_results.copy()
    df = df.dropna(subset=["ARI", "Silhouette", "iLISI"])

    methods = df["Method"].astype(str).tolist()
    metrics = ["ARI", "Silhouette", "iLISI"]

    x = np.arange(len(methods))
    width = 0.26

    plt.figure(figsize=(12, 5))
    for i, metric in enumerate(metrics):
        vals = df[metric].astype(float).to_numpy()
        plt.bar(x + (i - 1) * width, vals, width=width, label=metric)

    plt.xticks(x, methods, rotation=25, ha="right")
    plt.ylabel("Score")
    plt.title("Pancreas benchmark: ARI / Silhouette / iLISI by method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def df_to_latex_table_main_metrics(df_table: pd.DataFrame, caption: str, label: str) -> str:
    df_fmt = df_table.copy()
    for c in ["ARI", "Silhouette", "iLISI"]:
        df_fmt[c] = df_fmt[c].astype(float).map(lambda x: f"{x:.3f}")

    header = r"""\begin{table}[htbp]
\centering
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Method} & \textbf{ARI $\uparrow$} & \textbf{Sil $\uparrow$} & \textbf{iLISI $\uparrow$} \\ \midrule
"""
    rows = []
    for _, r in df_fmt.iterrows():
        rows.append(f"{r['Method']} & {r['ARI']} & {r['Silhouette']} & {r['iLISI']} \\\\")
    body = "\n".join(rows)

    footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return header + body + footer


# -----------------------------
# Data loading (canonical preprocessing)
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
        return base[4:-1]  # Baron
    if base.lower().startswith("data"):
        return base[4:]
    return base


def load_base_data(cfg: BenchConfig) -> ad.AnnData:
    """
    Load pancreas batches, apply the frozen canonical HVG file in the clean order
    when available (subset raw matrix first, then normalize/log), and otherwise
    fall back to the legacy dynamic-HVG path.
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
        a.obs["batch_id"] = int(i + 1)  # legacy 1..5

        adatas.append(a)

    adata = ad.concat(adatas, join="outer")
    adata.obs_names_make_unique()
    adata.var_names = pd.Index(adata.var_names.astype(str))

    cls_path = os.path.join(cfg.data_folder, cfg.class_label_file)
    cls = sio.loadmat(cls_path)["classLabel"].squeeze()

    class_name_map = {
        1: "MHC class II",
        2: "acinar",
        3: "ductal",
        4: "gamma",
        5: "macrophage",
        6: "alpha",
        7: "beta",
        8: "endothelial",
        9: "epsilon",
        10: "mast",
        11: "mesenchymal",
        12: "stellate",
        13: "delta",
        14: "schwann",
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

    used_frozen_hvgs = False
    if cfg.use_canonical_hvg_file:
        try:
            adata = _subset_with_frozen_canonical_hvgs(adata, cfg)
            used_frozen_hvgs = True
            logging.info(
                "Applying clean canonical-HVG workflow: subset raw feature matrix first, then normalize/log."
            )
        except FileNotFoundError as e:
            if not cfg.fallback_to_dynamic_hvgs_if_missing:
                raise
            logging.warning("%s -- falling back to dynamic per-batch HVG selection.", str(e))

    if used_frozen_hvgs:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        flavor = str(cfg.hvg_flavor)
        layer_for_hvg = None
        if flavor == "seurat_v3" and layer_for_hvg is None:
            logging.warning("HVG seurat_v3 requested but no counts layer -> fallback to cell_ranger.")
            flavor = "cell_ranger"

        if cfg.hvg_union_per_batch:
            hvgs = _select_hvgs_per_batch_union(
                adata,
                batch_key="batch",
                n_top=int(cfg.hvg_per_batch),
                flavor=flavor,
                layer=layer_for_hvg,
                mode=str(cfg.hvg_union_mode),
            )
            adata = adata[:, hvgs].copy()
            logging.info(
                "Using dynamic per-batch HVGs: mode=%s selected=%d md5=%s",
                str(cfg.hvg_union_mode),
                len(hvgs),
                md5_of_list([str(x) for x in hvgs]),
            )
        else:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=cfg.hvg,
                flavor=flavor,
                subset=True,
                layer=layer_for_hvg,
            )
            logging.info(
                "Using dynamic pooled HVGs: selected=%d md5=%s",
                int(adata.n_vars),
                md5_of_list([str(x) for x in adata.var_names.tolist()]),
            )

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=cfg.n_pcs)

    return adata


# -----------------------------
# Method runners
# -----------------------------
def run_scanorama(adata_in: ad.AnnData, cfg: BenchConfig) -> Tuple[ad.AnnData, str]:
    batches = []
    for b in list(adata_in.obs["batch"].cat.categories):
        batches.append(adata_in[adata_in.obs["batch"] == b].copy())

    corrected = scanorama.correct_scanpy(batches, return_dimred=True)
    adata = ad.concat(corrected, join="outer")
    adata.obs_names_make_unique()
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    adata.obs["class"] = adata.obs["class"].astype("category")
    # scanorama already stores X_scanorama
    adata.obsm["X_scanorama"] = _nan_guard(np.asarray(adata.obsm["X_scanorama"]), "X_scanorama")
    return adata, "X_scanorama"


def _coerce_embedding_to_nobs_by_dims(
    Z: Any,
    n_obs: int,
    context: str,
) -> np.ndarray:
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
            raise ValueError(
                f"{context}: unexpected 1D embedding shape {X.shape}; expected first axis == n_obs={n_obs}."
            )

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

    raise ValueError(
        f"{context}: could not align embedding to n_obs={n_obs}. Got shape={X.shape}."
    )


def run_harmony(adata_in: ad.AnnData, cfg: BenchConfig) -> Tuple[ad.AnnData, str]:
    if hm is None:
        raise RuntimeError("harmonypy not installed")

    adata = adata_in.copy()
    X_pca = _nan_guard(np.asarray(adata.obsm["X_pca"]), "X_pca_for_harmony")
    ho = hm.run_harmony(X_pca, adata.obs, ["batch"])
    X_harmony = _coerce_embedding_to_nobs_by_dims(ho.Z_corr, adata.n_obs, "X_harmony")
    adata.obsm["X_harmony"] = _nan_guard(X_harmony, "X_harmony")
    return adata, "X_harmony"


def run_bbknn(adata_in: ad.AnnData, cfg: BenchConfig) -> Tuple[ad.AnnData, str]:
    if bbknn is None:
        raise RuntimeError("bbknn not installed")

    adata = adata_in.copy()
    bbknn.bbknn(adata, batch_key="batch")  # uses X_pca
    return adata, "X_pca"


def _build_legacy_obs_names_from_current(adata_in: ad.AnnData) -> pd.Index:
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

    # Attempt 1: direct named-batch alignment
    df1 = df.reindex(adata_in.obs_names)
    match_rate_1 = float(df1.notna().all(axis=1).mean())
    if match_rate_1 >= 0.999:
        adata = adata_in.copy()
        adata.obsm[obsm_key] = _nan_guard(df1.to_numpy(), f"CSV[{obsm_key}]")
        return adata, obsm_key

    # Attempt 2: legacy alignment (Cell-<i>-Batch-<1..5>)
    legacy_names = _build_legacy_obs_names_from_current(adata_in)
    df2 = df.reindex(legacy_names)
    match_rate_2 = float(df2.notna().all(axis=1).mean())
    if match_rate_2 >= 0.999:
        adata = adata_in.copy()
        adata.obsm[obsm_key] = _nan_guard(df2.to_numpy(), f"CSV[{obsm_key}]")
        return adata, obsm_key

    # Attempt 3: normalized-name alignment
    obs_norm = pd.Index([_normalize_obs_name(x) for x in adata_in.obs_names.astype(str)])
    df_norm_index = pd.Index([_normalize_obs_name(x) for x in df.index.astype(str)])

    norm_to_row = {}
    dup = set()
    for orig, n in zip(df.index.astype(str), df_norm_index):
        if n in norm_to_row:
            dup.add(n)
        else:
            norm_to_row[n] = orig
    for n in dup:
        norm_to_row.pop(n, None)

    aligned_rows = [norm_to_row.get(n, None) for n in obs_norm]
    ok = [r is not None for r in aligned_rows]
    match_rate_3 = float(sum(ok) / max(1, len(ok)))

    if match_rate_3 >= 0.999:
        adata = adata_in.copy()
        df3 = df.loc[aligned_rows].copy()
        df3.index = adata.obs_names
        adata.obsm[obsm_key] = _nan_guard(df3.to_numpy(), f"CSV[{obsm_key}]")
        return adata, obsm_key

    missing1 = df1.index[df1.isna().any(axis=1)].tolist()[:10]
    missing2 = df2.index[df2.isna().any(axis=1)].tolist()[:10]
    missing3 = [adata_in.obs_names[i] for i, r in enumerate(aligned_rows) if r is None][:10]
    raise ValueError(
        f"Embedding CSV alignment failed.\n"
        f"CSV: {csv_path}\n"
        f"Attempt1 (named-batch) match={match_rate_1*100:.2f}% missing ex={missing1}\n"
        f"Attempt2 (legacy numeric-batch) match={match_rate_2*100:.2f}% missing ex={missing2}\n"
        f"Attempt3 (normalized) match={match_rate_3*100:.2f}% missing ex={missing3}\n"
        f"Fix: ensure CSV rownames match either:\n"
        f"  named:  Cell-<i>-Batch-<Baron|Muraro|Segerstolpe|Wang|Xin>\n"
        f"  legacy: Cell-<i>-Batch-<1..5>\n"
        f"  or match after removing '-1' and replacing '.' with '-'."
    )


def run_seurat_csv(adata_in: ad.AnnData, cfg: BenchConfig) -> Tuple[ad.AnnData, str]:
    csv_path = os.path.join(cfg.out_folder, cfg.seurat_csv_name)
    return _load_embedding_csv(adata_in, csv_path, "X_seurat")


def run_online_inmf_csv(adata_in: ad.AnnData, cfg: BenchConfig) -> Tuple[ad.AnnData, str]:
    csv_path = os.path.join(cfg.out_folder, cfg.online_inmf_csv_name)
    return _load_embedding_csv(adata_in, csv_path, "X_online_inmf")


def run_diffmap(adata_in: ad.AnnData, cfg: BenchConfig) -> Tuple[ad.AnnData, str]:
    """
    DiffusionMap embedding from Scanpy.
    Uses neighbors graph built on PCA (X_pca).
    """
    adata = adata_in.copy()
    sc.pp.neighbors(
        adata,
        use_rep=_rep_for_neighbors(adata, "X_pca", int(cfg.neighbor_n_dims)),
        n_neighbors=int(cfg.n_neighbors),
        metric=str(cfg.neighbor_metric),
    )
    sc.tl.diffmap(adata, n_comps=int(cfg.diffmap_n_comps))
    if cfg.diffmap_embed_key not in adata.obsm:
        raise KeyError("Expected diffusion map embedding missing at obsm['X_diffmap'].")
    adata.obsm[cfg.diffmap_embed_key] = _nan_guard(np.asarray(adata.obsm[cfg.diffmap_embed_key]), "X_diffmap")
    return adata, cfg.diffmap_embed_key


def run_phate(adata_in: ad.AnnData, cfg: BenchConfig) -> Tuple[ad.AnnData, str]:
    """
    PHATE embedding (optional dependency).
    Runs PHATE on PCA (X_pca) for stability and fairness.
    """
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


def run_genodr_on_existing_key(
    adata_in: ad.AnnData,
    cfg: BenchConfig,
    in_key: str,
    out_key: str,
) -> Tuple[ad.AnnData, str]:
    if gp is None:
        raise RuntimeError("genomap.genoDR not available")

    adata = adata_in.copy()
    if in_key not in adata.obsm:
        raise KeyError(f"Missing embedding '{in_key}' in adata.obsm")

    X = _nan_guard(np.asarray(adata.obsm[in_key]), f"genoDR_in[{in_key}]")
    n_clusters = max(2, int(adata.obs["class"].nunique()))

    Z = gp.genoDR(
        X,
        n_dim=int(cfg.genodr_dim),
        n_clusters=int(n_clusters),
        colNum=int(cfg.genodr_col),
        rowNum=int(cfg.genodr_row),
    )
    adata.obsm[out_key] = _nan_guard(np.asarray(Z), f"genoDR_out[{out_key}]")
    return adata, out_key


# -----------------------------
# One method runner
# -----------------------------
def run_one_method(method_name: str, adata_base: ad.AnnData, cfg: BenchConfig) -> Dict[str, object]:
    t0 = time.time()

    if method_name == "Scanorama":
        adata, embed_key = run_scanorama(adata_base, cfg)
        rep_for_neighbors = embed_key

    elif method_name == "Harmony":
        adata, embed_key = run_harmony(adata_base, cfg)
        rep_for_neighbors = embed_key

    elif method_name == "BBKNN":
        adata, embed_key = run_bbknn(adata_base, cfg)
        rep_for_neighbors = None  # graph built by bbknn

    elif method_name == "Seurat":
        adata, embed_key = run_seurat_csv(adata_base, cfg)
        rep_for_neighbors = embed_key

    elif method_name == "Online_iNMF":
        adata, embed_key = run_online_inmf_csv(adata_base, cfg)
        rep_for_neighbors = embed_key

    elif method_name == "DiffusionMap":
        adata, embed_key = run_diffmap(adata_base, cfg)
        rep_for_neighbors = embed_key

    elif method_name == "PHATE":
        adata, embed_key = run_phate(adata_base, cfg)
        rep_for_neighbors = embed_key

    elif method_name == "GenoDR(Scanorama)":
        ad_scan, scan_key = run_scanorama(adata_base, cfg)
        adata, embed_key = run_genodr_on_existing_key(ad_scan, cfg, scan_key, "X_genodr_scanorama")
        rep_for_neighbors = embed_key

    elif method_name == "GenoDR(Harmony)":
        ad_har, har_key = run_harmony(adata_base, cfg)
        adata, embed_key = run_genodr_on_existing_key(ad_har, cfg, har_key, "X_genodr_harmony")
        rep_for_neighbors = embed_key

    elif method_name == "GenoDR(Seurat)":
        ad_seu, seu_key = run_seurat_csv(adata_base, cfg)
        adata, embed_key = run_genodr_on_existing_key(ad_seu, cfg, seu_key, "X_genodr_seurat")
        rep_for_neighbors = embed_key

    elif method_name == "GenoDR(Online_iNMF)":
        ad_inmf, inmf_key = run_online_inmf_csv(adata_base, cfg)
        adata, embed_key = run_genodr_on_existing_key(ad_inmf, cfg, inmf_key, "X_genodr_online_inmf")
        rep_for_neighbors = embed_key

    elif method_name == "GenoDR(DiffusionMap)":
        ad_dm, dm_key = run_diffmap(adata_base, cfg)
        adata, embed_key = run_genodr_on_existing_key(ad_dm, cfg, dm_key, "X_genodr_diffmap")
        rep_for_neighbors = embed_key

    elif method_name == "GenoDR(PHATE)":
        ad_ph, ph_key = run_phate(adata_base, cfg)
        adata, embed_key = run_genodr_on_existing_key(ad_ph, cfg, ph_key, "X_genodr_phate")
        rep_for_neighbors = embed_key

    else:
        raise ValueError(f"Unknown method: {method_name}")

    cluster_key = "leiden"

    if rep_for_neighbors is not None and cfg.strict_use_rep:
        _ = _rep_for_neighbors(adata, rep_for_neighbors, int(cfg.neighbor_n_dims))

    # BBKNN: graph already exists
    if method_name == "BBKNN":
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

    metrics = compute_metrics(adata, embed_key_for_metrics=embed_key, cfg=cfg, cluster_key=cluster_key)

    # UMAP uses neighbor graph already computed by clustering step (or bbknn graph)
    sc.tl.umap(adata, random_state=cfg.seed)

    safe_name = sanitize_method_name(method_name)
    save_umap_png(
        adata,
        "class",
        os.path.join(cfg.out_folder, f"UMAP_{safe_name}_class.png"),
        title=f"{method_name} (class)",
    )
    save_umap_png(
        adata,
        "batch",
        os.path.join(cfg.out_folder, f"UMAP_{safe_name}_batch.png"),
        title=f"{method_name} (batch)",
    )

    cm = confusion_matrix_table(adata, "class", cluster_key, normalize="index")
    cm.to_csv(os.path.join(cfg.out_folder, f"Confusion_{safe_name}.csv"))
    save_confusion_heatmap(
        cm,
        os.path.join(cfg.out_folder, f"Heatmap_Confusion_{safe_name}.png"),
        title=f"Confusion (row-normalized): {method_name}",
    )

    runtime_s = float(time.time() - t0)
    return {
        "Method": method_name,
        "EmbedKey": embed_key,
        "LeidenRes": float(used_res),
        "Runtime_s": float(runtime_s),
        **metrics,
    }


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    sc.settings.verbosity = 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    set_all_seeds(CFG.seed)
    ensure_out_dir(CFG.out_folder)

    # dependency warnings
    if hm is None:
        logging.warning("harmonypy not installed -> Harmony and GenoDR(Harmony) will be skipped.")
    if bbknn is None:
        logging.warning("bbknn not installed -> BBKNN will be skipped.")
    if gp is None:
        logging.warning("genomap.genoDR not available -> GenoDR refinements will be skipped.")
    if phate is None:
        logging.warning("phate not installed -> PHATE and GenoDR(PHATE) will be skipped. (pip install phate)")

    logging.info("Loading pancreas base data (canonical preprocessing)...")
    adata_base = load_base_data(CFG)

    logging.info(
        f"Base data: n_obs={adata_base.n_obs}, n_vars={adata_base.n_vars}, "
        f"batches={adata_base.obs['batch'].nunique()}, classes={adata_base.obs['class'].nunique()}"
    )
    log_preprocess_fingerprint(adata_base, CFG, tag="BASE")

    # Full method list (as requested)
    methods_all = [
        "Scanorama",
        "Harmony",
        "BBKNN",
        "Seurat",
        "Online_iNMF",
        "DiffusionMap",
        "PHATE",
        "GenoDR(Scanorama)",
        "GenoDR(Harmony)",
        "GenoDR(Seurat)",
        "GenoDR(Online_iNMF)",
        "GenoDR(DiffusionMap)",
        "GenoDR(PHATE)",
    ]

    # Filter based on deps and presence of external embeddings
    filtered: List[str] = []
    for m in methods_all:
        if m in ("Harmony", "GenoDR(Harmony)") and hm is None:
            continue
        if m == "BBKNN" and bbknn is None:
            continue
        if m in ("PHATE", "GenoDR(PHATE)") and phate is None:
            continue
        if m.startswith("GenoDR(") and gp is None:
            continue

        # External embeddings required for Seurat / Online_iNMF (+ their GenoDR versions)
        if m in ("Seurat", "GenoDR(Seurat)"):
            if not os.path.exists(os.path.join(CFG.out_folder, CFG.seurat_csv_name)):
                logging.warning(f"Missing Seurat CSV -> skipping {m}")
                continue
        if m in ("Online_iNMF", "GenoDR(Online_iNMF)"):
            if not os.path.exists(os.path.join(CFG.out_folder, CFG.online_inmf_csv_name)):
                logging.warning(f"Missing Online_iNMF CSV -> skipping {m}")
                continue

        filtered.append(m)

    logging.info(f"Methods to run: {filtered}")

    results: List[Dict[str, object]] = []
    for m in filtered:
        logging.info(f"Running method: {m}")
        try:
            res = run_one_method(m, adata_base, CFG)
            results.append(res)
            logging.info(
                f"Done {m}: ARI={res['ARI']:.3f}, RI={res['RI']:.3f}, "
                f"Sil={res['Silhouette']:.3f}, iLISI={res['iLISI']:.3f}, "
                f"LeidenRes={res['LeidenRes']:.2f}, Runtime_s={res['Runtime_s']:.1f}"
            )
        except Exception as e:
            logging.exception(f"FAILED method {m}: {e}")
            results.append({"Method": m, "Error": str(e)})

    df = pd.DataFrame(results)

    out_csv = os.path.join(CFG.out_folder, "benchmark_results_pancreas.csv")
    df.to_csv(out_csv, index=False)
    logging.info(f"Saved results: {out_csv}")

    if {"ARI", "Silhouette", "iLISI"}.issubset(df.columns):
        ranked = df.dropna(subset=["ARI"]).sort_values(["ARI", "Silhouette"], ascending=False)
        ranked.to_csv(os.path.join(CFG.out_folder, "benchmark_ranked_pancreas.csv"), index=False)
        logging.info("Saved: benchmark_ranked_pancreas.csv")

    # Paper-facing table (NO RI)
    table_cols = ["Method", "ARI", "Silhouette", "iLISI"]
    df_table = df.dropna(subset=["ARI"])[table_cols].copy()

    out_table_csv = os.path.join(CFG.out_folder, "table_pancreas_main.csv")
    df_table.to_csv(out_table_csv, index=False)
    logging.info(f"Saved paper-facing table CSV (no RI): {out_table_csv}")

    out_table_tex = os.path.join(CFG.out_folder, "table_pancreas_main.tex")
    latex_str = df_to_latex_table_main_metrics(
        df_table=df_table,
        caption="Comparative Performance of Integration Methods on the Pancreas Dataset",
        label="tab:pancreas_results_main",
    )
    with open(out_table_tex, "w", encoding="utf-8") as f:
        f.write(latex_str)
    logging.info(f"Saved paper-facing LaTeX table (no RI): {out_table_tex}")

    out_bar_png = os.path.join(CFG.out_folder, "fig_pancreas_bars_main_metrics.png")
    try:
        save_grouped_bar_main_metrics(df_table, out_bar_png)
        logging.info(f"Saved bar chart (no RI): {out_bar_png}")
    except Exception as e:
        logging.warning(f"Could not save bar chart: {e}")


if __name__ == "__main__":
    main()
