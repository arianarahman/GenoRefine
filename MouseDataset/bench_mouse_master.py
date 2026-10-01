# Purpose: Canonical end-to-end benchmark for a single mouse atlas AnnData (.h5ad), specifically
#          Tabula Muris Senis droplet+facs processed annotations.
# Author: Ariana Rahman (Arizona State University)

"""
FILE: bench_mouse_master_v7.py
-------------------------------------------------------------------------------
Date: April 2026 (v8)

PURPOSE
    Canonical end-to-end benchmark for a single mouse atlas AnnData (.h5ad),
    specifically Tabula Muris Senis droplet+facs processed annotations.

    v8 changes:
      - use the cleaner canonical-HVG workflow: subset genes first, then normalize/log
      - preserve the raw counts layer for strict Seurat-v3-style HVG handling
      - keep the Harmony orientation fix across harmonypy versions
      - keep all GenoDR methods enabled and log detailed failure diagnostics
      - compute canonical HVG MD5 in the same way as make_mouse_canonical_hvg_v3.py

    Methods compared (depending on availability):
      - Baselines: Scanorama, Harmony, BBKNN, (optional) Seurat, Online_iNMF,py
                   DiffusionMap, PHATE
      - GenoDR refinements: GenoDR(Scanorama), GenoDR(Harmony),
                            (optional) GenoDR(Seurat), GenoDR(Online_iNMF),
                            GenoDR(DiffusionMap), GenoDR(PHATE)

CONSISTENCY GUARANTEE
    - Same preprocessing: normalize_total(1e4) + log1p
    - Same HVG selection: seurat_v3, top 2000 genes
    - Same scaling + PCA baseline: 50 PCs
    - Same clustering: Leiden tuned to match #classes (if labels exist)
    - Same metrics: ARI, RI, Silhouette, iLISI (+ optional kBET)

INPUTS
    Required:
      - CFG.data_file: mouse AnnData (.h5ad)

    Optional (external embeddings saved as CSV):
      - CFG.seurat_csv: rows indexed by cell IDs (adata.obs_names), cols are dims
      - CFG.online_inmf_csv: same format

OUTPUTS (written to CFG.out_folder)
    - benchmark_results_mouse.csv
    - benchmark_ranked_mouse.csv
    - table_mouse_main.csv
    - table_mouse_main.tex
    - fig_mouse_bars_main_metrics.png
    - UMAP_mouse_<Method>_class.png
    - UMAP_mouse_<Method>_batch.png
    - Confusion_mouse_<Method>.csv
    - Heatmap_Confusion_mouse_<Method>.png
-------------------------------------------------------------------------------
"""

import os
import time
import logging
import hashlib
import re
from importlib import metadata as importlib_metadata
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scanpy as sc
import anndata as ad
import scanorama

from sklearn.metrics import adjusted_rand_score, rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

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

# Optional kBET (rarely installed)
try:
    from scib.metrics import kBET as scib_kbet  # type: ignore
except Exception:
    scib_kbet = None


# -----------------------------
# Config
# -----------------------------
@dataclass
class MouseConfig:
    # Update this path to your local file
    data_file: str = r"./Dataset/tabula-muris_sub50k_combined.h5ad"
    out_folder: str = "./Benchmark_Mouse_Out"
    seed: int = 0

    # Frozen canonical HVG artifact
    use_canonical_hvg: bool = True
    canonical_hvg_file: str = "./Benchmark_Mouse_Out/mouse_hvg_canonical.txt"
    canonical_hvg_md5_file: Optional[str] = "./Benchmark_Mouse_Out/mouse_hvg_canonical.md5"
    require_canonical_hvg: bool = True

    # Optional external embeddings (exported from R, rows aligned to adata.obs_names)
    seurat_csv: str = "./Benchmark_Mouse_Out/X_seurat_mouse.csv"
    online_inmf_csv: str = "./Benchmark_Mouse_Out/X_online_inmf_mouse.csv"

    # Fixed keys for Tabula Muris Senis droplet (from your inspection)
    batch_key_fixed: Optional[str] = "dataset"  # fixed for mouse benchmark
    label_key_fixed: Optional[str] = None  # auto (match pancreas/HIE)

    # Fallback candidates (only if fixed missing)
    batch_key_candidates: Tuple[str, ...] = (
        "dataset", "study", "tech", "technology", "platform", "protocol", "method", "batch", "Batch",
        "sample", "Sample", "library", "run", "lane",
        "mouse.id", "mouse_id", "mouse", "animal", "animal_id",
        "donor", "individual", "subject",
    )
    label_key_candidates: Tuple[str, ...] = (
        "class", "celltype", "cell_type", "cell_type_label", "cell_ontology_class", "annotation", "label",
        "Cluster", "cluster", "celltypes", "CellType",
    )

    # Preprocess
    do_normalize_log1p: bool = True
    do_hvg: bool = True
    hvg: int = 2000
    hvg_flavor: str = "seurat_v3"

    # Fair feature selection across batches
    hvg_union_per_batch: bool = True
    hvg_per_batch: int = 2000
    hvg_union_mode: str = "union"  # union or intersection
    do_scale: bool = True
    max_scale_value: float = 10.0
    n_pcs: int = 50
    n_neighbors: int = 15

    neighbor_metric: str = "cosine"
    neighbor_n_dims: int = 30
    strict_use_rep: bool = True
    do_kmeans_eval: bool = True
    # Leiden tuning
    leiden_target_n_clusters_if_labels_exist: bool = True
    leiden_resolution_default: float = 0.5
    leiden_res_grid: Tuple[float, ...] = tuple(np.round(np.linspace(0.2, 1.6, 15), 2))

    # iLISI
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

    # Optional kBET
    run_kbet: bool = False

    # Export embeddings
    export_embeddings: bool = True
    embeddings_folder_name: str = "Embeddings"


CFG = MouseConfig()


# -----------------------------
# Utilities
# -----------------------------
def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def md5_of_list(items: List[str]) -> str:
    """
    Compute an MD5 over a list of strings using LF separators and no trailing
    newline, matching make_mouse_canonical_hvg_v2.py.
    """
    payload = "\n".join(map(str, items)).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_canonical_hvg_list(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        if "gene" in df.columns:
            genes = df["gene"].astype(str).tolist()
        else:
            genes = df.iloc[:, 0].astype(str).tolist()
    else:
        with open(path, "r", encoding="utf-8") as f:
            genes = [line.strip() for line in f if line.strip()]

    out: List[str] = []
    seen = set()
    for g in genes:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def safe_pkg_version(pkg_name: str, module_name: Optional[str] = None) -> str:
    try:
        return importlib_metadata.version(pkg_name)
    except Exception:
        try:
            if module_name is None:
                module_name = pkg_name
            mod = __import__(module_name)
            return str(getattr(mod, "__version__", "unknown"))
        except Exception:
            return "unavailable"


def get_genodr_env_info() -> Dict[str, str]:
    return {
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "tensorflow": safe_pkg_version("tensorflow"),
        "keras": safe_pkg_version("keras"),
        "genomap": safe_pkg_version("genomap"),
        "scanpy": safe_pkg_version("scanpy"),
        "anndata": safe_pkg_version("anndata"),
        "numpy": safe_pkg_version("numpy"),
        "harmonypy": safe_pkg_version("harmonypy"),
    }


_GENODR_PATCH_DONE = False


def ensure_genodr_keras3_compatibility() -> Dict[str, Any]:
    """
    Patch genomap's old Sequential-based CAE to a Functional API version that
    works with current Keras/TensorFlow, and patch save_weights() file naming
    for Keras 3.
    """
    global _GENODR_PATCH_DONE

    info: Dict[str, Any] = {
        "patch_attempted": True,
        "patch_applied": False,
        "save_weights_patch_applied": False,
        "details": "",
    }

    if gp is None:
        info["details"] = "genomap.genoDR import unavailable"
        return info

    if _GENODR_PATCH_DONE:
        info["patch_applied"] = True
        info["save_weights_patch_applied"] = True
        info["details"] = "already patched in current process"
        return info

    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, Input
        from tensorflow.keras.models import Model
        import genomap.utils.ConvDEC as convdec

        def patched_CAE(input_shape=(28, 28, 1), filters=[32, 64, 128, 10]):
            if input_shape[0] % 8 == 0:
                pad3 = "same"
            else:
                pad3 = "valid"

            x_in = Input(shape=input_shape, name="genodr_input")
            x = Conv2D(filters[0], 15, strides=2, padding="same", activation="relu", name="conv1")(x_in)
            x = Conv2D(filters[1], 5, strides=2, padding="same", activation="relu", name="conv2")(x)
            x = Conv2D(filters[2], 3, strides=2, padding=pad3, activation="relu", name="conv3")(x)
            x = Flatten(name="flatten")(x)
            embedding = Dense(units=filters[3], name="embedding")(x)
            x = Dense(
                units=filters[2] * int(input_shape[0] / 8) * int(input_shape[0] / 8),
                activation="relu",
                name="dense_decode",
            )(embedding)
            x = Reshape(
                (int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2]),
                name="reshape_decode",
            )(x)
            x = Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation="relu", name="deconv3")(x)
            x = Conv2DTranspose(filters[0], 5, strides=2, padding="same", activation="relu", name="deconv2")(x)
            x_out = Conv2DTranspose(input_shape[2], 5, strides=2, padding="same", name="deconv1")(x)

            autoencoder = Model(inputs=x_in, outputs=x_out, name="genodr_cae")
            encoder = Model(inputs=x_in, outputs=embedding, name="genodr_encoder")
            return autoencoder, encoder

        convdec.CAE = patched_CAE
        convdec._mouse_v7_keras3_patch = True
        info["patch_applied"] = True

        if not getattr(tf.keras.Model, "_mouse_v7_save_weights_patch", False):
            orig_save_weights = tf.keras.Model.save_weights

            def compat_save_weights(self, filepath, *args, **kwargs):
                fp = str(filepath)
                patched = fp
                if fp.endswith(".h5") and not fp.endswith(".weights.h5"):
                    patched = fp[:-3] + ".weights.h5"
                    logging.warning(
                        "Patched Keras save_weights filepath for GenoDR compatibility: %s -> %s",
                        fp, patched,
                    )
                return orig_save_weights(self, patched, *args, **kwargs)

            tf.keras.Model.save_weights = compat_save_weights
            tf.keras.Model._mouse_v7_save_weights_patch = True
            info["save_weights_patch_applied"] = True
        else:
            info["save_weights_patch_applied"] = True

        _GENODR_PATCH_DONE = True
        info["details"] = "Applied functional-API CAE patch and save_weights patch for Keras/TensorFlow compatibility"
        return info

    except Exception as e:
        info["details"] = f"Patch attempt failed: {type(e).__name__}: {e}"
        return info


def log_preprocess_fingerprint(adata: ad.AnnData, tag: str = "BASE") -> None:
    var_names = [str(v) for v in adata.var_names.tolist()]
    obs_names = [str(o) for o in adata.obs_names.tolist()]
    msg = [
        f"[{tag}] n_obs={adata.n_obs}, n_vars={adata.n_vars}",
        f"[{tag}] var_md5={md5_of_list(var_names)}",
        f"[{tag}] obs_md5={md5_of_list(obs_names)}",
    ]
    if "X_pca" in adata.obsm:
        Xp = np.asarray(adata.obsm["X_pca"])
        msg.append(f"[{tag}] X_pca shape={Xp.shape}, var(PC1)={float(np.var(Xp[:, 0])):.6f}")
    if "canonical_hvg_md5" in adata.uns:
        msg.append(f"[{tag}] canonical_hvg_md5={adata.uns['canonical_hvg_md5']}")
    if "canonical_hvg_file" in adata.uns:
        msg.append(f"[{tag}] canonical_hvg_file={adata.uns['canonical_hvg_file']}")
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


def infer_key_from_obs(adata: ad.AnnData, candidates: Tuple[str, ...]) -> Optional[str]:
    for k in candidates:
        if k in adata.obs.columns:
            return k
    return None


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

def sanitize_method_name(name: str) -> str:
    s = name.replace(" ", "_")
    s = s.replace("(", "_").replace(")", "")
    s = s.replace("/", "_").replace("__", "_")
    return s


def export_embedding_csv(adata: ad.AnnData, embed_key: str, out_path: str) -> None:
    X = np.asarray(adata.obsm[embed_key])
    df = pd.DataFrame(X, index=adata.obs_names.astype(str))
    df.to_csv(out_path)
    logging.info(f"Exported embedding '{embed_key}' -> {out_path}  shape={X.shape}")


def _normalize_obs_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"-1$", "", s)
    s = re.sub(r"\.1$", "", s)
    s = s.replace(".", "-")
    return s


def load_embedding_csv_align(adata_in: ad.AnnData, csv_path: str, key: str) -> None:
    """Load an external embedding CSV and align rows to adata.obs_names.

    Matches pancreas/HIE benchmark behavior:
    - Attempt 1: direct alignment to obs_names
    - Attempt 2: normalized-name alignment (handles '-1', '.' vs '-', etc.)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, index_col=0)

    # Attempt 1: direct match
    df1 = df.reindex(adata_in.obs_names)
    match_rate_1 = float(df1.notna().all(axis=1).mean())
    if match_rate_1 >= 0.999:
        adata_in.obsm[key] = _nan_guard(df1.to_numpy(), f"CSV[{key}]")
        return

    # Attempt 2: normalized match
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
    match_rate_2 = float(sum(ok) / max(1, len(ok)))

    if match_rate_2 >= 0.999:
        df2 = df.loc[aligned_rows].copy()
        df2.index = adata_in.obs_names
        adata_in.obsm[key] = _nan_guard(df2.to_numpy(), f"CSV[{key}]")
        return

    missing_ex = [adata_in.obs_names[i] for i, r in enumerate(aligned_rows) if r is None][:10]
    raise ValueError(
        f"Embedding CSV alignment failed.\n"
        f"CSV: {csv_path}\n"
        f"Attempt1 (direct) match={match_rate_1*100:.2f}%\n"
        f"Attempt2 (normalized) match={match_rate_2*100:.2f}% missing ex={missing_ex}\n"
        f"Fix: ensure CSV rownames match adata.obs_names (or match after removing '-1' and replacing '.' with '-')."
    )

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


def save_umap_png(adata: ad.AnnData, color_key: str, out_png: str, title: str) -> None:
    sc.pl.umap(adata, color=color_key, show=False, title=title)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def confusion_matrix_table(
    adata: ad.AnnData,
    class_key: str,
    cluster_key: str = "leiden",
    normalize: str = "index",
) -> pd.DataFrame:
    return pd.crosstab(
        adata.obs[class_key].astype(str),
        adata.obs[cluster_key].astype(str),
        normalize=normalize,
    )


def save_confusion_heatmap(cm: pd.DataFrame, out_png: str, title: str) -> None:
    import seaborn as sns
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, cmap="YlGnBu")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# -----------------------------
# Data loading + preprocessing
# -----------------------------

def load_mouse_base(cfg: MouseConfig) -> Tuple[ad.AnnData, str, Optional[str]]:
    if not os.path.exists(cfg.data_file):
        raise FileNotFoundError(f"Could not find: {cfg.data_file}")

    adata = sc.read_h5ad(cfg.data_file)
    adata.var_names_make_unique()

    # Ensure a 'dataset' field exists
    if "dataset" not in adata.obs.columns:
        if "method" in adata.obs.columns:
            adata.obs["dataset"] = adata.obs["method"].astype(str)
        elif "technology" in adata.obs.columns:
            adata.obs["dataset"] = adata.obs["technology"].astype(str)
        elif "tech" in adata.obs.columns:
            adata.obs["dataset"] = adata.obs["tech"].astype(str)
        else:
            adata.obs["dataset"] = "mouse_dataset"

    if cfg.batch_key_fixed is not None and cfg.batch_key_fixed in adata.obs.columns:
        batch_key = cfg.batch_key_fixed
    else:
        batch_key = infer_key_from_obs(adata, cfg.batch_key_candidates)

    if cfg.label_key_fixed is not None and cfg.label_key_fixed in adata.obs.columns:
        label_key = cfg.label_key_fixed
    else:
        label_key = infer_key_from_obs(adata, cfg.label_key_candidates)

    if batch_key is None:
        raise ValueError(
            f"No batch key found. Expected '{cfg.batch_key_fixed}' or candidates {cfg.batch_key_candidates}. "
            f"Available keys: {list(adata.obs.columns)[:40]}"
        )
    if label_key is None:
        logging.warning(
            f"No label key found. Expected '{cfg.label_key_fixed}' or candidates {cfg.label_key_candidates}. "
            "ARI/RI and class UMAP will be skipped."
        )

    a = adata.copy()

    # normalize metadata names early so downstream preprocessing can use them
    a.obs["batch"] = a.obs[batch_key].astype(str).astype("category")
    if label_key is not None:
        a.obs["class"] = a.obs[label_key].astype(str).astype("category")
    class_msg = f"obs['{label_key}'] -> obs['class']" if label_key is not None else "None"
    logging.info(f"Resolved keys: batch=obs['{batch_key}'] -> obs['batch']; class={class_msg}")

    has_counts_layer = "counts" in a.layers.keys()
    counts_layer_name: Optional[str] = "counts" if has_counts_layer else None

    # ------------------------------------------------------------------
    # CLEAN ORDER:
    #   1) choose/subset genes first (canonical HVG file or fallback HVG logic)
    #   2) then normalize/log the matrix used downstream
    # This preserves the raw counts layer for strict Seurat-v3-style HVG handling.
    # ------------------------------------------------------------------
    used_canonical = False
    if cfg.use_canonical_hvg:
        if os.path.exists(cfg.canonical_hvg_file):
            canonical_genes = load_canonical_hvg_list(cfg.canonical_hvg_file)
            canonical_md5 = md5_of_list(canonical_genes)

            if cfg.canonical_hvg_md5_file is not None and os.path.exists(cfg.canonical_hvg_md5_file):
                expected_md5 = read_text_file(cfg.canonical_hvg_md5_file)
                if expected_md5 != canonical_md5:
                    logging.warning(
                        "Canonical HVG MD5 mismatch: file=%s computed=%s",
                        expected_md5, canonical_md5,
                    )
                else:
                    logging.info("Canonical HVG MD5 verified: %s", canonical_md5)

            present = [g for g in canonical_genes if g in a.var_names]
            missing_n = len(canonical_genes) - len(present)

            if len(present) < min(500, int(cfg.hvg)):
                raise ValueError(
                    f"Too few canonical HVGs present in dataset after matching var_names: "
                    f"{len(present)} / {len(canonical_genes)}"
                )

            if missing_n > 0:
                logging.warning(
                    "Canonical HVG file had %d genes not found in adata.var_names; using %d matched genes.",
                    missing_n, len(present),
                )
            else:
                logging.info("Loaded canonical HVGs from %s (%d genes).", cfg.canonical_hvg_file, len(present))

            a = a[:, present].copy()
            a.var["highly_variable"] = True
            a.uns["canonical_hvg_file"] = cfg.canonical_hvg_file
            a.uns["canonical_hvg_md5"] = canonical_md5
            a.uns["canonical_hvg_n_genes"] = int(len(present))
            used_canonical = True
        elif cfg.require_canonical_hvg:
            raise FileNotFoundError(
                f"Canonical HVG file not found: {cfg.canonical_hvg_file}\n"
                "Either create it first with make_mouse_canonical_hvg_v3.py "
                "or set require_canonical_hvg=False."
            )
        else:
            logging.warning(
                "Canonical HVG file not found: %s. Falling back to internal HVG selection.",
                cfg.canonical_hvg_file,
            )

    # Fallback to internal HVG logic only if canonical HVG is not being used
    if cfg.do_hvg and not used_canonical:
        flavor = str(cfg.hvg_flavor)
        layer_for_hvg = counts_layer_name if counts_layer_name is not None else None
        if flavor == "seurat_v3" and layer_for_hvg is None:
            logging.warning("HVG seurat_v3 requested but no counts layer -> fallback to cell_ranger.")
            flavor = "cell_ranger"

        if cfg.hvg_union_per_batch:
            hvgs = _select_hvgs_per_batch_union(
                a,
                batch_key="batch",
                n_top=int(cfg.hvg_per_batch),
                flavor=flavor,
                layer=layer_for_hvg,
                mode=str(cfg.hvg_union_mode),
            )
            a = a[:, hvgs].copy()
        else:
            sc.pp.highly_variable_genes(
                a,
                n_top_genes=int(cfg.hvg),
                flavor=flavor,
                subset=True,
                layer=layer_for_hvg,
            )

    # ------------------------------------------------------------------
    # Normalize/log AFTER feature selection.
    # If raw counts are available, leave the raw counts layer untouched and
    # create downstream a.X from that raw layer.
    # ------------------------------------------------------------------
    if cfg.do_normalize_log1p:
        if counts_layer_name is not None and counts_layer_name in a.layers.keys():
            logging.info("Preprocess: normalize_total/log1p on a.X copied from raw layer='counts' after gene subsetting.")
            a.X = a.layers[counts_layer_name].copy()
            sc.pp.normalize_total(a, target_sum=1e4)
            sc.pp.log1p(a)
        else:
            logging.info("Preprocess: normalize_total/log1p on a.X after gene subsetting (no counts layer).")
            sc.pp.normalize_total(a, target_sum=1e4)
            sc.pp.log1p(a)
    else:
        if counts_layer_name is not None and counts_layer_name in a.layers.keys():
            a.X = a.layers[counts_layer_name].copy()

    # scale + PCA
    if cfg.do_scale:
        sc.pp.scale(a, max_value=float(cfg.max_scale_value))
    sc.tl.pca(a, n_comps=int(cfg.n_pcs), random_state=int(cfg.seed))

    # external embeddings (optional)
    try:
        load_embedding_csv_align(a, cfg.seurat_csv, key="X_seurat")
    except Exception as e:
        logging.warning(f"Seurat CSV not loaded (method will be unavailable): {e}")

    try:
        load_embedding_csv_align(a, cfg.online_inmf_csv, key="X_online_inmf")
    except Exception as e:
        logging.warning(f"Online iNMF CSV not loaded (method will be unavailable): {e}")

    return a, "batch", ("class" if label_key is not None else None)



# -----------------------------
# Method runners
# -----------------------------
def run_scanorama(adata_in: ad.AnnData) -> Tuple[ad.AnnData, str]:
    a = adata_in.copy()
    batches = list(a.obs["batch"].cat.categories)
    adatas_list = [a[a.obs["batch"] == b].copy() for b in batches]
    corrected = scanorama.correct_scanpy(adatas_list, return_dimred=True)
    out = ad.concat(corrected, join="outer")
    out.obs["batch"] = out.obs["batch"].astype("category")
    if "class" in out.obs:
        out.obs["class"] = out.obs["class"].astype("category")
    return out, "X_scanorama"


def run_harmony(adata_in: ad.AnnData) -> Tuple[ad.AnnData, str]:
    if hm is None:
        raise RuntimeError("harmonypy not installed. Install it with: pip install harmonypy")

    a = adata_in.copy()
    X_pca = _nan_guard(np.asarray(a.obsm["X_pca"]), "X_pca_pre_harmony")
    ho = hm.run_harmony(X_pca, a.obs, "batch")

    Z_raw = getattr(ho, "Z_corr", None)
    if Z_raw is None:
        raise RuntimeError("Harmony returned no Z_corr embedding.")

    Z = _nan_guard(np.asarray(Z_raw), "X_harmony_raw")
    logging.info("Harmony raw output type=%s shape=%s", type(Z_raw).__name__, getattr(Z, "shape", None))

    if Z.ndim != 2:
        raise ValueError(f"Harmony output is not 2D after conversion to ndarray: shape={Z.shape}")

    # harmonypy versions differ. Prefer the orientation that matches n_obs x n_pcs.
    if Z.shape[0] == a.n_obs:
        X_harmony = Z
    elif Z.shape[1] == a.n_obs:
        X_harmony = Z.T
    elif Z.shape == X_pca.shape:
        X_harmony = Z
    elif Z.T.shape == X_pca.shape:
        X_harmony = Z.T
    else:
        raise ValueError(
            f"Harmony output shape {Z.shape} does not match n_obs={a.n_obs} "
            f"or X_pca shape={X_pca.shape}. Check harmonypy version/orientation."
        )

    a.obsm["X_harmony"] = _nan_guard(np.asarray(X_harmony), "X_harmony")
    logging.info("Stored Harmony embedding with shape=%s", a.obsm["X_harmony"].shape)
    return a, "X_harmony"


def run_bbknn(adata_in: ad.AnnData) -> Tuple[ad.AnnData, str]:
    if bbknn is None:
        raise RuntimeError("bbknn not installed")

    a = adata_in.copy()
    bbknn.bbknn(a, batch_key="batch")  # uses X_pca
    return a, "X_pca"


def run_external_embedding(adata_in: ad.AnnData, embed_key: str) -> Tuple[ad.AnnData, str]:
    a = adata_in.copy()
    if embed_key not in a.obsm:
        raise KeyError(f"Missing '{embed_key}' in adata.obsm. Check CSV creation/path.")
    return a, embed_key


def run_diffmap(adata_in: ad.AnnData, cfg: MouseConfig) -> Tuple[ad.AnnData, str]:
    a = adata_in.copy()
    sc.pp.neighbors(a, use_rep=_rep_for_neighbors(a, "X_pca", int(cfg.neighbor_n_dims)), n_neighbors=int(cfg.n_neighbors), metric=str(cfg.neighbor_metric))
    sc.tl.diffmap(a, n_comps=int(cfg.diffmap_n_comps))
    if cfg.diffmap_embed_key not in a.obsm:
        raise KeyError("Expected diffusion map embedding missing at obsm['X_diffmap'].")
    a.obsm[cfg.diffmap_embed_key] = _nan_guard(np.asarray(a.obsm[cfg.diffmap_embed_key]), "X_diffmap")
    return a, cfg.diffmap_embed_key


def run_phate(adata_in: ad.AnnData, cfg: MouseConfig) -> Tuple[ad.AnnData, str]:
    if phate is None:
        raise RuntimeError("phate not installed (pip install phate)")

    a = adata_in.copy()
    X = _nan_guard(np.asarray(a.obsm["X_pca"]), "X_pca_for_PHATE")

    ph = phate.PHATE(
        n_components=int(cfg.phate_n_components),
        knn=int(cfg.phate_knn),
        decay=int(cfg.phate_decay),
        t=cfg.phate_t,
        random_state=int(cfg.seed),
        verbose=False,
    )
    Z = ph.fit_transform(X)
    a.obsm[cfg.phate_embed_key] = _nan_guard(np.asarray(Z), "X_phate")
    return a, cfg.phate_embed_key


def run_genodr_on_key(adata_in: ad.AnnData, embed_key: str, out_key: str, cfg: MouseConfig) -> Tuple[ad.AnnData, str]:
    if gp is None:
        env = get_genodr_env_info()
        raise RuntimeError(
            "genomap.genoDR not available.\n"
            f"Environment: {env}\n"
            "Install the package/environment needed for GenoDR before rerunning."
        )

    a = adata_in.copy()
    if embed_key not in a.obsm:
        raise KeyError(
            f"Missing embedding '{embed_key}' in obsm for GenoDR. "
            f"Available embeddings: {list(a.obsm.keys())}"
        )

    X = _nan_guard(np.asarray(a.obsm[embed_key]), name=f"genoDR_in[{embed_key}]")

    if X.ndim != 2:
        raise ValueError(f"GenoDR input embedding must be 2D; got shape={X.shape}")

    if "class" in a.obs:
        n_clusters = max(2, int(a.obs["class"].nunique()))
    else:
        n_clusters = 10

    patch_info = ensure_genodr_keras3_compatibility()
    env = get_genodr_env_info()
    logging.info(
        "GenoDR setup for %s: input_shape=%s, dtype=%s, n_clusters=%d, n_dim=%d, col=%d, row=%d, env=%s, patch=%s",
        embed_key, X.shape, X.dtype, n_clusters, int(cfg.genodr_dim), int(cfg.genodr_col), int(cfg.genodr_row), env, patch_info
    )

    try:
        Z = gp.genoDR(
            X,
            n_dim=int(cfg.genodr_dim),
            n_clusters=int(n_clusters),
            colNum=int(cfg.genodr_col),
            rowNum=int(cfg.genodr_row),
        )
    except Exception as e:
        diagnostic = [
            "GenoDR execution failed.",
            f"embed_key={embed_key}",
            f"input_shape={tuple(X.shape)}",
            f"input_dtype={X.dtype}",
            f"n_clusters={n_clusters}",
            f"n_dim={int(cfg.genodr_dim)}",
            f"colNum={int(cfg.genodr_col)}",
            f"rowNum={int(cfg.genodr_row)}",
            f"env={env}",
            f"patch_info={patch_info}",
            f"original_exception={type(e).__name__}: {e}",
        ]
        msg = "\n".join(diagnostic)
        logging.error(msg)
        raise RuntimeError(msg) from e

    Z = _nan_guard(np.asarray(Z), name=f"genoDR_out[{out_key}]")
    if Z.ndim != 2:
        raise ValueError(f"GenoDR output must be 2D; got shape={Z.shape}")
    if Z.shape[0] != a.n_obs:
        raise ValueError(f"GenoDR output row count mismatch: got {Z.shape[0]}, expected {a.n_obs}")
    a.obsm[out_key] = Z
    return a, out_key


# -----------------------------
# Metrics + clustering
# -----------------------------
def cluster_and_score(
    adata: ad.AnnData,
    rep_for_neighbors: Optional[str],
    cfg: MouseConfig,
) -> Tuple[float, Dict[str, Any]]:
    cluster_key = "leiden"
    has_labels = "class" in adata.obs

    if rep_for_neighbors is None:
        # existing graph (BBKNN)
        if has_labels and cfg.leiden_target_n_clusters_if_labels_exist:
            used_res = tune_leiden_resolution_existing_graph(
                adata=adata,
                target_k=int(adata.obs["class"].nunique()),
                res_grid=list(cfg.leiden_res_grid),
                key_added=cluster_key,
                random_state=int(cfg.seed),
            )
        else:
            used_res = float(cfg.leiden_resolution_default)
            sc.tl.leiden(adata, resolution=used_res, key_added=cluster_key, random_state=int(cfg.seed))
    else:
        if has_labels and cfg.leiden_target_n_clusters_if_labels_exist:
            used_res = tune_leiden_resolution_to_target(
                adata=adata,
                use_rep=rep_for_neighbors,
                target_k=int(adata.obs["class"].nunique()),
                n_neighbors=int(cfg.n_neighbors),
                res_grid=list(cfg.leiden_res_grid),
                key_added=cluster_key,
                random_state=int(cfg.seed),
            )
        else:
            sc.pp.neighbors(adata, use_rep=_rep_for_neighbors(adata, rep_for_neighbors, int(cfg.neighbor_n_dims)), n_neighbors=int(cfg.n_neighbors), metric=str(cfg.neighbor_metric))
            used_res = float(cfg.leiden_resolution_default)
            sc.tl.leiden(adata, resolution=used_res, key_added=cluster_key, random_state=int(cfg.seed))

    y_pred = adata.obs[cluster_key].astype(str).to_numpy()
    out: Dict[str, Any] = {"LeidenRes": float(used_res)}

    if has_labels:
        y_true = adata.obs["class"].astype(str).to_numpy()
        out["ARI"] = float(adjusted_rand_score(y_true, y_pred))
        out["RI"] = float(rand_score(y_true, y_pred))
    else:
        out["ARI"] = float("nan")
        out["RI"] = float("nan")

    embed_for_sil = rep_for_neighbors if rep_for_neighbors is not None else "X_pca"
    try:
        X = np.asarray(adata.obsm[embed_for_sil])
        out["Silhouette"] = float(silhouette_score(X, y_pred))
    except Exception:
        out["Silhouette"] = float("nan")

    try:
        X = np.asarray(adata.obsm[embed_for_sil])
        out["iLISI"] = float(compute_lisi_python(X, adata.obs["batch"].astype(str).to_numpy(), k=int(cfg.lisi_k)))
    except Exception:
        out["iLISI"] = float("nan")

    if cfg.run_kbet and scib_kbet is not None:
        try:
            kb = scib_kbet(adata, batch_key="batch", label_key="class" if has_labels else None, embed=embed_for_sil)
            out["kBET"] = float(kb) if kb is not None else float("nan")
        except Exception:
            out["kBET"] = float("nan")
    else:
        out["kBET"] = float("nan")

    return float(used_res), out


# -----------------------------
# One method runner
# -----------------------------
def run_one_method(method_name: str, adata_base: ad.AnnData, cfg: MouseConfig) -> Dict[str, Any]:
    t0 = time.time()

    if method_name == "Scanorama":
        adata, embed_key = run_scanorama(adata_base)
        rep_for_neighbors = embed_key
        if cfg.strict_use_rep:
                _ = _rep_for_neighbors(adata, rep_for_neighbors, int(cfg.neighbor_n_dims))

    elif method_name == "Harmony":
        adata, embed_key = run_harmony(adata_base)
        rep_for_neighbors = embed_key
        if cfg.strict_use_rep:
                _ = _rep_for_neighbors(adata, rep_for_neighbors, int(cfg.neighbor_n_dims))

    elif method_name == "BBKNN":
        adata, embed_key = run_bbknn(adata_base)
        rep_for_neighbors = None  # graph exists; silhouette uses X_pca

    elif method_name == "Seurat":
        adata, embed_key = run_external_embedding(adata_base, "X_seurat")
        rep_for_neighbors = embed_key
        if cfg.strict_use_rep:
                _ = _rep_for_neighbors(adata, rep_for_neighbors, int(cfg.neighbor_n_dims))

    elif method_name == "Online_iNMF":
        adata, embed_key = run_external_embedding(adata_base, "X_online_inmf")
        rep_for_neighbors = embed_key
        if cfg.strict_use_rep:
                _ = _rep_for_neighbors(adata, rep_for_neighbors, int(cfg.neighbor_n_dims))

    elif method_name == "DiffusionMap":
        adata, embed_key = run_diffmap(adata_base, cfg)
        rep_for_neighbors = embed_key
        if cfg.strict_use_rep:
                _ = _rep_for_neighbors(adata, rep_for_neighbors, int(cfg.neighbor_n_dims))

    elif method_name == "PHATE":
        adata, embed_key = run_phate(adata_base, cfg)
        rep_for_neighbors = embed_key
        if cfg.strict_use_rep:
                _ = _rep_for_neighbors(adata, rep_for_neighbors, int(cfg.neighbor_n_dims))

    elif method_name == "GenoDR(Scanorama)":
        ad_scan, scan_key = run_scanorama(adata_base)
        adata, embed_key = run_genodr_on_key(ad_scan, scan_key, "X_genodr_scanorama", cfg)
        rep_for_neighbors = embed_key
        if cfg.strict_use_rep:
                _ = _rep_for_neighbors(adata, rep_for_neighbors, int(cfg.neighbor_n_dims))

    elif method_name == "GenoDR(Harmony)":
        ad_har, har_key = run_harmony(adata_base)
        adata, embed_key = run_genodr_on_key(ad_har, har_key, "X_genodr_harmony", cfg)
        rep_for_neighbors = embed_key
        if cfg.strict_use_rep:
                _ = _rep_for_neighbors(adata, rep_for_neighbors, int(cfg.neighbor_n_dims))

    elif method_name == "GenoDR(Seurat)":
        ad_seu, seu_key = run_external_embedding(adata_base, "X_seurat")
        adata, embed_key = run_genodr_on_key(ad_seu, seu_key, "X_genodr_seurat", cfg)
        rep_for_neighbors = embed_key
        if cfg.strict_use_rep:
                _ = _rep_for_neighbors(adata, rep_for_neighbors, int(cfg.neighbor_n_dims))

    elif method_name == "GenoDR(Online_iNMF)":
        ad_inmf, inmf_key = run_external_embedding(adata_base, "X_online_inmf")
        adata, embed_key = run_genodr_on_key(ad_inmf, inmf_key, "X_genodr_online_inmf", cfg)
        rep_for_neighbors = embed_key
        if cfg.strict_use_rep:
                _ = _rep_for_neighbors(adata, rep_for_neighbors, int(cfg.neighbor_n_dims))

    elif method_name == "GenoDR(DiffusionMap)":
        ad_dm, dm_key = run_diffmap(adata_base, cfg)
        adata, embed_key = run_genodr_on_key(ad_dm, dm_key, "X_genodr_diffmap", cfg)
        rep_for_neighbors = embed_key
        if cfg.strict_use_rep:
                _ = _rep_for_neighbors(adata, rep_for_neighbors, int(cfg.neighbor_n_dims))

    elif method_name == "GenoDR(PHATE)":
        ad_ph, ph_key = run_phate(adata_base, cfg)
        adata, embed_key = run_genodr_on_key(ad_ph, ph_key, "X_genodr_phate", cfg)
        rep_for_neighbors = embed_key
        if cfg.strict_use_rep:
                _ = _rep_for_neighbors(adata, rep_for_neighbors, int(cfg.neighbor_n_dims))

    else:
        raise ValueError(f"Unknown method: {method_name}")

    used_res, metrics = cluster_and_score(adata, rep_for_neighbors=rep_for_neighbors, cfg=cfg)

    # UMAP
    sc.tl.umap(adata, random_state=int(cfg.seed))

    safe = sanitize_method_name(method_name)
    save_umap_png(
        adata,
        "batch",
        os.path.join(cfg.out_folder, f"UMAP_mouse_{safe}_batch.png"),
        title=f"Mouse: {method_name} (batch)",
    )
    if "class" in adata.obs:
        save_umap_png(
            adata,
            "class",
            os.path.join(cfg.out_folder, f"UMAP_mouse_{safe}_class.png"),
            title=f"Mouse: {method_name} (class)",
        )

        cm = confusion_matrix_table(adata, class_key="class", cluster_key="leiden", normalize="index")
        cm.to_csv(os.path.join(cfg.out_folder, f"Confusion_mouse_{safe}.csv"))
        save_confusion_heatmap(
            cm,
            os.path.join(cfg.out_folder, f"Heatmap_Confusion_mouse_{safe}.png"),
            title=f"Mouse confusion (row-normalized): {method_name}",
        )

    # Export embeddings
    if cfg.export_embeddings:
        emb_dir = os.path.join(cfg.out_folder, cfg.embeddings_folder_name)
        ensure_out_dir(emb_dir)
        out_emb = os.path.join(emb_dir, f"X_{safe}.csv")
        export_embedding_csv(adata, embed_key if method_name != "BBKNN" else "X_pca", out_emb)

    runtime_s = float(time.time() - t0)

    return {
        "Method": method_name,
        "Status": "OK",
        "EmbedKey": embed_key if method_name != "BBKNN" else "X_pca",
        "Runtime_s": runtime_s,
        **metrics,
    }




def build_failure_result(method_name: str, exc: Exception) -> Dict[str, Any]:
    return {
        "Method": method_name,
        "Status": "FAILED",
        "Runtime_s": float("nan"),
        "ARI": float("nan"),
        "RI": float("nan"),
        "Silhouette": float("nan"),
        "iLISI": float("nan"),
        "kBET": float("nan"),
        "ErrorType": type(exc).__name__,
        "Error": str(exc),
    }


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
        rows.append(f"{r['Method']} & {r['ARI']} & {r['Silhouette']} & {r['iLISI']} \\")
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

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    sc.settings.verbosity = 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    set_all_seeds(int(CFG.seed))
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
    if CFG.run_kbet and scib_kbet is None:
        logging.warning("kBET requested but scib.metrics.kBET is not available -> kBET will be NaN.")

    logging.info("Loading mouse dataset + canonical preprocessing ...")
    adata_base, _, class_key_norm = load_mouse_base(CFG)

    logging.info(
        f"Base data: n_obs={adata_base.n_obs}, n_vars={adata_base.n_vars}, "
        f"batches={adata_base.obs['batch'].nunique()}, "
        f"labels={'yes' if class_key_norm is not None else 'no'}"
    )
    log_preprocess_fingerprint(adata_base, tag="BASE")

    methods = [
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

    # Filter methods based on deps and available embeddings.
    # Intentionally do NOT skip GenoDR methods due to missing dependencies; let
    # them run and fail with detailed diagnostics.
    filtered: List[str] = []
    for m in methods:
        if m == "Harmony" and hm is None:
            continue
        if m == "BBKNN" and bbknn is None:
            continue
        if m == "PHATE" and phate is None:
            continue
        if m == "Seurat" and "X_seurat" not in adata_base.obsm:
            continue
        if m == "Online_iNMF" and "X_online_inmf" not in adata_base.obsm:
            continue
        filtered.append(m)

    logging.info(f"Methods to run: {filtered}")

    results: List[Dict[str, Any]] = []
    for m in filtered:
        logging.info(f"Running method: {m}")
        try:
            res = run_one_method(m, adata_base, CFG)
            results.append(res)
            logging.info(
                f"Done {m}: ARI={res['ARI']:.3f}, RI={res['RI']:.3f}, "
                f"Sil={res['Silhouette']:.3f}, iLISI={res['iLISI']:.3f}, "
                f"Runtime_s={res['Runtime_s']:.1f}"
            )
        except Exception as e:
            logging.exception(f"FAILED method {m}: {e}")
            results.append(build_failure_result(m, e))

    # ------------------------------------------------------------------
    # Core benchmark outputs (Pancreas/HIE-style naming)
    # ------------------------------------------------------------------
    df_results = pd.DataFrame(results)
    out_results_csv = os.path.join(CFG.out_folder, "benchmark_results_mouse.csv")
    df_results.to_csv(out_results_csv, index=False)

    # Ranked view
    sort_cols = [c for c in ["ARI", "Silhouette", "iLISI"] if c in df_results.columns]
    df_ranked = df_results.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
    out_ranked_csv = os.path.join(CFG.out_folder, "benchmark_ranked_mouse.csv")
    df_ranked.to_csv(out_ranked_csv, index=False)

    # Paper-facing main table (no RI)
    main_cols = [c for c in ["Method", "ARI", "Silhouette", "iLISI"] if c in df_ranked.columns]
    df_table = df_ranked[main_cols].copy()
    out_table_csv = os.path.join(CFG.out_folder, "table_mouse_main.csv")
    df_table.to_csv(out_table_csv, index=False)

    out_table_tex = os.path.join(CFG.out_folder, "table_mouse_main.tex")
    try:
        def _fmt(x):
            try:
                return f"{float(x):.3f}"
            except Exception:
                return str(x)

        with open(out_table_tex, "w", encoding="utf-8") as f:
            f.write(
                df_table.to_latex(
                    index=False,
                    escape=False,
                    formatters={c: _fmt for c in df_table.columns if c != "Method"},
                )
            )
    except Exception as e:
        logging.warning(f"Could not write LaTeX table: {e}")

    # Grouped bar chart (ARI / Silhouette / iLISI)
    try:
        plot_metrics = [m for m in ["ARI", "Silhouette", "iLISI"] if m in df_ranked.columns]
        if len(plot_metrics) >= 2:
            df_plot = df_ranked[["Method"] + plot_metrics].copy()
            x = np.arange(len(df_plot["Method"]))
            width = 0.8 / len(plot_metrics)

            fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(df_plot)), 4.5))
            for j, met in enumerate(plot_metrics):
                vals = pd.to_numeric(df_plot[met], errors="coerce").fillna(0.0).values
                ax.bar(x + (j - (len(plot_metrics) - 1) / 2) * width, vals, width, label=met)

            ax.set_xticks(x)
            ax.set_xticklabels(df_plot["Method"], rotation=45, ha="right")
            ax.set_ylabel("Score")
            ax.set_title("Mouse benchmark: main metrics")
            ax.legend(frameon=False)
            fig.tight_layout()

            out_bar_png = os.path.join(CFG.out_folder, "fig_mouse_bars_main_metrics.png")
            fig.savefig(out_bar_png, dpi=300)
            plt.close(fig)
    except Exception as e:
        logging.warning(f"Could not write bar figure: {e}")

    logging.info(f"Saved results: {out_results_csv}")
    logging.info(f"Saved ranked:  {out_ranked_csv}")
    logging.info(f"Saved table:   {out_table_csv}")
    logging.info(f"Saved tex:     {out_table_tex}")


if __name__ == "__main__":
    main()