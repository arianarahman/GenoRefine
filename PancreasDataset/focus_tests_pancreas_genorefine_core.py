"""
FILE: focus_tests_pancreas_canonical_v7.py
-------------------------------------------------------------------------------
Author: Ariana Rahman
Affiliation: Arizona State University / Stanford University
Date: February 2026
-------------------------------------------------------------------------------

PURPOSE
    Focus tests to support the paper narrative that GenoRefine functions as a
    post-integration manifold stabilizer rather than a label-maximizer.

    v7 is aligned to bench_pancreas_multi_master_v13.py:
      - Human-readable batch names: Baron, Muraro, Segerstolpe, Wang, Xin
      - obs_names: Cell-<i>-Batch-<BatchName>
      - Backward-compatible alignment for R-exported embeddings (legacy rownames)
      - Restricts focus tests to Scanorama, Harmony, Seurat, and Online iNMF plus GenoRefine refinements
      - Adds paper-ready UMAP exports per method (class + batch + leiden)
      - Adds run metadata fingerprint CSV
      - Hardens Scanorama metadata re-attachment with match-rate checks
      - Uses frozen pancreas_hvg_canonical.txt from make_pancreas_canonical_hvg.py
      - Applies clean canonical workflow: raw load -> subset canonical HVGs -> normalize/log -> scale/PCA
      - Adds Harmony orientation helper, non-interactive plotting, and in-process GenoRefine backend/Keras patch

FOCUS TESTS (MAIN-PAPER READY)
    Test 0) Cluster ↔ Class Confusion Matrices (counts + normalized variants)
    Test 1) Resolution Sensitivity (Leiden stability vs resolution)
    Test 2) Intra-Class Compactness (mean/median within-class distances)
    Test 3) iLISI–Silhouette Trade-off (from canonical benchmark results CSV)
    Test 4) kNN Stability vs Label Purity (baseline to GenoRefine pairs only)
    Test 5) Topology Preservation (between-class centroid geometry)
    Test 6) Per-cell Silhouette Width (optionally sampled)

INPUTS
    Required:
      - ./Dataset/*.mat and classLabel.mat
    Optional:
      - ./Benchmark_Out/benchmark_results_pancreas.csv (Test 3)
      - ./Benchmark_Out/X_seurat_pancreas.csv (Seurat embedding; optional)
      - ./Benchmark_Out/X_online_inmf_pancreas.csv (Online iNMF embedding; optional)

OUTPUTS (written to CFG.out_folder, default: ./Benchmark_Out)
    Run metadata:
      - focus_run_metadata.csv

    UMAPs (per method, if enabled):
      - UMAP_<Method>_class.png
      - UMAP_<Method>_batch.png
      - UMAP_<Method>_leiden.png

    Test 0:
      - confusion_counts_<Method>.csv
      - confusion_classnorm_<Method>.csv
      - confusion_clusternorm_<Method>.csv

    Test 1:
      - focus_resolution_sensitivity.csv
      - fig_resolution_ari.png
      - fig_resolution_nclusters.png

    Test 2:
      - focus_intraclass_compactness.csv
      - fig_intraclass_compactness_mean.png
      - fig_intraclass_compactness_heatmap_median.png

    Test 3:
      - fig_ilisi_sil_tradeoff.png

    Test 4:
      - focus_knn_stability_purity.csv
      - focus_knn_stability_purity_summary.csv
      - fig_knn_stability_vs_purity.png

    Test 5:
      - focus_topology_preservation.csv
      - fig_topology_preservation.png

    Test 6:
      - focus_silhouette_per_cell.csv
      - focus_silhouette_summary.csv
      - fig_silhouette_boxplot.png
      - fig_silhouette_negative_fraction.png

DEPENDENCIES
    scanpy, anndata, scanorama, scikit-learn
    Optional: harmonypy, genomap, scipy (spearmanr)
-------------------------------------------------------------------------------
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")  # non-interactive backend for Windows/headless batch runs

import sys
import time
import logging
import hashlib
import gc
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Callable, Any

import numpy as np
import pandas as pd
import scipy.io as sio

import matplotlib
matplotlib.use("Agg", force=True)

import anndata as ad
import scanpy as sc
import scanorama

from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_samples

# Optional deps
try:
    import harmonypy as hm  # type: ignore
except Exception:
    hm = None

try:
    import genomap.genoDR as gp  # type: ignore
except Exception:
    gp = None


# Optional stats dep
try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


# -----------------------------
# Config (match master_v9 + canonical preprocessing)
# -----------------------------
@dataclass
class FocusConfig:
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

    # Canonical preprocessing
    hvg: int = 2000
    hvg_flavor: str = "seurat_v3"

    # Frozen canonical HVG file produced by make_pancreas_canonical_hvg.py.
    # This mirrors bench_pancreas_multi_master_v13.py.
    use_canonical_hvg_file: bool = True
    canonical_hvg_file: str = "pancreas_hvg_canonical.txt"
    canonical_hvg_md5_file: str = "pancreas_hvg_canonical.md5"
    check_canonical_hvg_md5_file: bool = True
    require_exact_canonical_hvg_match: bool = True

    n_pcs: int = 50
    n_neighbors: int = 15
    neighbor_metric: str = "cosine"
    neighbor_n_dims: int = 30

    # Leiden sweep (stability test)
    leiden_resolutions: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0)

    # Leiden tuning grid for the confusion-matrix clustering (target #classes)
    leiden_target_n_clusters: bool = True
    leiden_res_grid: Tuple[float, ...] = tuple(np.round(np.linspace(0.2, 1.6, 15), 2))
    leiden_resolution_default: float = 0.5

    # Compactness sampling
    compactness_max_cells_per_class: int = 500
    compactness_random_state: int = 0

    # Test 4: kNN stability/purity
    knn_k: int = 30

    # GenoRefine params
    genodr_dim: int = 32
    genodr_col: int = 33
    genodr_row: int = 33

    # Silhouette per-cell (scalability guard)
    silhouette_max_cells: int = 5000
    silhouette_random_state: int = 0

    # Optional external embeddings (R exports) — support v9 + legacy rownames
    seurat_csv_name: str = "X_seurat_pancreas.csv"
    online_inmf_csv_name: str = "X_online_inmf_pancreas.csv"


    # Outputs
    save_umaps: bool = True
    save_umap_colors: Tuple[str, ...] = ("class", "batch", "leiden")
    umap_random_state: int = 0

    # Metadata attachment strictness
    scanorama_meta_match_min_rate: float = 0.999

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


CFG = FocusConfig()


# -----------------------------
# Utilities
# -----------------------------
def set_all_seeds(seed: int):
    np.random.seed(seed)


def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize_method_name(name: str) -> str:
    s = str(name).replace(" ", "_")
    s = s.replace("(", "_").replace(")", "")
    s = s.replace("/", "_").replace("__", "_")
    return s


def _nan_guard(X: np.ndarray, name: str) -> np.ndarray:
    X = np.asarray(X)
    if not np.isfinite(X).all():
        logging.warning(f"{name}: NaN/Inf found; replacing with 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


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


def _get_pkg_version(name: str) -> str:
    try:
        import importlib.metadata as im
        return im.version(name)
    except Exception:
        try:
            import pkg_resources
            return pkg_resources.get_distribution(name).version
        except Exception:
            return "NA"




def md5_of_list(items: List[str]) -> str:
    """MD5 over a list using one LF after each item, matching pancreas master_v13."""
    h = hashlib.md5()
    for s in items:
        h.update(str(s).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def md5_no_trailing_newline(items: List[str]) -> str:
    """MD5 over a list using LF separators and no trailing newline; useful for R-style checks."""
    payload = "\n".join(map(str, items)).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def get_genodr_env_info() -> Dict[str, str]:
    return {
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "tensorflow": _get_pkg_version("tensorflow"),
        "keras": _get_pkg_version("keras"),
        "genomap": _get_pkg_version("genomap"),
        "scanpy": _get_pkg_version("scanpy"),
        "anndata": _get_pkg_version("anndata"),
        "numpy": _get_pkg_version("numpy"),
        "harmonypy": _get_pkg_version("harmonypy") if hm is not None else "NA",
    }


_GENODR_PATCH_DONE = False


def ensure_genodr_keras3_compatibility() -> Dict[str, Any]:
    """Patch the GenoRefine backend for current Keras behavior inside this process.

    The backend builds a Sequential CAE and then queries ``model.input``
    before the model has been called. Current Keras also requires
    ``save_weights`` paths to end in ``.weights.h5``. This in-process patch
    avoids editing site-packages and makes the focus tests behave like the
    current master/robustness benchmark scripts.
    """
    global _GENODR_PATCH_DONE

    info: Dict[str, Any] = {
        "patch_attempted": True,
        "patch_applied": False,
        "save_weights_patch_applied": False,
        "clustering_layer_patch_applied": False,
        "details": "",
    }

    if gp is None:
        info["details"] = "genomap.genoDR import unavailable"
        return info

    if _GENODR_PATCH_DONE:
        info.update({
            "patch_applied": True,
            "save_weights_patch_applied": True,
            "clustering_layer_patch_applied": True,
            "details": "already patched in current process",
        })
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
        try:
            convdec.ConvDEC.__init__.__globals__["CAE"] = patched_CAE
        except Exception:
            pass
        try:
            import genomap.utils.ConvIDEC as convidec
            if hasattr(convidec, "CAE"):
                convidec.CAE = patched_CAE
            if hasattr(convidec, "ConvDEC"):
                convidec.ConvDEC.__init__.__globals__["CAE"] = patched_CAE
        except Exception:
            pass
        info["patch_applied"] = True

        # Keras 3 / TensorShape compatibility for GenoMap's ClusteringLayer.
        try:
            import genomap.utils.FcDEC as fcdec
            cls = fcdec.ClusteringLayer
            if not getattr(cls, "_focus_v7_inputshape_patch", False):
                orig_build = cls.build

                class _ShapeCompat:
                    def __init__(self, shape):
                        self._shape = list(shape)
                    def as_list(self):
                        return list(self._shape)

                def compat_build(self, input_shape):
                    if not hasattr(input_shape, "as_list"):
                        shape = input_shape
                        if isinstance(shape, (list, tuple)) and len(shape) > 0 and isinstance(shape[0], (list, tuple)):
                            shape = shape[0]
                        input_shape = _ShapeCompat(shape)
                    return orig_build(self, input_shape)

                cls.build = compat_build
                cls._focus_v7_inputshape_patch = True
            info["clustering_layer_patch_applied"] = True
        except Exception as e:
            info["clustering_layer_patch_applied"] = False
            info["details"] += f" ClusteringLayer patch failed: {type(e).__name__}: {e}."

        def patch_save_weights(model_cls: Any, flag_name: str) -> bool:
            if getattr(model_cls, flag_name, False):
                return True
            original = model_cls.save_weights

            def compat_save_weights(self, filepath, *args, **kwargs):
                fp = str(filepath)
                patched = fp
                if fp.endswith(".h5") and not fp.endswith(".weights.h5"):
                    patched = fp[:-3] + ".weights.h5"
                    logging.warning("Patched Keras save_weights filepath for GenoRefine: %s -> %s", fp, patched)
                return original(self, patched, *args, **kwargs)

            model_cls.save_weights = compat_save_weights
            setattr(model_cls, flag_name, True)
            return True

        ok_tf = patch_save_weights(tf.keras.Model, "_focus_v7_save_weights_patch")
        ok_keras = True
        try:
            import keras
            ok_keras = patch_save_weights(keras.Model, "_focus_v7_save_weights_patch")
        except Exception:
            ok_keras = True

        info["save_weights_patch_applied"] = bool(ok_tf and ok_keras)
        _GENODR_PATCH_DONE = True
        if not info["details"]:
            info["details"] = "Applied GenoRefine Functional-API CAE, ClusteringLayer input_shape, and save_weights patches"
        return info

    except Exception as e:
        info["details"] = f"Patch attempt failed: {type(e).__name__}: {e}"
        return info


def _rep_for_neighbors(adata: ad.AnnData, rep_key: str, n_dims: int) -> str:
    """Return an obsm key safe for neighbors, slicing to n_dims when needed."""
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


def _harmony_to_cells_by_dim(Z: np.ndarray, n_obs: int, name: str = "X_harmony") -> np.ndarray:
    Z = np.asarray(Z)
    if Z.ndim != 2:
        raise ValueError(f"{name}: expected a 2D array, got shape {Z.shape}")
    if Z.shape[0] == n_obs:
        return Z
    if Z.shape[1] == n_obs:
        logging.info("%s: transposing Harmony output from shape %s to (%d, %d)", name, tuple(Z.shape), n_obs, Z.shape[0])
        return Z.T
    raise ValueError(f"{name}: cannot align Harmony output shape {Z.shape} to n_obs={n_obs}")

def write_run_metadata(cfg: FocusConfig, methods: List[str]) -> None:
    rows = [{
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.replace("\n", " "),
        "seed": int(cfg.seed),
        "hvg": int(cfg.hvg),
        "hvg_flavor": str(cfg.hvg_flavor),
        "canonical_hvg_file": str(getattr(cfg, "canonical_hvg_file", "NA")),
        "canonical_hvg_required": bool(getattr(cfg, "use_canonical_hvg_file", False)),
        "n_pcs": int(cfg.n_pcs),
        "n_neighbors": int(cfg.n_neighbors),
        "methods_run": ";".join(methods),
        "scanpy": _get_pkg_version("scanpy"),
        "anndata": _get_pkg_version("anndata"),
        "scanorama": _get_pkg_version("scanorama"),
        "scikit_learn": _get_pkg_version("scikit-learn"),
        "harmonypy": _get_pkg_version("harmonypy") if hm is not None else "NA",
        "genomap": _get_pkg_version("genomap") if gp is not None else "NA",
        "scipy": _get_pkg_version("scipy"),
        "numpy": _get_pkg_version("numpy"),
        "pandas": _get_pkg_version("pandas"),
    }]
    out_csv = os.path.join(cfg.out_folder, "focus_run_metadata.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logging.info(f"Saved: {out_csv}")


# -----------------------------
# Data loading (match master_v9 naming + legacy support)
# -----------------------------
def load_base_data(cfg: FocusConfig) -> ad.AnnData:
    """Load raw pancreas batches without normalization or HVG selection.

    This mirrors bench_pancreas_multi_master_v13.py. The canonical HVG
    subsetting and normalize/log/scale/PCA steps are performed in
    preprocess_with_frozen_hvgs().
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
    adata.var_names = pd.Index(adata.var_names.astype(str))

    # Should already be unique due to batch name; avoid make_unique unless required.
    if adata.obs_names.has_duplicates:
        logging.warning("obs_names had duplicates; making unique.")
        adata.obs_names_make_unique()

    # Attach class labels
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

    # Stable batch category order (matches cfg.data_files order)
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


def preprocess_with_frozen_hvgs(adata_raw: ad.AnnData, cfg: FocusConfig) -> ad.AnnData:
    """Subset raw matrix to canonical HVGs first, then normalize/log/scale/PCA."""
    txt_path = os.path.join(cfg.out_folder, cfg.canonical_hvg_file)
    requested_hvgs = _read_canonical_hvg_file(txt_path)

    present = set(adata_raw.var_names.astype(str).tolist())
    matched_hvgs = [g for g in requested_hvgs if g in present]
    missing_count = int(len(requested_hvgs) - len(matched_hvgs))

    if len(matched_hvgs) == 0:
        raise ValueError(f"No canonical HVGs matched adata.var_names. File={txt_path}")
    if cfg.require_exact_canonical_hvg_match and missing_count > 0:
        raise ValueError(
            f"Canonical HVG mismatch: requested={len(requested_hvgs)}, "
            f"matched={len(matched_hvgs)}, missing={missing_count}"
        )

    md5_master = md5_of_list(matched_hvgs)
    md5_r_style = md5_no_trailing_newline(matched_hvgs)
    logging.info(
        "Using frozen canonical HVG file: %s | requested=%d matched=%d missing=%d md5_master=%s md5_r_style=%s",
        txt_path, len(requested_hvgs), len(matched_hvgs), missing_count, md5_master, md5_r_style
    )

    md5_path = os.path.join(cfg.out_folder, cfg.canonical_hvg_md5_file)
    if cfg.check_canonical_hvg_md5_file and os.path.exists(md5_path):
        expected = open(md5_path, "r", encoding="utf-8").read().strip().splitlines()[0].strip()
        if expected and expected not in {md5_master, md5_r_style}:
            logging.warning(
                "Canonical HVG md5 differs from %s. expected=%s master_style=%s r_style=%s",
                md5_path, expected, md5_master, md5_r_style,
            )

    adata = adata_raw[:, matched_hvgs].copy()
    logging.info("Applying clean canonical-HVG workflow: subset raw matrix first, then normalize/log.")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=int(cfg.n_pcs), random_state=int(cfg.seed))

    adata.uns["canonical_hvg_file"] = txt_path
    adata.uns["canonical_hvg_md5"] = md5_master
    adata.uns["canonical_hvg_n"] = int(len(matched_hvgs))
    return adata


def log_preprocess_fingerprint(adata: ad.AnnData, cfg: FocusConfig, tag: str = "BASE") -> None:
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


# -----------------------------
# Embedding runners (aligned to v9 behavior)
# -----------------------------
def _reattach_meta_strict(
    adata_out: ad.AnnData,
    meta: pd.DataFrame,
    min_match_rate: float,
    tag: str,
) -> None:
    # Ensure we can reindex meta by obs_names
    idx = adata_out.obs_names.astype(str)
    meta_idx = meta.index.astype(str)
    # Fast match-rate estimate
    match_rate = float(pd.Index(idx).isin(meta_idx).mean())
    if match_rate < min_match_rate:
        ex_missing = [x for x in idx if x not in set(meta_idx)][:10]
        raise ValueError(
            f"{tag}: metadata reattachment failed: match_rate={match_rate*100:.2f}% (<{min_match_rate*100:.2f}%). "
            f"Example missing obs_names: {ex_missing}\n"
            f"Likely cause: obs_names changed (e.g., make_unique) or Scanorama returned altered names."
        )
    adata_out.obs["batch"] = meta.loc[idx, "batch"].astype("category")
    adata_out.obs["class"] = meta.loc[idx, "class"].astype("category")
    if "batch_id" in meta.columns:
        adata_out.obs["batch_id"] = meta.loc[idx, "batch_id"].astype(int)


def run_scanorama(adata_in: ad.AnnData, cfg: FocusConfig) -> Tuple[ad.AnnData, str]:
    # Split by categorical batch order
    batches = []
    for b in list(adata_in.obs["batch"].cat.categories):
        batches.append(adata_in[adata_in.obs["batch"] == b].copy())

    # Keep meta from the input
    meta = adata_in.obs[["batch", "class", "batch_id"]].copy()
    meta.index = meta.index.astype(str)

    corrected = scanorama.correct_scanpy(batches, return_dimred=True)
    adata = ad.concat(corrected, join="outer")

    # Avoid make_unique unless necessary; but verify and reattach strictly.
    if adata.obs_names.has_duplicates:
        logging.warning("Scanorama output has duplicate obs_names; making unique (may break meta attach).")
        adata.obs_names_make_unique()

    _reattach_meta_strict(
        adata_out=adata,
        meta=meta,
        min_match_rate=float(cfg.scanorama_meta_match_min_rate),
        tag="Scanorama",
    )

    adata.obsm["X_scanorama"] = _nan_guard(np.asarray(adata.obsm["X_scanorama"]), "X_scanorama")
    return adata, "X_scanorama"


def run_harmony(adata_in: ad.AnnData, cfg: FocusConfig) -> Tuple[ad.AnnData, str]:
    if hm is None:
        raise RuntimeError("harmonypy not installed")
    adata = adata_in.copy()
    Xp = _nan_guard(np.asarray(adata.obsm["X_pca"]), "X_pca_for_harmony")
    ho = hm.run_harmony(Xp, adata.obs, "batch")
    Z = _harmony_to_cells_by_dim(ho.Z_corr, adata.n_obs, "X_harmony")
    adata.obsm["X_harmony"] = _nan_guard(Z, "X_harmony")
    return adata, "X_harmony"


def run_seurat_csv(adata_in: ad.AnnData, cfg: FocusConfig) -> Tuple[ad.AnnData, str]:
    csv_path = os.path.join(cfg.out_folder, cfg.seurat_csv_name)
    return _load_embedding_csv(adata_in, csv_path, "X_seurat")


def run_online_inmf_csv(adata_in: ad.AnnData, cfg: FocusConfig) -> Tuple[ad.AnnData, str]:
    csv_path = os.path.join(cfg.out_folder, cfg.online_inmf_csv_name)
    return _load_embedding_csv(adata_in, csv_path, "X_online_inmf")


def run_genodr(adata_in: ad.AnnData, base_embed_key: str, out_key: str, cfg: FocusConfig) -> ad.AnnData:
    if gp is None:
        raise RuntimeError("genomap.genoDR not available")
    if base_embed_key not in adata_in.obsm:
        raise KeyError(f"Missing base embedding '{base_embed_key}' for GenoRefine.")
    patch_info = ensure_genodr_keras3_compatibility()
    logging.info("GenoRefine backend compatibility patch: %s", patch_info.get("details", patch_info))
    X = _nan_guard(np.asarray(adata_in.obsm[base_embed_key]), f"genorefine_in[{base_embed_key}]")
    n_clusters = max(2, int(adata_in.obs["class"].nunique()))
    Z = gp.genoDR(
        X,
        n_dim=int(cfg.genodr_dim),
        n_clusters=int(n_clusters),
        colNum=int(cfg.genodr_col),
        rowNum=int(cfg.genodr_row),
    )
    adata_out = adata_in.copy()
    adata_out.obsm[out_key] = _nan_guard(np.asarray(Z), f"genorefine_out[{out_key}]")
    return adata_out


# -----------------------------
# Leiden tuner for confusion matrices
# -----------------------------
def tune_leiden_to_target_k(
    adata: ad.AnnData,
    rep_key: str,
    target_k: int,
    n_neighbors: int,
    res_grid: List[float],
    random_state: int,
    key_added: str = "leiden",
) -> float:
    safe_rep = _rep_for_neighbors(adata, rep_key, int(CFG.neighbor_n_dims))
    sc.pp.neighbors(adata, use_rep=safe_rep, n_neighbors=n_neighbors, metric=str(CFG.neighbor_metric))

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


def confusion_tables(
    adata: ad.AnnData,
    class_key: str = "class",
    cluster_key: str = "leiden",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = pd.crosstab(
        adata.obs[class_key].astype(str),
        adata.obs[cluster_key].astype(str),
        normalize=False,
    )
    class_norm = pd.crosstab(
        adata.obs[class_key].astype(str),
        adata.obs[cluster_key].astype(str),
        normalize="index",
    )
    cluster_norm = pd.crosstab(
        adata.obs[class_key].astype(str),
        adata.obs[cluster_key].astype(str),
        normalize="columns",
    )
    return counts, class_norm, cluster_norm


def run_confusion_matrix_export(
    adata: ad.AnnData,
    embed_key: str,
    method: str,
    cfg: FocusConfig,
) -> Dict[str, float]:
    a = adata.copy()
    target_k = int(a.obs["class"].nunique())

    if cfg.leiden_target_n_clusters:
        used_res = tune_leiden_to_target_k(
            a, rep_key=embed_key, target_k=target_k,
            n_neighbors=int(cfg.n_neighbors),
            res_grid=list(cfg.leiden_res_grid),
            random_state=int(cfg.seed),
            key_added="leiden",
        )
    else:
        safe_rep = _rep_for_neighbors(a, embed_key, int(cfg.neighbor_n_dims))
        sc.pp.neighbors(a, use_rep=safe_rep, n_neighbors=int(cfg.n_neighbors), metric=str(cfg.neighbor_metric))
        sc.tl.leiden(a, resolution=float(cfg.leiden_resolution_default), key_added="leiden", random_state=int(cfg.seed))
        used_res = float(cfg.leiden_resolution_default)

    counts, class_norm, cluster_norm = confusion_tables(a, class_key="class", cluster_key="leiden")

    safe_m = sanitize_method_name(method)
    counts.to_csv(os.path.join(cfg.out_folder, f"confusion_counts_{safe_m}.csv"))
    class_norm.to_csv(os.path.join(cfg.out_folder, f"confusion_classnorm_{safe_m}.csv"))
    cluster_norm.to_csv(os.path.join(cfg.out_folder, f"confusion_clusternorm_{safe_m}.csv"))

    return {
        "Method": method,
        "EmbedKey": embed_key,
        "LeidenRes_used": float(used_res),
        "NClusters": int(a.obs["leiden"].nunique()),
        "NClasses": int(target_k),
    }


# -----------------------------
# UMAP export helpers
# -----------------------------
def save_umap_png(adata: ad.AnnData, color_key: str, out_png: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    sc.pl.umap(adata, color=color_key, show=False, title=title)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close("all")
    gc.collect()


def export_umaps_for_method(
    adata: ad.AnnData,
    embed_key: str,
    method: str,
    cfg: FocusConfig,
) -> None:
    if not cfg.save_umaps:
        return
    a = adata.copy()
    safe_rep = _rep_for_neighbors(a, embed_key, int(cfg.neighbor_n_dims))
    sc.pp.neighbors(a, use_rep=safe_rep, n_neighbors=int(cfg.n_neighbors), metric=str(cfg.neighbor_metric))
    sc.tl.umap(a, random_state=int(cfg.umap_random_state))

    # If leiden not present, compute a reasonable one (target classes)
    if "leiden" not in a.obs:
        target_k = int(a.obs["class"].nunique())
        tune_leiden_to_target_k(
            a, rep_key=embed_key, target_k=target_k,
            n_neighbors=int(cfg.n_neighbors),
            res_grid=list(cfg.leiden_res_grid),
            random_state=int(cfg.seed),
            key_added="leiden",
        )

    safe_m = sanitize_method_name(method)
    for ck in cfg.save_umap_colors:
        if ck not in a.obs.columns and ck not in ("class", "batch"):
            continue
        out_png = os.path.join(cfg.out_folder, f"UMAP_{safe_m}_{ck}.png")
        save_umap_png(a, ck, out_png, title=f"{method} (UMAP; {ck})")


# -----------------------------
# Test 1: Resolution sensitivity
# -----------------------------
def run_resolution_sensitivity(
    adata: ad.AnnData,
    embed_key: str,
    method_name: str,
    pair_name: str,
    cfg: FocusConfig,
) -> pd.DataFrame:
    a = adata.copy()
    safe_rep = _rep_for_neighbors(a, embed_key, int(cfg.neighbor_n_dims))
    sc.pp.neighbors(a, use_rep=safe_rep, n_neighbors=cfg.n_neighbors, metric=str(cfg.neighbor_metric))

    rows = []
    y_true = a.obs["class"].astype(str).to_numpy()

    for r in cfg.leiden_resolutions:
        key_added = f"leiden_r_{str(r).replace('.', '_')}"
        sc.tl.leiden(a, resolution=float(r), key_added=key_added, random_state=cfg.seed)
        y_pred = a.obs[key_added].astype(str).to_numpy()
        ari = adjusted_rand_score(y_true, y_pred)
        nclust = int(a.obs[key_added].nunique())
        rows.append({
            "Pair": pair_name,
            "Method": method_name,
            "EmbedKey": embed_key,
            "Resolution": float(r),
            "NClusters": nclust,
            "ARI": float(ari),
        })

    return pd.DataFrame(rows)


def plot_resolution_curves(df: pd.DataFrame, out_ari_png: str, out_k_png: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for m in df["Method"].unique():
        sub = df[df["Method"] == m].sort_values("Resolution")
        plt.plot(sub["Resolution"], sub["ARI"], marker="o", label=m)
    plt.xlabel("Leiden resolution")
    plt.ylabel("ARI")
    plt.title("Resolution sensitivity: ARI vs Leiden resolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_ari_png, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    for m in df["Method"].unique():
        sub = df[df["Method"] == m].sort_values("Resolution")
        plt.plot(sub["Resolution"], sub["NClusters"], marker="o", label=m)
    plt.xlabel("Leiden resolution")
    plt.ylabel("Number of clusters")
    plt.title("Resolution sensitivity: #clusters vs Leiden resolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_k_png, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# Test 2: Intra-class compactness (mean + median)
# -----------------------------
def estimate_intraclass_compactness(X: np.ndarray, labels: np.ndarray, cfg: FocusConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.compactness_random_state)
    labels = labels.astype(str)
    uniq = np.unique(labels)

    rows = []
    per_class_means = []
    per_class_medians = []

    for c in uniq:
        idx = np.where(labels == c)[0]
        if idx.size < 3:
            continue

        take = min(idx.size, cfg.compactness_max_cells_per_class)
        sel = rng.choice(idx, size=take, replace=False) if idx.size > take else idx

        Xc = X[sel]
        D = pairwise_distances(Xc, metric="euclidean")
        triu = D[np.triu_indices_from(D, k=1)]

        mean_d = float(np.mean(triu)) if triu.size else float("nan")
        median_d = float(np.median(triu)) if triu.size else float("nan")
        var_d = float(np.var(triu)) if triu.size else float("nan")

        rows.append({
            "Class": c,
            "N_used": int(take),
            "MeanPairDist": mean_d,
            "MedianPairDist": median_d,
            "VarPairDist": var_d,
        })

        if not np.isnan(mean_d):
            per_class_means.append(mean_d)
        if not np.isnan(median_d):
            per_class_medians.append(median_d)

    df = pd.DataFrame(rows)

    agg = pd.DataFrame([{
        "Class": "__ALL__",
        "N_used": int(df["N_used"].sum()) if len(df) else 0,
        "MeanPairDist": float(np.mean(per_class_means)) if per_class_means else float("nan"),
        "MedianPairDist": float(np.mean(per_class_medians)) if per_class_medians else float("nan"),
        "VarPairDist": float(np.var(per_class_means)) if per_class_means else float("nan"),
    }])

    return pd.concat([df, agg], ignore_index=True)


def plot_compactness_bars_mean(df_all: pd.DataFrame, out_png: str) -> None:
    import matplotlib.pyplot as plt

    df_agg = df_all[df_all["Class"] == "__ALL__"].copy()
    df_agg = df_agg.sort_values("MeanPairDist", ascending=True)

    plt.figure(figsize=(9, 4))
    plt.bar(df_agg["Method"].astype(str), df_agg["MeanPairDist"].astype(float))
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Mean within-class pairwise distance (sampled)")
    plt.title("Intra-class compactness (mean; lower is tighter)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_compactness_heatmap(
    df_all: pd.DataFrame,
    value_col: str,
    out_png: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    df = df_all[df_all["Class"] != "__ALL__"].copy()
    if df.empty:
        return

    pivot = df.pivot_table(index="Class", columns="Method", values=value_col, aggfunc="mean")
    pivot = pivot.sort_index(axis=0)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    mat = pivot.to_numpy(dtype=float)
    if mat.size == 0 or np.all(np.isnan(mat)):
        return

    plt.figure(figsize=(10, max(4, 0.35 * pivot.shape[0])))
    im = plt.imshow(mat, aspect="auto")
    plt.colorbar(im, fraction=0.02, pad=0.02)
    plt.yticks(np.arange(pivot.shape[0]), pivot.index.astype(str))
    plt.xticks(np.arange(pivot.shape[1]), pivot.columns.astype(str), rotation=25, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# Test 3: iLISI vs Silhouette trade-off
# -----------------------------
def _clean_method_label_for_plot(label: str) -> str:
    """Normalize historical backend labels to manuscript-facing GenoRefine labels."""
    s = str(label)
    s = s.replace("Geno" + "DR", "GenoRefine")
    s = s.replace("Geno" + "Intig", "GenoRefine")
    s = s.replace("Online_iNMF", "Online iNMF")
    return s


def plot_ilisi_sil_tradeoff(results_csv: str, out_png: str) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(results_csv)
    df = df.dropna(subset=["Silhouette", "iLISI", "Method"]).copy()
    df["Method"] = df["Method"].map(_clean_method_label_for_plot)

    keep_methods = {
        "Scanorama", "GenoRefine(Scanorama)",
        "Harmony", "GenoRefine(Harmony)",
        "Seurat", "GenoRefine(Seurat)",
        "Online iNMF", "GenoRefine(Online iNMF)",
    }
    df = df[df["Method"].isin(keep_methods)].copy()
    if df.empty:
        logging.warning("No selected methods found in benchmark_results_pancreas.csv for iLISI-Silhouette trade-off plot.")
        return

    plt.figure(figsize=(7, 5))
    plt.scatter(df["iLISI"], df["Silhouette"])

    key_methods = keep_methods
    for _, r in df.iterrows():
        if str(r["Method"]) in key_methods:
            plt.text(float(r["iLISI"]) + 0.01, float(r["Silhouette"]) + 0.005, str(r["Method"]), fontsize=8)

    def arrow(m0: str, m1: str):
        if not ((df["Method"] == m0).any() and (df["Method"] == m1).any()):
            return
        a = df[df["Method"] == m0].iloc[0]
        b = df[df["Method"] == m1].iloc[0]
        plt.annotate(
            "",
            xy=(float(b["iLISI"]), float(b["Silhouette"])),
            xytext=(float(a["iLISI"]), float(a["Silhouette"])),
            arrowprops=dict(arrowstyle="->", lw=2),
        )

    arrow("Scanorama", "GenoRefine(Scanorama)")
    arrow("Harmony", "GenoRefine(Harmony)")
    arrow("Seurat", "GenoRefine(Seurat)")
    arrow("Online iNMF", "GenoRefine(Online iNMF)")

    plt.xlabel("iLISI (batch mixing)")
    plt.ylabel("Silhouette (cluster separation)")
    plt.title("Structure-mixing trade-off (arrows: baseline to GenoRefine)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

# -----------------------------
# Test 4: kNN stability vs label purity
# -----------------------------
def knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    X = np.asarray(X)
    n = X.shape[0]
    k_eff = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_eff)
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    idx = idx[:, 1:] if idx.shape[1] > 1 else idx
    return idx[:, : min(k, idx.shape[1])]


def knn_label_purity(idx: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    y_true = y_true.astype(str)
    neigh_labels = y_true[idx]
    same = (neigh_labels == y_true[:, None])
    return same.mean(axis=1).astype(float)


def knn_jaccard_overlap(idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    n = idx_a.shape[0]
    out = np.zeros(n, dtype=float)
    for i in range(n):
        sa = set(map(int, idx_a[i]))
        sb = set(map(int, idx_b[i]))
        inter = len(sa.intersection(sb))
        union = len(sa.union(sb))
        out[i] = (inter / union) if union > 0 else 0.0
    return out


def run_knn_stability_purity(
    X_base: np.ndarray,
    X_ref: np.ndarray,
    y_true: np.ndarray,
    pair_name: str,
    cfg: FocusConfig,
    obs_names: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    k = int(cfg.knn_k)
    idx_base = knn_indices(X_base, k=k)
    idx_ref = knn_indices(X_ref, k=k)

    purity_base = knn_label_purity(idx_base, y_true)
    purity_ref = knn_label_purity(idx_ref, y_true)
    stability = knn_jaccard_overlap(idx_base, idx_ref)

    df_cell = pd.DataFrame({
        "Pair": pair_name,
        "Cell": obs_names if obs_names is not None else np.arange(len(y_true)),
        "Class": y_true.astype(str),
        "Purity_Base": purity_base,
        "Purity_Ref": purity_ref,
        "Delta_Purity": purity_ref - purity_base,
        "Stability_Jaccard": stability,
    })

    rows = []
    for cls in sorted(df_cell["Class"].unique()):
        sub = df_cell[df_cell["Class"] == cls]
        rows.append({
            "Pair": pair_name,
            "Group": cls,
            "N": int(len(sub)),
            "Purity_Base_Mean": float(sub["Purity_Base"].mean()),
            "Purity_Ref_Mean": float(sub["Purity_Ref"].mean()),
            "Delta_Purity_Mean": float(sub["Delta_Purity"].mean()),
            "Stability_Mean": float(sub["Stability_Jaccard"].mean()),
        })
    rows.append({
        "Pair": pair_name,
        "Group": "__ALL__",
        "N": int(len(df_cell)),
        "Purity_Base_Mean": float(df_cell["Purity_Base"].mean()),
        "Purity_Ref_Mean": float(df_cell["Purity_Ref"].mean()),
        "Delta_Purity_Mean": float(df_cell["Delta_Purity"].mean()),
        "Stability_Mean": float(df_cell["Stability_Jaccard"].mean()),
    })

    df_sum = pd.DataFrame(rows)
    return df_cell, df_sum


def plot_knn_stability_vs_purity(df_cell: pd.DataFrame, out_png: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    plt.scatter(df_cell["Stability_Jaccard"].astype(float), df_cell["Delta_Purity"].astype(float), s=10)
    plt.axhline(0.0)
    plt.xlabel(f"Neighborhood stability (Jaccard overlap of kNN, k={CFG.knn_k})")
    plt.ylabel("Delta label purity (refined minus baseline)")
    plt.title("kNN stability vs label purity (baseline to GenoRefine)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# Test 5: Topology preservation (centroid geometry)
# -----------------------------
def _class_centroids(X: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    labels = labels.astype(str)
    classes = sorted(np.unique(labels))
    cents = []
    keep = []
    for c in classes:
        idx = np.where(labels == c)[0]
        if idx.size < 2:
            continue
        cents.append(np.mean(X[idx], axis=0))
        keep.append(c)
    return np.asarray(cents), keep


def topology_preservation_centroids(
    X_pca: np.ndarray,
    X_embed: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, int]:
    if spearmanr is None:
        return float("nan"), 0

    C_pca, classes = _class_centroids(X_pca, labels)
    C_emb, classes2 = _class_centroids(X_embed, labels)

    if len(classes) == 0 or len(classes2) == 0:
        return float("nan"), 0

    common = sorted(set(classes).intersection(set(classes2)))
    if len(common) < 3:
        return float("nan"), len(common)

    def select(C, cls_list):
        idx = [cls_list.index(c) for c in common]
        return C[idx]

    C_pca = select(C_pca, classes)
    C_emb = select(C_emb, classes2)

    D_pca = pairwise_distances(C_pca, metric="euclidean")
    D_emb = pairwise_distances(C_emb, metric="euclidean")

    v_pca = D_pca[np.triu_indices_from(D_pca, k=1)]
    v_emb = D_emb[np.triu_indices_from(D_emb, k=1)]

    if v_pca.size < 3:
        return float("nan"), len(common)

    rho = float(spearmanr(v_pca, v_emb).correlation)
    return rho, len(common)


def plot_topology_preservation(df: pd.DataFrame, out_png: str) -> None:
    import matplotlib.pyplot as plt

    if df.empty:
        return

    d = df.sort_values("TopoPreserve_Spearman_Centroids", ascending=False).copy()

    plt.figure(figsize=(10, 4))
    plt.bar(d["Method"].astype(str), d["TopoPreserve_Spearman_Centroids"].astype(float))
    plt.xticks(rotation=25, ha="right")
    plt.ylim(-0.05, 1.05)
    plt.ylabel("Spearman corr (PCA centroid distances vs embedding centroid distances)")
    plt.title("Topology preservation (between-class geometry)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# Test 6: Silhouette per cell (subsampled if needed)
# -----------------------------
def silhouette_per_cell(
    X: np.ndarray,
    labels: np.ndarray,
    method: str,
    cfg: FocusConfig,
    obs_names: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labels = labels.astype(str)
    n = X.shape[0]

    uniq = np.unique(labels)
    if uniq.size < 2 or n < 3:
        df_cell = pd.DataFrame(columns=["Method", "Cell", "Class", "Silhouette", "IsSampled"])
        df_sum = pd.DataFrame([{
            "Method": method,
            "N": int(n),
            "N_used": 0,
            "Sil_Mean": float("nan"),
            "Sil_Median": float("nan"),
            "Frac_Negative": float("nan"),
        }])
        return df_cell, df_sum

    rng = np.random.default_rng(cfg.silhouette_random_state)
    take = min(int(cfg.silhouette_max_cells), n)
    if take < n:
        sel = rng.choice(np.arange(n), size=take, replace=False)
        X_use = X[sel]
        y_use = labels[sel]
        cell_ids = [obs_names[i] for i in sel] if obs_names is not None else sel.astype(int)
        is_sampled = True
    else:
        X_use = X
        y_use = labels
        cell_ids = obs_names if obs_names is not None else np.arange(n)
        is_sampled = False

    X_use = np.nan_to_num(X_use)

    _, counts_use = np.unique(y_use, return_counts=True)
    if np.any(counts_use < 2):
        df_cell = pd.DataFrame({
            "Method": method,
            "Cell": cell_ids,
            "Class": y_use.astype(str),
            "Silhouette": np.nan,
            "IsSampled": bool(is_sampled),
        })
        df_sum = pd.DataFrame([{
            "Method": method,
            "N": int(n),
            "N_used": int(take),
            "Sil_Mean": float("nan"),
            "Sil_Median": float("nan"),
            "Frac_Negative": float("nan"),
        }])
        return df_cell, df_sum

    sil = silhouette_samples(X_use, y_use, metric="euclidean").astype(float)

    df_cell = pd.DataFrame({
        "Method": method,
        "Cell": cell_ids,
        "Class": y_use.astype(str),
        "Silhouette": sil,
        "IsSampled": bool(is_sampled),
    })

    frac_neg = float(np.mean(sil < 0.0)) if sil.size else float("nan")
    df_sum = pd.DataFrame([{
        "Method": method,
        "N": int(n),
        "N_used": int(take),
        "Sil_Mean": float(np.mean(sil)) if sil.size else float("nan"),
        "Sil_Median": float(np.median(sil)) if sil.size else float("nan"),
        "Frac_Negative": frac_neg,
    }])

    return df_cell, df_sum


def plot_silhouette_boxplot(df_cell: pd.DataFrame, out_png: str) -> None:
    import matplotlib.pyplot as plt

    if df_cell.empty:
        return

    methods = sorted(df_cell["Method"].unique())
    data = [df_cell[df_cell["Method"] == m]["Silhouette"].dropna().astype(float).to_numpy() for m in methods]

    plt.figure(figsize=(11, 4))
    plt.boxplot(data, labels=methods, showfliers=False)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Silhouette (per-cell; sampled if needed)")
    plt.title("Silhouette width distribution per method")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_silhouette_negative_fraction(df_sum: pd.DataFrame, out_png: str) -> None:
    import matplotlib.pyplot as plt

    if df_sum.empty or "Frac_Negative" not in df_sum.columns:
        return

    d = df_sum.sort_values("Frac_Negative", ascending=True).copy()

    plt.figure(figsize=(10, 4))
    plt.bar(d["Method"].astype(str), d["Frac_Negative"].astype(float))
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Fraction of cells with silhouette < 0")
    plt.title("Negative silhouette fraction (misplaced-cell tail)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# Method registry + dependency filtering
# -----------------------------
class MethodSpec:
    def __init__(
        self,
        name: str,
        runner: Callable[[ad.AnnData, FocusConfig], Tuple[ad.AnnData, str]],
        genodr_out_key: Optional[str] = None,
        pair_group: Optional[str] = None,  # for labeling baseline to GenoRefine pairs
    ):
        self.name = name
        self.runner = runner
        self.genodr_out_key = genodr_out_key
        self.pair_group = pair_group


def method_specs(cfg: FocusConfig) -> List[MethodSpec]:
    """Return only the four focus-test backbones used in the manuscript.

    The focused structural validation is restricted to Scanorama, Harmony,
    Seurat, and Online iNMF, along with their GenoRefine-refined embeddings.
    Reference baselines outside these four backbones are intentionally
    excluded from this focus-test script.
    """
    specs: List[MethodSpec] = []

    # Python backbones
    specs.append(MethodSpec(
        "Scanorama",
        lambda a, c: run_scanorama(a, c),
        genodr_out_key="X_genorefine_scanorama",
        pair_group="Scanorama",
    ))
    if hm is not None:
        specs.append(MethodSpec(
            "Harmony",
            lambda a, c: run_harmony(a, c),
            genodr_out_key="X_genorefine_harmony",
            pair_group="Harmony",
        ))
    else:
        logging.warning("harmonypy not installed -> Harmony and GenoRefine(Harmony) will be skipped.")

    # External R embeddings
    seurat_path = os.path.join(cfg.out_folder, cfg.seurat_csv_name)
    if os.path.exists(seurat_path):
        specs.append(MethodSpec(
            "Seurat",
            lambda a, c: run_seurat_csv(a, c),
            genodr_out_key="X_genorefine_seurat",
            pair_group="Seurat",
        ))
    else:
        logging.warning(f"Missing Seurat CSV: {seurat_path} -> Seurat and GenoRefine(Seurat) will be skipped.")

    inmf_path = os.path.join(cfg.out_folder, cfg.online_inmf_csv_name)
    if os.path.exists(inmf_path):
        specs.append(MethodSpec(
            "Online iNMF",
            lambda a, c: run_online_inmf_csv(a, c),
            genodr_out_key="X_genorefine_online_inmf",
            pair_group="Online iNMF",
        ))
    else:
        logging.warning(f"Missing Online iNMF CSV: {inmf_path} -> Online iNMF and GenoRefine(Online iNMF) will be skipped.")

    return specs

# -----------------------------
# Main
# -----------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    set_all_seeds(CFG.seed)
    ensure_out_dir(CFG.out_folder)

    v7_csv = os.path.join(CFG.out_folder, "benchmark_results_pancreas.csv")
    if not os.path.exists(v7_csv):
        logging.warning(
            f"Missing {v7_csv}. Run run_benchmark_all_v7.py first to generate it. "
            "Test 3 will be skipped if this file is absent."
        )

    logging.info("Loading pancreas raw dataset with canonical v13 preprocessing...")
    adata_raw = load_base_data(CFG)
    adata_base = preprocess_with_frozen_hvgs(adata_raw, CFG)
    logging.info(f"Base: n_obs={adata_base.n_obs}, n_vars={adata_base.n_vars}, batches={adata_base.obs['batch'].nunique()}")
    log_preprocess_fingerprint(adata_base, CFG, tag="BASE")
    logging.info("Runtime Python executable: %s", sys.executable)
    logging.info("Package environment: %s", get_genodr_env_info())
    if gp is not None:
        logging.info("GenoRefine backend compatibility patch startup: %s", ensure_genodr_keras3_compatibility())

    # Build method list (and GenoRefine list if available)
    base_specs = method_specs(CFG)

    methods_run: List[str] = []
    embed_store: Dict[str, Tuple[ad.AnnData, str]] = {}  # method -> (adata, embed_key)

    # Run base embeddings
    for spec in base_specs:
        logging.info(f"Running embedding: {spec.name}")
        try:
            ad_m, k_m = spec.runner(adata_base, CFG)
            embed_store[spec.name] = (ad_m, k_m)
            methods_run.append(spec.name)
        except Exception as e:
            logging.warning(f"Skipping {spec.name}: {e}")

    # Run GenoRefine refinements on available base embeddings
    genodr_methods: List[str] = []
    genodr_pairs: List[Tuple[str, str]] = []  # (base_method, genodr_method)

    if gp is None:
        logging.warning("genomap.genoDR not available -> GenoRefine refinements will be skipped.")
    else:
        for spec in base_specs:
            if spec.name not in embed_store:
                continue
            if spec.genodr_out_key is None:
                continue
            base_method = spec.name
            gen_name = f"GenoRefine({base_method})"
            logging.info(f"Running GenoRefine refinement: {gen_name}")
            try:
                ad_base_m, k_base = embed_store[base_method]
                ad_g = run_genodr(ad_base_m, base_embed_key=k_base, out_key=spec.genodr_out_key, cfg=CFG)
                embed_store[gen_name] = (ad_g, spec.genodr_out_key)
                genodr_methods.append(gen_name)
                methods_run.append(gen_name)
                genodr_pairs.append((base_method, gen_name))
            except Exception as e:
                logging.warning(f"Skipping {gen_name}: {e}")

    # Write run metadata
    write_run_metadata(CFG, methods_run)

    # Accumulators
    res_dfs: List[pd.DataFrame] = []
    compact_rows: List[pd.DataFrame] = []
    topo_rows: List[Dict] = []
    sil_cells_all: List[pd.DataFrame] = []
    sil_sum_all: List[pd.DataFrame] = []

    # ---- Test 0: Confusion matrices for all methods (base + GenoRefine) ----
    logging.info("Test 0: Confusion matrices for all methods...")
    for m in methods_run:
        if m not in embed_store:
            continue
        ad_m, k_m = embed_store[m]
        try:
            info = run_confusion_matrix_export(ad_m, embed_key=k_m, method=m, cfg=CFG)
            logging.info(f"  {m}: res={info['LeidenRes_used']:.2f}, k={info['NClusters']}")
        except Exception as e:
            logging.warning(f"  Confusion export failed for {m}: {e}")

    # Also export UMAPs for all methods (paper-ready)
    if CFG.save_umaps:
        logging.info("UMAP exports for all methods...")
        for m in methods_run:
            if m not in embed_store:
                continue
            ad_m, k_m = embed_store[m]
            try:
                export_umaps_for_method(ad_m, embed_key=k_m, method=m, cfg=CFG)
            except Exception as e:
                logging.warning(f"  UMAP export failed for {m}: {e}")

    # Build pair labels for Test 1/4
    pair_label_for_method: Dict[str, str] = {}
    for base_m, gen_m in genodr_pairs:
        pair_label_for_method[base_m] = f"{base_m} to {gen_m}"
        pair_label_for_method[gen_m] = f"{base_m} to {gen_m}"
    for m in methods_run:
        pair_label_for_method.setdefault(m, m)

    # ---- Test 1: Resolution sensitivity for all methods ----
    logging.info("Test 1: Resolution sensitivity for all methods...")
    for m in methods_run:
        if m not in embed_store:
            continue
        ad_m, k_m = embed_store[m]
        try:
            res_dfs.append(run_resolution_sensitivity(
                ad_m, embed_key=k_m, method_name=m, pair_name=pair_label_for_method[m], cfg=CFG
            ))
        except Exception as e:
            logging.warning(f"  Resolution sensitivity failed for {m}: {e}")

    # ---- Test 2: Intra-class compactness for all methods ----
    logging.info("Test 2: Intra-class compactness for all methods...")
    for m in methods_run:
        if m not in embed_store:
            continue
        ad_m, k_m = embed_store[m]
        try:
            y_true = ad_m.obs["class"].astype(str).to_numpy()
            X = _nan_guard(np.asarray(ad_m.obsm[k_m]), f"compactness_in[{m}]")
            df_c = estimate_intraclass_compactness(X, y_true, CFG)
            df_c["Method"] = m
            compact_rows.append(df_c)
        except Exception as e:
            logging.warning(f"  Compactness failed for {m}: {e}")

    # ---- Test 5: Topology preservation for all methods ----
    logging.info("Test 5: Topology preservation for all methods...")
    for m in methods_run:
        if m not in embed_store:
            continue
        ad_m, k_m = embed_store[m]
        try:
            y_true = ad_m.obs["class"].astype(str).to_numpy()
            X_pca = np.asarray(ad_m.obsm["X_pca"])
            X_emb = _nan_guard(np.asarray(ad_m.obsm[k_m]), f"topo_in[{m}]")
            rho, ncls = topology_preservation_centroids(X_pca, X_emb, y_true)
            topo_rows.append({"Method": m, "TopoPreserve_Spearman_Centroids": rho, "NClasses_Used": int(ncls)})
        except Exception as e:
            logging.warning(f"  Topology preservation failed for {m}: {e}")

    # ---- Test 6: Silhouette per cell for all methods ----
    logging.info("Test 6: Silhouette per cell for all methods...")
    for m in methods_run:
        if m not in embed_store:
            continue
        ad_m, k_m = embed_store[m]
        try:
            y_true = ad_m.obs["class"].astype(str).to_numpy()
            X = _nan_guard(np.asarray(ad_m.obsm[k_m]), f"silhouette_in[{m}]")
            df_cell, df_sum = silhouette_per_cell(X, y_true, m, CFG, ad_m.obs_names.tolist())
            sil_cells_all.append(df_cell)
            sil_sum_all.append(df_sum)
        except Exception as e:
            logging.warning(f"  Silhouette-per-cell failed for {m}: {e}")

    # ---- Save + plot Test 1 ----
    df_res = pd.concat(res_dfs, ignore_index=True) if res_dfs else pd.DataFrame()
    out_res_csv = os.path.join(CFG.out_folder, "focus_resolution_sensitivity.csv")
    df_res.to_csv(out_res_csv, index=False)
    logging.info(f"Saved: {out_res_csv}")
    if len(df_res):
        plot_resolution_curves(
            df_res,
            os.path.join(CFG.out_folder, "fig_resolution_ari.png"),
            os.path.join(CFG.out_folder, "fig_resolution_nclusters.png"),
        )

    # ---- Save + plot Test 2 ----
    df_comp = pd.concat(compact_rows, ignore_index=True) if compact_rows else pd.DataFrame()
    out_comp_csv = os.path.join(CFG.out_folder, "focus_intraclass_compactness.csv")
    df_comp.to_csv(out_comp_csv, index=False)
    logging.info(f"Saved: {out_comp_csv}")
    if len(df_comp):
        plot_compactness_bars_mean(df_comp, os.path.join(CFG.out_folder, "fig_intraclass_compactness_mean.png"))
        plot_compactness_heatmap(
            df_comp,
            value_col="MedianPairDist",
            out_png=os.path.join(CFG.out_folder, "fig_intraclass_compactness_heatmap_median.png"),
            title="Per-class compactness heatmap (median pairwise distance; lower is tighter)",
        )

    # ---- Test 3 plot ----
    if os.path.exists(v7_csv):
        plot_ilisi_sil_tradeoff(v7_csv, os.path.join(CFG.out_folder, "fig_ilisi_sil_tradeoff.png"))

    # ---- Test 4: kNN stability vs purity (baseline to GenoRefine pairs only) ----
    df_cells_all: List[pd.DataFrame] = []
    df_sum_all: List[pd.DataFrame] = []

    if genodr_pairs:
        logging.info("Test 4: kNN stability vs label purity (baseline to GenoRefine pairs)...")
        for base_m, gen_m in genodr_pairs:
            if base_m not in embed_store or gen_m not in embed_store:
                continue
            ad_b, k_b = embed_store[base_m]
            ad_g, k_g = embed_store[gen_m]
            try:
                y_true = ad_b.obs["class"].astype(str).to_numpy()
                df_cell, df_sum = run_knn_stability_purity(
                    X_base=_nan_guard(np.asarray(ad_b.obsm[k_b]), f"knn_base[{base_m}]"),
                    X_ref=_nan_guard(np.asarray(ad_g.obsm[k_g]), f"knn_ref[{gen_m}]"),
                    y_true=y_true,
                    pair_name=f"{base_m} to {gen_m}",
                    cfg=CFG,
                    obs_names=ad_b.obs_names.tolist(),
                )
                df_cells_all.append(df_cell)
                df_sum_all.append(df_sum)
            except Exception as e:
                logging.warning(f"  Test 4 failed for {base_m} to {gen_m}: {e}")
    else:
        logging.warning("No GenoRefine pairs available -> Test 4 will be empty.")

    df_knn = pd.concat(df_cells_all, ignore_index=True) if df_cells_all else pd.DataFrame()
    out_knn_csv = os.path.join(CFG.out_folder, "focus_knn_stability_purity.csv")
    df_knn.to_csv(out_knn_csv, index=False)
    logging.info(f"Saved: {out_knn_csv}")

    df_knn_sum = pd.concat(df_sum_all, ignore_index=True) if df_sum_all else pd.DataFrame()
    out_knn_sum_csv = os.path.join(CFG.out_folder, "focus_knn_stability_purity_summary.csv")
    df_knn_sum.to_csv(out_knn_sum_csv, index=False)
    logging.info(f"Saved: {out_knn_sum_csv}")

    if len(df_knn):
        plot_knn_stability_vs_purity(df_knn, os.path.join(CFG.out_folder, "fig_knn_stability_vs_purity.png"))

    # ---- Save + plot Test 5 ----
    df_topo = pd.DataFrame(topo_rows)
    out_topo_csv = os.path.join(CFG.out_folder, "focus_topology_preservation.csv")
    df_topo.to_csv(out_topo_csv, index=False)
    logging.info(f"Saved: {out_topo_csv}")
    if len(df_topo):
        plot_topology_preservation(df_topo, os.path.join(CFG.out_folder, "fig_topology_preservation.png"))

    # ---- Save + plot Test 6 ----
    df_sil_cells = pd.concat(sil_cells_all, ignore_index=True) if sil_cells_all else pd.DataFrame()
    out_sil_cells_csv = os.path.join(CFG.out_folder, "focus_silhouette_per_cell.csv")
    df_sil_cells.to_csv(out_sil_cells_csv, index=False)
    logging.info(f"Saved: {out_sil_cells_csv}")

    df_sil_sum = pd.concat(sil_sum_all, ignore_index=True) if sil_sum_all else pd.DataFrame()
    out_sil_sum_csv = os.path.join(CFG.out_folder, "focus_silhouette_summary.csv")
    df_sil_sum.to_csv(out_sil_sum_csv, index=False)
    logging.info(f"Saved: {out_sil_sum_csv}")

    if len(df_sil_cells):
        plot_silhouette_boxplot(df_sil_cells, os.path.join(CFG.out_folder, "fig_silhouette_boxplot.png"))
    if len(df_sil_sum):
        plot_silhouette_negative_fraction(df_sil_sum, os.path.join(CFG.out_folder, "fig_silhouette_negative_fraction.png"))

    logging.info("Done (canonical v7).")


if __name__ == "__main__":
    main()
