# Purpose: Canonical scalability harness for the 5-dataset Pancreas integration suite, aligned
#          to bench_pancreas_multi_master_v13.py.
# Author: Ariana Rahman (Arizona State University)

"""
=============================================================================
Script Name:    bench_pancreas_scalability_canonical_v3.py
Date:           April 2026

DESCRIPTION
    Canonical scalability harness for the 5-dataset Pancreas integration suite,
    aligned to bench_pancreas_multi_master_v13.py.

    Canonical preprocessing policy:
      raw .mat matrices
        -> canonical cell IDs
        -> proportional cell subsampling for each scale
        -> subset raw matrix to frozen pancreas_hvg_canonical.txt
        -> normalize_total/log1p
        -> scale
        -> PCA
        -> run method timing

    Measures:
      - Runtime (seconds)
      - Peak memory (MB) via psutil RSS polling thread

    Dataset sizes:
      - By default uses FRACTIONS of the full dataset: {25%, 50%, 75%, 100%}
      - Sampling is WITHOUT REPLACEMENT and preserves per-batch proportions.

    Methods (compute-only):
      - Scanorama
      - GenoIntig / GenoDR(Scanorama)
      - Harmony (if installed)
      - GenoDR(Harmony) (if installed)
      - BBKNN (if installed)
      - NMF (sklearn) baseline

OUTPUTS
  - ./Scalability_Results_Pancreas/Scalability_Results_Pancreas.csv
  - ./Scalability_Results_Pancreas/Figure_Scalability_Pancreas.png
  - ./Scalability_Results_Pancreas/scalability_preprocess_fingerprint.csv

USAGE
  cd PancreasMultistudyDataset
  python bench_pancreas_scalability.py
=============================================================================
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import time
import gc
import logging
import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import scanpy as sc
import anndata as ad
import scanorama
import scipy.io as sio
from scipy import sparse

from sklearn.decomposition import NMF

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
    import psutil  # type: ignore
except Exception:
    psutil = None

try:
    import genomap.genoDR as gp  # type: ignore
except Exception:
    gp = None


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class PancreasScalabilityConfig:
    data_folder: str = "./Dataset"

    # Scalability outputs are separate from benchmark outputs.
    out_folder: str = "./Scalability_Results_Pancreas"

    # Canonical HVG file comes from make_pancreas_canonical_hvg.py / master v13.
    benchmark_out_folder: str = "./Benchmark_Out"
    canonical_hvg_file: str = "pancreas_hvg_canonical.txt"
    canonical_hvg_md5_file: str = "pancreas_hvg_canonical.md5"
    require_exact_canonical_hvg_match: bool = True

    seed: int = 0

    data_files: Tuple[str, ...] = (
        "dataBaronX.mat",
        "dataMuraroX.mat",
        "dataScapleX.mat",
        "dataWangX.mat",
        "dataXinX.mat",
    )
    class_label_file: str = "classLabel.mat"

    hvg: int = 2000
    hvg_flavor: str = "seurat_v3"
    n_pcs: int = 50

    # Match master_v13 neighbor policy where relevant.
    neighbor_metric: str = "cosine"
    neighbor_n_dims: int = 30

    # Dataset sizes: fractions of full dataset.
    size_fractions: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)

    # Methods
    nmf_components: int = 30
    nmf_max_iter: int = 500

    # GenoDR params
    genodr_dim: int = 32
    genodr_col: int = 33
    genodr_row: int = 33

    # Batch name normalization
    batch_token_to_name: Dict[str, str] = None

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


CFG = PancreasScalabilityConfig()


# -----------------------------
# General utilities
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


def _get_pkg_version(name: str) -> str:
    try:
        import importlib.metadata as im
        return im.version(name)
    except Exception:
        return "NA"


def _get_process_rss_mb() -> float:
    if psutil is None:
        return float("nan")
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)


def measure_runtime_and_peak_rss(
    fn: Callable[[], Any],
    poll_interval_s: float = 0.02,
) -> Tuple[float, float, str]:
    """Measure runtime + peak RSS during execution by polling process memory."""
    gc.collect()
    start_rss = _get_process_rss_mb()
    peak_rss = start_rss
    start_t = time.time()
    status = "Success"

    if psutil is None:
        try:
            fn()
        except Exception as e:
            status = f"Failed: {type(e).__name__}: {e}"
        end_t = time.time()
        return (end_t - start_t), float("nan"), status

    import threading

    stop_flag = {"stop": False}

    def poller():
        nonlocal peak_rss
        while not stop_flag["stop"]:
            try:
                rss = _get_process_rss_mb()
                if rss > peak_rss:
                    peak_rss = rss
            except Exception:
                pass
            time.sleep(poll_interval_s)

    th = threading.Thread(target=poller, daemon=True)
    th.start()

    try:
        fn()
    except Exception as e:
        status = f"Failed: {type(e).__name__}: {e}"
    finally:
        stop_flag["stop"] = True
        th.join(timeout=1.0)

    end_t = time.time()
    end_rss = _get_process_rss_mb()
    peak_rss = max(peak_rss, end_rss)
    return (end_t - start_t), float(peak_rss), status


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
    base = os.path.splitext(os.path.basename(fn))[0]
    if base.lower().startswith("data") and base.endswith("X"):
        return base[4:-1]
    if base.lower().startswith("data"):
        return base[4:]
    return base


def _nan_to_zero_inplace(X) -> None:
    if sparse.issparse(X):
        if X.data.size and (np.isnan(X.data).any() or np.isinf(X.data).any()):
            X.data = np.nan_to_num(X.data, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        arr = np.asarray(X)
        if np.isnan(arr).any() or np.isinf(arr).any():
            arr[:] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _nan_guard(X: np.ndarray, name: str) -> np.ndarray:
    X = np.asarray(X)
    if not np.isfinite(X).all():
        logging.warning("%s: NaN/Inf found; replacing with 0.", name)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# -----------------------------
# GenoDR / Keras compatibility
# -----------------------------
_GENODR_PATCH_DONE = False


def ensure_genodr_keras3_compatibility() -> Dict[str, Any]:
    """Patch GenoMap/GenoDR for current Keras behavior inside this process."""
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
        info["patch_applied"] = True

        def patch_save_weights(model_cls: Any, flag_name: str) -> bool:
            if getattr(model_cls, flag_name, False):
                return True
            original = model_cls.save_weights

            def compat_save_weights(self, filepath, *args, **kwargs):
                fp = str(filepath)
                patched = fp
                if fp.endswith(".h5") and not fp.endswith(".weights.h5"):
                    patched = fp[:-3] + ".weights.h5"
                    logging.warning("Patched Keras save_weights path: %s -> %s", fp, patched)
                return original(self, patched, *args, **kwargs)

            model_cls.save_weights = compat_save_weights
            setattr(model_cls, flag_name, True)
            return True

        ok_tf = patch_save_weights(tf.keras.Model, "_pancreas_scalability_save_weights_patch")
        ok_keras = True
        try:
            import keras
            ok_keras = patch_save_weights(keras.Model, "_pancreas_scalability_save_weights_patch")
        except Exception:
            ok_keras = True
        info["save_weights_patch_applied"] = bool(ok_tf and ok_keras)

        # Patch ClusteringLayer.build to handle tuple input_shape under newer Keras.
        try:
            import genomap.utils.FcDEC as fcdec
            ClusteringLayer = getattr(fcdec, "ClusteringLayer", None)
            if ClusteringLayer is not None and not getattr(ClusteringLayer, "_pancreas_scalability_build_patch", False):
                original_build = ClusteringLayer.build

                def compat_build(self, input_shape):
                    try:
                        return original_build(self, input_shape)
                    except AttributeError:
                        shape = input_shape
                        if hasattr(shape, "as_list"):
                            shape = shape.as_list()
                        if isinstance(shape, (list, tuple)) and len(shape) > 0 and isinstance(shape[0], (list, tuple)):
                            shape = shape[0]
                        input_dim = int(shape[-1])
                        try:
                            from tensorflow.keras import backend as K
                            self.clusters = self.add_weight(
                                name="clusters",
                                shape=(self.n_clusters, input_dim),
                                initializer="glorot_uniform",
                            )
                            self.built = True
                        except Exception:
                            # If the internal API is different, re-raise the original failure context.
                            raise

                ClusteringLayer.build = compat_build
                ClusteringLayer._pancreas_scalability_build_patch = True
            info["clustering_layer_patch_applied"] = True
        except Exception as e:
            logging.warning("Could not patch ClusteringLayer.build: %s", e)

        _GENODR_PATCH_DONE = True
        info["details"] = "Applied GenoDR Functional-API CAE, save_weights, and ClusteringLayer compatibility patches"
        return info

    except Exception as e:
        info["details"] = f"Patch attempt failed: {type(e).__name__}: {e}"
        return info


# -----------------------------
# Canonical pancreas loading/preprocessing
# -----------------------------
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
    if not hvgs:
        raise ValueError(f"Canonical HVG file is empty: {path}")
    return hvgs


def load_pancreas_raw(cfg: PancreasScalabilityConfig) -> ad.AnnData:
    """Load raw pancreas .mat files with canonical cell names and class labels."""
    adatas: List[ad.AnnData] = []
    batch_names_in_order: List[str] = []

    for i, fn in enumerate(cfg.data_files):
        path = os.path.join(cfg.data_folder, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        d = sio.loadmat(path)
        key = _find_mat_data_key(d)
        X = np.asarray(d[key])
        a = ad.AnnData(X)

        token = _extract_batch_token_from_filename(fn)
        batch_name = cfg.batch_token_to_name.get(token, token)
        batch_names_in_order.append(batch_name)

        a.obs_names = [f"Cell-{j+1}-Batch-{batch_name}" for j in range(a.n_obs)]
        a.obs["batch"] = batch_name
        a.obs["batch_id"] = int(i + 1)
        adatas.append(a)

    adata = ad.concat(adatas, join="outer")
    if adata.obs_names.has_duplicates:
        adata.obs_names_make_unique()
    adata.var_names = pd.Index(adata.var_names.astype(str))

    cls_path = os.path.join(cfg.data_folder, cfg.class_label_file)
    if not os.path.exists(cls_path):
        raise FileNotFoundError(cls_path)
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
    return adata


def preprocess_with_canonical_hvgs(adata_raw: ad.AnnData, cfg: PancreasScalabilityConfig) -> ad.AnnData:
    """Subset raw features to frozen canonical HVGs, then normalize/log/scale/PCA."""
    hvg_path = os.path.join(cfg.benchmark_out_folder, cfg.canonical_hvg_file)
    requested_hvgs = _read_canonical_hvg_file(hvg_path)

    present = set(adata_raw.var_names.astype(str).tolist())
    matched_hvgs = [g for g in requested_hvgs if g in present]
    missing_count = len(requested_hvgs) - len(matched_hvgs)
    if len(matched_hvgs) == 0:
        raise ValueError(f"No canonical HVGs matched adata.var_names. File={hvg_path}")
    if cfg.require_exact_canonical_hvg_match and missing_count > 0:
        raise ValueError(
            f"Canonical HVG mismatch: requested={len(requested_hvgs)} matched={len(matched_hvgs)} missing={missing_count}"
        )

    logging.info(
        "Using frozen canonical HVG file: %s | requested=%d matched=%d missing=%d md5=%s",
        hvg_path,
        len(requested_hvgs),
        len(matched_hvgs),
        missing_count,
        md5_of_list(matched_hvgs),
    )

    a = adata_raw[:, matched_hvgs].copy()
    _nan_to_zero_inplace(a.X)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    _nan_to_zero_inplace(a.X)
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=int(cfg.n_pcs), random_state=int(cfg.seed))

    a.uns["canonical_hvg_file"] = hvg_path
    a.uns["canonical_hvg_md5"] = md5_of_list(matched_hvgs)
    a.uns["canonical_hvg_n"] = len(matched_hvgs)
    return a


def write_preprocess_fingerprint(adata: ad.AnnData, cfg: PancreasScalabilityConfig, tag: str) -> Dict[str, Any]:
    row = {
        "Tag": tag,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_batches": int(adata.obs["batch"].nunique()),
        "n_classes": int(adata.obs["class"].nunique()),
        "var_md5": md5_of_list([str(v) for v in adata.var_names.tolist()]),
        "obs_md5": md5_of_list([str(o) for o in adata.obs_names.tolist()]),
        "canonical_hvg_file": str(adata.uns.get("canonical_hvg_file", "")),
        "canonical_hvg_md5": str(adata.uns.get("canonical_hvg_md5", "")),
        "X_pca_shape": str(tuple(adata.obsm["X_pca"].shape)) if "X_pca" in adata.obsm else "",
        "var_PC1": float(np.var(np.asarray(adata.obsm["X_pca"])[:, 0])) if "X_pca" in adata.obsm else float("nan"),
    }
    return row


# -----------------------------
# Proportional subsampling
# -----------------------------
def subsample_preserve_batch_proportions(
    adata: ad.AnnData,
    target_n: int,
    seed: int,
    batch_key: str = "batch",
) -> ad.AnnData:
    if target_n >= adata.n_obs:
        return adata.copy()

    rng = np.random.default_rng(seed)
    batches = adata.obs[batch_key].astype(str).to_numpy()
    uniq, counts = np.unique(batches, return_counts=True)
    props = counts / counts.sum()

    alloc = np.floor(props * target_n).astype(int)
    if target_n >= len(uniq):
        alloc = np.maximum(alloc, 1)

    diff = target_n - int(alloc.sum())
    if diff > 0:
        rema = (props * target_n) - np.floor(props * target_n)
        order = np.argsort(-rema)
        for i in range(diff):
            alloc[order[i % len(order)]] += 1
    elif diff < 0:
        order = np.argsort(-alloc)
        for i in range(-diff):
            j = order[i % len(order)]
            if alloc[j] > 0:
                alloc[j] -= 1

    for i, u in enumerate(uniq):
        alloc[i] = min(int(alloc[i]), int((batches == u).sum()))

    picked_idx: List[int] = []
    for u, k in zip(uniq, alloc):
        idx_u = np.where(batches == u)[0]
        if k > 0:
            picked_idx.extend(rng.choice(idx_u, size=int(k), replace=False).tolist())

    picked_idx = np.array(picked_idx, dtype=int)
    if picked_idx.size < target_n:
        remaining = np.setdiff1d(np.arange(adata.n_obs), picked_idx, assume_unique=False)
        extra = rng.choice(remaining, size=int(target_n - picked_idx.size), replace=False)
        picked_idx = np.concatenate([picked_idx, extra])

    rng.shuffle(picked_idx)
    return adata[picked_idx].copy()


# -----------------------------
# Method wrappers
# -----------------------------
def _harmony_to_cells_by_dim(Z: np.ndarray, n_obs: int, name: str = "X_harmony") -> np.ndarray:
    Z = np.asarray(Z)
    if Z.shape[0] == n_obs:
        return Z
    if Z.shape[1] == n_obs:
        logging.info("%s: transposing Harmony output from shape %s", name, Z.shape)
        return Z.T
    raise ValueError(f"{name}: cannot align Harmony output shape {Z.shape} to n_obs={n_obs}")


def _reattach_meta_strict(adata_out: ad.AnnData, meta: pd.DataFrame, tag: str) -> None:
    idx = adata_out.obs_names.astype(str)
    meta.index = meta.index.astype(str)
    match_rate = float(pd.Index(idx).isin(meta.index).mean())
    if match_rate < 0.999:
        missing = [x for x in idx if x not in set(meta.index)][:10]
        raise ValueError(f"{tag}: metadata reattachment failed. match_rate={match_rate:.3f}; missing={missing}")
    for col in ["batch", "class", "batch_id"]:
        if col in meta.columns:
            adata_out.obs[col] = meta.loc[idx, col]
    adata_out.obs["batch"] = adata_out.obs["batch"].astype("category")
    adata_out.obs["class"] = adata_out.obs["class"].astype("category")


def run_scanorama_method(adata_in: ad.AnnData) -> ad.AnnData:
    batches = list(adata_in.obs["batch"].cat.categories)
    adatas_list = [adata_in[adata_in.obs["batch"] == b].copy() for b in batches]
    meta = adata_in.obs[["batch", "class", "batch_id"]].copy()
    corrected = scanorama.correct_scanpy(adatas_list, return_dimred=True)
    adata_int = ad.concat(corrected, join="outer")
    if adata_int.obs_names.has_duplicates:
        adata_int.obs_names_make_unique()
    _reattach_meta_strict(adata_int, meta, "Scanorama")
    adata_int.obsm["X_scanorama"] = _nan_guard(np.asarray(adata_int.obsm["X_scanorama"]), "X_scanorama")
    return adata_int


def run_harmony_method(adata_in: ad.AnnData) -> ad.AnnData:
    if hm is None:
        raise RuntimeError("harmonypy is not installed.")
    a = adata_in.copy()
    X_pca = _nan_guard(np.asarray(a.obsm["X_pca"]), "X_pca_for_harmony")
    ho = hm.run_harmony(X_pca, a.obs, "batch")
    a.obsm["X_harmony"] = _nan_guard(_harmony_to_cells_by_dim(ho.Z_corr, a.n_obs, "X_harmony"), "X_harmony")
    return a


def run_genodr_harmony_method(adata_in: ad.AnnData, cfg: PancreasScalabilityConfig) -> ad.AnnData:
    if gp is None:
        raise RuntimeError("genomap.genoDR not available (install/verify genomap).")
    ensure_genodr_keras3_compatibility()
    a = run_harmony_method(adata_in)
    X = _nan_guard(np.asarray(a.obsm["X_harmony"]), "genoDR_in[X_harmony]")
    n_clusters = max(2, int(a.obs["class"].nunique()))
    Z = gp.genoDR(
        X,
        n_dim=int(cfg.genodr_dim),
        n_clusters=int(n_clusters),
        colNum=int(cfg.genodr_col),
        rowNum=int(cfg.genodr_row),
    )
    a.obsm["X_genodr_harmony"] = _nan_guard(np.asarray(Z), "X_genodr_harmony")
    return a


def run_bbknn_method(adata_in: ad.AnnData) -> ad.AnnData:
    if bbknn is None:
        raise RuntimeError("bbknn is not installed.")
    a = adata_in.copy()
    bbknn.bbknn(a, batch_key="batch")
    return a


def run_nmf_method(adata_in: ad.AnnData, cfg: PancreasScalabilityConfig) -> ad.AnnData:
    a = adata_in.copy()
    X = a.X
    X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X_dense = np.asarray(X_dense, dtype=np.float32)
    min_val = float(np.min(X_dense)) if X_dense.size else 0.0
    if min_val < 0:
        X_dense = X_dense - min_val
    model = NMF(
        n_components=int(cfg.nmf_components),
        init="nndsvda",
        max_iter=int(cfg.nmf_max_iter),
        random_state=int(cfg.seed),
    )
    a.obsm["X_nmf"] = model.fit_transform(X_dense)
    return a


def run_genointig_method(adata_in: ad.AnnData, cfg: PancreasScalabilityConfig) -> ad.AnnData:
    if gp is None:
        raise RuntimeError("genomap.genoDR not available (install/verify genomap).")
    ensure_genodr_keras3_compatibility()
    adata_int = run_scanorama_method(adata_in)
    if "X_scanorama" not in adata_int.obsm:
        raise RuntimeError("Scanorama did not produce X_scanorama.")
    X = _nan_guard(np.asarray(adata_int.obsm["X_scanorama"]), "genoDR_in[X_scanorama]")
    n_clusters = max(2, int(adata_int.obs["class"].nunique()))
    Z = gp.genoDR(
        X,
        n_dim=int(cfg.genodr_dim),
        n_clusters=int(n_clusters),
        colNum=int(cfg.genodr_col),
        rowNum=int(cfg.genodr_row),
    )
    adata_int.obsm["X_genointig"] = _nan_guard(np.asarray(Z), "X_genointig")
    return adata_int


def build_methods(cfg: PancreasScalabilityConfig) -> Dict[str, Callable[[ad.AnnData], ad.AnnData]]:
    methods: Dict[str, Callable[[ad.AnnData], ad.AnnData]] = {
        "Scanorama": run_scanorama_method,
        "GenoIntig": lambda a: run_genointig_method(a, cfg),
    }
    if hm is not None:
        methods["Harmony"] = run_harmony_method
        methods["GenoDR(Harmony)"] = lambda a: run_genodr_harmony_method(a, cfg)
    if bbknn is not None:
        methods["BBKNN"] = run_bbknn_method
    methods["NMF"] = lambda a: run_nmf_method(a, cfg)
    return methods


# -----------------------------
# Plotting
# -----------------------------
def plot_scalability(df: pd.DataFrame, out_dir: str) -> None:
    fig = plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    for method in df["Method"].unique():
        sub = df[df["Method"] == method].sort_values("Cells")
        ax1.plot(sub["Cells"], sub["Runtime_s"], marker="o", label=method)
    ax1.set_title("Runtime Scalability (Pancreas)")
    ax1.set_xlabel("Number of Cells")
    ax1.set_ylabel("Time (seconds)")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    for method in df["Method"].unique():
        sub = df[df["Method"] == method].sort_values("Cells")
        ax2.plot(sub["Cells"], sub["PeakRSS_MB"], marker="o", label=method)
    ax2.set_title("Memory Scalability (Pancreas)")
    ax2.set_xlabel("Number of Cells")
    ax2.set_ylabel("Peak RSS (MB)")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(out_dir, "Figure_Scalability_Pancreas.png")
    plt.savefig(save_path, dpi=600)
    logging.info("Figure saved to %s", save_path)
    plt.close(fig)
    plt.close("all")


# -----------------------------
# Main
# -----------------------------
def main():
    """Measure runtime and peak memory across proportionally subsampled pancreas dataset sizes."""
    sc.settings.verbosity = 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ensure_out_dir(CFG.out_folder)
    set_all_seeds(CFG.seed)

    logging.info("Runtime Python executable: %s", os.sys.executable)
    logging.info(
        "Package environment: scanpy=%s anndata=%s numpy=%s tensorflow=%s keras=%s genomap=%s harmonypy=%s",
        _get_pkg_version("scanpy"),
        _get_pkg_version("anndata"),
        _get_pkg_version("numpy"),
        _get_pkg_version("tensorflow"),
        _get_pkg_version("keras"),
        _get_pkg_version("genomap"),
        _get_pkg_version("harmonypy"),
    )

    if psutil is None:
        logging.warning("psutil not installed -> peak memory will be NaN. Install with: pip install psutil")
    if gp is None:
        logging.warning("genomap.genoDR not available -> GenoIntig / GenoDR(Harmony) may fail.")
    if bbknn is None:
        logging.warning("bbknn not installed -> BBKNN will be skipped.")
    if hm is None:
        logging.warning("harmonypy not installed -> Harmony + GenoDR(Harmony) will be skipped.")

    if gp is not None:
        logging.info("GenoDR compatibility patch: %s", ensure_genodr_keras3_compatibility())

    logging.info("Loading raw Pancreas data with canonical cell IDs...")
    adata_raw_full = load_pancreas_raw(CFG)
    n_full = int(adata_raw_full.n_obs)

    logging.info(
        "Pancreas raw full: n_obs=%d, n_vars=%d, batches=%d, classes=%d",
        n_full,
        adata_raw_full.n_vars,
        adata_raw_full.obs["batch"].nunique(),
        adata_raw_full.obs["class"].nunique(),
    )

    sizes = sorted({max(200, int(round(f * n_full))) for f in CFG.size_fractions})
    sizes = [s for s in sizes if s <= n_full]
    if sizes[-1] != n_full:
        sizes.append(n_full)
    logging.info("Scalability sizes (cells): %s", sizes)

    methods = build_methods(CFG)
    logging.info("Methods: %s", list(methods.keys()))

    results: List[Dict[str, object]] = []
    fingerprints: List[Dict[str, Any]] = []

    for target_n in sizes:
        logging.info("--- Dataset Size: %d cells ---", target_n)
        adata_raw_sub = subsample_preserve_batch_proportions(adata_raw_full, target_n=target_n, seed=CFG.seed)
        logging.info("Preprocessing subset with frozen canonical HVGs...")
        adata_sub = preprocess_with_canonical_hvgs(adata_raw_sub, CFG)
        fingerprints.append(write_preprocess_fingerprint(adata_sub, CFG, tag=f"cells_{target_n}"))

        for method_name, fn in methods.items():
            logging.info("Running %s on %d cells...", method_name, target_n)
            adata_run = adata_sub.copy()

            def _runner():
                _ = fn(adata_run)

            runtime_s, peak_mb, status = measure_runtime_and_peak_rss(_runner)
            results.append(
                {
                    "Method": method_name,
                    "Cells": int(target_n),
                    "Runtime_s": float(runtime_s),
                    "PeakRSS_MB": float(peak_mb),
                    "Status": status,
                }
            )
            if status.startswith("Success"):
                logging.info("  -> Time: %.2fs | Peak RSS: %.2f MB", runtime_s, peak_mb)
            else:
                logging.error("  -> %s failed at %d: %s", method_name, target_n, status)

            del adata_run
            gc.collect()

        del adata_sub, adata_raw_sub
        gc.collect()

    df = pd.DataFrame(results)
    out_csv = os.path.join(CFG.out_folder, "Scalability_Results_Pancreas.csv")
    df.to_csv(out_csv, index=False)
    logging.info("Saved results: %s", out_csv)

    fp_csv = os.path.join(CFG.out_folder, "scalability_preprocess_fingerprint.csv")
    pd.DataFrame(fingerprints).to_csv(fp_csv, index=False)
    logging.info("Saved preprocessing fingerprints: %s", fp_csv)

    ok_df = df[df["Status"].astype(str).str.startswith("Success")].copy()
    if not ok_df.empty:
        plot_scalability(ok_df, CFG.out_folder)
        print("\n--- Scalability Summary (Runtime / Peak RSS) ---")
        print(ok_df.pivot(index="Cells", columns="Method", values=["Runtime_s", "PeakRSS_MB"]))
    else:
        logging.error("No successful scalability results produced.")


if __name__ == "__main__":
    main()
