#!/usr/bin/env python3
"""
bench_hie_robustness_genorefine_core_mirror.py
-------------------------------------------------------------------------------
Strict canonical, core-method HIE robustness benchmark.

This script is the Python driver for the strict HIE robustness workflow:

    1) python make_hie_canonical_hvg.py
    2) python make_hie_robustness_hvg.py
    3) Rscript seurat_integration_hie_robustness.R
    4) Rscript rliger_online_inmf_hie_robustness.R
    5) python bench_hie_robustness_genorefine_core_mirror.py

What makes this strict:
    - Uses the frozen canonical HIE HVG file only.
    - Reads the exact cells_*.csv manifests produced by make_hie_robustness_hvg.py.
    - For each manifest/context, evaluates Python methods on the same cells.
    - Loads condition-specific R embeddings:
          X_seurat_hie_<context>.csv
          X_online_inmf_hie_<context>.csv
      rather than reusing full-dataset R embeddings.
    - Fails by default if any required manifest/R embedding/canonical audit is missing.

Default input/output layout:
    INPUT:
        ./Dataset/human_pancreas_norm_complexBatch.h5ad
        ./Benchmark_Hie_Out/hie_hvg_canonical.txt
        ./Benchmark_Hie_Out/hie_hvg_canonical.md5
        ./Benchmark_Hie_Out/Robustness_R_Embeddings/cells_*.csv
        ./Benchmark_Hie_Out/Robustness_R_Embeddings/X_seurat_hie_<context>.csv
        ./Benchmark_Hie_Out/Robustness_R_Embeddings/X_online_inmf_hie_<context>.csv

    OUTPUT:
        ./HIE_Robustness_Out/robustness_all_hie.csv
        ./HIE_Robustness_Out/robustness_imbalance_hie.csv
        ./HIE_Robustness_Out/robustness_nonoverlap_hie.csv
        ./HIE_Robustness_Out/purity_entropy_cluster_tables_hie.csv
        ./HIE_Robustness_Out/purity_entropy_imbalance_allclusters_hie.csv
        ./HIE_Robustness_Out/purity_entropy_nonoverlap_allclusters_hie.csv
        ./HIE_Robustness_Out/purity_entropy_imbalance_summary_hie.csv
        ./HIE_Robustness_Out/purity_entropy_nonoverlap_summary_hie.csv
        ./HIE_Robustness_Out/table_extreme_imbalance_hie.csv
        ./HIE_Robustness_Out/fig_imbalance_curves_ARI.png
        ./HIE_Robustness_Out/fig_imbalance_curves_Silhouette.png
        ./HIE_Robustness_Out/fig_imbalance_curves_iLISI.png
        ./HIE_Robustness_Out/external_embedding_validation_hie.csv

Run:
    cd <HIE project folder>
    python bench_hie_robustness_genorefine_core_mirror.py

Optional environment override:
    set HIE_BASE_DIR=C:/path/to/HIEDataset
    python bench_hie_robustness_genorefine_core_mirror.py
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.metrics import adjusted_rand_score, rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# Optional method dependencies
# -----------------------------
try:
    import scanorama  # type: ignore
except Exception:
    scanorama = None

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
class HieMultiRobustConfig:
    base_dir: str = "."
    data_file: str = "./Dataset/human_pancreas_norm_complexBatch.h5ad"
    benchmark_out_folder: str = "./Benchmark_Hie_Out"
    robustness_folder: str = "Robustness_R_Embeddings"
    out_folder: str = "./HIE_Robustness_Out"
    seed: int = 0

    # Strict canonical HVG artifacts.
    canonical_hvg_file: str = "./Benchmark_Hie_Out/hie_hvg_canonical.txt"
    canonical_hvg_md5_file: str = "./Benchmark_Hie_Out/hie_hvg_canonical.md5"
    hvg: int = 2000
    require_exact_canonical_hvg_count: bool = True
    require_canonical_md5_match: bool = True
    require_all_canonical_genes_present: bool = True

    # Strict manifest/R-output artifacts.
    manifest_summary_csv: str = "./Benchmark_Hie_Out/Robustness_R_Embeddings/hie_robustness_manifest_summary.csv"
    require_r_embeddings: bool = True
    require_external_exact_rows: bool = True
    seurat_prefix: str = "X_seurat_hie_"
    online_inmf_prefix: str = "X_online_inmf_hie_"

    # HIE obs key policy.
    batch_key_fixed: str = "tech"
    label_key_fixed: str = "celltype"
    batch_key_candidates: Tuple[str, ...] = (
        "tech", "technology", "platform", "protocol", "method", "batch", "Batch",
        "dataset", "study", "donor", "orig.ident", "source",
    )
    label_key_candidates: Tuple[str, ...] = (
        "celltype", "class", "cell_type", "CellType", "label", "labels", "annotation",
    )

    # Preprocessing. Strict robustness normalizes within each manifest condition.
    do_normalize_log1p: bool = True
    normalize_target_sum: float = 1e4
    counts_layer_preference: str = "counts"
    do_scale: bool = True
    max_scale_value: float = 10.0
    n_pcs: int = 50
    n_neighbors: int = 15

    # Metrics / clustering.
    lisi_k: int = 90
    leiden_target_n_clusters_if_labels_exist: bool = True
    leiden_resolution_default: float = 0.5
    leiden_res_grid: Tuple[float, ...] = tuple(np.round(np.linspace(0.2, 1.6, 15), 2))

    # NMF baseline (disabled in core GenoRefine robustness runs).
    nmf_components: int = 30
    nmf_max_iter: int = 500

    # DiffusionMap / PHATE baselines (disabled in core GenoRefine robustness runs).
    diffmap_n_comps: int = 30
    diffmap_embed_key: str = "X_diffmap"
    phate_n_components: int = 30
    phate_knn: int = 15
    phate_decay: int = 40
    phate_t: str = "auto"
    phate_embed_key: str = "X_phate"

    # GenoRefine params passed to the genoDR backend.
    genodr_dim: int = 32
    genodr_col: int = 33
    genodr_row: int = 33

    # Methods. Use "AUTO" to run the four core backbones and their GenoRefine variants.
    methods: str = "AUTO"

    # Output controls.
    entropy_eps: float = 1e-12
    save_per_condition_purity_entropy_tables: bool = True
    save_umap: bool = False
    save_umap_contexts: Tuple[str, ...] = ()  # empty => all contexts if save_umap=True


CFG = HieMultiRobustConfig()


@dataclass
class RobustCondition:
    context: str
    test: str
    manifest_csv: str
    frac: Optional[float] = None
    downsample_batch: str = ""
    removed_batch: str = ""
    removed_class: str = ""
    n_cells_manifest: Optional[int] = None
    n_batches_manifest: Optional[int] = None
    n_classes_manifest: Optional[int] = None
    extras: Dict[str, Any] = None  # type: ignore[assignment]

    def as_result_fields(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "Context": self.context,
            "Test": self.test,
            "ManifestCSV": self.manifest_csv,
            "Frac": self.frac if self.frac is not None else np.nan,
            "DownsampleBatch": self.downsample_batch,
            "RemovedBatch": self.removed_batch,
            "RemovedClass": self.removed_class,
            "NCellsManifest": self.n_cells_manifest if self.n_cells_manifest is not None else np.nan,
            "NBatchesManifest": self.n_batches_manifest if self.n_batches_manifest is not None else np.nan,
            "NClassesManifest": self.n_classes_manifest if self.n_classes_manifest is not None else np.nan,
        }
        if self.extras:
            for k, v in self.extras.items():
                if k not in out:
                    out[k] = v
        return out


# -----------------------------
# General utilities
# -----------------------------
def set_all_seeds(seed: int) -> None:
    np.random.seed(int(seed))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(str(path)):
        return os.path.normpath(str(path))
    return os.path.normpath(os.path.join(base_dir, str(path)))


def md5_no_trailing_newline(items: Sequence[str]) -> str:
    payload = "\n".join(map(str, items)).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def md5_with_trailing_newline(items: Sequence[str]) -> str:
    h = hashlib.md5()
    for s in items:
        h.update(str(s).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def sanitize_tag(text: Any) -> str:
    s = str(text).strip()
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "NA"


def sanitize_method_name(name: str) -> str:
    return sanitize_tag(str(name).replace("(", "_").replace(")", "").replace("/", "_"))


def _nan_guard(X: np.ndarray, name: str) -> np.ndarray:
    X = np.asarray(X)
    if not np.isfinite(X).all():
        logging.warning("%s: NaN/Inf found; replacing with 0.", name)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def infer_key_from_obs(adata: ad.AnnData, fixed: str, candidates: Tuple[str, ...], require_two_levels: bool) -> Optional[str]:
    if fixed in adata.obs.columns:
        vals = adata.obs[fixed].dropna().astype(str)
        if (vals.nunique() >= 2) or (not require_two_levels and vals.nunique() >= 1):
            return fixed
    for k in candidates:
        if k not in adata.obs.columns:
            continue
        vals = adata.obs[k].dropna().astype(str)
        if (vals.nunique() >= 2) or (not require_two_levels and vals.nunique() >= 1):
            return k
    return None


def parse_methods(method_text: str, methods_all: List[str]) -> List[str]:
    if str(method_text).strip().upper() == "AUTO":
        return methods_all

    alias = {
        "Online_iNMF": "Online iNMF",
    }
    requested_raw = [x.strip() for x in str(method_text).split(",") if x.strip()]
    requested = [alias.get(x, x) for x in requested_raw]
    valid = set(methods_all)
    bad = [m for m in requested if m not in valid]
    if bad:
        raise ValueError(f"Unknown method(s): {bad}. Valid methods: {methods_all}")
    return requested


def safe_float_or_none(x: Any) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def context_from_manifest_path(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    if stem.startswith("cells_"):
        return stem[len("cells_"):]
    return stem


def infer_test_from_context(ctx: str) -> str:
    if str(ctx).startswith("imbalance_"):
        return "imbalance"
    if str(ctx).startswith("nonoverlap_"):
        return "nonoverlap"
    return "unknown"


def infer_frac_from_context(ctx: str) -> Optional[float]:
    # Handles suffixes such as frac1_00, frac0_50, frac0_25, frac0_10.
    m = re.search(r"(?:^|_)frac(\d+)_(\d+)(?:$|_)", str(ctx))
    if not m:
        return None
    return float(f"{int(m.group(1))}.{m.group(2)}")


def should_save_umap_for_context(cfg: HieMultiRobustConfig, ctx: str) -> bool:
    if not cfg.save_umap:
        return False
    if not cfg.save_umap_contexts:
        return True
    return str(ctx) in set(map(str, cfg.save_umap_contexts))


# -----------------------------
# Data and canonical HVGs
# -----------------------------
def read_hie_raw(cfg: HieMultiRobustConfig) -> Tuple[ad.AnnData, str, Optional[str], Optional[str]]:
    if not os.path.exists(cfg.data_file):
        raise FileNotFoundError(f"Could not find HIE .h5ad: {cfg.data_file}")

    logging.info("Reading HIE data: %s", cfg.data_file)
    if str(cfg.data_file).lower().endswith(".h5ad"):
        a = sc.read_h5ad(cfg.data_file)
    else:
        a = sc.read(cfg.data_file)

    a.var_names_make_unique()
    if a.obs_names.has_duplicates:
        raise ValueError("HIE AnnData obs_names are not unique; strict manifests require unique cell IDs.")

    batch_key = infer_key_from_obs(a, cfg.batch_key_fixed, cfg.batch_key_candidates, require_two_levels=True)
    label_key = infer_key_from_obs(a, cfg.label_key_fixed, cfg.label_key_candidates, require_two_levels=False)

    if batch_key is None:
        raise ValueError(
            f"No usable batch key found. Tried fixed='{cfg.batch_key_fixed}' and candidates "
            f"{cfg.batch_key_candidates}. Available obs keys: {list(a.obs.columns)[:50]}"
        )

    if label_key is None:
        logging.warning(
            "No label key found. Tried fixed='%s' and candidates %s. ARI/RI/purity outputs will be NaN.",
            cfg.label_key_fixed, cfg.label_key_candidates,
        )

    a.obs["batch"] = a.obs[batch_key].astype(str).astype("category")
    if label_key is not None:
        a.obs["class"] = a.obs[label_key].astype(str).astype("category")

    # Stable batch_id for validation/readability.
    batch_levels_sorted = sorted(pd.unique(a.obs["batch"].astype(str)).tolist())
    batch_to_id = {b: i + 1 for i, b in enumerate(batch_levels_sorted)}
    a.obs["batch_id"] = a.obs["batch"].astype(str).map(batch_to_id).astype(int)

    counts_layer_name: Optional[str] = None
    if cfg.counts_layer_preference and cfg.counts_layer_preference in a.layers.keys():
        counts_layer_name = str(cfg.counts_layer_preference)

    logging.info(
        "Loaded HIE: n_obs=%d n_vars=%d batch_key=%s label_key=%s counts_layer=%s",
        a.n_obs, a.n_vars, batch_key, label_key, counts_layer_name,
    )
    logging.info("Batch counts: %s", a.obs["batch"].astype(str).value_counts().to_dict())
    if "class" in a.obs:
        logging.info("Class count: %d", int(a.obs["class"].nunique()))

    return a, batch_key, label_key, counts_layer_name


def read_canonical_hvgs(cfg: HieMultiRobustConfig, adata: ad.AnnData) -> List[str]:
    if not os.path.exists(cfg.canonical_hvg_file):
        raise FileNotFoundError(
            f"Canonical HVG file not found: {cfg.canonical_hvg_file}. Run make_hie_canonical_hvg.py first."
        )

    with open(cfg.canonical_hvg_file, "r", encoding="utf-8") as f:
        genes_raw = [line.strip() for line in f if line.strip()]

    if not genes_raw:
        raise ValueError(f"Canonical HVG file is empty: {cfg.canonical_hvg_file}")

    duplicated = sorted(set(g for g in genes_raw if genes_raw.count(g) > 1))
    if duplicated:
        raise ValueError(
            f"Canonical HVG file contains duplicate genes ({len(duplicated)}). Examples: {duplicated[:10]}"
        )

    genes = [str(g) for g in genes_raw]

    if cfg.require_exact_canonical_hvg_count and len(genes) != int(cfg.hvg):
        raise ValueError(
            f"Expected exactly {cfg.hvg} canonical HVGs, but found {len(genes)} in {cfg.canonical_hvg_file}."
        )

    md5_r = md5_no_trailing_newline(genes)
    md5_master = md5_with_trailing_newline(genes)
    logging.info(
        "Loaded canonical HIE HVGs: n=%d md5_no_trailing_newline=%s md5_trailing_newline=%s",
        len(genes), md5_r, md5_master,
    )

    if cfg.canonical_hvg_md5_file:
        if not os.path.exists(cfg.canonical_hvg_md5_file):
            if cfg.require_canonical_md5_match:
                raise FileNotFoundError(f"Canonical HVG md5 file not found: {cfg.canonical_hvg_md5_file}")
            logging.warning("Canonical HVG md5 file not found: %s", cfg.canonical_hvg_md5_file)
        else:
            with open(cfg.canonical_hvg_md5_file, "r", encoding="utf-8") as f:
                expected = next((line.strip() for line in f if line.strip()), "")
            if not expected:
                if cfg.require_canonical_md5_match:
                    raise ValueError(f"Canonical HVG md5 file is empty: {cfg.canonical_hvg_md5_file}")
                logging.warning("Canonical HVG md5 file is empty: %s", cfg.canonical_hvg_md5_file)
            elif expected != md5_r:
                msg = f"Canonical HVG md5 mismatch: expected={expected} loaded={md5_r}"
                if cfg.require_canonical_md5_match:
                    raise ValueError(msg)
                logging.warning(msg)
            else:
                logging.info("Canonical HVG md5 check passed: %s", expected)

    missing = [g for g in genes if g not in adata.var_names]
    if missing:
        msg = (
            f"Canonical HVG file contains {len(missing)} genes absent from HIE AnnData var_names. "
            f"Examples: {missing[:10]}"
        )
        if cfg.require_all_canonical_genes_present:
            raise ValueError(msg)
        logging.warning(msg)
        genes = [g for g in genes if g in adata.var_names]

    return genes




def _read_simple_gene_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def validate_hvg_audit_file(
    label: str,
    txt_path: str,
    md5_path: str,
    canonical_hvgs: List[str],
    cfg: HieMultiRobustConfig,
    required: bool,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "AuditLabel": label,
        "HVGFile": txt_path,
        "MD5File": md5_path,
        "Required": bool(required),
        "Status": "OK",
        "Error": "",
    }

    if not os.path.exists(txt_path):
        row.update({"Status": "MISSING", "Error": f"Missing HVG audit file: {txt_path}"})
        if required:
            raise FileNotFoundError(row["Error"])
        logging.warning("%s", row["Error"])
        return row

    genes = _read_simple_gene_list(txt_path)
    row["NGenes"] = int(len(genes))
    row["ComputedMD5"] = md5_no_trailing_newline(genes)
    row["CanonicalMD5"] = md5_no_trailing_newline(canonical_hvgs)

    if cfg.require_exact_canonical_hvg_count and len(genes) != int(cfg.hvg):
        row.update({"Status": "BAD_N", "Error": f"Expected {cfg.hvg} genes, found {len(genes)}"})
        if required:
            raise ValueError(f"{label}: {row['Error']}")

    if genes != canonical_hvgs:
        first_diff = None
        for i, (a, b) in enumerate(zip(genes, canonical_hvgs)):
            if a != b:
                first_diff = i
                break
        if first_diff is None and len(genes) != len(canonical_hvgs):
            first_diff = min(len(genes), len(canonical_hvgs))
        row.update({"Status": "ORDER_MISMATCH", "Error": f"Does not exactly match canonical HVG order; first_diff={first_diff}"})
        if required:
            raise ValueError(f"{label}: {row['Error']}")

    expected_md5 = ""
    if os.path.exists(md5_path):
        with open(md5_path, "r", encoding="utf-8") as f:
            expected_md5 = next((line.strip() for line in f if line.strip()), "")
    elif required and cfg.require_canonical_md5_match:
        row.update({"Status": "MISSING_MD5", "Error": f"Missing HVG audit MD5 file: {md5_path}"})
        raise FileNotFoundError(f"{label}: {row['Error']}")

    row["ExpectedMD5FromFile"] = expected_md5
    if expected_md5 and expected_md5 != row["ComputedMD5"]:
        row.update({"Status": "MD5_MISMATCH", "Error": f"Expected MD5 {expected_md5}, computed {row['ComputedMD5']}"})
        if required and cfg.require_canonical_md5_match:
            raise ValueError(f"{label}: {row['Error']}")

    if row["Status"] == "OK":
        logging.info("Validated HVG audit: %s", label)
    else:
        logging.warning("HVG audit %s status=%s error=%s", label, row["Status"], row["Error"])
    return row


def validate_robustness_hvg_audits(
    cfg: HieMultiRobustConfig,
    canonical_hvgs: List[str],
    methods_requested: List[str],
) -> pd.DataFrame:
    robust_dir = os.path.join(cfg.benchmark_out_folder, cfg.robustness_folder)
    needs_seurat = any(m in ("Seurat", "GenoRefine(Seurat)") for m in methods_requested)
    needs_inmf = any(m in ("Online iNMF", "GenoRefine(Online iNMF)") for m in methods_requested)

    specs = [
        (
            "make_hie_robustness_hvg",
            os.path.join(robust_dir, "hie_robustness_hvg_used.txt"),
            os.path.join(robust_dir, "hie_robustness_hvg_used.md5"),
            True,
        ),
        (
            "seurat_integration_hie_robustness",
            os.path.join(robust_dir, "seurat_hie_robustness_hvg_used.txt"),
            os.path.join(robust_dir, "seurat_hie_robustness_hvg_used.md5"),
            bool(needs_seurat and cfg.require_r_embeddings),
        ),
        (
            "rliger_online_inmf_hie_robustness",
            os.path.join(robust_dir, "online_inmf_hie_robustness_hvg_used.txt"),
            os.path.join(robust_dir, "online_inmf_hie_robustness_hvg_used.md5"),
            bool(needs_inmf and cfg.require_r_embeddings),
        ),
    ]

    rows = [
        validate_hvg_audit_file(label, txt, md5, canonical_hvgs, cfg, required=required)
        for label, txt, md5, required in specs
    ]
    return pd.DataFrame(rows)

def normalize_condition(adata_condition_raw: ad.AnnData, cfg: HieMultiRobustConfig, counts_layer_name: Optional[str]) -> ad.AnnData:
    a = adata_condition_raw.copy()

    # Remove unused categories so batch iteration never creates empty batches.
    a.obs["batch"] = a.obs["batch"].astype(str).astype("category")
    if "class" in a.obs:
        a.obs["class"] = a.obs["class"].astype(str).astype("category")

    if cfg.do_normalize_log1p:
        if counts_layer_name is not None and counts_layer_name in a.layers.keys():
            logging.debug("Condition preprocess: using layer='%s' as X for normalize_total/log1p", counts_layer_name)
            a.X = a.layers[counts_layer_name].copy()
        else:
            logging.debug("Condition preprocess: using a.X for normalize_total/log1p")
        sc.pp.normalize_total(a, target_sum=float(cfg.normalize_target_sum))
        sc.pp.log1p(a)

    return a


def preprocess_with_canonical_hvgs(adata_norm: ad.AnnData, hvgs: List[str], cfg: HieMultiRobustConfig) -> ad.AnnData:
    missing = [g for g in hvgs if g not in adata_norm.var_names]
    if missing:
        raise ValueError(f"Preprocess missing {len(missing)} canonical HVGs. Examples: {missing[:10]}")

    a = adata_norm[:, hvgs].copy()
    if cfg.do_scale:
        sc.pp.scale(a, max_value=float(cfg.max_scale_value))
    n_pcs_use = min(int(cfg.n_pcs), int(a.n_obs) - 1, int(a.n_vars) - 1)
    if n_pcs_use < 2:
        raise ValueError(f"Too few cells/genes for PCA: n_obs={a.n_obs}, n_vars={a.n_vars}")
    sc.tl.pca(a, n_comps=int(n_pcs_use), random_state=int(cfg.seed))
    a.obsm["X_pca"] = _nan_guard(np.asarray(a.obsm["X_pca"]), "X_pca")
    return a


def log_preprocess_fingerprint(adata: ad.AnnData, tag: str) -> None:
    var_names = [str(v) for v in adata.var_names.tolist()]
    obs_names = [str(o) for o in adata.obs_names.tolist()]
    msg = [
        f"[{tag}] n_obs={adata.n_obs}, n_vars={adata.n_vars}",
        f"[{tag}] batches={adata.obs['batch'].nunique() if 'batch' in adata.obs else 'NA'}",
        f"[{tag}] classes={adata.obs['class'].nunique() if 'class' in adata.obs else 'NA'}",
        f"[{tag}] var_md5={md5_with_trailing_newline(var_names)}",
        f"[{tag}] obs_md5={md5_with_trailing_newline(obs_names)}",
    ]
    if "X_pca" in adata.obsm:
        x0 = np.asarray(adata.obsm["X_pca"])
        msg.append(f"[{tag}] X_pca shape={x0.shape}, var(PC1)={float(np.var(x0[:, 0])):.6f}")
    logging.info("\n".join(msg))


# -----------------------------
# Conditions / manifests / external embeddings
# -----------------------------
def resolve_manifest_candidate(path_value: Any, ctx: str, cfg: HieMultiRobustConfig) -> str:
    robust_dir = os.path.join(cfg.benchmark_out_folder, cfg.robustness_folder)
    candidates: List[str] = []

    if path_value is not None and not pd.isna(path_value) and str(path_value).strip():
        p = str(path_value).strip()
        candidates.append(p if os.path.isabs(p) else resolve_path(p, cfg.base_dir))
        candidates.append(os.path.join(robust_dir, os.path.basename(p)))

    candidates.append(os.path.join(robust_dir, f"cells_{ctx}.csv"))

    seen: set[str] = set()
    for c in candidates:
        c_norm = os.path.normpath(c)
        if c_norm in seen:
            continue
        seen.add(c_norm)
        if os.path.exists(c_norm):
            return c_norm

    return os.path.normpath(candidates[-1])


def load_conditions(cfg: HieMultiRobustConfig) -> List[RobustCondition]:
    robust_dir = os.path.join(cfg.benchmark_out_folder, cfg.robustness_folder)
    summary_path = cfg.manifest_summary_csv
    conditions: List[RobustCondition] = []

    if os.path.exists(summary_path):
        logging.info("Loading robustness manifest summary: %s", summary_path)
        df = pd.read_csv(summary_path)
        if "Context" not in df.columns:
            raise ValueError(f"Manifest summary missing Context column: {summary_path}")
        for _, row in df.iterrows():
            ctx = str(row["Context"])
            manifest_csv = resolve_manifest_candidate(row.get("ManifestCSV", ""), ctx, cfg)
            test = str(row.get("Test", infer_test_from_context(ctx)))
            frac = safe_float_or_none(row.get("Frac", np.nan))
            extras = {k: row[k] for k in df.columns if k not in {
                "Context", "Test", "ManifestCSV", "Frac", "DownsampleBatch", "RemovedBatch", "RemovedClass",
                "NCells", "NBatches", "NClasses",
            }}
            conditions.append(
                RobustCondition(
                    context=ctx,
                    test=test,
                    manifest_csv=manifest_csv,
                    frac=frac,
                    downsample_batch=str(row.get("DownsampleBatch", "")) if not pd.isna(row.get("DownsampleBatch", "")) else "",
                    removed_batch=str(row.get("RemovedBatch", "")) if not pd.isna(row.get("RemovedBatch", "")) else "",
                    removed_class=str(row.get("RemovedClass", "")) if not pd.isna(row.get("RemovedClass", "")) else "",
                    n_cells_manifest=int(row["NCells"]) if "NCells" in row and not pd.isna(row["NCells"]) else None,
                    n_batches_manifest=int(row["NBatches"]) if "NBatches" in row and not pd.isna(row["NBatches"]) else None,
                    n_classes_manifest=int(row["NClasses"]) if "NClasses" in row and not pd.isna(row["NClasses"]) else None,
                    extras=extras,
                )
            )
    else:
        logging.warning("Manifest summary not found; discovering cells_*.csv directly in %s", robust_dir)
        manifest_files = sorted(glob.glob(os.path.join(robust_dir, "cells_*.csv")))
        for p in manifest_files:
            ctx = context_from_manifest_path(p)
            conditions.append(
                RobustCondition(
                    context=ctx,
                    test=infer_test_from_context(ctx),
                    manifest_csv=p,
                    frac=infer_frac_from_context(ctx),
                    extras={},
                )
            )

    if not conditions:
        raise FileNotFoundError(
            f"No robustness manifests found. Run make_hie_robustness_hvg.py first. Expected summary or cells_*.csv under {robust_dir}"
        )

    missing = [c.manifest_csv for c in conditions if not os.path.exists(c.manifest_csv)]
    if missing:
        raise FileNotFoundError(f"Missing manifest CSV(s): {missing[:10]}")

    logging.info("Loaded %d robustness condition(s): %s", len(conditions), [c.context for c in conditions])
    return conditions


def validate_and_subset_by_manifest(adata_raw: ad.AnnData, condition: RobustCondition) -> Tuple[ad.AnnData, pd.DataFrame]:
    manifest = pd.read_csv(condition.manifest_csv)
    if "cell_id" not in manifest.columns:
        raise ValueError(f"Manifest lacks cell_id column: {condition.manifest_csv}")

    manifest["cell_id"] = manifest["cell_id"].astype(str)
    if manifest["cell_id"].isna().any() or (manifest["cell_id"].str.len() == 0).any():
        raise ValueError(f"Manifest contains empty/NA cell_id values: {condition.manifest_csv}")
    if manifest["cell_id"].duplicated().any():
        dup = manifest.loc[manifest["cell_id"].duplicated(), "cell_id"].head(10).tolist()
        raise ValueError(f"Manifest has duplicate cell IDs for context={condition.context}. Examples: {dup}")

    ids = manifest["cell_id"].tolist()
    missing = [x for x in ids if x not in adata_raw.obs_names]
    if missing:
        raise ValueError(
            f"Manifest for context={condition.context} contains {len(missing)} cells absent from HIE AnnData. "
            f"Examples: {missing[:10]}"
        )

    a = adata_raw[ids].copy()

    if "batch" in manifest.columns:
        m_batch = manifest["batch"].astype(str).to_numpy()
        d_batch = a.obs["batch"].astype(str).to_numpy()
        mismatch = np.where(m_batch != d_batch)[0]
        if len(mismatch) > 0:
            i = int(mismatch[0])
            raise ValueError(
                f"Manifest batch mismatch for context={condition.context}, cell={ids[i]}: "
                f"manifest={m_batch[i]} data={d_batch[i]}"
            )

    if "class" in manifest.columns and "class" in a.obs:
        m_class = manifest["class"].astype(str).to_numpy()
        d_class = a.obs["class"].astype(str).to_numpy()
        mismatch = np.where(m_class != d_class)[0]
        if len(mismatch) > 0:
            i = int(mismatch[0])
            raise ValueError(
                f"Manifest class mismatch for context={condition.context}, cell={ids[i]}: "
                f"manifest={m_class[i]} data={d_class[i]}"
            )

    condition.n_cells_manifest = int(a.n_obs)
    condition.n_batches_manifest = int(a.obs["batch"].nunique())
    condition.n_classes_manifest = int(a.obs["class"].nunique()) if "class" in a.obs else None

    return a, manifest


def external_embedding_path(cfg: HieMultiRobustConfig, prefix: str, ctx: str) -> str:
    robust_dir = os.path.join(cfg.benchmark_out_folder, cfg.robustness_folder)
    return os.path.join(robust_dir, f"{prefix}{ctx}.csv")


def load_embedding_csv_align(
    adata: ad.AnnData,
    csv_path: str,
    key: str,
    require_exact_rows: bool,
) -> Dict[str, Any]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing embedding CSV: {csv_path}")

    df = pd.read_csv(csv_path, index_col=0)
    df.index = df.index.astype(str)
    obs = pd.Index(adata.obs_names.astype(str))

    if df.index.duplicated().any():
        dup = df.index[df.index.duplicated()].unique().tolist()[:10]
        raise ValueError(f"Embedding CSV '{csv_path}' has duplicate row IDs. Examples: {dup}")

    missing = obs[~obs.isin(df.index)].tolist()
    extra = df.index[~df.index.isin(obs)].tolist()

    if missing:
        raise ValueError(
            f"Embedding CSV '{csv_path}' is missing {len(missing)} manifest cells. Examples: {missing[:10]}"
        )
    if extra and require_exact_rows:
        raise ValueError(
            f"Embedding CSV '{csv_path}' has {len(extra)} extra rows not in manifest condition. Examples: {extra[:10]}"
        )
    if extra:
        logging.warning("Embedding CSV '%s' has %d extra rows; ignoring extras.", csv_path, len(extra))

    df = df.loc[obs, :]
    X = _nan_guard(df.to_numpy(dtype=np.float32, copy=True), name=f"CSV[{key}]")
    if X.shape[0] != adata.n_obs:
        raise ValueError(f"Aligned embedding {key} row count {X.shape[0]} != adata.n_obs {adata.n_obs}")
    if X.shape[1] < 1:
        raise ValueError(f"Embedding CSV '{csv_path}' has no dimensions.")

    adata.obsm[key] = X
    logging.info("Loaded embedding '%s' from %s with shape=%s", key, csv_path, X.shape)

    return {
        "Key": key,
        "CSV": csv_path,
        "Rows": int(X.shape[0]),
        "Dims": int(X.shape[1]),
        "MissingRows": int(len(missing)),
        "ExtraRows": int(len(extra)),
        "Variance": float(np.var(X)),
    }


def load_condition_external_embeddings(
    adata_pp: ad.AnnData,
    condition: RobustCondition,
    cfg: HieMultiRobustConfig,
    methods_requested: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    validation_rows: List[Dict[str, Any]] = []

    specs: List[Tuple[str, str]] = []
    if methods_requested is None or any(m in ("Seurat", "GenoRefine(Seurat)") for m in methods_requested):
        specs.append(("X_seurat", cfg.seurat_prefix))
    if methods_requested is None or any(m in ("Online iNMF", "GenoRefine(Online iNMF)") for m in methods_requested):
        specs.append(("X_online_inmf", cfg.online_inmf_prefix))

    for key, prefix in specs:
        path = external_embedding_path(cfg, prefix, condition.context)
        try:
            row = load_embedding_csv_align(
                adata_pp,
                path,
                key=key,
                require_exact_rows=bool(cfg.require_external_exact_rows),
            )
            row.update({"Context": condition.context, "Status": "OK"})
            validation_rows.append(row)
        except Exception as e:
            if cfg.require_r_embeddings:
                raise
            logging.warning("Could not load %s for context=%s: %s", key, condition.context, e)
            validation_rows.append(
                {
                    "Context": condition.context,
                    "Key": key,
                    "CSV": path,
                    "Status": "MISSING_OR_INVALID",
                    "Error": str(e),
                }
            )

    return validation_rows


# -----------------------------
# Metrics / clustering helpers
# -----------------------------
def compute_lisi_python(X: np.ndarray, labels: np.ndarray, k: int = 90) -> float:
    X = np.asarray(X)
    labels = np.asarray(labels)
    n = X.shape[0]
    if n <= 1:
        return float("nan")
    k = min(int(k), n - 1)
    if k < 1:
        return float("nan")

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
        simpson = float(np.sum(probs ** 2))
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


def cluster_purity_entropy(
    adata: ad.AnnData,
    class_key: str = "class",
    cluster_key: str = "leiden",
    eps: float = 1e-12,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    if class_key not in adata.obs or cluster_key not in adata.obs:
        summary = {"Purity_MeanWeighted": float("nan"), "Entropy_MeanWeighted": float("nan")}
        return summary, pd.DataFrame(columns=["Cluster", "N", "DominantClass", "Purity", "Entropy"])

    y = adata.obs[class_key].astype(str)
    z = adata.obs[cluster_key].astype(str)

    tab = pd.crosstab(z, y)
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

    df = pd.DataFrame(
        {
            "Cluster": tab.index.astype(str),
            "N": n_per_cluster.astype(int),
            "DominantClass": dom_class.astype(str),
            "Purity": purity.astype(float),
            "Entropy": entropy.astype(float),
        }
    ).sort_values("N", ascending=False)

    if n_total <= 0:
        summary = {"Purity_MeanWeighted": float("nan"), "Entropy_MeanWeighted": float("nan")}
    else:
        w = n_per_cluster / n_total
        summary = {
            "Purity_MeanWeighted": float(np.sum(w * purity)),
            "Entropy_MeanWeighted": float(np.sum(w * entropy)),
        }

    return summary, df


def cluster_and_score(
    adata: ad.AnnData,
    rep_for_neighbors: Optional[str],
    cfg: HieMultiRobustConfig,
    embed_key_for_metrics: str,
) -> Tuple[float, Dict[str, Any]]:
    cluster_key = "leiden"
    has_labels = "class" in adata.obs

    if rep_for_neighbors is None:
        if has_labels and cfg.leiden_target_n_clusters_if_labels_exist:
            used_res = tune_leiden_resolution_existing_graph(
                adata=adata,
                target_k=max(2, int(adata.obs["class"].nunique())),
                res_grid=list(cfg.leiden_res_grid),
                key_added=cluster_key,
                random_state=int(cfg.seed),
            )
        else:
            used_res = float(cfg.leiden_resolution_default)
            sc.tl.leiden(adata, resolution=used_res, key_added=cluster_key, random_state=int(cfg.seed))
    else:
        if rep_for_neighbors not in adata.obsm:
            raise KeyError(f"Missing representation '{rep_for_neighbors}' in adata.obsm")
        if has_labels and cfg.leiden_target_n_clusters_if_labels_exist:
            used_res = tune_leiden_resolution_to_target(
                adata=adata,
                use_rep=rep_for_neighbors,
                target_k=max(2, int(adata.obs["class"].nunique())),
                n_neighbors=int(cfg.n_neighbors),
                res_grid=list(cfg.leiden_res_grid),
                key_added=cluster_key,
                random_state=int(cfg.seed),
            )
        else:
            sc.pp.neighbors(adata, use_rep=rep_for_neighbors, n_neighbors=int(cfg.n_neighbors))
            used_res = float(cfg.leiden_resolution_default)
            sc.tl.leiden(adata, resolution=used_res, key_added=cluster_key, random_state=int(cfg.seed))

    y_pred = adata.obs[cluster_key].astype(str).to_numpy()
    out: Dict[str, Any] = {"LeidenRes": float(used_res), "NClusters": int(adata.obs[cluster_key].nunique())}

    if has_labels:
        y_true = adata.obs["class"].astype(str).to_numpy()
        out["ARI"] = float(adjusted_rand_score(y_true, y_pred))
        out["RI"] = float(rand_score(y_true, y_pred))
    else:
        out["ARI"] = float("nan")
        out["RI"] = float("nan")

    try:
        X = _nan_guard(np.asarray(adata.obsm[embed_key_for_metrics]), f"metrics_in[{embed_key_for_metrics}]")
        if len(np.unique(y_pred)) >= 2 and X.shape[0] > len(np.unique(y_pred)):
            out["Silhouette"] = float(silhouette_score(X, y_pred))
        else:
            out["Silhouette"] = float("nan")
    except Exception as e:
        logging.debug("Silhouette failed: %s", e)
        out["Silhouette"] = float("nan")

    try:
        X = _nan_guard(np.asarray(adata.obsm[embed_key_for_metrics]), f"ilisi_in[{embed_key_for_metrics}]")
        out["iLISI"] = float(compute_lisi_python(X, adata.obs["batch"].astype(str).to_numpy(), k=int(cfg.lisi_k)))
    except Exception as e:
        logging.debug("iLISI failed: %s", e)
        out["iLISI"] = float("nan")

    return float(used_res), out


def save_umap_png(adata: ad.AnnData, color_key: str, out_png: str, title: str) -> None:
    sc.pl.umap(adata, color=color_key, show=False, title=title)
    import matplotlib.pyplot as plt

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# Method runners
# -----------------------------
def run_scanorama(adata_in: ad.AnnData) -> Tuple[ad.AnnData, str]:
    if scanorama is None:
        raise RuntimeError("scanorama not installed")
    a = adata_in.copy()
    batches = sorted(a.obs["batch"].astype(str).unique().tolist())
    adatas_list = [a[a.obs["batch"].astype(str) == b].copy() for b in batches]
    adatas_list = [x for x in adatas_list if x.n_obs > 0]
    if len(adatas_list) < 2:
        raise RuntimeError("Scanorama requires at least two non-empty batches")
    corrected = scanorama.correct_scanpy(adatas_list, return_dimred=True)
    out = ad.concat(corrected, join="outer")
    out.obs_names_make_unique()
    out.obs["batch"] = out.obs["batch"].astype(str).astype("category")
    if "class" in out.obs:
        out.obs["class"] = out.obs["class"].astype(str).astype("category")
    out.obsm["X_scanorama"] = _nan_guard(np.asarray(out.obsm["X_scanorama"]), "X_scanorama")
    return out, "X_scanorama"


def run_harmony(adata_in: ad.AnnData) -> Tuple[ad.AnnData, str]:
    if hm is None:
        raise RuntimeError("harmonypy not installed")
    a = adata_in.copy()
    Xp = _nan_guard(np.asarray(a.obsm["X_pca"]), "X_pca_for_harmony")
    ho = hm.run_harmony(Xp, a.obs, "batch")
    a.obsm["X_harmony"] = _nan_guard(ho.Z_corr.T, "X_harmony")
    return a, "X_harmony"


def run_bbknn(adata_in: ad.AnnData) -> Tuple[ad.AnnData, str]:
    if bbknn is None:
        raise RuntimeError("bbknn not installed")
    a = adata_in.copy()
    bbknn.bbknn(a, batch_key="batch")
    return a, "X_pca"


def run_nmf(adata_in: ad.AnnData, cfg: HieMultiRobustConfig) -> Tuple[ad.AnnData, str]:
    if NMF is None:
        raise RuntimeError("sklearn NMF not available")
    a = adata_in.copy()
    X = a.X
    X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X_dense = np.asarray(X_dense, dtype=np.float64)
    min_val = float(np.min(X_dense))
    if min_val < 0:
        X_dense = X_dense - min_val
    n_components = min(int(cfg.nmf_components), max(2, min(X_dense.shape) - 1))
    model = NMF(
        n_components=n_components,
        init="nndsvda",
        max_iter=int(cfg.nmf_max_iter),
        random_state=int(cfg.seed),
    )
    a.obsm["X_nmf"] = _nan_guard(model.fit_transform(X_dense), "X_nmf")
    return a, "X_nmf"


def run_external_embedding(adata_in: ad.AnnData, embed_key: str) -> Tuple[ad.AnnData, str]:
    a = adata_in.copy()
    if embed_key not in a.obsm:
        raise KeyError(f"Missing '{embed_key}' in adata.obsm. Check condition-specific R CSV output.")
    a.obsm[embed_key] = _nan_guard(np.asarray(a.obsm[embed_key]), embed_key)
    return a, embed_key


def run_diffmap(adata_in: ad.AnnData, cfg: HieMultiRobustConfig) -> Tuple[ad.AnnData, str]:
    a = adata_in.copy()
    sc.pp.neighbors(a, use_rep="X_pca", n_neighbors=int(cfg.n_neighbors))
    n_comps = min(int(cfg.diffmap_n_comps), max(2, a.n_obs - 1))
    sc.tl.diffmap(a, n_comps=n_comps)
    if cfg.diffmap_embed_key not in a.obsm:
        raise KeyError("Expected diffusion map embedding missing at obsm['X_diffmap'].")
    a.obsm[cfg.diffmap_embed_key] = _nan_guard(np.asarray(a.obsm[cfg.diffmap_embed_key]), "X_diffmap")
    return a, cfg.diffmap_embed_key


def run_phate(adata_in: ad.AnnData, cfg: HieMultiRobustConfig) -> Tuple[ad.AnnData, str]:
    if phate is None:
        raise RuntimeError("phate not installed (pip install phate)")
    a = adata_in.copy()
    X = _nan_guard(np.asarray(a.obsm["X_pca"]), "X_pca_for_PHATE")
    n_components = min(int(cfg.phate_n_components), max(2, X.shape[1]))
    ph = phate.PHATE(
        n_components=n_components,
        knn=int(cfg.phate_knn),
        decay=int(cfg.phate_decay),
        t=cfg.phate_t,
        random_state=int(cfg.seed),
        verbose=False,
    )
    Z = ph.fit_transform(X)
    a.obsm[cfg.phate_embed_key] = _nan_guard(np.asarray(Z), "X_phate")
    return a, cfg.phate_embed_key


def run_genodr_on_key(adata_in: ad.AnnData, in_key: str, out_key: str, cfg: HieMultiRobustConfig) -> Tuple[ad.AnnData, str]:
    if gp is None:
        raise RuntimeError("genomap.genoDR backend not available (install genomap)")
    a = adata_in.copy()
    if in_key not in a.obsm:
        raise KeyError(f"Missing embedding '{in_key}' in adata.obsm")
    X = _nan_guard(np.asarray(a.obsm[in_key]), f"genorefine_in[{in_key}]")

    n_clusters = max(2, int(a.obs["class"].nunique())) if "class" in a.obs else 10
    Z = gp.genoDR(
        X,
        n_dim=int(cfg.genodr_dim),
        n_clusters=int(n_clusters),
        colNum=int(cfg.genodr_col),
        rowNum=int(cfg.genodr_row),
    )
    a.obsm[out_key] = _nan_guard(np.asarray(Z), f"genorefine_out[{out_key}]")
    return a, out_key


def run_method_and_score(
    method: str,
    adata_pp: ad.AnnData,
    cfg: HieMultiRobustConfig,
    condition: RobustCondition,
) -> Tuple[Dict[str, Any], ad.AnnData, pd.DataFrame]:
    t0 = time.time()

    if method == "Scanorama":
        adata, embed_key = run_scanorama(adata_pp)
        rep_for_neighbors = embed_key
        embed_key_for_metrics = embed_key
    elif method == "Harmony":
        adata, embed_key = run_harmony(adata_pp)
        rep_for_neighbors = embed_key
        embed_key_for_metrics = embed_key
    elif method == "Seurat":
        adata, embed_key = run_external_embedding(adata_pp, "X_seurat")
        rep_for_neighbors = embed_key
        embed_key_for_metrics = embed_key
    elif method == "Online iNMF":
        adata, embed_key = run_external_embedding(adata_pp, "X_online_inmf")
        rep_for_neighbors = embed_key
        embed_key_for_metrics = embed_key
    elif method == "GenoRefine(Scanorama)":
        ad_scan, scan_key = run_scanorama(adata_pp)
        adata, embed_key = run_genodr_on_key(ad_scan, scan_key, "X_genorefine_scanorama", cfg)
        rep_for_neighbors = embed_key
        embed_key_for_metrics = embed_key
    elif method == "GenoRefine(Harmony)":
        ad_har, har_key = run_harmony(adata_pp)
        adata, embed_key = run_genodr_on_key(ad_har, har_key, "X_genorefine_harmony", cfg)
        rep_for_neighbors = embed_key
        embed_key_for_metrics = embed_key
    elif method == "GenoRefine(Seurat)":
        ad_seu, seu_key = run_external_embedding(adata_pp, "X_seurat")
        adata, embed_key = run_genodr_on_key(ad_seu, seu_key, "X_genorefine_seurat", cfg)
        rep_for_neighbors = embed_key
        embed_key_for_metrics = embed_key
    elif method == "GenoRefine(Online iNMF)":
        ad_inmf, inmf_key = run_external_embedding(adata_pp, "X_online_inmf")
        adata, embed_key = run_genodr_on_key(ad_inmf, inmf_key, "X_genorefine_online_inmf", cfg)
        rep_for_neighbors = embed_key
        embed_key_for_metrics = embed_key
    else:
        raise ValueError(f"Unknown method: {method}")

    used_res, metrics = cluster_and_score(
        adata,
        rep_for_neighbors=rep_for_neighbors,
        cfg=cfg,
        embed_key_for_metrics=embed_key_for_metrics,
    )

    if "class" in adata.obs:
        pe_summary, pe_table = cluster_purity_entropy(
            adata,
            class_key="class",
            cluster_key="leiden",
            eps=float(cfg.entropy_eps),
        )
    else:
        pe_summary = {"Purity_MeanWeighted": float("nan"), "Entropy_MeanWeighted": float("nan")}
        pe_table = pd.DataFrame(columns=["Cluster", "N", "DominantClass", "Purity", "Entropy"])

    runtime_s = float(time.time() - t0)
    result = {
        **condition.as_result_fields(),
        "Method": method,
        "EmbedKey": embed_key_for_metrics,
        "LeidenRes": float(used_res),
        "Runtime_s": runtime_s,
        "NCells": int(adata.n_obs),
        "NBatches": int(adata.obs["batch"].nunique()),
        "NClasses": int(adata.obs["class"].nunique()) if "class" in adata.obs else np.nan,
        **metrics,
        "Purity_MeanWeighted": float(pe_summary.get("Purity_MeanWeighted", np.nan)),
        "Entropy_MeanWeighted": float(pe_summary.get("Entropy_MeanWeighted", np.nan)),
    }

    pe_table = pe_table.copy()
    pe_table["Context"] = condition.context
    pe_table["Test"] = condition.test
    pe_table["Frac"] = condition.frac if condition.frac is not None else np.nan
    pe_table["Method"] = method

    return result, adata, pe_table


# -----------------------------
# Method filtering
# -----------------------------
def methods_all_default() -> List[str]:
    """Core robustness methods used in the paper-facing HIE analysis."""
    return [
        "Scanorama",
        "Harmony",
        "Seurat",
        "Online iNMF",
        "GenoRefine(Scanorama)",
        "GenoRefine(Harmony)",
        "GenoRefine(Seurat)",
        "GenoRefine(Online iNMF)",
    ]


def filter_methods_for_deps(methods_requested: List[str], adata_pp: ad.AnnData) -> List[str]:
    """Filter unavailable dependencies for the four core backbones and GenoRefine variants."""
    filtered: List[str] = []
    for m in methods_requested:
        if m in ("Scanorama", "GenoRefine(Scanorama)") and scanorama is None:
            logging.warning("Skipping %s because scanorama is not installed.", m)
            continue
        if m in ("Harmony", "GenoRefine(Harmony)") and hm is None:
            logging.warning("Skipping %s because harmonypy is not installed.", m)
            continue
        if m.startswith("GenoRefine(") and gp is None:
            logging.warning("Skipping %s because the genomap.genoDR backend is not available.", m)
            continue
        if m in ("Seurat", "GenoRefine(Seurat)") and "X_seurat" not in adata_pp.obsm:
            logging.warning("Skipping %s because condition-specific X_seurat is unavailable.", m)
            continue
        if m in ("Online iNMF", "GenoRefine(Online iNMF)") and "X_online_inmf" not in adata_pp.obsm:
            logging.warning("Skipping %s because condition-specific X_online_inmf is unavailable.", m)
            continue
        filtered.append(m)
    return filtered


# -----------------------------
# Benchmark driver
# -----------------------------
def summarize_ok(df: pd.DataFrame, group_cols: List[str], out_csv: str) -> pd.DataFrame:
    if df.empty or "ARI" not in df.columns:
        out = pd.DataFrame()
        out.to_csv(out_csv, index=False)
        return out

    metric_cols = [
        c for c in [
            "ARI", "RI", "Silhouette", "iLISI", "Purity_MeanWeighted", "Entropy_MeanWeighted", "Runtime_s",
        ] if c in df.columns
    ]
    df_ok = df[df["ARI"].notna()].copy()
    if df_ok.empty:
        out = pd.DataFrame()
    else:
        out = df_ok.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()
    out.to_csv(out_csv, index=False)
    return out


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def write_hie_pancreas_style_outputs(
    df_imb: pd.DataFrame,
    df_non: pd.DataFrame,
    pe_all: pd.DataFrame,
    cfg: HieMultiRobustConfig,
) -> Dict[str, str]:
    """Write HIE outputs mirroring the pancreas robustness script.

    The HIE driver already writes strict HIE-specific outputs, including external
    embedding validation and run metadata. This helper adds the convenience
    outputs used by the pancreas robustness workflow: extreme imbalance table,
    split all-cluster purity/entropy files, and imbalance metric curves.
    """
    outputs: Dict[str, str] = {}

    # Extreme imbalance summary: retain the full condition and strongest 10% condition.
    extreme_csv = os.path.join(cfg.out_folder, "table_extreme_imbalance_hie.csv")
    if not df_imb.empty and "ARI" in df_imb.columns and "Frac" in df_imb.columns:
        df_ok = df_imb.dropna(subset=["ARI"]).copy()
        df_ext = df_ok[df_ok["Frac"].isin([1.0, 0.10])].copy()
    else:
        df_ext = pd.DataFrame()
    df_ext.to_csv(extreme_csv, index=False)
    outputs["extreme_imbalance_csv"] = extreme_csv

    # Split all-cluster purity/entropy tables by robustness test, matching pancreas outputs.
    imb_allclusters_csv = os.path.join(cfg.out_folder, "purity_entropy_imbalance_allclusters_hie.csv")
    non_allclusters_csv = os.path.join(cfg.out_folder, "purity_entropy_nonoverlap_allclusters_hie.csv")
    if not pe_all.empty and "Test" in pe_all.columns:
        pe_all[pe_all["Test"].astype(str) == "imbalance"].copy().to_csv(imb_allclusters_csv, index=False)
        pe_all[pe_all["Test"].astype(str) == "nonoverlap"].copy().to_csv(non_allclusters_csv, index=False)
    else:
        pd.DataFrame().to_csv(imb_allclusters_csv, index=False)
        pd.DataFrame().to_csv(non_allclusters_csv, index=False)
    outputs["purity_entropy_imbalance_allclusters_csv"] = imb_allclusters_csv
    outputs["purity_entropy_nonoverlap_allclusters_csv"] = non_allclusters_csv

    # Imbalance metric curves, using the same file names as pancreas inside the HIE output folder.
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        if not df_imb.empty:
            for metric in ["ARI", "Silhouette", "iLISI"]:
                fig_csv_key = f"fig_imbalance_curves_{metric}"
                out_png = os.path.join(cfg.out_folder, f"fig_imbalance_curves_{metric}.png")
                if metric not in df_imb.columns or "Method" not in df_imb.columns or "Frac" not in df_imb.columns:
                    outputs[fig_csv_key] = out_png
                    continue

                df_plot = df_imb.dropna(subset=[metric]).copy()
                if df_plot.empty:
                    outputs[fig_csv_key] = out_png
                    continue

                plt.figure(figsize=(9, 6))
                for method in sorted(df_plot["Method"].dropna().astype(str).unique()):
                    sub = df_plot[df_plot["Method"].astype(str) == method].sort_values("Frac")
                    plt.plot(sub["Frac"], sub[metric], marker="o", label=method)

                plt.xlabel("Fraction of cells retained in downsampled batch")
                plt.ylabel(metric)
                plt.title(f"Robustness to dataset imbalance: {metric}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_png, dpi=300)
                plt.close()
                outputs[fig_csv_key] = out_png
    except Exception as e:
        logging.warning("Could not generate HIE imbalance metric curves: %s", e)

    return outputs


def run_benchmark(cfg: HieMultiRobustConfig) -> None:
    set_all_seeds(cfg.seed)
    ensure_dir(cfg.out_folder)

    adata_raw, batch_key, label_key, counts_layer_name = read_hie_raw(cfg)
    canonical_hvgs = read_canonical_hvgs(cfg, adata_raw)
    conditions = load_conditions(cfg)

    methods_requested = parse_methods(cfg.methods, methods_all_default())
    logging.info("Requested methods: %s", methods_requested)

    hvg_audit_df = validate_robustness_hvg_audits(cfg, canonical_hvgs, methods_requested)
    hvg_audit_csv = os.path.join(cfg.out_folder, "hvg_audit_validation_hie.csv")
    hvg_audit_df.to_csv(hvg_audit_csv, index=False)

    all_rows: List[Dict[str, Any]] = []
    all_pe_tables: List[pd.DataFrame] = []
    embedding_validation_rows: List[Dict[str, Any]] = []

    for i, condition in enumerate(conditions, start=1):
        logging.info("=== [%d/%d] Context: %s (%s) ===", i, len(conditions), condition.context, condition.test)
        t_condition = time.time()

        ad_raw_cond, manifest = validate_and_subset_by_manifest(adata_raw, condition)
        ad_norm_cond = normalize_condition(ad_raw_cond, cfg, counts_layer_name=counts_layer_name)
        ad_pp = preprocess_with_canonical_hvgs(ad_norm_cond, canonical_hvgs, cfg)

        if i == 1:
            log_preprocess_fingerprint(ad_pp, tag=f"{condition.context}_PREPROCESSED")

        validation = load_condition_external_embeddings(ad_pp, condition, cfg, methods_requested)
        embedding_validation_rows.extend(validation)

        methods = filter_methods_for_deps(methods_requested, ad_pp)
        logging.info("Methods for context=%s: %s", condition.context, methods)

        if not methods:
            logging.warning("No methods available for context=%s", condition.context)
            continue

        for method in methods:
            logging.info("  Running %s", method)
            try:
                row, adata_method, pe_tbl = run_method_and_score(method, ad_pp, cfg, condition)
                all_rows.append(row)
                all_pe_tables.append(pe_tbl)

                if cfg.save_per_condition_purity_entropy_tables:
                    safe_m = sanitize_method_name(method)
                    out_tbl = os.path.join(
                        cfg.out_folder,
                        f"purity_entropy_{sanitize_tag(condition.context)}_{safe_m}_hie.csv",
                    )
                    pe_tbl.to_csv(out_tbl, index=False)

                if should_save_umap_for_context(cfg, condition.context):
                    sc.tl.umap(adata_method, random_state=int(cfg.seed))
                    safe_m = sanitize_method_name(method)
                    if "class" in adata_method.obs:
                        save_umap_png(
                            adata_method,
                            "class",
                            os.path.join(cfg.out_folder, f"UMAP_{sanitize_tag(condition.context)}_{safe_m}_class.png"),
                            title=f"{condition.context} {method} (class)",
                        )
                    save_umap_png(
                        adata_method,
                        "batch",
                        os.path.join(cfg.out_folder, f"UMAP_{sanitize_tag(condition.context)}_{safe_m}_batch.png"),
                        title=f"{condition.context} {method} (batch)",
                    )

            except Exception as e:
                logging.exception("Method failed: context=%s method=%s", condition.context, method)
                all_rows.append(
                    {
                        **condition.as_result_fields(),
                        "Method": method,
                        "Error": str(e),
                        "Runtime_s": float("nan"),
                    }
                )

        logging.info(
            "Finished context=%s in %.2f s", condition.context, float(time.time() - t_condition)
        )

    df_all = pd.DataFrame(all_rows)
    all_csv = os.path.join(cfg.out_folder, "robustness_all_hie.csv")
    df_all.to_csv(all_csv, index=False)

    if not df_all.empty and "Test" in df_all.columns:
        df_imb = df_all[df_all["Test"].astype(str) == "imbalance"].copy()
        df_non = df_all[df_all["Test"].astype(str) == "nonoverlap"].copy()
    else:
        df_imb = pd.DataFrame()
        df_non = pd.DataFrame()

    imb_csv = os.path.join(cfg.out_folder, "robustness_imbalance_hie.csv")
    non_csv = os.path.join(cfg.out_folder, "robustness_nonoverlap_hie.csv")
    df_imb.to_csv(imb_csv, index=False)
    df_non.to_csv(non_csv, index=False)

    pe_all = pd.concat(all_pe_tables, ignore_index=True) if all_pe_tables else pd.DataFrame()
    pe_all_csv = os.path.join(cfg.out_folder, "purity_entropy_cluster_tables_hie.csv")
    pe_all.to_csv(pe_all_csv, index=False)

    val_df = pd.DataFrame(embedding_validation_rows)
    val_csv = os.path.join(cfg.out_folder, "external_embedding_validation_hie.csv")
    val_df.to_csv(val_csv, index=False)

    mirrored_outputs = write_hie_pancreas_style_outputs(df_imb, df_non, pe_all, cfg)

    summarize_ok(
        df_all,
        ["Test", "Context", "Method"],
        os.path.join(cfg.out_folder, "robustness_context_summary_hie.csv"),
    )
    summarize_ok(
        df_imb,
        ["Method", "Frac"],
        os.path.join(cfg.out_folder, "purity_entropy_imbalance_summary_hie.csv"),
    )
    summarize_ok(
        df_non,
        ["Method"],
        os.path.join(cfg.out_folder, "purity_entropy_nonoverlap_summary_hie.csv"),
    )

    run_meta = {
        "script": "bench_hie_robustness_genorefine_core_mirror.py",
        "config": asdict(cfg),
        "data": {
            "data_file": cfg.data_file,
            "n_obs": int(adata_raw.n_obs),
            "n_vars": int(adata_raw.n_vars),
            "batch_key_resolved": batch_key,
            "label_key_resolved": label_key,
            "counts_layer_name": counts_layer_name,
        },
        "canonical_hvg": {
            "file": cfg.canonical_hvg_file,
            "n": len(canonical_hvgs),
            "md5_no_trailing_newline": md5_no_trailing_newline(canonical_hvgs),
            "md5_trailing_newline": md5_with_trailing_newline(canonical_hvgs),
        },
        "conditions": [c.as_result_fields() for c in conditions],
        "outputs": {
            "all_csv": all_csv,
            "imbalance_csv": imb_csv,
            "nonoverlap_csv": non_csv,
            "purity_entropy_cluster_tables_csv": pe_all_csv,
            "external_embedding_validation_csv": val_csv,
            "hvg_audit_validation_csv": hvg_audit_csv,
            **mirrored_outputs,
        },
        "hvg_audit_validation": hvg_audit_df.to_dict(orient="records"),
    }
    write_json(os.path.join(cfg.out_folder, "bench_hie_multi_robustness_run_config.json"), run_meta)

    print()
    print("=" * 80)
    print("Strict HIE core-method robustness benchmark complete")
    print(f"Conditions: {len(conditions)}")
    print(f"Rows written: {len(df_all)}")
    print(f"All results: {all_csv}")
    print(f"Imbalance results: {imb_csv}")
    print(f"Nonoverlap results: {non_csv}")
    print(f"External embedding validation: {val_csv}")
    print("=" * 80)
    print()


# -----------------------------
# CLI
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run strict canonical HIE core-method robustness benchmark.")
    p.add_argument("--base-dir", default=os.environ.get("HIE_BASE_DIR", CFG.base_dir), help="HIE project folder")
    p.add_argument("--data-file", default=CFG.data_file, help="HIE .h5ad path, relative to base-dir or absolute")
    p.add_argument("--benchmark-out", default=CFG.benchmark_out_folder, help="Benchmark_Hie_Out folder")
    p.add_argument("--robustness-folder", default=CFG.robustness_folder, help="Robustness subfolder under benchmark-out")
    p.add_argument("--out-folder", default=CFG.out_folder, help="Python benchmark output folder")
    p.add_argument("--canonical-hvg-file", default=CFG.canonical_hvg_file, help="Canonical HIE HVG txt path")
    p.add_argument("--canonical-hvg-md5-file", default=CFG.canonical_hvg_md5_file, help="Canonical HIE HVG md5 path")
    p.add_argument("--manifest-summary-csv", default=CFG.manifest_summary_csv, help="HIE robustness manifest summary CSV")
    p.add_argument("--seed", type=int, default=CFG.seed, help="Random seed")
    p.add_argument("--methods", default=CFG.methods, help="AUTO or comma-separated method list")
    p.add_argument("--batch-key", default=CFG.batch_key_fixed, help="Preferred/fixed batch key")
    p.add_argument("--label-key", default=CFG.label_key_fixed, help="Preferred/fixed label key")
    p.add_argument("--n-pcs", type=int, default=CFG.n_pcs, help="Number of PCA components")
    p.add_argument("--neighbors", type=int, default=CFG.n_neighbors, help="Neighbors for graphs")
    p.add_argument("--lisi-k", type=int, default=CFG.lisi_k, help="k for iLISI")
    p.add_argument("--no-require-r-embeddings", action="store_true", help="Do not fail if condition-specific Seurat/Online iNMF CSVs are missing")
    p.add_argument("--allow-external-extra-rows", action="store_true", help="Allow extra rows in R embedding CSVs")
    p.add_argument("--allow-md5-missing-or-mismatch", action="store_true", help="Do not fail on canonical HVG md5 missing/mismatch")
    p.add_argument("--allow-missing-canonical-genes", action="store_true", help="Do not fail if some canonical genes are absent from AnnData")
    p.add_argument("--save-umap", action="store_true", help="Save UMAP PNGs for each method/context")
    p.add_argument("--save-umap-contexts", default="", help="Comma-separated contexts to save UMAPs for; empty means all if --save-umap")
    p.add_argument("--save-per-condition-purity", action="store_true", help="Write one purity/entropy table per condition and method")
    return p


def config_from_args(args: argparse.Namespace) -> HieMultiRobustConfig:
    cfg = HieMultiRobustConfig()
    cfg.base_dir = os.path.abspath(str(args.base_dir))
    cfg.data_file = resolve_path(args.data_file, cfg.base_dir)
    cfg.benchmark_out_folder = resolve_path(args.benchmark_out, cfg.base_dir)
    cfg.robustness_folder = str(args.robustness_folder)
    cfg.out_folder = resolve_path(args.out_folder, cfg.base_dir)
    cfg.canonical_hvg_file = resolve_path(args.canonical_hvg_file, cfg.base_dir)
    cfg.canonical_hvg_md5_file = resolve_path(args.canonical_hvg_md5_file, cfg.base_dir)
    cfg.manifest_summary_csv = resolve_path(args.manifest_summary_csv, cfg.base_dir)
    cfg.seed = int(args.seed)
    cfg.methods = str(args.methods)
    cfg.batch_key_fixed = str(args.batch_key)
    cfg.label_key_fixed = str(args.label_key)
    cfg.n_pcs = int(args.n_pcs)
    cfg.n_neighbors = int(args.neighbors)
    cfg.lisi_k = int(args.lisi_k)
    cfg.require_r_embeddings = not bool(args.no_require_r_embeddings)
    cfg.require_external_exact_rows = not bool(args.allow_external_extra_rows)
    cfg.require_canonical_md5_match = not bool(args.allow_md5_missing_or_mismatch)
    cfg.require_all_canonical_genes_present = not bool(args.allow_missing_canonical_genes)
    cfg.save_umap = bool(args.save_umap)
    cfg.save_umap_contexts = tuple(x.strip() for x in str(args.save_umap_contexts).split(",") if x.strip())
    cfg.save_per_condition_purity_entropy_tables = bool(args.save_per_condition_purity)
    return cfg


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)

    ensure_dir(cfg.out_folder)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(cfg.out_folder, "bench_hie_multi_robustness.log"), mode="w"),
            logging.StreamHandler(),
        ],
    )
    sc.settings.verbosity = 0

    # Dependency notes. Methods will also be filtered per context.
    if scanorama is None:
        logging.warning("scanorama not installed -> Scanorama and GenoRefine(Scanorama) will be skipped.")
    if hm is None:
        logging.warning("harmonypy not installed -> Harmony and GenoRefine(Harmony) will be skipped.")
    if gp is None:
        logging.warning("genomap.genoDR backend not available -> GenoRefine refinements will be skipped.")

    run_benchmark(cfg)


if __name__ == "__main__":
    main()
