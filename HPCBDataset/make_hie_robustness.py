#!/usr/bin/env python3
"""
FILE: make_hie_robustness_hvg.py
-------------------------------------------------------------------------------
Generate exact cell manifests for STRICT HIE robustness experiments.

Run this AFTER make_hie_canonical_hvg.py and BEFORE the strict HIE Seurat /
Online iNMF robustness R scripts. The goal is to make Python and R use exactly
these same perturbed cell sets while all methods use the same frozen canonical
HIE HVGs.

This script does NOT recompute HVGs. It validates and audits the frozen
canonical file:

    ./Benchmark_Hie_Out/hie_hvg_canonical.txt
    ./Benchmark_Hie_Out/hie_hvg_canonical.md5

Default outputs:

    ./Benchmark_Hie_Out/Robustness_R_Embeddings/cells_imbalance_<batch>_frac1_00.csv
    ./Benchmark_Hie_Out/Robustness_R_Embeddings/cells_imbalance_<batch>_frac0_50.csv
    ./Benchmark_Hie_Out/Robustness_R_Embeddings/cells_imbalance_<batch>_frac0_25.csv
    ./Benchmark_Hie_Out/Robustness_R_Embeddings/cells_imbalance_<batch>_frac0_10.csv
    ./Benchmark_Hie_Out/Robustness_R_Embeddings/cells_nonoverlap_<batch>_<class>.csv
    ./Benchmark_Hie_Out/Robustness_R_Embeddings/hie_robustness_manifest_summary.csv
    ./Benchmark_Hie_Out/Robustness_R_Embeddings/hie_robustness_manifest_config.json
    ./Benchmark_Hie_Out/Robustness_R_Embeddings/hie_robustness_hvg_used.txt
    ./Benchmark_Hie_Out/Robustness_R_Embeddings/hie_robustness_hvg_used.md5

Design choices:
    - Uses fixed batch key 'tech' when present, matching the HIE canonical HVG
      script and HIE R scripts.
    - Uses fixed label key 'celltype' when present.
    - AUTO imbalance batch = largest batch by cell count.
    - AUTO nonoverlap batch = largest batch by cell count.
    - AUTO nonoverlap class = most frequent class within the resolved
      nonoverlap batch.
    - Strict canonical behavior: fail if the canonical HVG file is missing,
      has duplicate genes, has the wrong length, has an MD5 mismatch, or
      contains genes absent from the HIE .h5ad feature space.

Run:
    cd <HIE project folder>
    python make_hie_robustness_hvg.py

Optional environment override:
    set HIE_BASE_DIR=C:/path/to/HIEDataset
    python make_hie_robustness_hvg.py

Optional examples:
    python make_hie_robustness_hvg.py --imbalance-batch 10X --nonoverlap-batch 10X --nonoverlap-class Enterocyte
    python make_hie_robustness_hvg.py --fracs 1.0,0.5,0.25,0.1
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class HIERobustnessManifestConfig:
    base_dir: str = "."
    data_file: str = "./Dataset/human_pancreas_norm_complexBatch.h5ad"
    out_folder: str = "./Benchmark_Hie_Out"
    robustness_folder: str = "Robustness_R_Embeddings"

    canonical_hvg_file: str = "./Benchmark_Hie_Out/hie_hvg_canonical.txt"
    canonical_hvg_md5_file: str = "./Benchmark_Hie_Out/hie_hvg_canonical.md5"
    n_canonical_hvgs: int = 2000
    require_exact_canonical_hvg_count: bool = True
    require_canonical_md5_match: bool = True
    require_all_canonical_genes_present: bool = True

    seed: int = 0
    fixed_batch_key: str = "tech"
    fixed_label_key: str = "celltype"
    batch_key_candidates: Tuple[str, ...] = (
        "tech", "technology", "platform", "protocol", "method", "batch", "Batch",
        "dataset", "study", "donor", "orig.ident", "source",
    )
    label_key_candidates: Tuple[str, ...] = (
        "celltype", "class", "cell_type", "CellType", "label", "labels", "annotation",
    )

    imbalance_batch: str = "AUTO"
    imbalance_fracs: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.10)
    nonoverlap_batch: str = "AUTO"
    nonoverlap_class: str = "AUTO"

    summary_csv: str = "hie_robustness_manifest_summary.csv"
    config_json: str = "hie_robustness_manifest_config.json"
    hvg_used_txt: str = "hie_robustness_hvg_used.txt"
    hvg_used_md5: str = "hie_robustness_hvg_used.md5"


CFG = HIERobustnessManifestConfig()


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def set_all_seeds(seed: int) -> None:
    np.random.seed(int(seed))


def resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def md5_no_trailing_newline(items: Sequence[str]) -> str:
    payload = "\n".join(map(str, items)).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def sanitize_tag(text: Any) -> str:
    s = str(text)
    s = s.strip()
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "NA"


def frac_tag(frac: float) -> str:
    return f"frac{float(frac):.2f}".replace(".", "_")


def parse_fracs(text: str) -> Tuple[float, ...]:
    vals: List[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        x = float(part)
        if not (0 < x <= 1.0):
            raise ValueError(f"Fractions must be in (0, 1]. Got {x}")
        vals.append(x)
    if not vals:
        raise ValueError("At least one fraction is required.")
    return tuple(vals)


def pick_key(obs: pd.DataFrame, fixed: Optional[str], candidates: Tuple[str, ...], require_two_levels: bool) -> Optional[str]:
    if fixed is not None and fixed in obs.columns:
        vals = obs[fixed].dropna().astype(str)
        if (vals.nunique() >= 2) or (not require_two_levels and vals.nunique() >= 1):
            return fixed

    for key in candidates:
        if key not in obs.columns:
            continue
        vals = obs[key].dropna().astype(str)
        if (vals.nunique() >= 2) or (not require_two_levels and vals.nunique() >= 1):
            return key
    return None


def largest_category(series: pd.Series) -> str:
    vc = series.astype(str).value_counts(dropna=False)
    if vc.empty:
        raise ValueError("Cannot choose largest category from an empty series.")
    return str(vc.index[0])


def most_frequent_class_within_batch(adata: ad.AnnData, batch: str) -> str:
    mask = adata.obs["batch"].astype(str) == str(batch)
    if int(mask.sum()) == 0:
        raise ValueError(f"Batch '{batch}' has no cells; cannot choose AUTO nonoverlap class.")
    vc = adata.obs.loc[mask, "class"].astype(str).value_counts(dropna=False)
    if vc.empty:
        raise ValueError(f"Batch '{batch}' has no class labels; cannot choose AUTO nonoverlap class.")
    return str(vc.index[0])


def context_tag(test_tag: str, batch: str, frac: Optional[float] = None, celltype: Optional[str] = None) -> str:
    if test_tag == "imbalance":
        if frac is None:
            raise ValueError("frac is required for imbalance context tags.")
        return f"imbalance_{sanitize_tag(batch)}_{frac_tag(float(frac))}"
    if test_tag == "nonoverlap":
        if celltype is None:
            raise ValueError("celltype is required for nonoverlap context tags.")
        return f"nonoverlap_{sanitize_tag(batch)}_{sanitize_tag(celltype)}"
    raise ValueError(f"Unknown test tag: {test_tag}")


# -----------------------------------------------------------------------------
# Canonical HVG validation
# -----------------------------------------------------------------------------
def read_and_validate_canonical_hvgs(cfg: HIERobustnessManifestConfig, adata: ad.AnnData) -> Tuple[List[str], Dict[str, Any]]:
    hvg_path = str(cfg.canonical_hvg_file)
    md5_path = str(cfg.canonical_hvg_md5_file)

    if not os.path.exists(hvg_path):
        raise FileNotFoundError(
            f"Canonical HIE HVG file not found: {hvg_path}. Run make_hie_canonical_hvg.py first."
        )

    with open(hvg_path, "r", encoding="utf-8") as f:
        raw = [line.strip() for line in f]
    genes = [g for g in raw if g]

    if not genes:
        raise ValueError(f"Canonical HIE HVG file is empty: {hvg_path}")

    duplicated = sorted(pd.Series(genes)[pd.Series(genes).duplicated()].unique().tolist())
    if duplicated:
        raise ValueError(
            f"Canonical HIE HVG file contains duplicate genes: {len(duplicated)}. "
            f"Examples: {duplicated[:10]}"
        )

    if cfg.require_exact_canonical_hvg_count and len(genes) != int(cfg.n_canonical_hvgs):
        raise ValueError(
            f"Expected exactly {cfg.n_canonical_hvgs} canonical HIE HVGs, but found {len(genes)} in {hvg_path}."
        )

    md5_value = md5_no_trailing_newline(genes)
    expected_md5: Optional[str] = None
    md5_match: Optional[bool] = None

    if os.path.exists(md5_path):
        with open(md5_path, "r", encoding="utf-8") as f:
            vals = [line.strip() for line in f if line.strip()]
        expected_md5 = vals[0] if vals else None
        if expected_md5:
            md5_match = (expected_md5 == md5_value)
            if cfg.require_canonical_md5_match and not md5_match:
                raise ValueError(
                    f"Canonical HIE HVG MD5 mismatch. Expected {expected_md5} from {md5_path}, "
                    f"computed {md5_value} from {hvg_path}."
                )
    elif cfg.require_canonical_md5_match:
        raise FileNotFoundError(f"Canonical HIE HVG MD5 file not found: {md5_path}")

    var_names = set(map(str, adata.var_names.tolist()))
    missing = [g for g in genes if g not in var_names]
    if cfg.require_all_canonical_genes_present and missing:
        raise ValueError(
            f"Canonical HIE HVG file contains genes absent from the .h5ad var_names. "
            f"Missing={len(missing)}; examples={missing[:10]}"
        )

    meta = {
        "canonical_hvg_file": os.path.abspath(hvg_path),
        "canonical_hvg_md5_file": os.path.abspath(md5_path) if os.path.exists(md5_path) else None,
        "n_canonical_hvgs": int(len(genes)),
        "computed_md5_no_trailing_newline": md5_value,
        "expected_md5": expected_md5,
        "md5_match": md5_match,
        "missing_from_adata_var_names": int(len(missing)),
        "top10_hvgs": genes[:10],
    }
    return genes, meta


def write_hvg_audit_files(genes: List[str], md5_value: str, out_dir: str, cfg: HIERobustnessManifestConfig) -> Dict[str, str]:
    hvg_txt = os.path.join(out_dir, cfg.hvg_used_txt)
    hvg_md5 = os.path.join(out_dir, cfg.hvg_used_md5)

    with open(hvg_txt, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(genes))
        f.write("\n")
    with open(hvg_md5, "w", encoding="utf-8", newline="\n") as f:
        f.write(md5_value + "\n")

    return {"hvg_used_txt": hvg_txt, "hvg_used_md5": hvg_md5}


# -----------------------------------------------------------------------------
# Data loading and metadata normalization
# -----------------------------------------------------------------------------
def load_hie_raw(cfg: HIERobustnessManifestConfig) -> Tuple[ad.AnnData, str, str, Dict[str, Any]]:
    if not os.path.exists(cfg.data_file):
        raise FileNotFoundError(f"Could not find HIE .h5ad file: {cfg.data_file}")

    a = ad.read_h5ad(cfg.data_file)
    a.var_names_make_unique()

    batch_key = pick_key(a.obs, cfg.fixed_batch_key, cfg.batch_key_candidates, require_two_levels=True)
    label_key = pick_key(a.obs, cfg.fixed_label_key, cfg.label_key_candidates, require_two_levels=False)

    if batch_key is None:
        raise ValueError(
            f"No usable HIE batch key found. Tried fixed='{cfg.fixed_batch_key}' and candidates "
            f"{cfg.batch_key_candidates}. Available obs keys: {list(a.obs.columns)[:50]}"
        )
    if label_key is None:
        raise ValueError(
            f"No usable HIE label key found. Tried fixed='{cfg.fixed_label_key}' and candidates "
            f"{cfg.label_key_candidates}. Available obs keys: {list(a.obs.columns)[:50]}"
        )

    a.obs["batch"] = a.obs[batch_key].astype(str).astype("category")
    a.obs["class"] = a.obs[label_key].astype(str).astype("category")

    # Stable numeric batch_id for audit/readability. The R scripts should use
    # cell_id and batch names, not this integer, for alignment.
    batch_levels_sorted = sorted(pd.unique(a.obs["batch"].astype(str)).tolist())
    batch_to_id = {b: i + 1 for i, b in enumerate(batch_levels_sorted)}
    a.obs["batch_id"] = a.obs["batch"].astype(str).map(batch_to_id).astype(int)

    if a.obs_names.has_duplicates:
        raise ValueError("HIE .h5ad obs_names are not unique; strict cell manifests require unique cell IDs.")
    if a.var_names.has_duplicates:
        raise ValueError("HIE .h5ad var_names are not unique after var_names_make_unique(); cannot validate HVGs safely.")

    meta = {
        "data_file": os.path.abspath(cfg.data_file),
        "n_obs": int(a.n_obs),
        "n_vars": int(a.n_vars),
        "batch_key_resolved": batch_key,
        "label_key_resolved": label_key,
        "n_batches": int(a.obs["batch"].nunique()),
        "n_classes": int(a.obs["class"].nunique()),
        "batch_counts": a.obs["batch"].astype(str).value_counts().to_dict(),
        "class_counts": a.obs["class"].astype(str).value_counts().to_dict(),
    }
    return a, batch_key, label_key, meta


# -----------------------------------------------------------------------------
# Perturbations and manifest writing
# -----------------------------------------------------------------------------
def make_imbalance_dataset(adata: ad.AnnData, batch_to_downsample: str, frac: float, seed: int) -> ad.AnnData:
    rng = np.random.default_rng(int(seed))
    batch_str = adata.obs["batch"].astype(str)

    mask_other = batch_str != str(batch_to_downsample)
    mask_target = ~mask_other

    ad_other = adata[mask_other].copy()
    ad_target = adata[mask_target].copy()

    if ad_target.n_obs == 0:
        raise ValueError(f"Imbalance batch '{batch_to_downsample}' has zero cells.")

    n_keep = int(np.floor(ad_target.n_obs * float(frac)))
    if n_keep < 2:
        raise ValueError(
            f"Too few cells retained for imbalance batch '{batch_to_downsample}' at frac={frac}: n_keep={n_keep}"
        )

    if float(frac) >= 0.999:
        keep_idx = np.arange(ad_target.n_obs)
    else:
        keep_idx = rng.choice(ad_target.n_obs, size=n_keep, replace=False)

    ad_target = ad_target[keep_idx].copy()
    out = ad.concat([ad_other, ad_target], join="outer")
    out.obs_names_make_unique()
    out.obs["batch"] = out.obs["batch"].astype("category")
    out.obs["class"] = out.obs["class"].astype("category")
    out.obs["batch_id"] = out.obs["batch_id"].astype(int)
    return out


def make_nonoverlap_dataset(adata: ad.AnnData, batch: str, celltype: str) -> ad.AnnData:
    batch_str = adata.obs["batch"].astype(str)
    class_str = adata.obs["class"].astype(str)
    drop = (batch_str == str(batch)) & (class_str == str(celltype))

    n_drop = int(drop.sum())
    if n_drop == 0:
        raise ValueError(f"Nonoverlap removal would drop zero cells: batch='{batch}', class='{celltype}'.")

    out = adata[~drop].copy()
    out.obs_names_make_unique()
    out.obs["batch"] = out.obs["batch"].astype("category")
    out.obs["class"] = out.obs["class"].astype("category")
    out.obs["batch_id"] = out.obs["batch_id"].astype(int)
    return out


def write_manifest(adata: ad.AnnData, out_dir: str, ctx: str) -> str:
    ensure_dir(out_dir)
    out_csv = os.path.join(out_dir, f"cells_{ctx}.csv")

    if adata.obs_names.has_duplicates:
        raise ValueError(f"Condition '{ctx}' has duplicate cell IDs; refusing to write manifest.")

    df = pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "batch": adata.obs["batch"].astype(str).to_numpy(),
            "class": adata.obs["class"].astype(str).to_numpy(),
            "batch_id": adata.obs["batch_id"].astype(int).to_numpy(),
        }
    )
    df.to_csv(out_csv, index=False)
    return out_csv


def summarize_condition(adata: ad.AnnData, ctx: str, out_csv: str, test: str, **extra: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "Context": ctx,
        "Test": test,
        "ManifestCSV": out_csv,
        "NCells": int(adata.n_obs),
        "NBatches": int(adata.obs["batch"].nunique()),
        "NClasses": int(adata.obs["class"].nunique()),
    }
    row.update(extra)
    return row


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate strict HIE robustness cell manifests using frozen canonical HVGs.")
    p.add_argument("--base-dir", default=os.environ.get("HIE_BASE_DIR", CFG.base_dir), help="HIE project folder")
    p.add_argument("--data-file", default=CFG.data_file, help="HIE .h5ad path, relative to base-dir or absolute")
    p.add_argument("--out-folder", default=CFG.out_folder, help="Benchmark output folder, relative to base-dir or absolute")
    p.add_argument("--robustness-folder", default=CFG.robustness_folder, help="Subfolder for strict R embeddings/manifests")
    p.add_argument("--canonical-hvg-file", default=CFG.canonical_hvg_file, help="Canonical HVG txt path")
    p.add_argument("--canonical-hvg-md5-file", default=CFG.canonical_hvg_md5_file, help="Canonical HVG md5 path")
    p.add_argument("--n-canonical-hvgs", type=int, default=CFG.n_canonical_hvgs, help="Expected canonical HVG count")
    p.add_argument("--seed", type=int, default=CFG.seed, help="Random seed for downsampling")
    p.add_argument("--batch-key", default=CFG.fixed_batch_key, help="Fixed/preferred HIE batch key")
    p.add_argument("--label-key", default=CFG.fixed_label_key, help="Fixed/preferred HIE label key")
    p.add_argument("--imbalance-batch", default=CFG.imbalance_batch, help="Batch to downsample, or AUTO")
    p.add_argument("--fracs", default=",".join(map(str, CFG.imbalance_fracs)), help="Comma-separated imbalance fractions")
    p.add_argument("--nonoverlap-batch", default=CFG.nonoverlap_batch, help="Batch for nonoverlap removal, or AUTO")
    p.add_argument("--nonoverlap-class", default=CFG.nonoverlap_class, help="Class to remove from nonoverlap batch, or AUTO")
    p.add_argument(
        "--allow-md5-missing-or-mismatch",
        action="store_true",
        help="Do not fail if canonical md5 file is missing or mismatched. Not recommended for strict runs.",
    )
    p.add_argument(
        "--allow-missing-canonical-genes",
        action="store_true",
        help="Do not fail if canonical genes are missing from the .h5ad var_names. Not recommended for strict runs.",
    )
    return p


def config_from_args(args: argparse.Namespace) -> HIERobustnessManifestConfig:
    base_dir = os.path.abspath(args.base_dir)
    cfg = HIERobustnessManifestConfig()
    cfg.base_dir = base_dir
    cfg.data_file = resolve_path(args.data_file, base_dir)
    cfg.out_folder = resolve_path(args.out_folder, base_dir)
    cfg.robustness_folder = str(args.robustness_folder)
    cfg.canonical_hvg_file = resolve_path(args.canonical_hvg_file, base_dir)
    cfg.canonical_hvg_md5_file = resolve_path(args.canonical_hvg_md5_file, base_dir)
    cfg.n_canonical_hvgs = int(args.n_canonical_hvgs)
    cfg.seed = int(args.seed)
    cfg.fixed_batch_key = str(args.batch_key)
    cfg.fixed_label_key = str(args.label_key)
    cfg.imbalance_batch = str(args.imbalance_batch)
    cfg.imbalance_fracs = parse_fracs(args.fracs)
    cfg.nonoverlap_batch = str(args.nonoverlap_batch)
    cfg.nonoverlap_class = str(args.nonoverlap_class)
    cfg.require_canonical_md5_match = not bool(args.allow_md5_missing_or_mismatch)
    cfg.require_all_canonical_genes_present = not bool(args.allow_missing_canonical_genes)
    return cfg


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_all_seeds(cfg.seed)

    robust_dir = os.path.join(cfg.out_folder, cfg.robustness_folder)
    ensure_dir(robust_dir)

    logging.info("Loading HIE .h5ad and resolving metadata keys ...")
    adata, batch_key, label_key, data_meta = load_hie_raw(cfg)

    logging.info("Validating frozen canonical HIE HVGs ...")
    canonical_hvgs, hvg_meta = read_and_validate_canonical_hvgs(cfg, adata)
    audit_paths = write_hvg_audit_files(
        canonical_hvgs,
        hvg_meta["computed_md5_no_trailing_newline"],
        robust_dir,
        cfg,
    )

    batch_series = adata.obs["batch"].astype(str)
    class_series = adata.obs["class"].astype(str)

    imbalance_batch = cfg.imbalance_batch
    if str(imbalance_batch).upper() == "AUTO":
        imbalance_batch = largest_category(batch_series)
    if str(imbalance_batch) not in set(batch_series):
        raise ValueError(
            f"Imbalance batch '{imbalance_batch}' is absent from HIE batches: {sorted(set(batch_series))}"
        )

    nonoverlap_batch = cfg.nonoverlap_batch
    if str(nonoverlap_batch).upper() == "AUTO":
        nonoverlap_batch = largest_category(batch_series)
    if str(nonoverlap_batch) not in set(batch_series):
        raise ValueError(
            f"Nonoverlap batch '{nonoverlap_batch}' is absent from HIE batches: {sorted(set(batch_series))}"
        )

    nonoverlap_class = cfg.nonoverlap_class
    if str(nonoverlap_class).upper() == "AUTO":
        nonoverlap_class = most_frequent_class_within_batch(adata, str(nonoverlap_batch))
    if str(nonoverlap_class) not in set(class_series):
        raise ValueError(
            f"Nonoverlap class '{nonoverlap_class}' is absent from HIE classes: {sorted(set(class_series))[:50]}"
        )

    logging.info("Resolved imbalance batch: %s", imbalance_batch)
    logging.info("Resolved nonoverlap removal: batch=%s, class=%s", nonoverlap_batch, nonoverlap_class)

    summary_rows: List[Dict[str, Any]] = []

    for frac in cfg.imbalance_fracs:
        ctx = context_tag("imbalance", batch=str(imbalance_batch), frac=float(frac))
        pert = make_imbalance_dataset(adata, str(imbalance_batch), float(frac), seed=int(cfg.seed))
        out_csv = write_manifest(pert, robust_dir, ctx)
        n_target_original = int((batch_series == str(imbalance_batch)).sum())
        n_target_kept = int((pert.obs["batch"].astype(str) == str(imbalance_batch)).sum())
        summary_rows.append(
            summarize_condition(
                pert,
                ctx,
                out_csv,
                test="imbalance",
                DownsampleBatch=str(imbalance_batch),
                Frac=float(frac),
                DownsampleBatchOriginalN=n_target_original,
                DownsampleBatchKeptN=n_target_kept,
                RemovedBatch="",
                RemovedClass="",
                RemovedCellsN=0,
            )
        )
        logging.info("Wrote %s: n=%d -> %s", ctx, pert.n_obs, out_csv)

    ctx = context_tag("nonoverlap", batch=str(nonoverlap_batch), celltype=str(nonoverlap_class))
    n_removed = int(((adata.obs["batch"].astype(str) == str(nonoverlap_batch)) & (adata.obs["class"].astype(str) == str(nonoverlap_class))).sum())
    pert = make_nonoverlap_dataset(adata, str(nonoverlap_batch), str(nonoverlap_class))
    out_csv = write_manifest(pert, robust_dir, ctx)
    summary_rows.append(
        summarize_condition(
            pert,
            ctx,
            out_csv,
            test="nonoverlap",
            DownsampleBatch="",
            Frac=np.nan,
            DownsampleBatchOriginalN=0,
            DownsampleBatchKeptN=0,
            RemovedBatch=str(nonoverlap_batch),
            RemovedClass=str(nonoverlap_class),
            RemovedCellsN=n_removed,
        )
    )
    logging.info("Wrote %s: n=%d, removed=%d -> %s", ctx, pert.n_obs, n_removed, out_csv)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(robust_dir, cfg.summary_csv)
    summary_df.to_csv(summary_path, index=False)

    config_payload: Dict[str, Any] = {
        "script": "make_hie_robustness_hvg.py",
        "timestamp_utc": utc_now_iso(),
        "config": asdict(cfg),
        "resolved": {
            "batch_key": batch_key,
            "label_key": label_key,
            "imbalance_batch": str(imbalance_batch),
            "nonoverlap_batch": str(nonoverlap_batch),
            "nonoverlap_class": str(nonoverlap_class),
            "robust_dir": os.path.abspath(robust_dir),
        },
        "data": data_meta,
        "canonical_hvg": hvg_meta,
        "audit_paths": audit_paths,
        "summary_csv": summary_path,
        "conditions": summary_rows,
    }

    config_path = os.path.join(robust_dir, cfg.config_json)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, default=str)

    print()
    print("=" * 80)
    print("Strict HIE robustness manifests generated")
    print(f"Data file: {cfg.data_file}")
    print(f"Batch key: {batch_key} | Label key: {label_key}")
    print(f"Canonical HVGs: {len(canonical_hvgs)} | MD5: {hvg_meta['computed_md5_no_trailing_newline']}")
    print(f"Resolved imbalance batch: {imbalance_batch}")
    print(f"Resolved nonoverlap: batch={nonoverlap_batch}, class={nonoverlap_class}, removed={n_removed}")
    print(f"Manifest folder: {robust_dir}")
    print(f"Summary CSV: {summary_path}")
    print(f"Config JSON: {config_path}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
