#!/usr/bin/env python3
"""
make_pancreas_canonical_hvg_v2.py

PURPOSE
    Generate one frozen canonical HVG list for the pancreas benchmark using a
    true separate-batch-first workflow.

WHAT IS NEW IN V2
    - Each pancreas .mat file is loaded and preprocessed independently.
    - HVGs are computed within each batch before any cross-batch aggregation.
    - No concatenation is used for HVG selection.
    - Cross-batch aggregation happens only at the gene-statistics level:
        frequency across batches + mean within-batch rank.
    - Preserves the benchmark-compatible fallback from seurat_v3 to
      cell_ranger when no counts layer exists.

PRIMARY OUTPUTS
    - pancreas_hvg_canonical.txt
    - pancreas_hvg_canonical.csv
    - pancreas_hvg_canonical.md5

ADDITIONAL AUDIT OUTPUTS
    - pancreas_hvg_batch_summary.csv
    - pancreas_hvg_membership.csv
    - pancreas_hvg_manifest.json

RANKING RULE
    Genes are ranked by:
      1) higher frequency across batches
      2) lower mean within-batch rank
      3) alphabetical gene name

IMPORTANT NOTE ABOUT FEATURE IDS
    By default, this script preserves the benchmark's implicit feature identity
    behavior: it does NOT try to replace matrix columns with gene symbols from
    the .mat files. That keeps the resulting canonical list compatible with the
    current pancreas benchmark loader.

    If you later want real gene symbols, update BOTH this script and the
    benchmark loader together.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio


# ============================================================
# CONFIG
# ============================================================
@dataclass
class CanonicalHVGConfig:
    data_folder: str = "./Dataset"
    out_dir: str = "./Benchmark_Out"

    data_files: Tuple[str, ...] = (
        "dataBaronX.mat",
        "dataMuraroX.mat",
        "dataScapleX.mat",  # benchmark token used for Segerstolpe
        "dataWangX.mat",
        "dataXinX.mat",
    )

    # Keep this mapping aligned with the benchmark loader
    batch_token_to_name: Dict[str, str] = None  # filled in __post_init__

    # Per-batch HVGs -> aggregated canonical list
    per_batch_top: int = 2000
    final_top: int = 2000
    hvg_flavor: str = "seurat_v3"

    # Benchmark-style preprocessing
    normalize_target_sum: float = 1e4
    allow_flavor_fallback: bool = True
    fail_on_mixed_batch_flavors: bool = False

    # Optional strict gene filter
    require_nonzero_in_every_batch: bool = False

    # Optional audit outputs
    save_membership_table: bool = True

    # Keep False if you want exact compatibility with current benchmark loader
    use_gene_names_from_mat: bool = False

    # Reproducibility
    seed: int = 0
    script_version: str = "v2"

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


CFG = CanonicalHVGConfig()


# ============================================================
# SMALL DATA CLASSES
# ============================================================
@dataclass
class BatchHVGResult:
    ranked_hvgs: List[str]
    used_flavor: str
    fallback_used: bool
    strategy: str


@dataclass
class BatchPayload:
    batch_name: str
    batch_id: int
    source_file: str
    mat_key: str
    adata: ad.AnnData


# ============================================================
# HELPERS
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def md5_for_gene_list(genes: List[str]) -> str:
    payload = "\n".join(map(str, genes)).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def sanitize_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(text))
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def _find_mat_data_key(d: Dict[str, Any]) -> str:
    keys = [k for k in d.keys() if not k.startswith("__")]
    starts = [k for k in keys if k.lower().startswith("data")]
    if starts:
        return starts[0]
    contains = [k for k in keys if "data" in k.lower()]
    if contains:
        return contains[0]
    if not keys:
        raise KeyError("No non-metadata keys found in .mat file.")
    return keys[0]


def _extract_batch_token_from_filename(fn: str) -> str:
    base = os.path.splitext(os.path.basename(fn))[0]
    if base.lower().startswith("data") and base.endswith("X"):
        return base[4:-1]
    if base.lower().startswith("data"):
        return base[4:]
    return base


def _flatten_textlike(x: Any) -> List[str]:
    arr = np.asarray(x)
    if arr.size == 0:
        return []
    arr = arr.squeeze()
    if arr.ndim == 0:
        return [str(arr.item())]
    out: List[str] = []
    for item in arr.ravel(order="C"):
        if isinstance(item, bytes):
            out.append(item.decode("utf-8", errors="ignore"))
        elif isinstance(item, np.ndarray):
            if item.size == 1:
                out.append(str(np.asarray(item).reshape(-1)[0]))
            else:
                out.append("".join(map(str, np.asarray(item).reshape(-1).tolist())))
        else:
            out.append(str(item))
    return out


def _extract_optional_gene_names(d: Dict[str, Any], n_vars: int) -> Optional[List[str]]:
    candidate_keys = [
        "gene",
        "genes",
        "gene_names",
        "genenames",
        "var_names",
        "features",
        "feature_names",
        "symbols",
    ]
    lower_map = {str(k).lower(): k for k in d.keys()}
    for ck in candidate_keys:
        if ck in lower_map:
            raw = d[lower_map[ck]]
            vals = _flatten_textlike(raw)
            if len(vals) == int(n_vars):
                return [str(v) for v in vals]
    return None


def matrix_from_layer(adata: ad.AnnData, layer: Optional[str]):
    if layer is not None and layer in adata.layers:
        return adata.layers[layer]
    return adata.X


def nonzero_gene_mask(adata: ad.AnnData, layer: Optional[str]) -> np.ndarray:
    X = matrix_from_layer(adata, layer)
    sums = np.asarray(X.sum(axis=0)).ravel()
    return sums > 0


def rank_hvgs_from_var(var_df: pd.DataFrame) -> List[str]:
    """
    Convert Scanpy HVG output into a deterministic ordered list.
    Preference order:
      1) highly_variable_rank ascending
      2) variances_norm descending
      3) dispersions_norm descending
      4) dispersions descending
      5) alphabetical gene name
    """
    if "highly_variable" not in var_df.columns:
        raise KeyError("Expected column 'highly_variable' missing from var table.")

    hv = var_df.loc[var_df["highly_variable"].fillna(False)].copy()
    if hv.empty:
        return []

    hv["gene"] = hv.index.astype(str)

    if "highly_variable_rank" in hv.columns and hv["highly_variable_rank"].notna().any():
        hv["__rank"] = pd.to_numeric(hv["highly_variable_rank"], errors="coerce")
        hv = hv.sort_values(["__rank", "gene"], ascending=[True, True])
        return hv["gene"].tolist()

    if "variances_norm" in hv.columns and hv["variances_norm"].notna().any():
        hv["__score"] = pd.to_numeric(hv["variances_norm"], errors="coerce")
        hv = hv.sort_values(["__score", "gene"], ascending=[False, True])
        return hv["gene"].tolist()

    if "dispersions_norm" in hv.columns and hv["dispersions_norm"].notna().any():
        hv["__score"] = pd.to_numeric(hv["dispersions_norm"], errors="coerce")
        hv = hv.sort_values(["__score", "gene"], ascending=[False, True])
        return hv["gene"].tolist()

    if "dispersions" in hv.columns and hv["dispersions"].notna().any():
        hv["__score"] = pd.to_numeric(hv["dispersions"], errors="coerce")
        hv = hv.sort_values(["__score", "gene"], ascending=[False, True])
        return hv["gene"].tolist()

    return sorted(hv["gene"].tolist())


# ============================================================
# BATCH LOADING
# ============================================================
def load_one_pancreas_batch(cfg: CanonicalHVGConfig, file_name: str, batch_id: int) -> BatchPayload:
    path = os.path.join(cfg.data_folder, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing pancreas .mat file: {path}")

    d = sio.loadmat(path)
    key = _find_mat_data_key(d)
    X = np.asarray(d[key])

    a = ad.AnnData(X)

    if cfg.use_gene_names_from_mat:
        gene_names = _extract_optional_gene_names(d, a.n_vars)
        if gene_names is not None:
            a.var_names = pd.Index([str(x) for x in gene_names])
            a.var_names_make_unique()

    sc.pp.normalize_total(a, target_sum=float(cfg.normalize_target_sum))
    sc.pp.log1p(a)

    token = _extract_batch_token_from_filename(file_name)
    batch_name = cfg.batch_token_to_name.get(token, token)

    a.obs_names = [f"Cell-{j + 1}-Batch-{batch_name}" for j in range(a.n_obs)]
    a.obs["batch"] = batch_name
    a.obs["batch_id"] = int(batch_id)
    a.var_names = pd.Index(a.var_names.astype(str))

    return BatchPayload(
        batch_name=batch_name,
        batch_id=int(batch_id),
        source_file=file_name,
        mat_key=key,
        adata=a,
    )


def load_pancreas_batches(cfg: CanonicalHVGConfig) -> List[BatchPayload]:
    batches: List[BatchPayload] = []
    for i, fn in enumerate(cfg.data_files, start=1):
        batches.append(load_one_pancreas_batch(cfg, fn, i))
    return batches


def prepare_hvg_context_from_batches(
    cfg: CanonicalHVGConfig,
    batches: List[BatchPayload],
) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """
    Freeze benchmark-compatible flavor logic for pancreas.

    Since the benchmark has no counts layer for these .mat inputs, requesting
    seurat_v3 should start from cell_ranger for consistency.
    """
    if not batches:
        raise ValueError("No batches provided.")

    has_counts_layer = any("counts" in bp.adata.layers for bp in batches)
    layer_for_hvg = None
    requested_flavor = str(cfg.hvg_flavor)
    initial_flavor = requested_flavor

    if requested_flavor == "seurat_v3" and layer_for_hvg is None:
        logging.warning(
            "Requested hvg_flavor='seurat_v3' but no counts layer exists. "
            "Using initial flavor='cell_ranger' to mirror the pancreas benchmark."
        )
        initial_flavor = "cell_ranger"

    feature_md5_by_batch = {
        bp.batch_name: md5_for_gene_list(bp.adata.var_names.astype(str).tolist())
        for bp in batches
    }
    n_vars_by_batch = {bp.batch_name: int(bp.adata.n_vars) for bp in batches}
    reference_names = batches[0].adata.var_names.astype(str).tolist()
    identical_var_names_all_batches = all(
        bp.adata.var_names.astype(str).tolist() == reference_names for bp in batches[1:]
    )

    prep_meta = {
        "has_counts_layer": bool(has_counts_layer),
        "layer_for_hvg": layer_for_hvg,
        "requested_flavor": requested_flavor,
        "initial_flavor": initial_flavor,
        "normalize_target_sum": float(cfg.normalize_target_sum),
        "preprocess_applied": f"normalize_total({cfg.normalize_target_sum})+log1p on X independently per .mat batch",
        "use_gene_names_from_mat": bool(cfg.use_gene_names_from_mat),
        "n_vars_by_batch": n_vars_by_batch,
        "feature_md5_by_batch": feature_md5_by_batch,
        "identical_var_names_all_batches": bool(identical_var_names_all_batches),
    }
    return layer_for_hvg, initial_flavor, prep_meta


# ============================================================
# HVG COMPUTATION
# ============================================================
def compute_ranked_hvgs_for_batch(
    adata_batch: ad.AnnData,
    n_top: int,
    flavor: str,
    layer: Optional[str],
    context: str,
    allow_flavor_fallback: bool,
) -> BatchHVGResult:
    """
    Compute a deterministic ranked HVG list for one batch.

    Primary attempt:
      - run requested flavor directly

    Fallback logic, if enabled:
      1) if flavor == seurat_v3 -> try cell_ranger
      2) remove all-zero genes, then try a safer flavor:
           seurat (or cell_ranger if already seurat)
    """
    n_top_use = min(int(n_top), int(adata_batch.n_vars))
    layer_use = layer if (layer is not None and layer in adata_batch.layers) else None

    def _run(aobj: ad.AnnData, flavor_local: str, layer_local: Optional[str]) -> List[str]:
        sc.pp.highly_variable_genes(
            aobj,
            n_top_genes=int(min(n_top_use, aobj.n_vars)),
            flavor=str(flavor_local),
            layer=layer_local,
            subset=False,
        )
        return rank_hvgs_from_var(aobj.var)

    try:
        genes = _run(adata_batch.copy(), flavor, layer_use)
        return BatchHVGResult(
            ranked_hvgs=genes,
            used_flavor=str(flavor),
            fallback_used=False,
            strategy="primary",
        )
    except Exception as e1:
        logging.warning(
            "HVG selection failed in %s with flavor=%s. Error: %s",
            context, flavor, str(e1),
        )
        if not allow_flavor_fallback:
            raise RuntimeError(
                f"HVG selection failed in {context} with flavor={flavor} and "
                "allow_flavor_fallback=False."
            ) from e1

    if str(flavor) == "seurat_v3":
        fallback = "cell_ranger"
        try:
            genes = _run(adata_batch.copy(), fallback, layer_use)
            logging.warning(
                "Recovered HVG selection in %s using fallback flavor=%s",
                context, fallback,
            )
            return BatchHVGResult(
                ranked_hvgs=genes,
                used_flavor=fallback,
                fallback_used=True,
                strategy="fallback_direct",
            )
        except Exception as e2:
            logging.warning(
                "Fallback HVG selection also failed in %s with flavor=%s. Error: %s",
                context, fallback, str(e2),
            )

    keep = nonzero_gene_mask(adata_batch, layer_use)
    if int(keep.sum()) == 0:
        raise ValueError(f"All genes are zero in {context}; cannot compute HVGs safely.")

    trimmed = adata_batch[:, keep].copy()
    safer_flavor = "seurat" if str(flavor) != "seurat" else "cell_ranger"
    safer_layer = layer_use if (layer_use is not None and layer_use in trimmed.layers) else None

    try:
        genes = _run(trimmed, safer_flavor, safer_layer)
        logging.warning(
            "Recovered HVG selection in %s after removing all-zero genes using fallback flavor=%s",
            context, safer_flavor,
        )
        return BatchHVGResult(
            ranked_hvgs=genes,
            used_flavor=safer_flavor,
            fallback_used=True,
            strategy="fallback_trimmed_nonzero",
        )
    except Exception as e3:
        raise RuntimeError(
            f"HVG selection failed for {context}.\n"
            f"Original flavor={flavor}, layer={layer_use}\n"
            f"Fallback flavor={safer_flavor}, layer={safer_layer}\n"
            f"Final error: {e3}"
        ) from e3


def compute_nonzero_in_every_batch_genes_from_batches(
    batches: List[BatchPayload],
    layer: Optional[str],
) -> Set[str]:
    nonzero_sets: List[Set[str]] = []
    for bp in batches:
        keep = nonzero_gene_mask(bp.adata, layer)
        genes = set(bp.adata.var_names[keep].astype(str).tolist())
        nonzero_sets.append(genes)
    if not nonzero_sets:
        return set()
    return set.intersection(*nonzero_sets)


# ============================================================
# AUDIT OUTPUTS
# ============================================================
def build_membership_table(
    hvg_table: pd.DataFrame,
    batch_rank_maps: Dict[str, Dict[str, int]],
    canonical_genes: Set[str],
) -> pd.DataFrame:
    membership = hvg_table.copy()

    for batch_name, rank_map in batch_rank_maps.items():
        safe = sanitize_name(batch_name)
        rank_col = f"rank_{safe}"
        in_col = f"in_{safe}"
        membership[rank_col] = membership["gene"].map(rank_map).astype("Int64")
        membership[in_col] = membership[rank_col].notna()

    membership["is_canonical"] = membership["gene"].isin(canonical_genes)
    return membership


def write_manifest_json(path: str, payload: Dict[str, Any]) -> None:
    def _convert(obj: Any):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, set):
            return sorted(obj)
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_convert)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    sc.settings.verbosity = 0
    np.random.seed(int(CFG.seed))

    ensure_dir(CFG.out_dir)

    logging.info("Loading pancreas batches for canonical HVG generation ...")
    batches = load_pancreas_batches(CFG)
    if len(batches) == 0:
        raise RuntimeError("No pancreas batches were loaded.")

    layer_for_hvg, initial_flavor, prep_meta = prepare_hvg_context_from_batches(CFG, batches)
    uniq_batches = [bp.batch_name for bp in batches]

    logging.info("Number of batches: %d", len(uniq_batches))
    logging.info("Batch order: %s", uniq_batches)
    logging.info("Requested HVG flavor: %s", CFG.hvg_flavor)
    logging.info("Initial HVG flavor: %s", initial_flavor)
    logging.info("Layer used for HVG: %s", layer_for_hvg if layer_for_hvg is not None else "X")
    logging.info("Per-batch top HVGs: %d", CFG.per_batch_top)
    logging.info("Final frozen HVGs: %d", CFG.final_top)

    nonzero_all_batches_genes: Optional[Set[str]] = None
    if CFG.require_nonzero_in_every_batch:
        logging.info("Computing genes that are nonzero in every batch ...")
        nonzero_all_batches_genes = compute_nonzero_in_every_batch_genes_from_batches(
            batches=batches,
            layer=layer_for_hvg,
        )
        logging.info("Genes nonzero in every batch: %d", len(nonzero_all_batches_genes))

    gene_freq = Counter()
    gene_rank_sum = Counter()
    batch_summary_rows: List[Dict[str, Any]] = []
    batch_rank_maps: Dict[str, Dict[str, int]] = {}

    for i, bp in enumerate(batches, start=1):
        sub = bp.adata
        context = f"batch={bp.batch_name}"

        logging.info(
            "[%d/%d] Computing HVGs for %s | cells=%d genes=%d | source=%s",
            i, len(batches), context, sub.n_obs, sub.n_vars, bp.source_file,
        )

        if sub.n_obs < 2:
            logging.warning("Skipping %s because it has <2 cells.", context)
            batch_summary_rows.append(
                {
                    "batch": bp.batch_name,
                    "status": "SKIPPED_TOO_FEW_CELLS",
                    "source_file": bp.source_file,
                    "mat_key": bp.mat_key,
                    "n_cells": int(sub.n_obs),
                    "n_genes": int(sub.n_vars),
                    "used_layer": layer_for_hvg if layer_for_hvg is not None else "X",
                    "requested_flavor": str(CFG.hvg_flavor),
                    "initial_flavor": initial_flavor,
                    "used_flavor": "SKIPPED",
                    "fallback_used": False,
                    "strategy": "skipped",
                    "n_hvgs": 0,
                }
            )
            batch_rank_maps[bp.batch_name] = {}
            continue

        result = compute_ranked_hvgs_for_batch(
            adata_batch=sub,
            n_top=CFG.per_batch_top,
            flavor=initial_flavor,
            layer=layer_for_hvg,
            context=context,
            allow_flavor_fallback=bool(CFG.allow_flavor_fallback),
        )

        ranked_hvgs = result.ranked_hvgs[: min(len(result.ranked_hvgs), int(CFG.per_batch_top))]
        batch_rank_maps[bp.batch_name] = {gene: rank for rank, gene in enumerate(ranked_hvgs, start=1)}

        for rank, gene in enumerate(ranked_hvgs, start=1):
            gene_freq[gene] += 1
            gene_rank_sum[gene] += rank

        batch_summary_rows.append(
            {
                "batch": bp.batch_name,
                "status": "OK",
                "source_file": bp.source_file,
                "mat_key": bp.mat_key,
                "n_cells": int(sub.n_obs),
                "n_genes": int(sub.n_vars),
                "used_layer": layer_for_hvg if layer_for_hvg is not None else "X",
                "requested_flavor": str(CFG.hvg_flavor),
                "initial_flavor": initial_flavor,
                "used_flavor": result.used_flavor,
                "fallback_used": bool(result.fallback_used),
                "strategy": result.strategy,
                "n_hvgs": int(len(ranked_hvgs)),
            }
        )

    if len(gene_freq) == 0:
        raise RuntimeError("No HVGs were collected from any batch.")

    used_flavors = sorted({
        row["used_flavor"]
        for row in batch_summary_rows
        if row["status"] == "OK" and row["used_flavor"] not in ("", "SKIPPED")
    })

    if CFG.fail_on_mixed_batch_flavors and len(used_flavors) > 1:
        raise RuntimeError(
            "Mixed batch HVG flavors detected and fail_on_mixed_batch_flavors=True.\n"
            f"Used flavors: {used_flavors}"
        )

    hvg_table = pd.DataFrame(
        {
            "gene": list(gene_freq.keys()),
            "freq": [int(gene_freq[g]) for g in gene_freq],
            "mean_rank": [float(gene_rank_sum[g] / gene_freq[g]) for g in gene_freq],
        }
    )

    hvg_table = hvg_table.sort_values(
        ["freq", "mean_rank", "gene"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    if nonzero_all_batches_genes is not None:
        before_n = len(hvg_table)
        hvg_table = hvg_table[hvg_table["gene"].isin(nonzero_all_batches_genes)].reset_index(drop=True)
        logging.info(
            "Applied require_nonzero_in_every_batch filter: %d -> %d genes",
            before_n, len(hvg_table),
        )

    if len(hvg_table) < int(CFG.final_top):
        logging.warning(
            "Only %d unique HVGs remain across batches, which is fewer than FINAL_TOP=%d.",
            len(hvg_table), CFG.final_top,
        )

    canonical = hvg_table.head(int(CFG.final_top)).copy()
    canonical_genes = canonical["gene"].astype(str).tolist()
    canonical_gene_set = set(canonical_genes)
    canonical_md5 = md5_for_gene_list(canonical_genes)

    txt_path = os.path.join(CFG.out_dir, "pancreas_hvg_canonical.txt")
    csv_path = os.path.join(CFG.out_dir, "pancreas_hvg_canonical.csv")
    md5_path = os.path.join(CFG.out_dir, "pancreas_hvg_canonical.md5")
    summary_path = os.path.join(CFG.out_dir, "pancreas_hvg_batch_summary.csv")
    membership_path = os.path.join(CFG.out_dir, "pancreas_hvg_membership.csv")
    manifest_path = os.path.join(CFG.out_dir, "pancreas_hvg_manifest.json")

    canonical["gene"].to_csv(txt_path, index=False, header=False)
    canonical.to_csv(csv_path, index=False)

    with open(md5_path, "w", encoding="utf-8") as f:
        f.write(canonical_md5 + "\n")

    batch_summary_df = pd.DataFrame(batch_summary_rows)
    batch_summary_df.to_csv(summary_path, index=False)

    if CFG.save_membership_table:
        membership_df = build_membership_table(
            hvg_table=hvg_table,
            batch_rank_maps=batch_rank_maps,
            canonical_genes=canonical_gene_set,
        )
        membership_df.to_csv(membership_path, index=False)

    manifest: Dict[str, Any] = {
        "script_name": os.path.basename(__file__) if "__file__" in globals() else "make_pancreas_canonical_hvg_v2.py",
        "script_version": CFG.script_version,
        "timestamp_utc": utc_now_iso(),
        "config": asdict(CFG),
        "input": {
            "data_folder": CFG.data_folder,
            "data_files": list(CFG.data_files),
            "batch_key": "batch",
            "batch_levels": uniq_batches,
            "n_batches": int(len(uniq_batches)),
            "total_n_obs": int(sum(bp.adata.n_obs for bp in batches)),
            "n_vars_by_batch": {bp.batch_name: int(bp.adata.n_vars) for bp in batches},
        },
        "preprocessing": prep_meta,
        "batch_hvg_summary": batch_summary_rows,
        "used_flavors": used_flavors,
        "mixed_batch_flavors": bool(len(used_flavors) > 1),
        "nonzero_in_every_batch_gene_count": (
            int(len(nonzero_all_batches_genes)) if nonzero_all_batches_genes is not None else None
        ),
        "outputs": {
            "canonical_txt": txt_path,
            "canonical_csv": csv_path,
            "canonical_md5": md5_path,
            "batch_summary_csv": summary_path,
            "membership_csv": membership_path if CFG.save_membership_table else None,
            "manifest_json": manifest_path,
        },
        "canonical_result": {
            "n_canonical_genes": int(len(canonical_genes)),
            "canonical_md5": canonical_md5,
            "top10_genes": canonical_genes[:10],
        },
    }
    write_manifest_json(manifest_path, manifest)

    print()
    print("=" * 80)
    print(f"Saved {len(canonical_genes)} canonical HVGs")
    print(f"TXT: {txt_path}")
    print(f"CSV: {csv_path}")
    print(f"MD5: {md5_path}")
    print(f"MD5 value: {canonical_md5}")
    print(f"Batch summary: {summary_path}")
    if CFG.save_membership_table:
        print(f"Membership table: {membership_path}")
    print(f"Manifest: {manifest_path}")
    print("=" * 80)
    print()

    logging.info("Done.")


if __name__ == "__main__":
    main()
