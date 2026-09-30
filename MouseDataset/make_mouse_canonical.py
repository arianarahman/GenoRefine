#!/usr/bin/env python3
"""
make_mouse_canonical_hvg_v3.py

PURPOSE
    Generate one frozen canonical HVG list for the mouse benchmark.

WHAT IS NEW IN V3
    - Keeps the v2 output structure, ranking rule, and audit tables.
    - Uses raw layer='counts' directly for per-batch HVG selection when
      hvg_flavor='seurat_v3' and a counts layer exists.
    - Avoids normalize_total/log1p on the counts layer before Seurat-v3 HVG
      selection, which is methodologically cleaner than the v2 behavior.
    - Still falls back to cell_ranger when seurat_v3 is requested but no
      counts layer exists.
    - Preserves pragmatic fallback handling for difficult HVG cases.

PRIMARY OUTPUTS
    - mouse_hvg_canonical.txt
    - mouse_hvg_canonical.csv
    - mouse_hvg_canonical.md5

ADDITIONAL AUDIT OUTPUTS
    - mouse_hvg_batch_summary.csv
    - mouse_hvg_membership.csv
    - mouse_hvg_manifest.json

RANKING RULE
    Genes are ranked by:
      1) higher frequency across batches
      2) lower mean within-batch rank
      3) alphabetical gene name
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

import numpy as np
import pandas as pd
import scanpy as sc


# ============================================================
# CONFIG
# ============================================================
@dataclass
class CanonicalHVGConfig:
    data_file: str = r"./Dataset/tabula-muris_sub50k_combined.h5ad"
    out_dir: str = "./Benchmark_Mouse_Out"

    # Freeze this once and reuse it everywhere
    batch_key: str = "dataset"

    # Per-batch HVGs -> aggregated canonical list
    per_batch_top: int = 2000
    final_top: int = 2000
    hvg_flavor: str = "seurat_v3"

    # Benchmark-style preprocessing
    mirror_benchmark_preprocess: bool = True
    normalize_target_sum: float = 1e4

    # Counts / flavor behavior
    prefer_counts_layer: bool = True
    require_counts_for_seurat_v3: bool = False
    allow_flavor_fallback: bool = True
    fail_on_mixed_batch_flavors: bool = False

    # Optional strict gene filter
    require_nonzero_in_every_batch: bool = False

    # Optional audit outputs
    save_membership_table: bool = True

    # Reproducibility
    seed: int = 0
    script_version: str = "v3"


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


def matrix_from_layer(adata, layer: Optional[str]):
    if layer is not None and layer in adata.layers:
        return adata.layers[layer]
    return adata.X


def nonzero_gene_mask(adata, layer: Optional[str]) -> np.ndarray:
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


def prepare_hvg_source(
    adata,
    cfg: CanonicalHVGConfig,
) -> Tuple[Any, Optional[str], str, Dict[str, Any]]:
    """
    Prepare a temporary AnnData object used only for canonical HVG generation.

    V3 behavior:
      - if hvg_flavor='seurat_v3' and a counts layer exists:
            use raw layer='counts' directly for HVG selection
      - otherwise:
            prepare logged values for HVG selection by applying
            normalize_total + log1p to the preferred source
    """
    a = adata.copy()
    has_counts_layer = "counts" in a.layers
    layer_for_hvg = "counts" if (cfg.prefer_counts_layer and has_counts_layer) else None
    requested_flavor = str(cfg.hvg_flavor)
    initial_flavor = requested_flavor

    if requested_flavor == "seurat_v3" and layer_for_hvg is None:
        if cfg.require_counts_for_seurat_v3:
            raise RuntimeError(
                "Requested hvg_flavor='seurat_v3' but no 'counts' layer exists and "
                "require_counts_for_seurat_v3=True."
            )
        logging.warning(
            "Requested hvg_flavor='seurat_v3' but no 'counts' layer exists. "
            "Using initial flavor='cell_ranger' instead."
        )
        initial_flavor = "cell_ranger"

    # V3 CHANGE:
    # If strict Seurat-v3 on counts is possible, keep raw counts untouched.
    if requested_flavor == "seurat_v3" and layer_for_hvg is not None:
        logging.info(
            "Using raw layer='counts' directly for HVG selection; "
            "counts layer is left unmodified before Seurat-v3 HVG computation."
        )
        preprocess_applied = "raw counts in layer='counts' used directly for Seurat-v3 HVG selection"
    else:
        preprocess_applied = "none"
        if cfg.mirror_benchmark_preprocess:
            if layer_for_hvg is not None:
                logging.info(
                    "Preparing logged layer='counts' for HVG selection: normalize_total(%s) + log1p",
                    cfg.normalize_target_sum,
                )
                sc.pp.normalize_total(a, target_sum=float(cfg.normalize_target_sum), layer=layer_for_hvg)
                sc.pp.log1p(a, layer=layer_for_hvg)
                a.X = a.layers[layer_for_hvg]
                preprocess_applied = f"normalize_total({cfg.normalize_target_sum})+log1p on layer='counts'"
            else:
                logging.info(
                    "Preparing logged X for HVG selection: normalize_total(%s) + log1p",
                    cfg.normalize_target_sum,
                )
                sc.pp.normalize_total(a, target_sum=float(cfg.normalize_target_sum))
                sc.pp.log1p(a)
                preprocess_applied = f"normalize_total({cfg.normalize_target_sum})+log1p on X"

    prep_meta = {
        "has_counts_layer": bool(has_counts_layer),
        "layer_for_hvg": layer_for_hvg,
        "requested_flavor": requested_flavor,
        "initial_flavor": initial_flavor,
        "mirror_benchmark_preprocess": bool(cfg.mirror_benchmark_preprocess),
        "normalize_target_sum": float(cfg.normalize_target_sum),
        "preprocess_applied": preprocess_applied,
        "strict_raw_counts_seurat_v3": bool(requested_flavor == "seurat_v3" and layer_for_hvg is not None),
    }
    return a, layer_for_hvg, initial_flavor, prep_meta


def compute_ranked_hvgs_for_batch(
    adata_batch,
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

    def _run(aobj, flavor_local: str, layer_local: Optional[str]) -> List[str]:
        sc.pp.highly_variable_genes(
            aobj,
            n_top_genes=int(min(n_top_use, aobj.n_vars)),
            flavor=str(flavor_local),
            layer=layer_local,
            subset=False,
        )
        return rank_hvgs_from_var(aobj.var)

    # Primary attempt
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

    # Fallback 1: seurat_v3 -> cell_ranger
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

    # Fallback 2: remove all-zero genes and try a safer flavor
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


def compute_nonzero_in_every_batch_genes(
    adata,
    batch_key: str,
    batch_levels: List[str],
    layer: Optional[str],
) -> Set[str]:
    mask = np.ones(adata.n_vars, dtype=bool)
    batches = adata.obs[batch_key].astype(str).to_numpy()
    for b in batch_levels:
        sub = adata[batches == b]
        mask &= nonzero_gene_mask(sub, layer)
    return set(adata.var_names[mask].astype(str).tolist())


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
    def _convert(obj):
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

    if not os.path.exists(CFG.data_file):
        raise FileNotFoundError(f"Input file not found: {CFG.data_file}")

    logging.info("Reading AnnData: %s", CFG.data_file)
    adata_raw = sc.read_h5ad(CFG.data_file)
    adata_raw.var_names_make_unique()

    if CFG.batch_key not in adata_raw.obs.columns:
        raise KeyError(
            f"Missing batch key '{CFG.batch_key}' in adata.obs.\n"
            f"Available keys: {list(adata_raw.obs.columns)}"
        )

    adata_hvg, layer_for_hvg, initial_flavor, prep_meta = prepare_hvg_source(adata_raw, CFG)

    batches = adata_hvg.obs[CFG.batch_key].astype(str)
    uniq_batches = sorted(pd.unique(batches).tolist())

    if len(uniq_batches) < 2:
        logging.warning(
            "Batch key '%s' has only %d level(s). This is unusual for integration benchmarking.",
            CFG.batch_key, len(uniq_batches),
        )

    logging.info("Batch key: %s", CFG.batch_key)
    logging.info("Number of batches: %d", len(uniq_batches))
    logging.info("Requested HVG flavor: %s", CFG.hvg_flavor)
    logging.info("Initial HVG flavor: %s", initial_flavor)
    logging.info("Layer used for HVG: %s", layer_for_hvg if layer_for_hvg is not None else "X")
    logging.info("Per-batch top HVGs: %d", CFG.per_batch_top)
    logging.info("Final frozen HVGs: %d", CFG.final_top)

    nonzero_all_batches_genes: Optional[Set[str]] = None
    if CFG.require_nonzero_in_every_batch:
        logging.info("Computing genes that are nonzero in every batch ...")
        nonzero_all_batches_genes = compute_nonzero_in_every_batch_genes(
            adata=adata_hvg,
            batch_key=CFG.batch_key,
            batch_levels=uniq_batches,
            layer=layer_for_hvg,
        )
        logging.info(
            "Genes nonzero in every batch: %d",
            len(nonzero_all_batches_genes),
        )

    gene_freq = Counter()
    gene_rank_sum = Counter()
    batch_summary_rows: List[Dict[str, Any]] = []
    batch_rank_maps: Dict[str, Dict[str, int]] = {}

    batches_arr = adata_hvg.obs[CFG.batch_key].astype(str).to_numpy()

    for i, b in enumerate(uniq_batches, start=1):
        sub = adata_hvg[batches_arr == b].copy()
        context = f"batch={b}"

        logging.info(
            "[%d/%d] Computing HVGs for %s | cells=%d genes=%d",
            i, len(uniq_batches), context, sub.n_obs, sub.n_vars,
        )

        if sub.n_obs < 2:
            logging.warning("Skipping %s because it has <2 cells.", context)
            batch_summary_rows.append(
                {
                    "batch": b,
                    "status": "SKIPPED_TOO_FEW_CELLS",
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
            batch_rank_maps[b] = {}
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
        batch_rank_maps[b] = {gene: rank for rank, gene in enumerate(ranked_hvgs, start=1)}

        for rank, gene in enumerate(ranked_hvgs, start=1):
            gene_freq[gene] += 1
            gene_rank_sum[gene] += rank

        batch_summary_rows.append(
            {
                "batch": b,
                "status": "OK",
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

    txt_path = os.path.join(CFG.out_dir, "mouse_hvg_canonical.txt")
    csv_path = os.path.join(CFG.out_dir, "mouse_hvg_canonical.csv")
    md5_path = os.path.join(CFG.out_dir, "mouse_hvg_canonical.md5")
    summary_path = os.path.join(CFG.out_dir, "mouse_hvg_batch_summary.csv")
    membership_path = os.path.join(CFG.out_dir, "mouse_hvg_membership.csv")
    manifest_path = os.path.join(CFG.out_dir, "mouse_hvg_manifest.json")

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
        "script_name": os.path.basename(__file__) if "__file__" in globals() else "make_mouse_canonical_hvg_v3.py",
        "script_version": CFG.script_version,
        "timestamp_utc": utc_now_iso(),
        "config": asdict(CFG),
        "input": {
            "data_file": CFG.data_file,
            "n_obs": int(adata_raw.n_obs),
            "n_vars": int(adata_raw.n_vars),
            "batch_key": CFG.batch_key,
            "batch_levels": uniq_batches,
            "n_batches": int(len(uniq_batches)),
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