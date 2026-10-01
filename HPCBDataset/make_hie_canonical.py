# Purpose: Generate the frozen canonical HVG file for the HIE benchmark so that
#          bench_hie_master_v6.py, seurat_integration_hie_v4.R, and rliger_online_inmf_hie_v7.R
#          all use the same fixed feature space.
# Author: Ariana Rahman (Arizona State University)

"""
FILE: make_hie_canonical_hvg.py
-------------------------------------------------------------------------------
Purpose
    Generate the frozen canonical HVG file for the HIE benchmark so that
    bench_hie_master_v6.py, seurat_integration_hie_v4.R, and
    rliger_online_inmf_hie_v7.R all use the same fixed feature space.

Default input/output
    INPUT:
        ./Dataset/human_pancreas_norm_complexBatch.h5ad
    OUTPUT:
        ./Benchmark_Hie_Out/hie_hvg_canonical.txt
        ./Benchmark_Hie_Out/hie_hvg_canonical.md5
        ./Benchmark_Hie_Out/hie_hvg_canonical_metadata.json
        ./Benchmark_Hie_Out/hie_hvg_ranked_candidates.csv

Design
    - Uses fixed batch key 'tech' when present, matching the HIE benchmark/R files.
    - Uses fixed label key 'celltype' when present for metadata only.
    - Preferentially computes HVGs on layer='counts' when available.
    - Uses Scanpy's batch-aware highly_variable_genes(..., batch_key='tech')
      to create a deterministic top-2000 canonical HVG list.
    - Writes genes in canonical order and fails if fewer than 2000 genes are selected.

Run
    cd <HIE project folder>
    python make_hie_canonical.py

Optional environment override
    set HIE_BASE_DIR=C:\\path\\to\\3.4-ScanoramaDataset
    python make_hie_canonical_hvg.py
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad


@dataclass
class CanonicalHIEHVGConfig:
    base_dir: str = "."
    data_file: str = "./Dataset/human_pancreas_norm_complexBatch.h5ad"
    out_folder: str = "./Benchmark_Hie_Out"
    out_hvg_file: str = "hie_hvg_canonical.txt"
    out_md5_file: str = "hie_hvg_canonical.md5"
    out_metadata_file: str = "hie_hvg_canonical_metadata.json"
    out_ranked_candidates_file: str = "hie_hvg_ranked_candidates.csv"

    seed: int = 0
    n_top_genes: int = 2000
    hvg_flavor: str = "seurat_v3"
    normalize_target_sum: float = 1e4

    fixed_batch_key: str = "tech"
    fixed_label_key: str = "celltype"
    batch_key_candidates: Tuple[str, ...] = (
        "tech", "technology", "platform", "protocol", "method", "batch", "Batch",
        "dataset", "study", "donor", "orig.ident", "source",
    )
    label_key_candidates: Tuple[str, ...] = (
        "celltype", "class", "cell_type", "CellType", "label", "labels", "annotation",
    )

    prefer_counts_layer: bool = True
    fallback_flavor_if_no_counts: str = "cell_ranger"
    require_exact_n_top: bool = True
    overwrite: bool = True


def set_all_seeds(seed: int) -> None:
    np.random.seed(int(seed))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pick_key(obs: pd.DataFrame, fixed: Optional[str], candidates: Tuple[str, ...]) -> Optional[str]:
    if fixed is not None and fixed in obs.columns:
        return fixed
    for key in candidates:
        if key in obs.columns:
            vals = obs[key].dropna().astype(str)
            if vals.nunique() >= 1:
                return key
    return None


def md5_no_trailing_newline(items: List[str]) -> str:
    payload = "\n".join(map(str, items)).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def md5_master_style_trailing_newline(items: List[str]) -> str:
    h = hashlib.md5()
    for s in items:
        h.update(str(s).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def matrix_has_nonzero_gene_values(a: ad.AnnData, layer: Optional[str]) -> np.ndarray:
    X = a.layers[layer] if layer is not None else a.X
    sums = np.asarray(X.sum(axis=0)).ravel()
    return sums > 0


def rank_hvgs_from_var(var: pd.DataFrame) -> pd.DataFrame:
    """Return a ranked HVG table from Scanpy var annotations."""
    df = var.copy()
    df["gene"] = df.index.astype(str)

    if "highly_variable" not in df.columns:
        raise ValueError("Scanpy did not create var['highly_variable'].")

    # Common Scanpy columns for batch-aware HVG include:
    # highly_variable_nbatches, highly_variable_rank, means, variances_norm.
    for col in ["highly_variable_nbatches", "highly_variable_rank", "means", "variances", "variances_norm"]:
        if col not in df.columns:
            df[col] = np.nan

    selected = df[df["highly_variable"].astype(bool)].copy()
    if selected.empty:
        raise ValueError("No highly variable genes selected.")

    selected["_nbatches_sort"] = selected["highly_variable_nbatches"].fillna(0).astype(float)
    selected["_rank_sort"] = selected["highly_variable_rank"].fillna(np.inf).astype(float)
    selected["_varnorm_sort"] = selected["variances_norm"].fillna(-np.inf).astype(float)

    selected = selected.sort_values(
        by=["_nbatches_sort", "_rank_sort", "_varnorm_sort", "gene"],
        ascending=[False, True, False, True],
        kind="mergesort",
    )

    selected.insert(0, "canonical_order", np.arange(1, selected.shape[0] + 1))
    return selected[
        [
            "canonical_order", "gene", "highly_variable", "highly_variable_nbatches",
            "highly_variable_rank", "means", "variances", "variances_norm",
        ]
    ]


def run_hvg_selection(a: ad.AnnData, batch_key: str, cfg: CanonicalHIEHVGConfig) -> Tuple[List[str], pd.DataFrame, Dict[str, Any]]:
    """Compute batch-aware canonical HVGs for HIE."""
    layer_for_hvg: Optional[str] = "counts" if (cfg.prefer_counts_layer and "counts" in a.layers) else None
    flavor = str(cfg.hvg_flavor)
    fallback_used = False
    attempted: List[Dict[str, Any]] = []

    if flavor == "seurat_v3" and layer_for_hvg is None:
        logging.warning(
            "Requested flavor='seurat_v3' but no counts layer was found. Falling back to flavor='%s'.",
            cfg.fallback_flavor_if_no_counts,
        )
        flavor = str(cfg.fallback_flavor_if_no_counts)
        fallback_used = True

    def _try_hvg(aobj: ad.AnnData, flavor_local: str, layer_local: Optional[str]) -> pd.DataFrame:
        sc.pp.highly_variable_genes(
            aobj,
            n_top_genes=int(cfg.n_top_genes),
            flavor=str(flavor_local),
            batch_key=str(batch_key),
            layer=layer_local,
            subset=False,
        )
        return rank_hvgs_from_var(aobj.var)

    try:
        attempted.append({"flavor": flavor, "layer": layer_for_hvg, "strategy": "primary"})
        ranked = _try_hvg(a.copy(), flavor, layer_for_hvg)
    except Exception as e1:
        logging.warning("Primary HVG selection failed: %s", str(e1))
        attempted[-1]["error"] = str(e1)

        # Fallback 1: if seurat_v3 failed, try cell_ranger on the same matrix/layer.
        fallback_flavor = "cell_ranger" if flavor != "cell_ranger" else "seurat"
        try:
            attempted.append({"flavor": fallback_flavor, "layer": layer_for_hvg, "strategy": "flavor_fallback"})
            ranked = _try_hvg(a.copy(), fallback_flavor, layer_for_hvg)
            fallback_used = True
            flavor = fallback_flavor
        except Exception as e2:
            logging.warning("Fallback HVG selection failed: %s", str(e2))
            attempted[-1]["error"] = str(e2)

            # Fallback 2: remove all-zero genes, then try safer flavor.
            keep = matrix_has_nonzero_gene_values(a, layer_for_hvg)
            if keep.sum() < int(cfg.n_top_genes):
                raise RuntimeError(
                    f"Too few nonzero genes after filtering: {keep.sum()} < {cfg.n_top_genes}"
                ) from e2
            a_trim = a[:, keep].copy()
            safer_flavor = "seurat" if fallback_flavor != "seurat" else "cell_ranger"
            attempted.append({"flavor": safer_flavor, "layer": layer_for_hvg, "strategy": "zero_filter_fallback"})
            ranked = _try_hvg(a_trim, safer_flavor, layer_for_hvg)
            fallback_used = True
            flavor = safer_flavor

    genes = ranked["gene"].astype(str).tolist()
    if len(genes) > int(cfg.n_top_genes):
        genes = genes[: int(cfg.n_top_genes)]
        ranked = ranked.iloc[: int(cfg.n_top_genes)].copy()

    if cfg.require_exact_n_top and len(genes) != int(cfg.n_top_genes):
        raise ValueError(
            f"Expected exactly {cfg.n_top_genes} canonical HVGs, but got {len(genes)}."
        )

    meta = {
        "batch_key": batch_key,
        "batch_levels": sorted(pd.unique(a.obs[batch_key].astype(str)).tolist()),
        "n_batches": int(a.obs[batch_key].astype(str).nunique()),
        "layer_for_hvg": layer_for_hvg,
        "hvg_flavor_used": flavor,
        "hvg_flavor_requested": cfg.hvg_flavor,
        "fallback_used": bool(fallback_used),
        "attempted": attempted,
    }
    return genes, ranked, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen canonical HVGs for HIE benchmark.")
    parser.add_argument("--base-dir", default=os.environ.get("HIE_BASE_DIR", "."), help="HIE project folder")
    parser.add_argument("--data-file", default=None, help="Path to HIE h5ad relative to base-dir or absolute")
    parser.add_argument("--out-folder", default=None, help="Output folder relative to base-dir or absolute")
    parser.add_argument("--n-top", type=int, default=2000, help="Number of canonical HVGs")
    parser.add_argument("--batch-key", default=None, help="Fixed batch key; default='tech'")
    parser.add_argument("--label-key", default=None, help="Fixed label key; default='celltype'")
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if canonical files already exist")
    args = parser.parse_args()

    cfg = CanonicalHIEHVGConfig()
    cfg.base_dir = args.base_dir
    if args.data_file is not None:
        cfg.data_file = args.data_file
    if args.out_folder is not None:
        cfg.out_folder = args.out_folder
    if args.batch_key is not None:
        cfg.fixed_batch_key = args.batch_key
    if args.label_key is not None:
        cfg.fixed_label_key = args.label_key
    cfg.n_top_genes = int(args.n_top)
    cfg.overwrite = not bool(args.no_overwrite)

    os.chdir(cfg.base_dir)
    ensure_dir(cfg.out_folder)

    log_path = os.path.join(cfg.out_folder, "make_hie_canonical_hvg.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )

    set_all_seeds(cfg.seed)

    data_file = cfg.data_file
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Could not find HIE data file: {data_file}")

    out_hvg = os.path.join(cfg.out_folder, cfg.out_hvg_file)
    out_md5 = os.path.join(cfg.out_folder, cfg.out_md5_file)
    out_meta = os.path.join(cfg.out_folder, cfg.out_metadata_file)
    out_ranked = os.path.join(cfg.out_folder, cfg.out_ranked_candidates_file)

    if not cfg.overwrite:
        for p in [out_hvg, out_md5, out_meta, out_ranked]:
            if os.path.exists(p):
                raise FileExistsError(f"Output already exists and --no-overwrite was set: {p}")

    logging.info("Reading HIE h5ad: %s", data_file)
    adata = sc.read_h5ad(data_file)
    adata.var_names_make_unique()

    batch_key = pick_key(adata.obs, cfg.fixed_batch_key, cfg.batch_key_candidates)
    label_key = pick_key(adata.obs, cfg.fixed_label_key, cfg.label_key_candidates)
    if batch_key is None:
        raise ValueError(
            f"No batch key found. Tried fixed={cfg.fixed_batch_key} and candidates={cfg.batch_key_candidates}. "
            f"Available keys={list(adata.obs.columns)[:50]}"
        )

    adata.obs["batch"] = adata.obs[batch_key].astype(str).astype("category")
    if label_key is not None:
        adata.obs["class"] = adata.obs[label_key].astype(str).astype("category")

    logging.info("Resolved keys: batch=obs['%s']; label=%s", batch_key, label_key)
    logging.info("Data shape: n_obs=%d, n_vars=%d", adata.n_obs, adata.n_vars)
    logging.info("Layers: %s", list(adata.layers.keys()))
    logging.info("Batches: %s", dict(adata.obs[batch_key].astype(str).value_counts()))

    genes, ranked, hvg_meta = run_hvg_selection(adata, batch_key=batch_key, cfg=cfg)

    md5_r_style = md5_no_trailing_newline(genes)
    md5_master_style = md5_master_style_trailing_newline(genes)

    # Write canonical gene list with a trailing newline for readability.
    with open(out_hvg, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(genes))
        f.write("\n")

    # HIE R scripts compute md5 over paste(genes, collapse='\n'), no trailing newline.
    with open(out_md5, "w", encoding="utf-8", newline="\n") as f:
        f.write(md5_r_style + "\n")

    ranked.to_csv(out_ranked, index=False)

    metadata: Dict[str, Any] = {
        "script": "make_hie_canonical_hvg.py",
        "config": asdict(cfg),
        "data_file": os.path.abspath(data_file),
        "out_hvg_file": os.path.abspath(out_hvg),
        "n_obs": int(adata.n_obs),
        "n_vars_original": int(adata.n_vars),
        "n_canonical_hvgs": int(len(genes)),
        "batch_key_resolved": batch_key,
        "label_key_resolved": label_key,
        "md5_no_trailing_newline_R_style": md5_r_style,
        "md5_master_style_trailing_newline": md5_master_style,
        "hvg_selection": hvg_meta,
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Wrote canonical HVGs: %s", out_hvg)
    logging.info("Wrote canonical HVG md5 (R-style): %s", out_md5)
    logging.info("Wrote ranked candidates: %s", out_ranked)
    logging.info("Wrote metadata: %s", out_meta)
    logging.info("Canonical HVG md5 R-style=%s | master-style=%s", md5_r_style, md5_master_style)
    logging.info("Done.")


if __name__ == "__main__":
    main()
