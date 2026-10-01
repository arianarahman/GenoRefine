# Purpose: Generate exact cell manifests for strict pancreas robustness experiments.
# Author: Ariana Rahman (Arizona State University)

"""
FILE: make_pancreas_robustness_manifests_canonical_v1.py
-------------------------------------------------------------------------------
Generate exact cell manifests for strict pancreas robustness experiments.

Run this BEFORE the Seurat / Online iNMF robustness R scripts so that Python and
R use exactly the same perturbed cell sets.

Outputs:
  ./Benchmark_Out/Robustness_R_Embeddings/cells_imbalance_Baron_frac1_00.csv
  ./Benchmark_Out/Robustness_R_Embeddings/cells_imbalance_Baron_frac0_50.csv
  ./Benchmark_Out/Robustness_R_Embeddings/cells_imbalance_Baron_frac0_25.csv
  ./Benchmark_Out/Robustness_R_Embeddings/cells_imbalance_Baron_frac0_10.csv
  ./Benchmark_Out/Robustness_R_Embeddings/cells_nonoverlap_Baron_beta.csv
-------------------------------------------------------------------------------
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import scipy.io as sio
import anndata as ad


@dataclass
class ManifestConfig:
    data_folder: str = "./Dataset"
    out_folder: str = "./Benchmark_Out"
    r_robustness_folder: str = "Robustness_R_Embeddings"
    seed: int = 0

    data_files: Tuple[str, ...] = (
        "dataBaronX.mat",
        "dataMuraroX.mat",
        "dataScapleX.mat",
        "dataWangX.mat",
        "dataXinX.mat",
    )
    class_label_file: str = "classLabel.mat"

    batch_token_to_name: Dict[str, str] = None
    imbalance_batch_to_downsample: str = "Baron"
    imbalance_fracs: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.10)
    nonoverlap_batch: str = "Baron"
    nonoverlap_celltype: str = "beta"

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


CFG = ManifestConfig()


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


def _frac_tag(frac: float) -> str:
    return f"frac{float(frac):.2f}".replace(".", "_")


def _context_tag(cfg: ManifestConfig, test_tag: str, frac: Optional[float] = None) -> str:
    if test_tag == "imbalance":
        if frac is None:
            raise ValueError("frac is required for imbalance context")
        return f"imbalance_{cfg.imbalance_batch_to_downsample}_{_frac_tag(frac)}"
    if test_tag == "nonoverlap":
        return f"nonoverlap_{cfg.nonoverlap_batch}_{str(cfg.nonoverlap_celltype).replace(' ', '_')}"
    raise ValueError(test_tag)


def load_raw(cfg: ManifestConfig) -> ad.AnnData:
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
        a.obs["batch_id"] = int(i + 1)
        adatas.append(a)

    adata = ad.concat(adatas, join="outer")
    adata.obs_names_make_unique()

    cls = sio.loadmat(os.path.join(cfg.data_folder, cfg.class_label_file))["classLabel"].squeeze()
    class_name_map = {
        1: "MHC class II", 2: "acinar", 3: "ductal", 4: "gamma", 5: "macrophage",
        6: "alpha", 7: "beta", 8: "endothelial", 9: "epsilon", 10: "mast",
        11: "mesenchymal", 12: "stellate", 13: "delta", 14: "schwann",
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


def write_manifest(adata: ad.AnnData, cfg: ManifestConfig, context_tag: str) -> str:
    out_dir = os.path.join(cfg.out_folder, cfg.r_robustness_folder)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"cells_{context_tag}.csv")
    df = pd.DataFrame({
        "cell_id": adata.obs_names.astype(str),
        "batch": adata.obs["batch"].astype(str).to_numpy(),
        "class": adata.obs["class"].astype(str).to_numpy(),
        "batch_id": adata.obs["batch_id"].astype(int).to_numpy(),
    })
    df.to_csv(out_csv, index=False)
    return out_csv


def make_imbalance_dataset(adata: ad.AnnData, cfg: ManifestConfig, frac: float) -> ad.AnnData:
    rng = np.random.default_rng(cfg.seed)
    mask_other = adata.obs["batch"].astype(str) != cfg.imbalance_batch_to_downsample
    mask_target = ~mask_other

    ad_other = adata[mask_other].copy()
    ad_target = adata[mask_target].copy()

    n_keep = int(np.floor(ad_target.n_obs * float(frac)))
    if n_keep < 2:
        raise ValueError(f"Too few cells retained for frac={frac}: {n_keep}")

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


def make_nonoverlap_dataset(adata: ad.AnnData, cfg: ManifestConfig) -> ad.AnnData:
    b = adata.obs["batch"].astype(str)
    c = adata.obs["class"].astype(str)
    drop = (b == cfg.nonoverlap_batch) & (c == cfg.nonoverlap_celltype)
    out = adata[~drop].copy()
    out.obs_names_make_unique()
    out.obs["batch"] = out.obs["batch"].astype("category")
    out.obs["class"] = out.obs["class"].astype("category")
    out.obs["batch_id"] = out.obs["batch_id"].astype(int)
    return out


def main() -> None:
    adata = load_raw(CFG)
    print(f"Loaded raw pancreas: n_obs={adata.n_obs}, batches={adata.obs['batch'].nunique()}, classes={adata.obs['class'].nunique()}")

    for frac in CFG.imbalance_fracs:
        tag = _context_tag(CFG, "imbalance", frac)
        pert = make_imbalance_dataset(adata, CFG, frac)
        out = write_manifest(pert, CFG, tag)
        print(f"Wrote {tag}: n={pert.n_obs} -> {out}")

    tag = _context_tag(CFG, "nonoverlap", None)
    pert = make_nonoverlap_dataset(adata, CFG)
    out = write_manifest(pert, CFG, tag)
    print(f"Wrote {tag}: n={pert.n_obs} -> {out}")


if __name__ == "__main__":
    main()
