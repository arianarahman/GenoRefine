# Purpose: Validate prepare behavior and invariants for the spatial multisection workflow.
# Author: Ariana Rahman (Arizona State University)

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from ..prepare import (
    SpatialBundle,
    load_spatial_bundle,
    preprocess_bundle,
    read_positions,
    verify_manifest_urls,
)
from ..source_acquisition import DEFAULT_CONFIG, ROOT, load_spec, source_root


class _HarmonyOutput:
    def __init__(self, values: np.ndarray):
        self.Z_corr = values.copy()
        self.objective_harmony = [10.0, 9.0]


def _fake_harmony(values, metadata, batch_key, **kwargs):
    assert list(metadata) == [batch_key]
    assert kwargs["random_state"] == 0
    return _HarmonyOutput(np.asarray(values))


def _synthetic_bundle() -> SpatialBundle:
    rng = np.random.default_rng(11)
    n_cells, n_genes = 120, 80
    counts = rng.poisson(1.3, size=(n_cells, n_genes)).astype(np.int32)
    counts[:, 0] = 0
    counts[:2, 0] = 1
    sections = np.repeat(["s1", "s2", "s3"], 40)
    metadata = pd.DataFrame({
        "cell_id": [f"{section}::b{i:03d}" for i, section in enumerate(sections)],
        "barcode": [f"b{i:03d}" for i in range(n_cells)],
        "section": sections,
        "donor": np.repeat(["d1", "d2", "d3"], 40),
        "label": np.resize(np.asarray(["L1", "L2", "L3"], dtype="U"), n_cells),
        "array_row": np.arange(n_cells),
        "array_col": np.arange(n_cells) + 1,
        "pxl_row_in_fullres": np.arange(n_cells) + 2,
        "pxl_col_in_fullres": np.arange(n_cells) + 3,
        "hires_pxl_row": (np.arange(n_cells) + 2) * 0.15,
        "hires_pxl_col": (np.arange(n_cells) + 3) * 0.15,
    })
    return SpatialBundle(
        counts=sparse.csr_matrix(counts),
        gene_ids=np.asarray([f"g{i:03d}" for i in range(n_genes)], dtype="U"),
        gene_symbols=np.asarray([f"G{i:03d}" for i in range(n_genes)], dtype="U"),
        metadata=metadata,
        excluded=pd.DataFrame(),
        section_summary=[],
    )


def _preprocessing() -> dict:
    return {
        "pooled_gene_min_cells": 3,
        "normalize_total_target_sum": 10000.0,
        "log1p": True,
        "hvg": {"n_top_genes": 20, "flavor": "seurat", "batch_key": "section"},
        "scale": {"zero_center": True, "max_value": 10.0},
        "pca": {"n_components": 5, "svd_solver": "arpack", "random_state": 0},
        "harmony": {
            "batch_key": "section", "max_iter_harmony": 2,
            "epsilon_harmony": 0.0001, "random_state": 0, "thread_limit": 1,
        },
    }


def test_preprocessing_is_deterministic_and_labels_do_not_fit_the_embedding() -> None:
    bundle = _synthetic_bundle()
    first = preprocess_bundle(bundle, _preprocessing(), harmony_runner=_fake_harmony)
    relabeled = deepcopy(bundle)
    relabeled.metadata["label"] = "different-reference-label"
    second = preprocess_bundle(relabeled, _preprocessing(), harmony_runner=_fake_harmony)
    assert first.counts.shape == (120, 79)
    assert int(first.gene_metadata["highly_variable"].sum()) == 20
    assert first.pca.shape == (120, 5)
    assert first.harmony.shape == (120, 5)
    assert np.array_equal(first.pca, second.pca)
    assert np.array_equal(first.harmony, second.harmony)
    assert first.record["logical_hashes"]["harmony"] == second.record["logical_hashes"]["harmony"]
    assert not first.record["harmony_convergence"]["harmonypy_stop_criterion_met"]
    assert not first.record["harmony_convergence"]["strict_converged"]


def test_positions_reject_duplicate_barcodes(tmp_path: Path) -> None:
    path = tmp_path / "positions.csv"
    path.write_text("a,1,1,2,3,4\na,1,2,3,4,5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate barcode"):
        read_positions(path)


def test_real_sources_reconcile_before_preprocessing() -> None:
    spec = load_spec(DEFAULT_CONFIG)
    root = source_root(spec, ROOT)
    validation = verify_manifest_urls(spec, root)
    assert set(validation["verified_h5_urls"]) == {
        "151507", "151508", "151669", "151670", "151673", "151674",
    }
    assert set(validation["verified_image_hi_urls"]) == set(validation["verified_h5_urls"])
    bundle = load_spatial_bundle(spec, root)
    assert bundle.counts.shape == (22968, 33538)
    assert len(bundle.metadata) == 22968
    assert len(bundle.excluded) == 113
    assert bundle.metadata["cell_id"].is_unique
    assert bundle.excluded["cell_id"].is_unique
    assert sorted(bundle.metadata["donor"].unique()) == ["Br5292", "Br5595", "Br8100"]
    assert bundle.metadata["hires_pxl_col"].between(0, 1999.999).all()
    assert bundle.metadata["hires_pxl_row"].between(0, 1999.999).all()
