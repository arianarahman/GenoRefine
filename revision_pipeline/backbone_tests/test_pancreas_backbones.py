# Purpose: Integration-environment tests; intentionally not part of the minimal-refiner suite.
# Author: Ariana Rahman (Arizona State University)

"""Integration-environment tests; intentionally not part of the minimal-refiner suite."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from revision_pipeline.data.pancreas_backbones import PROCESS_ENV
os.environ.update(PROCESS_ENV)

import numpy as np

from revision_pipeline.data.pancreas_backbones import (
    align_output, copy_parent_artifacts, feature_positions, fit_harmony,
    fit_scanorama, load_features, preprocess, read_config, runtime_record,
)
from revision_pipeline.data.embeddings import cache_key, matrix_diagnostics
from revision_pipeline.data.store import Store, import_legacy
from revision_pipeline.integrity import canonical_hash, file_fingerprint
from revision_pipeline.runs import RunDirectory, write_json
from revision_pipeline.data.tests import test_data_store as fixtures


class BackboneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = Path(__file__).resolve().parents[2]
        self.cfg = read_config(self.project, "revision_pipeline/configs/pancreas_backbones.json")
        self.cfg["batch_order"] = ["A", "B"]
        self.cfg["preprocessing"]["pca_components"] = 4
        self.cfg["scanorama"].update(dimred=5, knn=3, verbose=0)
        self.cfg["harmony"].update(nclust=4, verbose=False)
        self.ids = [f"cell-{i}" for i in range(80)]
        self.batches = ["A"] * 40 + ["B"] * 40
        self.values = np.random.default_rng(8).poisson(3, (80, 20)).astype(np.float64)
        self.tokens = [str(i) for i in range(19, -1, -1)]
        self.positions = feature_positions(self.tokens, 20)

    def prepared(self):
        return preprocess(self.values, self.ids, self.batches, self.tokens, self.positions, self.cfg)

    def test_feature_order_is_positional_not_lexicographic(self):
        np.testing.assert_array_equal(feature_positions(["10", "2", "0"], 20), [10, 2, 0])

    def test_invalid_duplicate_blank_alias_and_out_of_range_features(self):
        for tokens in (["1", "1"], [""], ["01"], ["-1"], ["20"], ["GENE"]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                feature_positions(tokens, 20)

    def test_frozen_real_feature_files_validate_without_running_a_method(self):
        cfg = read_config(self.project, "revision_pipeline/configs/pancreas_backbones.json")
        tokens, positions, report = load_features(self.project, cfg)
        self.assertEqual(len(tokens), 2000)
        self.assertEqual(report["order_md5"], "4bef6a587e23cba721b7c38b72e04b60")
        np.testing.assert_array_equal(np.sort(positions), np.arange(2000))

    def test_preprocessing_matches_legacy_call_sequence_and_preserves_source(self):
        import anndata as ad
        import pandas as pd
        import scanpy as sc
        before = self.values.copy()
        data, report = self.prepared()
        old = ad.AnnData(self.values.copy(), obs=pd.DataFrame(index=self.ids),
                        var=pd.DataFrame(index=[str(i) for i in range(20)]))
        old = old[:, self.tokens].copy()
        sc.pp.normalize_total(old, target_sum=1e4)
        sc.pp.log1p(old)
        sc.pp.scale(old, max_value=10)
        sc.tl.pca(old, n_comps=4)
        np.testing.assert_array_equal(data.X, old.X)
        np.testing.assert_array_equal(data.obsm["X_pca"], old.obsm["X_pca"])
        np.testing.assert_array_equal(before, self.values)
        self.assertEqual(list(data.obs.columns), ["batch"])
        self.assertFalse(report["biological_labels_used"])
        self.assertEqual(list(data.var_names), self.tokens)
        self.assertEqual(data.obsm["X_pca"].dtype, np.float32)

    def test_negative_nan_empty_and_wrong_batches_rejected(self):
        for bad in (-1., np.nan):
            values = self.values.copy()
            values[0, 0] = bad
            with self.assertRaises(ValueError):
                preprocess(values, self.ids, self.batches, self.tokens, self.positions, self.cfg)
        values = self.values.copy()
        values[0] = 0
        with self.assertRaises(ValueError):
            preprocess(values, self.ids, self.batches, self.tokens, self.positions, self.cfg)
        with self.assertRaises(ValueError):
            preprocess(self.values, self.ids, ["unknown"] * 80, self.tokens, self.positions, self.cfg)

    def test_method_outputs_are_id_aligned_not_assumed(self):
        x, order, _ = align_output(np.array([[3., 4.], [1., 2.]], dtype=np.float32), ["b", "a"], ["a", "b"], 2)
        np.testing.assert_array_equal(x, [[1, 2], [3, 4]])
        np.testing.assert_array_equal(order, [1, 0])
        self.assertEqual(x.dtype, np.float32)
        with self.assertRaises(ValueError):
            align_output(np.ones((2, 2)), ["a", "a"], ["a", "b"], 2)
        with self.assertRaises(ValueError):
            align_output(np.ones((2, 3)), ["a", "b"], ["a", "b"], 2)

    def test_scanorama_small_real_backend_and_input_unchanged(self):
        data, _ = self.prepared()
        before = data.X.copy()
        values, order, report = fit_scanorama(data, self.cfg)
        self.assertEqual(values.shape, (80, 5))
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_array_equal(data.X, before)
        np.testing.assert_array_equal(order, np.arange(80))
        self.assertEqual(report["canonical_cell_order_sha256"], canonical_hash(self.ids))

    def test_harmony_small_real_backend_native_dtype_and_seed_repeat(self):
        data, _ = self.prepared()
        before = data.obsm["X_pca"].copy()
        first, _, report = fit_harmony(data, self.cfg)
        second, _, _ = fit_harmony(data, self.cfg)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(before, data.obsm["X_pca"])
        self.assertEqual(first.shape, (80, 4))
        self.assertEqual(report["harmony_resolved_nclust"], 4)

    def test_cache_identity_includes_method_parameters_and_runtime(self):
        args = dict(dataset_fingerprint="a" * 64, cell_ids=self.ids, preprocessing={"hvg": 2000},
                    method="Scanorama", backbone_seed=0, method_parameters={"dimred": 100}, runtime_fingerprint="runtime-A")
        self.assertNotEqual(cache_key(**args), cache_key(**dict(args, method_parameters={"dimred": 50})))
        self.assertNotEqual(cache_key(**args), cache_key(**dict(args, runtime_fingerprint="runtime-B")))

    def test_parent_unchanged_and_new_native_float32_loads(self):
        fixture = fixtures.StoreTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.make_config(blocked=True)
        parent = Store(import_legacy(fixture.root))
        old_manifest = file_fingerprint(parent.path / "run.json")
        with RunDirectory(fixture.root / "revision_pipeline/runs", kind="step3a_data_store", config={"test": True}) as run:
            copy_parent_artifacts(parent, run)
            index = copy.deepcopy(parent.index)
            old = parent.embedding("example", "baseline")
            values = np.asarray(old.values, dtype=np.float32)
            stem = "embeddings/example/new_native"
            np.save(run.artifact_path(stem + ".npy"), values, allow_pickle=False)
            np.save(run.artifact_path(stem + ".source_rows.npy"), np.arange(3), allow_pickle=False)
            meta = dict(old.metadata, **matrix_diagnostics(values))
            meta.update(id="new_native", kind="recomputed_baseline", values_path=stem + ".npy", source_rows_path=stem + ".source_rows.npy")
            run.write_json(stem + ".json", meta)
            index["embeddings"]["example"]["new_native"] = stem + ".json"
            run.write_json("store.json", index)
        child = Store(run.final_path)
        self.assertTrue(child.verify(fixture.root)["passed"])
        self.assertEqual(child.embedding("example", "new_native").values.dtype, np.float32)
        self.assertEqual(file_fingerprint(parent.path / "run.json"), old_manifest)
        self.assertEqual(parent.index["embeddings"]["example"].keys(), {"baseline"})
        with self.assertRaisesRegex(ValueError, "blocked"):
            child.dataset("example").named_labels()
        # Copies, not hardlinks: changing the child never damages the parent.
        with (child.path / "embeddings/example/baseline.npy").open("ab") as stream:
            stream.write(b"child-only-corruption")
        self.assertEqual(parent.embedding("example", "baseline").values.shape, (3, 2))

    def test_runtime_records_native_backend_hashes(self):
        info = runtime_record()
        self.assertIn("annoy_extension", info["source_or_binary_hashes"])
        self.assertFalse(info["evaluation"])
        self.assertFalse(info["genodr_training"])


if __name__ == "__main__":
    unittest.main()
