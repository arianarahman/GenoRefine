# Purpose: Validate pre3c behavior and invariants for the evaluate workflow.
# Author: Ariana Rahman (Arizona State University)

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.distance import cdist

from revision_pipeline.evaluate.config import EvaluationConfig, historical_profile
from revision_pipeline.evaluate.engine import graph_and_grid, select_rows
from revision_pipeline.evaluate.exact import exact_neighbors, exact_connectivities
from revision_pipeline.evaluate.cache import reuse_baseline
from revision_pipeline.integrity import canonical_hash
from revision_pipeline.runs import RunDirectory


ROOT = Path(__file__).resolve().parents[2]


def primary():
    return EvaluationConfig.from_dict(json.loads((ROOT/"configs/evaluation_primary_v1.json").read_text()))


class ExactGraphTests(unittest.TestCase):
    def test_full_weighted_graph_survives_row_permutation(self):
        import numba
        rng = np.random.default_rng(20260916)
        x = rng.integers(-3, 4, size=(160, 5)).astype(np.float64)
        x[1:6] = x[0]  # Duplicate coordinates and many equal-distance ties.
        ids = np.array([f"cell-{i:04d}" for i in range(len(x))])
        p = rng.permutation(len(x))
        inverse = np.argsort(p)
        old_threads = numba.get_num_threads()
        try:
            numba.set_num_threads(1)
            a_d, a_g, a_i, a_dist = exact_connectivities(x, ids, 15)
            b_d, b_g, b_i, b_dist = exact_connectivities(x[p], ids[p], 15)
        finally:
            numba.set_num_threads(old_threads)
        np.testing.assert_array_equal(ids[a_i], ids[p][b_i][inverse])
        self.assertEqual(a_dist.tobytes(), b_dist[inverse].tobytes())
        for first, second in ((a_d, b_d), (a_g, b_g)):
            second = second[inverse][:, inverse].tocsr()
            first.sort_indices()
            second.sort_indices()
            self.assertEqual(first.shape, second.shape)
            for key in ("indptr", "indices", "data"):
                self.assertEqual(getattr(first, key).dtype, getattr(second, key).dtype)
                self.assertEqual(getattr(first, key).tobytes(), getattr(second, key).tobytes())

    def test_primary_harmony_convergence_gate(self):
        from revision_pipeline.evaluate.runner import require_primary_convergence
        p = primary()
        with self.assertRaises(ValueError):
            require_primary_convergence("pancreas_five_study", "Harmony", {}, p)
        require_primary_convergence("hpcb", "Harmony", {}, p)
        require_primary_convergence("pancreas_five_study", "Scanorama", {}, p)
        require_primary_convergence("pancreas_five_study", "Harmony", {}, replace(p, purpose="development_only"))
        require_primary_convergence("pancreas_five_study", "Harmony", {"primary_convergence": {
            "policy_id": "pancreas_harmony_convergence_v1", "status": "passed"}}, p)

    def test_exact_matches_full_distance_sort_with_duplicates(self):
        x = np.array([[0., 0.], [0., 0.], [1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        ids = ["z", "a", "c", "b", "d", "e"]
        idx, distances = exact_neighbors(x, ids, 3, working_memory_mb=1)
        full = cdist(x, x)
        for i in range(len(x)):
            expected = sorted((j for j in range(len(x)) if j != i), key=lambda j: (full[i, j], ids[j]))[:3]
            np.testing.assert_array_equal(idx[i], expected)
            np.testing.assert_array_equal(distances[i], full[i, expected])

    def test_order_invariance_including_boundary_ties(self):
        x = np.zeros((30, 3))
        ids = np.array([f"cell-{i:02}" for i in range(len(x))])
        p = np.random.default_rng(3).permutation(len(x))
        a, da = exact_neighbors(x, ids, 4)
        b, db = exact_neighbors(x[p], ids[p], 4)
        np.testing.assert_array_equal(ids[a][p], ids[p][b])
        np.testing.assert_array_equal(da[p], db)
        self.assertFalse(np.any(a == np.arange(len(x))[:, None]))

    def test_block_memory_does_not_change_result(self):
        x = np.random.default_rng(5).normal(size=(410, 5)).astype(np.float32)
        ids = [f"c{i}" for i in range(len(x))]
        a, da = exact_neighbors(x, ids, 15, working_memory_mb=1)
        b, db = exact_neighbors(x, ids, 15, working_memory_mb=4)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(da, db)

    def test_invalid_inputs(self):
        x = np.zeros((4, 2))
        for ids, k in ((["a"]*4, 2), (["a", "b", "c", "d"], 4), (["a"], 2)):
            with self.assertRaises(ValueError):
                exact_neighbors(x, ids, k)

    def test_profile_is_deliberately_frozen(self):
        p = primary()
        self.assertFalse(p.label_informed)
        for change in ({"dimensions": 30}, {"n_neighbors": 14}, {"neighbor_backend": "scanpy_legacy"},
                       {"fixed_resolution": .6}, {"neighbor_threads": 2}, {"row_order": "historical_source"},
                       {"metric": "cosine"}, {"leiden_seeds": (0,)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(p, **change)

    def test_historical_settings_preserved(self):
        c = historical_profile("hpcb_main_tuned")
        self.assertEqual((c.dimensions, c.neighbor_threads, c.neighbor_backend, c.row_order),
                         (30, 24, "scanpy_legacy", "historical_source"))
        with self.assertRaises(ValueError):
            replace(c, neighbor_backend="exact_stable_id", neighbor_threads=1)

    def test_failed_mouse_calibration_is_explicit(self):
        c = historical_profile("mouse_main_tuned")
        rows = [{"leiden_seed": 0, "resolution": .2, "n_clusters": 10, "ARI": .8},
                {"leiden_seed": 0, "resolution": 1.6, "n_clusters": 60, "ARI": .57}]
        row = select_rows(rows, c, 155)[0]
        self.assertEqual(row["calibration"]["status"], "failed")
        self.assertTrue(row["calibration"]["selected_at_grid_boundary"])
        self.assertEqual(row["reference_count_difference"], -95)

    def test_exact_graph_uses_one_saved_graph_for_grid(self):
        from unittest.mock import patch
        from revision_pipeline.evaluate.exact import exact_connectivities
        x = np.random.default_rng(2).normal(size=(36, 4))
        c = replace(primary(), purpose="development_only", resolutions=(.2, .5), leiden_seeds=(0, 1))
        with patch("revision_pipeline.evaluate.engine.exact_connectivities", wraps=exact_connectivities) as graph:
            out = graph_and_grid(x, np.arange(36) % 3, [f"c{i}" for i in range(36)], c)
        self.assertEqual(graph.call_count, 1)
        self.assertEqual(len(out["grid"]), 4)
        self.assertEqual(out["graph"]["umap_neighbor_slots"], 16)
        self.assertTrue(all(row["calibration"] is None for row in out["selected"]))


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = {"evaluation": primary().to_dict(), "dataset": "fixture", "store_manifest": {"sha256": "store"},
                    "embeddings": ["Scanorama"], "refined_bundle": None}
        self.input = {"name": "Scanorama", "reference": {"values": "hash", "order": "hash"}}
        self.runtime = {"runtime": "locked"}
        with RunDirectory(self.root, kind="step3b_evaluation", config=self.cfg) as r:
            r.manifest["source_tree_sha256"] = "source"
            r.write_json("runtime_start.json", self.runtime)
            r.write_json("embedding_0/input.json", self.input)
            r.write_json("embedding_0/graph.json", {"graph": "test"})
            r.write_json("summary.json", {"evaluations": [{"timing": {"graph_seconds": 4.0}, "name": "Scanorama"}]})
        self.path = r.final_path

    def reuse(self, **changes):
        kwargs = dict(expected_config=self.cfg, expected_input=self.input, source_hash="source", current_runtime=self.runtime)
        kwargs.update(changes)
        with RunDirectory(self.root, kind="cache_test", config={}) as r:
            return reuse_baseline(self.path, run=r, **kwargs)

    def test_reuse_records_zero_new_computation_and_original_cost(self):
        result = self.reuse()
        self.assertEqual(result["timing"]["graph_seconds"], 0)
        self.assertEqual(result["timing"]["reused_original_timing"]["graph_seconds"], 4)

    def test_reject_changed_source_runtime_or_parent(self):
        for change in ({"source_hash": "changed"}, {"current_runtime": {}}, {"expected_input": {}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.reuse(**change)

    def test_reject_changed_protocol(self):
        cfg = dict(self.cfg, evaluation={"different": True})
        with self.assertRaises(ValueError):
            self.reuse(expected_config=cfg)

    def test_reject_tampering(self):
        (self.path/"embedding_0/graph.json").write_text("{}")
        with self.assertRaises(ValueError):
            self.reuse()
