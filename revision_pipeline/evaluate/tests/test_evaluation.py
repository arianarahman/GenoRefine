from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.metrics import silhouette_samples

from revision_pipeline.data.readers import array_hash
from revision_pipeline.data.store import DatasetView
from revision_pipeline.evaluate.config import EvaluationConfig, historical_profile
from revision_pipeline.evaluate.engine import graph_and_grid, graph_representation, metric_records, select_rows
from revision_pipeline.evaluate.inputs import load_refined_bundle, write_refined_bundle
from revision_pipeline.evaluate.metrics import (asw, d_batch, exclude_self, ilisi, isolated_asw,
                                                matrix, neighbors, overlap, purity)
from revision_pipeline.integrity import canonical_hash


def fixture(n=36):
    rng = np.random.default_rng(4)
    y = np.arange(n) % 3
    x = (rng.normal(0, .3, (n, 4)) + y[:, None]*4).astype(np.float64)
    b = np.where(y == 0, 0, np.arange(n) % 2)
    ids = [f"cell-{i}" for i in range(n)]
    record = {"cell_ids": ids, "obs": {"reference": y.tolist(), "batch": b.tolist()},
              "loader": {"label_key": "reference", "batch_key": "batch", "annotation_policy": {
                  "named_labels_allowed": False, "restriction_reason": "unresolved names",
                  "partition_description": "Anonymous in-house grouped Leiden partition"}},
              "registry": {"batch_evaluation": True}}
    return x, y, b, DatasetView(record, np.arange(n))


def config():
    return EvaluationConfig("unit_test", "development_only", None, "euclidean", 5, 0, (0, 1, 2),
                            (.2, .5, 1.), "fixed_resolution", .5, "preserve", "canonical", (),
                            3, 9, 3., 36, 0, 1, 8)


class ConfigurationTests(unittest.TestCase):
    def test_round_trip(self):
        self.assertEqual(EvaluationConfig.from_dict(json.loads(json.dumps(config().to_dict()))), config())

    def test_historical_branches(self):
        self.assertEqual(historical_profile("hpcb_main_tuned").dimensions, 30)
        self.assertEqual(historical_profile("hpcb_main_tuned").metric, "euclidean")
        self.assertIsNone(historical_profile("mouse_main_tuned").dimensions)
        self.assertIsNone(historical_profile("hpcb_robustness_tuned").dimensions)
        self.assertEqual(historical_profile("pancreas_focused_tuned").metric, "cosine")

    def test_primary_not_silently_frozen(self):
        with self.assertRaises(ValueError):
            replace(config(), purpose="primary")

    def test_invalid_settings(self):
        changes = [dict(resolutions=(.5, .2)), dict(resolutions=(float("nan"),)),
                   dict(leiden_seeds=(0, 0)), dict(leiden_seeds=(-1,)), dict(metric="mystery"),
                   dict(dimensions=0), dict(n_neighbors=1), dict(graph_seed=-1), dict(geometry_k=0),
                   dict(selection="fixed_resolution", fixed_resolution=.8), dict(lisi_perplexity=9.),
                   dict(metrics=("isolated_label_F1",)), dict(precision="float32"),
                   dict(row_order="historical_source"), dict(fixed_resolution=None), dict(neighbor_threads=0)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(config(), **change)

    def test_label_informed_flags(self):
        self.assertFalse(config().label_informed)
        for selection in ("best_ARI", "matched_reference_count"):
            self.assertTrue(replace(config(), selection=selection, fixed_resolution=None).label_informed)

    def test_selection_never_discards_grid(self):
        rows = [{"leiden_seed": s, "resolution": r, "n_clusters": k, "ARI": a}
                for s in config().leiden_seeds for r, k, a in ((.2, 2, .9), (.5, 4, .1), (1., 4, .7))]
        fixed = select_rows(rows, config(), 3)
        self.assertTrue(all(r["resolution"] == .5 for r in fixed))
        matched = select_rows(rows, replace(config(), selection="matched_reference_count", fixed_resolution=None), 3)
        self.assertTrue(all(r["resolution"] == .2 for r in matched))  # first tie
        oracle = select_rows(rows, replace(config(), selection="best_ARI", fixed_resolution=None), 3)
        self.assertTrue(all(r["resolution"] == .2 for r in oracle))
        self.assertEqual(len(rows), 9)

    def test_grid_only_does_not_select(self):
        self.assertEqual(select_rows([], replace(config(), selection="grid_only", fixed_resolution=None), 3), [])


class MetricTests(unittest.TestCase):
    def test_nonfirst_self(self):
        idx = np.array([[1, 0, 2], [2, 0, 1], [0, 1, 2]])
        out, _ = exclude_self(idx, np.zeros_like(idx, dtype=float))
        np.testing.assert_array_equal(out, [[1, 2], [2, 0], [0, 1]])

    def test_missing_self_due_to_ties(self):
        idx, _ = neighbors(np.zeros((12, 2)), 3)
        self.assertFalse(np.any(idx == np.arange(12)[:, None]))
        self.assertEqual(idx.shape, (12, 3))

    def test_duplicate_neighbor_rejected(self):
        with self.assertRaises(ValueError):
            exclude_self(np.array([[0, 0], [1, 0]]), np.zeros((2, 2)))

    def test_knn_hand_check_and_bug(self):
        x = np.array([[0.], [1.], [3.], [9.], [20.]])
        idx, _ = neighbors(x, 1)
        np.testing.assert_array_equal(idx[:, 0], [1, 0, 1, 2, 3])
        bug, _ = neighbors(x, 1, historical_bug=True)
        self.assertEqual(bug[0, 0], 2)

    def test_neighbor_k_not_silently_clipped(self):
        with self.assertRaises(ValueError):
            neighbors(np.zeros((4, 2)), 4)

    def test_jaccard_hand_check(self):
        result = overlap([[1, 2], [0, 2]], [[2, 3], [0, 2]])
        np.testing.assert_allclose(result, [1/3, 1])

    def test_purity_hand_check(self):
        np.testing.assert_array_equal(purity(np.array([[1], [0], [1]]), [0, 0, 1]), [1, 1, 0])

    def test_d_batch_uniform_includes_self(self):
        x = np.array([[0.], [.01], [10.], [10.01]])
        per, k = d_batch(x, [0, 1, 0, 1], k=2)
        self.assertEqual(k, 2)
        np.testing.assert_allclose(per, 2.)

    def test_d_batch_small_n_policy(self):
        per, k = d_batch(np.array([[0.], [1.]]), [0, 1])
        self.assertEqual(k, 1)
        np.testing.assert_array_equal(per, 1.)

    def test_single_batch_is_not_score_zero(self):
        x, y, _, _ = fixture()
        with self.assertRaises(ValueError):
            d_batch(x, np.zeros(len(x)))

    def test_asw_exact_against_sklearn(self):
        x, y, _, _ = fixture()
        np.testing.assert_allclose(asw(x, y, working_memory_mb=1), silhouette_samples(x, y), atol=1e-12)

    def test_asw_undefined_groups(self):
        x, _, _, _ = fixture()
        for y in (np.zeros(len(x)), np.arange(len(x))):
            with self.assertRaises(ValueError):
                asw(x, y)

    def test_isolated_asw_matches_scib_package(self):
        from scib_metrics import isolated_labels
        x, y, b, _ = fixture()
        score, groups, _ = isolated_asw(x, y, b, threshold=1)
        self.assertEqual(groups, [0])
        self.assertAlmostEqual(score, float(isolated_labels(x, y, b, iso_threshold=1)), places=5)

    def test_no_isolated_labels_is_missing(self):
        x, y, _, _ = fixture()
        score, groups, _ = isolated_asw(x, y, np.arange(len(x)) % 2, threshold=1)
        self.assertIsNone(score)
        self.assertEqual(groups, [])

    def test_lisi_matches_direct_api(self):
        from scib_metrics import ilisi_knn
        from scib_metrics.nearest_neighbors import NeighborsResults
        x, _, b, _ = fixture()
        score, raw = ilisi(x, b, k=9, perplexity=3., metric="euclidean")
        idx, dist = neighbors(x, 9)
        expected = ilisi_knn(NeighborsResults(indices=idx, distances=dist), b, perplexity=3., scale=True)
        self.assertAlmostEqual(score, float(expected), places=7)
        self.assertEqual(raw.shape, (len(x),))

    def test_bad_matrix(self):
        for x in (np.zeros(3), np.zeros((0, 2)), np.array([[np.nan]]), np.array([[1]])):
            with self.assertRaises(ValueError):
                matrix(x)

    def test_batch_and_name_gates(self):
        x, _, _, ds = fixture()
        ds.record["registry"]["batch_evaluation"] = False
        with self.assertRaises(ValueError):
            ds.named_labels()
        rows = metric_records(x, ds, replace(config(), metrics=("D_batch", "iLISI_scib_metrics", "isolated_label_ASW")), grid={"selected": []})
        self.assertTrue(all(r["status"] == "not_applicable" and r["value"] is None for r in rows))

    def test_sampled_scores_named_as_sampled(self):
        x, _, _, ds = fixture()
        rows = metric_records(x, ds, replace(config(), silhouette_max_cells=20, metrics=("reference_ASW",)), grid={"selected": []})
        self.assertEqual(rows[0]["metric"], "reference_ASW_subsample")
        self.assertEqual(rows[0]["scope"], "subsample")


class GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y, _, cls.ds = fixture()
        cls.result = graph_and_grid(cls.x, cls.y, cls.ds.cell_ids, config())

    def test_full_grid_times_seeds(self):
        self.assertEqual(len(self.result["grid"]), 9)
        self.assertEqual(self.result["partitions"].shape, (9, 36))
        self.assertEqual(len(self.result["selected"]), 3)

    def test_readonly_native_float32_gets_private_writable_buffer(self):
        x = self.x.astype(np.float32)
        x.flags.writeable = False
        original = x.tobytes()
        rep = graph_representation(x, config())
        self.assertTrue(rep.flags.writeable and rep.flags.c_contiguous)
        self.assertFalse(np.shares_memory(x, rep))
        self.assertEqual(rep.dtype, x.dtype)
        self.assertEqual(rep.tobytes(), original)
        rep[0, 0] += 100
        self.assertEqual(x.tobytes(), original)

    def test_fixed_rule_remains_label_free(self):
        self.assertTrue(all(not r["selection_label_informed"] for r in self.result["grid"]))
        self.assertTrue(all(r["resolution"] == .5 for r in self.result["selected"]))

    def test_labels_do_not_affect_graph_or_fitting(self):
        other = graph_and_grid(self.x, self.y[::-1], self.ds.cell_ids, config())
        np.testing.assert_array_equal(other["partitions"], self.result["partitions"])

    def test_graph_dimension_profile(self):
        other = graph_and_grid(self.x, self.y, self.ds.cell_ids, replace(config(), dimensions=2))
        self.assertEqual(other["graph"]["effective_dimensions"], 2)

    def test_graph_requires_ids_and_enough_cells(self):
        with self.assertRaises(ValueError):
            graph_and_grid(self.x, self.y, self.ds.cell_ids[:-1], config())


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "bundle"
        self.x, _, _, self.ds = fixture()
        self.parent = {"dataset_fingerprint": "d"*64, "embedding_id": "baseline",
                       "stored_values_file_sha256": "a"*64, "selected_values_sha256": array_hash(self.x),
                       "cell_order_sha256": canonical_hash(list(self.ds.cell_ids)), "shape": list(self.x.shape)}

    def write(self):
        write_refined_bundle(self.path, self.x, self.ds.cell_ids, parent_reference=self.parent, training_label_use="label_free")

    def test_exact_parent_and_canonical_alignment(self):
        self.write()
        loaded = load_refined_bundle(self.path, expected_parent=self.parent, output_cell_ids=self.ds.cell_ids[::-1])
        np.testing.assert_array_equal(loaded.values, self.x[::-1])
        self.assertEqual(loaded.metadata["pairing_status"], "exact_parent_reference_verified")

    def test_wrong_parent_rejected(self):
        self.write()
        with self.assertRaises(ValueError):
            load_refined_bundle(self.path, expected_parent=dict(self.parent, selected_values_sha256="b"*64), output_cell_ids=self.ds.cell_ids)

    def test_unknown_or_missing_ids_rejected(self):
        self.write()
        with self.assertRaises(ValueError):
            load_refined_bundle(self.path, expected_parent=self.parent, output_cell_ids=self.ds.cell_ids[:-1])

    def test_mutated_values_rejected(self):
        self.write()
        np.save(self.path / "values.npy", self.x+1)
        with self.assertRaises(ValueError):
            load_refined_bundle(self.path, expected_parent=self.parent, output_cell_ids=self.ds.cell_ids)

    def test_writer_requires_input_order(self):
        with self.assertRaises(ValueError):
            write_refined_bundle(self.path, self.x, self.ds.cell_ids[::-1], parent_reference=self.parent, training_label_use="unknown")

    def test_writer_refuses_overwrite(self):
        self.write()
        with self.assertRaises(FileExistsError):
            self.write()


if __name__ == "__main__":
    unittest.main()
