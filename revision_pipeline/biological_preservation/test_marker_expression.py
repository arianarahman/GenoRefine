# Purpose: Validate marker expression behavior and invariants for the biological preservation
#          workflow.
# Author: Ariana Rahman (Arizona State University)

import inspect
import unittest

import numpy as np
from scipy import sparse

from revision_pipeline.biological_preservation.marker_expression import (
    align_rows_to_canonical,
    cluster_marker_summary,
    compute_external_marker_scores,
    evaluate_external_marker_embedding,
    evaluate_external_marker_neighbors,
    exact_nonself_neighbors,
    match_external_marker_panel,
    normalize_log1p_sparse,
    prepare_external_marker_scores,
)


class ExpressionPreparationTests(unittest.TestCase):
    def test_sparse_normalization_matches_dense_and_is_float64(self):
        counts = np.asarray([[1, 3, 0], [0, 2, 2], [5, 0, 0]], dtype=np.int64)
        dense = normalize_log1p_sparse(counts, target_sum=100.0)
        sparse_result = normalize_log1p_sparse(sparse.coo_matrix(counts), target_sum=100.0)
        self.assertTrue(sparse.isspmatrix_csr(dense))
        self.assertEqual(dense.dtype, np.float64)
        np.testing.assert_array_equal(dense.toarray(), sparse_result.toarray())
        np.testing.assert_allclose(
            dense.toarray(), np.log1p(counts / counts.sum(axis=1)[:, None] * 100.0))

    def test_normalization_rejects_invalid_expression(self):
        for values in (
            [[1, -1]],
            [[1, np.nan]],
            [[1, np.inf]],
            [[0, 0], [1, 2]],
            np.empty((0, 2)),
        ):
            with self.subTest(values=repr(values)), self.assertRaises(ValueError):
                normalize_log1p_sparse(values)
        for target in (0, -1, np.inf, np.nan, True):
            with self.subTest(target=target), self.assertRaises(ValueError):
                normalize_log1p_sparse([[1, 2]], target_sum=target)

    def test_explicit_id_alignment_dense_sparse_and_labels(self):
        canonical = ["c", "a", "b"]
        observed = ["b", "c", "a"]
        values = np.asarray([[20, 21], [30, 31], [10, 11]])
        expected = np.asarray([[30, 31], [10, 11], [20, 21]])
        np.testing.assert_array_equal(
            align_rows_to_canonical(values, observed, canonical), expected)
        np.testing.assert_array_equal(
            align_rows_to_canonical(sparse.csr_matrix(values), observed, canonical).toarray(),
            expected)
        np.testing.assert_array_equal(
            align_rows_to_canonical(np.asarray(["B", "C", "A"]), observed, canonical),
            ["C", "A", "B"])
        with self.assertRaises(ValueError):
            align_rows_to_canonical(values, ["b", "b", "a"], canonical)

    def test_exact_feature_matching_and_minimum(self):
        features = ["CD3D", "CD3E", "TRAC", "MS4A1", "CD79A", "CD37"]
        panel = {"B": ["MS4A1", "CD79A", "CD37", "MISSING"],
                 "T": ["CD3D", "CD3E", "TRAC"]}
        match = match_external_marker_panel(features, panel, min_marker_genes=3)
        self.assertEqual(match["class_names"], ["B", "T"])
        self.assertEqual(match["matched_genes"]["B"], ["MS4A1", "CD79A", "CD37"])
        self.assertEqual(match["unmatched_genes"]["B"], ["MISSING"])
        self.assertFalse(match["panel_selection_used_reference_labels"])
        with self.assertRaises(ValueError):
            match_external_marker_panel(features, panel, min_marker_genes=4)
        with self.assertRaises(ValueError):
            match_external_marker_panel(features, {"T": ["CD3D", "CD3D", "TRAC"]},
                                        min_marker_genes=2)
        with self.assertRaises(ValueError):
            match_external_marker_panel(["Gene", "GENE"], {"x": ["gene"]},
                                        min_marker_genes=1, case_sensitive=False)

    def test_scoring_is_label_free_and_sparse_deterministic(self):
        counts, features, panel, _, _, ids = fixture()
        normalized = normalize_log1p_sparse(counts)
        first = compute_external_marker_scores(
            normalized, features, panel, min_marker_genes=3)
        second = compute_external_marker_scores(
            normalized.tocoo(), features, panel, min_marker_genes=3)
        np.testing.assert_array_equal(first["scores"], second["scores"])
        self.assertEqual(first["scores"].shape, (len(ids), 2))
        self.assertGreater(first["scores"][:6, 0].mean(), first["scores"][:6, 1].mean())
        self.assertGreater(first["scores"][6:, 1].mean(), first["scores"][6:, 0].mean())
        self.assertNotIn("label", inspect.signature(compute_external_marker_scores).parameters)
        self.assertFalse(first["report"]["score_selection_used_reference_labels"])
        with self.assertRaises(ValueError):
            compute_external_marker_scores(normalized[:, :-1], features, panel,
                                           min_marker_genes=3)


class ExactNeighborTests(unittest.TestCase):
    def test_canonical_index_ties_and_row_permutation(self):
        canonical = ["z", "a", "m", "b", "q", "c"]
        values = np.zeros((len(canonical), 3), dtype=np.float64)
        first, distances = exact_nonself_neighbors(values, canonical, canonical, 2,
                                                    working_memory_mb=1)
        self.assertEqual(canonical[first[0, 0]], "a")
        self.assertEqual(canonical[first[0, 1]], "m")
        self.assertFalse(np.any(first == np.arange(len(canonical))[:, None]))
        np.testing.assert_array_equal(distances, 0.0)

        permutation = np.asarray([4, 2, 0, 5, 1, 3])
        permuted_ids = np.asarray(canonical)[permutation].tolist()
        second, second_distances = exact_nonself_neighbors(
            values[permutation], permuted_ids, canonical, 2, working_memory_mb=1)
        first_by_id = {cell: tuple(np.asarray(canonical)[row])
                       for cell, row in zip(canonical, first)}
        second_by_id = {cell: tuple(np.asarray(permuted_ids)[row])
                        for cell, row in zip(permuted_ids, second)}
        self.assertEqual(first_by_id, second_by_id)
        self.assertEqual(distances.tobytes(), second_distances[np.argsort(permutation)].tobytes())

    def test_float64_reranking_resolves_float32_scale_tie(self):
        # At magnitude one, this delta is lost by float32 but retained by float64.
        values = np.asarray([[0.0], [1.0 + 5e-8], [1.0], [4.0]], dtype=np.float64)
        ids = ["query", "canonical-first-but-farther", "closer", "other"]
        neighbors, distances = exact_nonself_neighbors(values, ids, ids, 1)
        self.assertEqual(ids[neighbors[0, 0]], "closer")
        self.assertEqual(distances[0, 0], 1.0)

    def test_invalid_neighbor_inputs(self):
        values = np.zeros((4, 2))
        ids = ["a", "b", "c", "d"]
        for observed, canonical, k in (
            (["a", "a", "c", "d"], ids, 2),
            (ids, ["a", "b", "c", "x"], 2),
            (ids, ids, 4),
        ):
            with self.subTest(observed=observed, canonical=canonical, k=k), self.assertRaises(ValueError):
                exact_nonself_neighbors(values, observed, canonical, k)


class MarkerEvaluationTests(unittest.TestCase):
    def test_perfect_external_marker_neighborhood_metrics(self):
        counts, features, panel, labels, embedding, ids = fixture()
        prepared = prepare_external_marker_scores(
            sparse.csr_matrix(counts), features, ids, ids, panel,
            min_marker_genes=3)
        neighbors, _ = exact_nonself_neighbors(embedding, ids, ids, 2)
        result = evaluate_external_marker_neighbors(
            prepared["scores"], prepared["class_names"], labels,
            neighbors, ids)
        self.assertEqual(result["evaluation_sampling"], "all_cells_no_subsampling")
        self.assertEqual(result["k_nonself"], 2)
        self.assertTrue(result["marker_panel_selected_without_evaluated_embedding_or_score"])
        self.assertTrue(result["cell_level_labels_used_only_as_evaluation_strata"])
        self.assertNotIn("panel_selected_without_reference_labels", result)
        for row in result["per_class"].values():
            self.assertEqual(row["neighbor_marker_auroc"], 1.0)
            self.assertEqual(row["neighbor_marker_average_precision"], 1.0)
            self.assertGreater(row["neighbor_marker_contrast"], 0.0)
        self.assertEqual(result["macro"]["neighbor_marker_auroc"], 1.0)

    def test_full_evaluation_is_row_permutation_invariant(self):
        counts, features, panel, labels, embedding, ids = fixture()
        prepared = prepare_external_marker_scores(
            counts, features, ids, ids, panel, min_marker_genes=3)
        original = evaluate_external_marker_embedding(
            prepared, labels, ids, embedding, ids, ids, k=2,
            working_memory_mb=1)
        p = np.asarray([8, 0, 11, 3, 7, 2, 10, 1, 6, 5, 4, 9])
        permuted = evaluate_external_marker_embedding(
            prepared, labels[p], np.asarray(ids)[p].tolist(), embedding[p],
            np.asarray(ids)[p].tolist(), ids, k=2, working_memory_mb=1)
        self.assertEqual(original["per_class"], permuted["per_class"])
        self.assertEqual(original["macro"], permuted["macro"])
        self.assertEqual(original["neighbor_search"]["tie_break"], "canonical_cell_index")

    def test_rare_class_uses_its_only_cell(self):
        ids = [f"c{i}" for i in range(6)]
        classes = ["common", "rare"]
        labels = np.asarray(["common"] * 5 + ["rare"])
        scores = np.column_stack((np.arange(6, 0, -1), np.arange(6)))
        neighbors, _ = exact_nonself_neighbors(
            np.arange(6, dtype=float)[:, None], ids, ids, 2)
        result = evaluate_external_marker_neighbors(scores, classes, labels,
                                                    neighbors, ids)
        self.assertEqual(result["rare_class_cells_retained"]["rare"], 1)
        self.assertEqual(result["per_class"]["rare"]["positive_cells"], 1)
        self.assertTrue(np.isfinite(result["per_class"]["rare"]["neighbor_marker_auroc"]))

        thresholded = evaluate_external_marker_neighbors(
            scores, classes, labels, neighbors, ids,
            minimum_class_cells_for_macro=2)
        self.assertEqual(thresholded["macro_eligible_classes"], ["common"])
        self.assertEqual(thresholded["macro_excluded_classes"], ["rare"])
        self.assertIn("rare", thresholded["per_class"])

    def test_self_duplicate_and_missing_class_are_rejected(self):
        ids = ["a", "b", "c", "d"]
        scores = np.ones((4, 2))
        labels = ["A", "A", "B", "B"]
        with self.assertRaises(ValueError):
            evaluate_external_marker_neighbors(
                scores, ["A", "B"], labels,
                np.asarray([[0], [0], [0], [0]]), ids)
        with self.assertRaises(ValueError):
            evaluate_external_marker_neighbors(
                scores, ["A", "B"], labels,
                np.asarray([[1, 1], [0, 2], [0, 1], [0, 1]]), ids)
        valid = np.asarray([[1], [0], [0], [0]])
        with self.assertRaises(ValueError):
            evaluate_external_marker_neighbors(
                scores, ["A", "absent"], labels, valid, ids)

    def test_optional_cluster_marker_summary(self):
        scores = np.asarray([[2., 0.], [4., 0.], [0., 3.], [0., 5.]])
        summary = cluster_marker_summary(scores, ["A", "B"], ["x", "x", "y", "y"])
        self.assertEqual(summary["cluster_count"], 2)
        self.assertEqual(summary["clusters"][0]["mean_marker_scores"], {"A": 3.0, "B": 0.0})
        self.assertFalse(summary["selection_used_reference_labels"])


def fixture():
    ids = [f"cell-{i:02d}" for i in range(12)]
    labels = np.asarray(["A"] * 6 + ["B"] * 6)
    features = ["A1", "A2", "A3", "B1", "B2", "B3", "HK"]
    counts = np.ones((12, len(features)), dtype=np.float64)
    counts[:6, :3] = 20
    counts[6:, 3:6] = 20
    counts[:, 6] = 10
    panel = {"A": ["A1", "A2", "A3"], "B": ["B1", "B2", "B3"]}
    embedding = np.column_stack((
        np.r_[np.arange(6) * 0.01, 10 + np.arange(6) * 0.01],
        np.zeros(12),
    ))
    return counts, features, panel, labels, embedding, ids


if __name__ == "__main__":
    unittest.main()
