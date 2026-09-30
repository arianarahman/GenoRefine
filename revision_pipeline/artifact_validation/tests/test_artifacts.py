import unittest

import numpy as np

from revision_pipeline.artifact_validation.artifacts import (
    batch_simplex_artifact,
    regular_simplex_directions,
    target_local_warp_artifact,
)


class ArtifactConstructionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(91)
        self.n = 48
        self.d = 8
        self.ids = [f"cell-{i:03d}" for i in range(self.n)]
        self.x = rng.normal(size=(self.n, self.d))
        # Preserve all three batches within the 24-cell target population.
        self.batches = np.asarray(["batch-c", "batch-a", "batch-b"] * 16)
        self.labels = np.asarray(["target"] * 24 + ["other"] * 24)

    def test_regular_simplex_is_unit_radius_equal_edge_and_rotated(self):
        vertices = regular_simplex_directions(4, 9)
        np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 1.0, atol=1e-14)
        np.testing.assert_allclose(vertices.sum(axis=0), 0.0, atol=1e-14)
        distances = np.linalg.norm(vertices[:, None] - vertices[None, :], axis=2)
        edges = distances[np.triu_indices(4, k=1)]
        np.testing.assert_allclose(edges, edges[0], atol=1e-14)
        self.assertTrue(np.all(np.ptp(vertices, axis=0) > 1e-6))

    def test_batch_artifact_preserves_weighted_centroid_and_recovers_observed(self):
        # Deliberately unbalance batches to test cell-count weighting.
        batches = np.asarray(["b"] * 27 + ["a"] * 14 + ["c"] * 7)
        bundle = batch_simplex_artifact(
            self.x,
            batches,
            self.ids,
            self.ids,
            strength=0.75,
            scale_k=3,
            working_memory_mb=1,
        )
        np.testing.assert_array_equal(bundle.observed, bundle.recovered_observed())
        np.testing.assert_allclose(
            bundle.observed.mean(axis=0), self.x.mean(axis=0), rtol=0, atol=1e-14
        )
        self.assertGreater(bundle.metadata["clean_median_kth_neighbor_radius"], 0)
        self.assertEqual(bundle.counterfactuals.shape, (3, self.n, self.d))
        self.assertEqual(bundle.metadata["centering"], "observed_batch_size_weighted")

    def test_batch_artifact_is_invariant_to_input_row_permutation(self):
        permutation = np.random.default_rng(7).permutation(self.n)
        first = batch_simplex_artifact(
            self.x, self.batches, self.ids, self.ids, scale_k=3, working_memory_mb=1
        )
        second = batch_simplex_artifact(
            self.x[permutation],
            self.batches[permutation],
            [self.ids[i] for i in permutation],
            self.ids,
            scale_k=3,
            working_memory_mb=1,
        )
        np.testing.assert_array_equal(first.observed, second.observed)
        np.testing.assert_array_equal(first.counterfactuals, second.counterfactuals)
        self.assertEqual(first.batch_levels, second.batch_levels)

    def test_target_local_warp_contracts(self):
        bundle = target_local_warp_artifact(
            self.x,
            self.batches,
            self.labels,
            "target",
            self.ids,
            self.ids,
            strength=1.2,
            scale_k=3,
            working_memory_mb=1,
        )
        target = self.labels == "target"
        np.testing.assert_array_equal(bundle.observed, bundle.recovered_observed())
        np.testing.assert_array_equal(bundle.observed[~target], self.x[~target])
        for counterfactual in bundle.counterfactuals:
            np.testing.assert_array_equal(counterfactual[~target], self.x[~target])
        self.assertGreater(np.linalg.norm(bundle.observed[target] - self.x[target]), 0)
        loading = np.asarray(bundle.metadata["pca1_loading"])
        directions = np.asarray(bundle.metadata["batch_directions"])
        np.testing.assert_allclose(directions @ loading, 0.0, atol=2e-14)
        anchor = int(np.argmax(np.abs(loading)))
        self.assertGreaterEqual(loading[anchor], 0)
        self.assertTrue(bundle.metadata["non_target_rows_bitwise_unchanged"])

    def test_target_local_warp_is_invariant_to_input_row_permutation(self):
        permutation = np.random.default_rng(11).permutation(self.n)
        kwargs = dict(
            target_label="target",
            canonical_ids=self.ids,
            strength=0.5,
            scale_k=3,
            working_memory_mb=1,
        )
        first = target_local_warp_artifact(
            self.x,
            self.batches,
            self.labels,
            cell_ids=self.ids,
            **kwargs,
        )
        second = target_local_warp_artifact(
            self.x[permutation],
            self.batches[permutation],
            self.labels[permutation],
            cell_ids=[self.ids[i] for i in permutation],
            **kwargs,
        )
        np.testing.assert_array_equal(first.observed, second.observed)
        np.testing.assert_array_equal(first.counterfactuals, second.counterfactuals)
        self.assertEqual(first.metadata, second.metadata)

    def test_target_missing_from_a_batch_fails_closed(self):
        labels = self.labels.copy()
        labels[(self.batches == "batch-c") & (labels == "target")] = "other"
        with self.assertRaisesRegex(ValueError, "Every hypothetical batch"):
            target_local_warp_artifact(
                self.x,
                self.batches,
                labels,
                "target",
                self.ids,
                self.ids,
                scale_k=3,
                working_memory_mb=1,
            )

    def test_invalid_ids_dimension_and_zero_radius_fail_closed(self):
        with self.assertRaises(ValueError):
            batch_simplex_artifact(
                self.x,
                self.batches,
                self.ids[:-1] + [self.ids[0]],
                self.ids,
                scale_k=3,
            )
        with self.assertRaises(ValueError):
            regular_simplex_directions(5, 3)
        with self.assertRaisesRegex(ValueError, "radius"):
            batch_simplex_artifact(
                np.zeros((20, 4)),
                np.asarray(["a", "b"] * 10),
                [f"z{i}" for i in range(20)],
                [f"z{i}" for i in range(20)],
                scale_k=3,
                working_memory_mb=1,
            )


if __name__ == "__main__":
    unittest.main()
