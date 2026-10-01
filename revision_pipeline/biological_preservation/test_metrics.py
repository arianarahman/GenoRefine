# Purpose: Validate metrics behavior and invariants for the biological preservation workflow.
# Author: Ariana Rahman (Arizona State University)

import unittest

import numpy as np

from revision_pipeline.biological_preservation.metrics import centroid_geometry, neighborhood_jaccard


class PreservationMetricTests(unittest.TestCase):
    def test_neighborhood_jaccard(self):
        before = np.array([[1, 2, 3], [0, 2, 3]])
        after = np.array([[1, 2, 4], [0, 2, 3]])
        np.testing.assert_allclose(neighborhood_jaccard(before, after), [0.5, 1.0])

    def test_centroid_geometry_allows_different_dimensions_and_scale(self):
        labels = np.array(["a", "a", "b", "b", "c", "c"])
        reference = np.array([[0, 0], [0, .1], [2, 0], [2, .1], [0, 3], [0, 3.1]])
        refined = np.column_stack([reference * 7, np.ones(len(reference))])
        result = centroid_geometry(reference, refined, labels)
        self.assertAlmostEqual(result["centroid_distance_spearman"], 1.0)
        self.assertAlmostEqual(result["centroid_distance_pearson"], 1.0)
        np.testing.assert_allclose(
            list(result["reference_separation_ratio"].values()),
            list(result["refined_separation_ratio"].values()),
        )

    def test_invalid_neighbor_shape_rejected(self):
        with self.assertRaises(ValueError):
            neighborhood_jaccard(np.ones((2, 2)), np.ones((3, 2)))


if __name__ == "__main__":
    unittest.main()
