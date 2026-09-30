import unittest

import numpy as np

from revision_pipeline.broader_validation.common import EvaluationDataset
from revision_pipeline.broader_validation.score import spatial_overlap


class BroaderValidationTests(unittest.TestCase):
    def test_spatial_overlap_identity_is_one(self):
        coordinates = np.asarray([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.], [5., 0.], [6., 0.], [7., 0.]])
        result = spatial_overlap(coordinates, coordinates, k=3)
        self.assertAlmostEqual(result["mean_spatial_embedding_neighbor_jaccard"], 1.0)

    def test_nonspatial_is_not_forced(self):
        values = np.eye(4)
        self.assertIsNone(spatial_overlap(values, np.empty((4, 0))))

    def test_single_batch_is_explicitly_inapplicable(self):
        data = EvaluationDataset(("a", "b"), np.asarray([0, 1]), ("one", "one"), "test", False)
        with self.assertRaises(ValueError):
            data.batch_labels(for_mixing_metric=True)


if __name__ == "__main__":
    unittest.main()

