# Purpose: Validate marker crossfit behavior and invariants for the biological preservation
#          workflow.
# Author: Ariana Rahman (Arizona State University)

import unittest

import numpy as np
from scipy import sparse

from revision_pipeline.biological_preservation.marker_crossfit import (
    crossfit_signatures, deterministic_folds, normalized_log1p)


class MarkerCrossfitTests(unittest.TestCase):
    def test_fold_is_id_deterministic(self):
        ids = [f"cell-{i}" for i in range(30)]
        np.testing.assert_array_equal(deterministic_folds(ids), deterministic_folds(ids))
        self.assertEqual(set(deterministic_folds(ids)), {0, 1})

    def test_crossfit_separable_signatures(self):
        ids = [f"cell-{i}" for i in range(60)]
        labels = np.asarray(["a"] * 20 + ["b"] * 20 + ["c"] * 20)
        x = np.ones((60, 12), dtype=np.float32)
        x[:20, :4] += 10; x[20:40, 4:8] += 10; x[40:, 8:] += 10
        result = crossfit_signatures(sparse.csr_matrix(x), labels, ids,
                                     [f"g{i}" for i in range(12)], top_n=3,
                                     min_train=3, min_test=3)
        self.assertEqual(result["accuracy_eligible"], 1.0)
        self.assertEqual(result["eligible_class_count"], 3)

    def test_normalization_rejects_negative(self):
        with self.assertRaises(ValueError):
            normalized_log1p([[1, -1]])


if __name__ == "__main__":
    unittest.main()

