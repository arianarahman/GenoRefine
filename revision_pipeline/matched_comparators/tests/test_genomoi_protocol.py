# Purpose: Validate genomoi protocol behavior and invariants for the matched comparators
#          workflow.
# Author: Ariana Rahman (Arizona State University)

import unittest

from revision_pipeline.matched_comparators.common import implementation_record, protocol


class GenoMOIProtocolTests(unittest.TestCase):
    def test_frozen_protocol_and_label_free_k(self):
        spec, k, _, _ = protocol()
        self.assertEqual(spec["replicate_seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(k["n_clusters"], 14)
        self.assertFalse(k["reference_labels_used"])

    def test_relationship_is_not_mislabeled_independent(self):
        record = implementation_record()
        self.assertIn("not an independent", record["interpretation"])
        self.assertEqual(len(record["shared_calls"]), 6)


if __name__ == "__main__":
    unittest.main()
