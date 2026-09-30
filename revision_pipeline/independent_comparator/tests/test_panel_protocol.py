import json
import math
from pathlib import Path
import unittest

from revision_pipeline.independent_comparator.run_idec import (
    encoder_dimensions,
    select_case,
)


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "revision_pipeline/configs/independent_idec_panel_v2.json"


class IndependentIDECPanelProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    def test_panel_is_complete_and_unique(self):
        cases = self.spec["cases"]
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({case["id"] for case in cases}), 12)
        self.assertEqual(
            {(case["display_dataset"], case["embedding"]) for case in cases},
            {(dataset, embedding)
             for dataset in ("Pancreas", "HP-CB", "Mouse")
             for embedding in ("Scanorama", "Harmony", "Seurat", "Online_iNMF")},
        )

    def test_joint_budget_matches_two_pass_rule(self):
        batch = self.spec["batch_size"]
        for case in self.spec["cases"]:
            with self.subTest(case=case["id"]):
                self.assertEqual(case["joint_updates"], 2 * math.ceil(case["input_shape"][0] / batch))

    def test_dimensions_and_label_free_k_are_explicit(self):
        for raw in self.spec["cases"]:
            case = select_case(self.spec, raw["id"])
            with self.subTest(case=case["id"]):
                dims = encoder_dimensions(self.spec, case)
                self.assertEqual(dims[0], case["input_shape"][1])
                self.assertEqual(dims[-1], 32)
                self.assertGreater(case["n_clusters"], 1)
                self.assertTrue((ROOT / case["k_evidence"]).is_file())

    def test_unknown_case_is_rejected(self):
        with self.assertRaises(ValueError):
            select_case(self.spec, "not_a_case")


if __name__ == "__main__":
    unittest.main()
