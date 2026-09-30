import tempfile
from pathlib import Path
import unittest

import numpy as np

from revision_pipeline.pilot.stage_check import saved_features, source_bridge, stage_summary, untimed
from revision_pipeline.evaluate.config import EvaluationConfig
from revision_pipeline.pilot.common import read


class SavedStageTests(unittest.TestCase):
    def test_source_bridge_allows_only_docs_and_named_new_diagnostic(self):
        old = {"revision_pipeline/evaluate/engine.py": {"sha256": "a"}, "revision_pipeline/README.md": {"sha256": "b"}}
        current = {**old, "revision_pipeline/README.md": {"sha256": "c"},
                   "revision_pipeline/pilot/stage_check.py": {"sha256": "d"}}
        self.assertTrue(source_bridge(old, current)["unchanged_existing_non_documentation_sources"])
        current["revision_pipeline/evaluate/engine.py"] = {"sha256": "bad"}
        with self.assertRaises(ValueError):
            source_bridge(old, current)

    def test_new_unrecognized_source_is_rejected(self):
        with self.assertRaises(ValueError):
            source_bridge({}, {"revision_pipeline/evaluate/new_math.py": {"sha256": "a"}})

    def test_saved_features_exact_order_and_native_precision(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"stage.npz"
            values = np.ones((2, 3), dtype=np.float32)
            np.savez(path, embedding=values, cell_ids=np.array(["b", "a"]))
            np.testing.assert_array_equal(saved_features(path, ["b", "a"], 3), values)
            with self.assertRaises(ValueError):
                saved_features(path, ["a", "b"], 3)
            with self.assertRaises(ValueError):
                saved_features(path, ["b", "a"], 4)
            np.savez(path, embedding=values.astype(np.float64), cell_ids=np.array(["b", "a"]))
            with self.assertRaises(ValueError):
                saved_features(path, ["b", "a"], 3)

    def test_saved_nonfinite_features_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"stage.npz"
            np.savez(path, embedding=np.array([[np.nan]], dtype=np.float32), cell_ids=np.array(["a"]))
            with self.assertRaises(ValueError):
                saved_features(path, ["a"], 1)

    def test_untimed_preserves_scores_and_choices(self):
        self.assertEqual(untimed([{"ARI": .3, "resolution": .5, "seconds": 3, "graph_seconds": 4}]),
                         [{"ARI": .3, "resolution": .5}])

    def test_stage_deltas_telescope_without_selecting_scores(self):
        root = Path(__file__).resolve().parents[3]
        config = EvaluationConfig.from_dict(read(root/"revision_pipeline/configs/evaluation_primary_v1.json"))
        def grid(score):
            return [{"leiden_seed": s, "resolution": r, "n_clusters": 10, "ARI": score, "RI": score}
                    for s in config.leiden_seeds for r in config.resolutions]
        result = stage_summary(grid(.8), grid(.5), grid(.6), config)
        comp = result["comparisons"]
        self.assertEqual(comp["baseline_to_pretrain"]["grid_consistency"]["ARI"]["sign_counts"]["negative"], 45)
        self.assertEqual(comp["pretrain_to_joint"]["grid_consistency"]["ARI"]["sign_counts"]["positive"], 45)
        self.assertAlmostEqual(comp["baseline_to_pretrain"]["grid_consistency"]["ARI"]["anchor_mean_delta_over_Leiden_seeds"]
                             +comp["pretrain_to_joint"]["grid_consistency"]["ARI"]["anchor_mean_delta_over_Leiden_seeds"], -.2)
        with self.assertRaises(ValueError):
            stage_summary(grid(.8), grid(.5)[:-1], grid(.6), config)


if __name__ == "__main__":
    unittest.main()
