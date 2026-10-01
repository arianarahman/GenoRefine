# Purpose: Validate pilot behavior and invariants for the pilot workflow.
# Author: Ariana Rahman (Arizona State University)

import tempfile
from pathlib import Path
import unittest

import numpy as np

from revision_pipeline.pilot.common import coverage, identical, panel_workers, specification, target_range
from revision_pipeline.pilot.run import environment, last_line


class PilotContractTests(unittest.TestCase):
    def test_frozen_specification_and_distinct_seed_panel(self):
        root = Path(__file__).resolve().parents[3]
        spec = specification(root)
        self.assertEqual(spec["replicate_seeds"], [0, 1, 2, 3, 4])
        self.assertFalse(spec["run_training_now"])  # Explicit CLI authorization required.
        self.assertEqual(spec["training"]["pretrain_epochs"], 100)

    def test_hpcb_actual_partial_batch_coverage(self):
        n = 16382
        visits = np.ones(n, dtype=np.int64)
        visits[:2816] += 1
        result = coverage(visits, ["a"]*8000+["b"]*(n-8000), n=n, updates=300, batch_size=64, shuffle=False, seed=0)
        self.assertEqual(result["total_cell_visits"], 19198)
        self.assertEqual(result["cells_visited_more_than_once"], 2816)
        self.assertEqual(result["per_batch"][1]["repeat_visited_cells"], 0)

    def test_early_stop_coverage_not_claimed_full(self):
        result = coverage(np.array([1, 1, 1, 1, 0, 0]), ["a"]*3+["b"]*3,
                          n=6, updates=1, batch_size=4, shuffle=False, seed=0)
        self.assertEqual(result["unique_cell_fraction"], 4/6)
        self.assertEqual(result["per_batch"][1]["unvisited_cells"], 2)

    def test_invalid_coverage_fails(self):
        for visits in (np.array([1., 1.]), np.array([0, 0]), np.array([1, -1])):
            with self.assertRaises(ValueError):
                coverage(visits, ["a", "b"], n=2, updates=1, batch_size=2, shuffle=False, seed=0)

    def test_gate_requires_duplicate_and_both_measured_peaks(self):
        self.assertEqual(panel_workers([1024, 8*1024**3], True), 2)
        self.assertEqual(panel_workers([1024, 8*1024**3+1], True), 1)
        for peaks, passed in (([1024, 1024], False), ([1024], True), ([1024, None], True)):
            with self.assertRaises(ValueError):
                panel_workers(peaks, passed)

    def test_range_is_descriptive_not_an_acceptance_target(self):
        result = target_range({0: .1, 1: .2, 2: .15, 3: .18, 4: .12}, range(5), [.1, .2004, .463])
        self.assertTrue(result["targets"][0]["inside_observed_range"])
        self.assertFalse(result["targets"][1]["inside_observed_range"])
        self.assertTrue(result["targets"][1]["rounding_interval_intersects_range"])
        self.assertFalse(result["targets"][2]["rounding_interval_intersects_range"])

    def test_incomplete_or_nonfinite_panel_cannot_give_range(self):
        for scores in ({0: .4}, {0: float("nan"), 1: .4}):
            with self.assertRaises(ValueError):
                target_range(scores, [0, 1], [.463])

    def test_byte_comparison_is_not_close_enough(self):
        a = np.array([1.], dtype=np.float32)
        self.assertTrue(identical(a, a.copy()))
        self.assertFalse(identical(a, a.astype(np.float64)))
        self.assertFalse(identical(a, np.nextafter(a, np.float32(2))))

    def test_separate_environment_thread_policy(self):
        self.assertEqual(environment("historical")["NUMBA_NUM_THREADS"], "24")
        self.assertEqual(environment("training")["NUMBA_NUM_THREADS"], "1")
        self.assertEqual(environment("primary")["NUMBA_NUM_THREADS"], "1")

    def test_bounded_tail_read(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"output.txt"
            p.write_text("old\n"*10000+"final\n")
            self.assertEqual(last_line(p), "final")


if __name__ == "__main__":
    unittest.main()
