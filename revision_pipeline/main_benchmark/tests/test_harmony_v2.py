import unittest

from revision_pipeline.main_benchmark.harmony_v2 import objective_gate


class HarmonyV2GateTests(unittest.TestCase):
    def test_adjacent_stabilization(self):
        result = objective_gate([10.0, 9.0, 9.0005], 1e-4, 6)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["mode"], "adjacent_stabilization")

    def test_stable_period_two(self):
        values = [10.0, 9.0, 10.0001, 9.0001, 10.0002, 9.0002, 10.0003, 9.0003]
        result = objective_gate(values, 1e-4, 6)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["mode"], "stable_period_2")

    def test_unstable_period_two_fails(self):
        result = objective_gate([10.0, 9.0, 10.2, 9.2, 10.4, 9.4, 10.6, 9.6], 1e-4, 6)
        self.assertEqual(result["status"], "failed")


if __name__ == "__main__":
    unittest.main()
