# Purpose: Validate metrics behavior and invariants for the artifact validation workflow.
# Author: Ariana Rahman (Arizona State University)

import unittest

import numpy as np

from revision_pipeline.artifact_validation.metrics import (
    clean_neighbor_recovery,
    counterfactual_batch_sensitivity,
    evaluate_prespecified_pass_rule,
    preservation_metrics,
)


class CounterfactualMetricTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(51)
        self.x = rng.normal(size=(24, 5))
        self.ids = [f"c{i:02d}" for i in range(24)]
        self.batches = np.asarray(["d", "b", "a", "c"] * 6)
        self.levels = ("a", "b", "c", "d")
        shifts = np.asarray(
            [[-1, 0, 0, 0, 0], [0, -1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
            dtype=np.float64,
        )
        self.counterfactual = np.stack([self.x + shift for shift in shifts])
        lookup = {level: i for i, level in enumerate(self.levels)}
        index = np.asarray([lookup[value] for value in self.batches])
        self.observed = self.counterfactual[index, np.arange(len(self.x))]

    def metric(self, **changes):
        args = dict(
            clean_output=self.x,
            counterfactual_outputs=self.counterfactual,
            observed_output=self.observed,
            observed_batches=self.batches,
            batch_levels=self.levels,
            cell_ids=self.ids,
            canonical_ids=self.ids,
        )
        args.update(changes)
        return counterfactual_batch_sensitivity(**args)

    def test_formula_and_common_similarity_invariance(self):
        result = self.metric()
        numerator = np.mean(
            np.sum(
                (self.counterfactual - self.counterfactual.mean(axis=0, keepdims=True)) ** 2,
                axis=2,
            )
        )
        denominator = np.mean(
            np.sum((self.x - self.x.mean(axis=0)) ** 2, axis=1)
        )
        self.assertAlmostEqual(result["sensitivity_ratio"], numerator / denominator, places=14)
        self.assertFalse(result["collapsed_or_undefined"])

        rotation, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(5, 5)))
        transform = lambda a: 3.5 * a @ rotation + 7
        transformed = self.metric(
            clean_output=transform(self.x),
            counterfactual_outputs=transform(self.counterfactual),
            observed_output=transform(self.observed),
        )
        self.assertAlmostEqual(result["sensitivity_ratio"], transformed["sensitivity_ratio"], places=12)

    def test_row_permutation_is_canonical_and_identical(self):
        permutation = np.random.default_rng(9).permutation(len(self.x))
        first = self.metric()
        second = self.metric(
            clean_output=self.x[permutation],
            counterfactual_outputs=self.counterfactual[:, permutation],
            observed_output=self.observed[permutation],
            observed_batches=self.batches[permutation],
            cell_ids=[self.ids[i] for i in permutation],
        )
        self.assertEqual(first["sensitivity_ratio"], second["sensitivity_ratio"])
        self.assertEqual(first["clean_output_effective_rank"], second["clean_output_effective_rank"])

    def test_collapse_is_undefined_and_observed_mismatch_rejected(self):
        constant = np.ones_like(self.x)
        result = self.metric(
            clean_output=constant,
            counterfactual_outputs=np.stack([constant] * 4),
            observed_output=constant,
        )
        self.assertTrue(result["collapsed_or_undefined"])
        self.assertIsNone(result["sensitivity_ratio"])
        with self.assertRaisesRegex(ValueError, "reproduce"):
            self.metric(observed_output=self.observed + 0.1)


class NeighborhoodMetricTests(unittest.TestCase):
    def setUp(self):
        self.x = np.asarray(
            [[0, 0], [0, 0], [0, 1], [1, 0], [5, 5], [5, 5], [5, 6], [6, 5]],
            dtype=np.float64,
        )
        self.ids = [f"id{i}" for i in range(8)]
        self.labels = np.asarray(["rare", "rare", "a", "a", "b", "b", "b", "b"])

    def test_clean_recovery_identity_and_permutation_invariance_with_ties(self):
        first = clean_neighbor_recovery(
            self.x, self.x, self.ids, self.ids, self.ids, k=3, working_memory_mb=1
        )
        self.assertEqual(first["mean_clean_neighbor_jaccard"], 1.0)
        permutation = np.asarray([7, 3, 0, 6, 1, 4, 2, 5])
        second = clean_neighbor_recovery(
            self.x[permutation],
            self.x[permutation],
            [self.ids[i] for i in permutation],
            [self.ids[i] for i in permutation],
            self.ids,
            k=3,
            working_memory_mb=1,
        )
        np.testing.assert_array_equal(first["clean_neighbors"], second["clean_neighbors"])
        np.testing.assert_array_equal(first["candidate_neighbors"], second["candidate_neighbors"])

    def test_purity_rare_recall_and_target_fraction(self):
        result = preservation_metrics(
            self.x,
            self.labels,
            self.ids,
            self.ids,
            rare_groups=["rare"],
            target_label="b",
            k=3,
            working_memory_mb=1,
        )
        self.assertGreaterEqual(result["purity"]["mean_neighbor_purity"], 0)
        self.assertLessEqual(result["purity"]["mean_neighbor_purity"], 1)
        rare = result["rare_recall"]["groups"][0]
        # Both rare cells retrieve each other, so recall@3 has denominator one.
        self.assertEqual(rare["mean_same_class_recall_at_k"], 1.0)
        self.assertGreater(result["target_same_class_fraction"]["mean_target_same_class_fraction_at_k"], 0.5)


class PassRuleTests(unittest.TestCase):
    @staticmethod
    def rows():
        return [
            {
                "replicate_seed": seed,
                "sensitivity_before": 0.4,
                "sensitivity_after": 0.3,
                "clean_neighbor_jaccard_before": 0.5,
                "clean_neighbor_jaccard_after": 0.6,
                "delta_ARI": -0.01,
                "delta_purity": -0.005,
                "rare_recall_deltas": {"rare-a": -0.05, "rare-b": 0.0},
                "delta_target_same_class_fraction": -0.05,
                "collapsed": False,
            }
            for seed in range(5)
        ]

    def test_exact_floors_are_inclusive_and_complete_panel_passes(self):
        result = evaluate_prespecified_pass_rule(self.rows())
        self.assertTrue(result["passed"])
        self.assertEqual(result["sensitivity"]["positive_seeds"], 5)
        self.assertEqual(result["clean_neighbor_recovery"]["positive_seeds"], 5)

    def test_zero_is_not_improvement(self):
        rows = self.rows()
        for row in rows[:2]:
            row["sensitivity_after"] = row["sensitivity_before"]
            row["clean_neighbor_jaccard_after"] = row["clean_neighbor_jaccard_before"]
        result = evaluate_prespecified_pass_rule(rows)
        self.assertFalse(result["passed"])
        self.assertEqual(result["sensitivity"]["positive_seeds"], 3)
        self.assertEqual(result["clean_neighbor_recovery"]["positive_seeds"], 3)

    def test_any_safeguard_violation_fails(self):
        rows = self.rows()
        rows[4]["rare_recall_deltas"]["rare-a"] = -0.0500001
        result = evaluate_prespecified_pass_rule(rows)
        self.assertFalse(result["passed"])
        self.assertEqual(result["preservation"]["violations"][0]["metric"], "rare_recall:rare-a")

    def test_collapse_nonfinite_and_incomplete_panels_fail_closed(self):
        rows = self.rows()
        rows[1]["collapsed"] = True
        self.assertFalse(evaluate_prespecified_pass_rule(rows)["passed"])
        rows = self.rows()
        rows[2]["delta_ARI"] = float("nan")
        self.assertFalse(evaluate_prespecified_pass_rule(rows)["passed"])
        self.assertFalse(evaluate_prespecified_pass_rule(self.rows()[:-1])["passed"])


if __name__ == "__main__":
    unittest.main()
