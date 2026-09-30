from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from revision_pipeline.data.store import read_json
from revision_pipeline.evaluate.contrasts import anchor_counts, paired_clustering, select_count_match
from revision_pipeline.evaluate.engine import graph_and_grid
from revision_pipeline.evaluate.runner import evaluate_store
from test_evaluation import config, fixture


def grid():
    return [{"leiden_seed": s, "resolution": r, "n_clusters": k, "ARI": .5, "RI": .8,
             "partition_index": i, "training_label_use": "reference_labels_secondary"}
            for i, (s, r, k) in enumerate((s, r, k) for s in config().leiden_seeds
                                         for r, k in ((.2, 2), (.5, 4), (1., 8)))]


class ContrastTests(unittest.TestCase):
    def test_zero_changes_and_K_at_anchor(self):
        r = paired_clustering(grid(), grid(), config())
        self.assertEqual(r["grid_consistency"]["ARI"]["sign_counts"], {"positive": 0, "zero": 9, "negative": 0})
        self.assertTrue(all(m["status"] == "exact_match" and m["refined_resolution"] == .5 for m in r["matched_granularity"]))
        self.assertEqual([r["n_clusters"] for r in anchor_counts(grid(), config())], [4, 4, 4])

    def test_flips_are_retained(self):
        after = deepcopy(grid())
        for row in after:
            row["ARI"] += .1 if row["resolution"] == .5 else -.1
        r = paired_clustering(grid(), after, config())
        s = r["grid_consistency"]["ARI"]
        self.assertEqual(s["sign_counts"], {"positive": 3, "zero": 0, "negative": 6})
        self.assertTrue(s["resolution_or_seed_sensitive"])
        self.assertEqual(s["flips_vs_same_seed_anchor"], 6)
        self.assertEqual(len(r["grid_differences"]), 9)

    def test_matching_uses_each_baseline_seed_count(self):
        before, after = grid(), grid()
        before[4]["n_clusters"] = 8
        r = paired_clustering(before, after, config())
        self.assertEqual([m["refined_resolution"] for m in r["matched_granularity"]], [.5, 1., .5])

    def test_scores_never_choose_resolution(self):
        candidates = [{"resolution": .2, "n_clusters": 4, "ARI": -1.},
                      {"resolution": .5, "n_clusters": 2, "ARI": 1.}]
        self.assertEqual(select_count_match(candidates, 4)["resolution"], .2)
        candidates[0]["ARI"], candidates[1]["ARI"] = 1., -1.
        self.assertEqual(select_count_match(candidates, 4)["resolution"], .2)
        self.assertEqual(select_count_match([{k: v for k, v in c.items() if k != "ARI"} for c in candidates], 4)["resolution"], .2)

    def test_ties_use_anchor_then_lower_resolution_not_input_order(self):
        candidates = [{"resolution": .6, "n_clusters": 4}, {"resolution": .4, "n_clusters": 4},
                      {"resolution": .2, "n_clusters": 4}]
        self.assertEqual(select_count_match(candidates, 4)["resolution"], .4)
        self.assertEqual(select_count_match(list(reversed(candidates)), 4)["resolution"], .4)
        candidates.append({"resolution": .5, "n_clusters": 4})
        self.assertEqual(select_count_match(candidates, 4)["resolution"], .5)

    def test_no_exact_match_never_claims_matching(self):
        after = grid()
        for row in after:
            row["n_clusters"] += 10
        r = paired_clustering(grid(), after, config())
        self.assertTrue(all(m["status"] == "unmatched_closest_available" for m in r["matched_granularity"]))
        self.assertTrue(all(m["selected_at_grid_boundary"] for m in r["matched_granularity"]))

    def test_training_label_use_and_pairing_provenance_survive(self):
        r = paired_clustering(grid(), grid(), config(), verified=True)
        self.assertFalse(r["selection_label_informed"])
        self.assertEqual(r["matched_granularity"][0]["refined_training_label_use"], "reference_labels_secondary")
        self.assertEqual(r["pairing_status"], "exact_parent_reference_verified")

    def test_missing_duplicate_extra_or_nonfinite_results_fail(self):
        for changed in (grid()[:-1], grid()+[grid()[0]], [dict(grid()[0], resolution=9.)]+grid()[1:],
                        [dict(grid()[0], ARI=float("nan"))]+grid()[1:], [dict(grid()[0], n_clusters=0)]+grid()[1:]):
            with self.assertRaises(ValueError):
                paired_clustering(grid(), changed, config())

    def test_label_informed_anchor_selection_is_rejected(self):
        with self.assertRaises(ValueError):
            paired_clustering(grid(), grid(), replace(config(), selection="matched_reference_count", fixed_resolution=None))

    def test_numerical_zero_is_not_a_meaningful_effect_margin(self):
        after = grid()
        for row in after:
            row["ARI"] += 1e-14
        result = paired_clustering(grid(), after, config())
        self.assertFalse(result["sign_tolerance_is_effect_margin"])
        self.assertEqual(result["grid_consistency"]["ARI"]["sign_counts"]["zero"], 9)


class RunnerContractTests(unittest.TestCase):
    def test_new_pair_and_cached_baseline_emit_identical_contrasts(self):
        x, _, _, dataset = fixture()
        make = lambda name, values, kind: SimpleNamespace(values=values, cell_ids=dataset.cell_ids,
            metadata={"kind": kind, "pairing_status": "historical_input_pairing_unverified"},
            parent_reference=lambda: {"name": name, "ordered_values": values.tolist()})
        items = {"Scanorama": make("Scanorama", x, "baseline"),
                 "GenoDR_Scanorama": make("GenoDR_Scanorama", x[:, :2], "historical_refined")}
        fake = SimpleNamespace(dataset=lambda _: dataset, embedding=lambda _, name: items[name])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"store").mkdir()
            (root/"store/run.json").write_text("{}", encoding="utf-8")
            with patch("revision_pipeline.evaluate.runner.Store", return_value=fake), \
                    patch("revision_pipeline.evaluate.runner.snapshot", return_value={}), \
                    patch("revision_pipeline.evaluate.runner.runtime", return_value={"runtime": "fixture"}), \
                    patch("revision_pipeline.evaluate.runner.graph_and_grid", wraps=graph_and_grid) as graph:
                baseline = evaluate_store(root, root/"store", "fixture", "Scanorama", config())
                direct = evaluate_store(root, root/"store", "fixture", "Scanorama", config(), compare_to="GenoDR_Scanorama")
                cached = evaluate_store(root, root/"store", "fixture", "Scanorama", config(), compare_to="GenoDR_Scanorama",
                                        baseline_evaluation=baseline)
                self.assertEqual(graph.call_count, 4)  # 1 + 2 + 1; baseline graph not repeated by cache.
            a, b = [read_json(path/"pair/clustering_comparison.json") for path in (direct, cached)]
            self.assertEqual(a, b)
            self.assertEqual(len(a["grid_differences"]), 9)
            self.assertEqual(len(read_json(baseline/"summary.json")["evaluations"][0]["cluster_counts_at_0_5"]), 3)
