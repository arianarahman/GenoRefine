# Purpose: Validate panel behavior and invariants for the spatial multisection workflow.
# Author: Ariana Rahman (Arizona State University)

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from revision_pipeline.spatial_multisection.common import ROOT, evaluation_dict, load_metadata, specification
from revision_pipeline.spatial_multisection.consolidate import (
    DONOR_METRICS, SECTION_METRICS, aggregate_donors, aggregate_sections, expected_runs,
    validate_child_source_hashes,
)
from revision_pipeline.spatial_multisection.harmony_fixed import run_fixed_harmony
from revision_pipeline.spatial_multisection.score import spatial_metrics, validate_training_binding
from revision_pipeline.spatial_multisection.train_spagcn import _histology_adjacency
from revision_pipeline.step4_policy import derive_training_k


class _HarmonyOutput:
    def __init__(self, values, iterations=10):
        self.Z_corr = np.asarray(values).T
        self.objective_harmony = list(np.linspace(20.0, 10.0, iterations + 1))
        self.kmeans_rounds = [5] * iterations


class _FakeSpaGCN:
    def __init__(self):
        self.call = None

    def calculate_adj_matrix(self, **kwargs):
        self.call = kwargs
        n = len(kwargs["x"])
        return np.zeros((n, n), dtype=np.float32)


class PanelContractTests(unittest.TestCase):
    def test_locked_scope_and_distinct_neighbor_conventions(self):
        spec = specification()
        self.assertEqual(spec["sections"], ["151507", "151508", "151669", "151670", "151673", "151674"])
        self.assertEqual(spec["evaluation"]["spatial_neighbor_k"], 6)
        self.assertEqual(spec["evaluation"]["label_neighbor_purity_k"], 30)
        self.assertFalse(spec["cluster_count_selection"]["reference_labels_used"])
        self.assertFalse(spec["aggregation"]["p_values"])

    def test_expected_run_names_and_counts_cover_the_complete_131_job_plan(self):
        runs = expected_runs("fixture")
        self.assertEqual(len(runs["gr_training"]), 5)
        self.assertEqual(len(runs["spagcn_training"]), 30)
        self.assertEqual(len(runs["section_scores"]), 72)
        self.assertEqual(len(runs["donor_scores"]), 21)
        names = [path.name for value in runs.values()
                 for path in (value if isinstance(value, list) else [value])]
        self.assertEqual(len(names), 130)
        self.assertEqual(len(set(names)), 130)
        self.assertIn("fixture-score-section-151507-spagcn-s4", names)
        self.assertIn("fixture-score-donor-Br8100-gr-s4", names)

    def test_k_rule_uses_canonical_json_safe_evaluator_contract(self):
        ids = [f"cell-{index}" for index in range(8)]
        partitions = {
            0: np.asarray([0, 0, 0, 1, 1, 1, 1, 1]),
            1: np.asarray([0, 0, 1, 1, 2, 2, 2, 2]),
            2: np.asarray([0, 0, 1, 1, 1, 2, 2, 2]),
        }
        frozen = evaluation_dict()
        self.assertIsInstance(frozen["leiden_seeds"], list)
        decision = derive_training_k(partitions, ids, ids, frozen)
        self.assertEqual(decision["n_clusters"], 3)
        self.assertFalse(decision["reference_labels_used"])

    def test_foundation_ids_and_donors_are_complete(self):
        frame = load_metadata()
        self.assertEqual(len(frame), 22968)
        self.assertEqual(frame["cell_id"].nunique(), 22968)
        self.assertEqual(frame["section"].nunique(), 6)
        self.assertEqual(frame["donor"].nunique(), 3)

    def test_fixed_harmony_disables_outer_stop_and_requires_ten(self):
        pca = np.arange(120, dtype=np.float32).reshape(24, 5)
        sections = np.asarray(["a"] * 12 + ["b"] * 12)
        captured = {}

        def runner(values, metadata, key, **kwargs):
            captured.update(kwargs)
            self.assertEqual(key, "section")
            self.assertEqual(list(metadata.columns), ["section"])
            return _HarmonyOutput(values, 10)

        values, record = run_fixed_harmony(pca, sections, runner=runner)
        np.testing.assert_array_equal(values, pca)
        self.assertTrue(np.isneginf(captured["epsilon_harmony"]))
        self.assertEqual(captured["max_iter_harmony"], 10)
        self.assertEqual(record["iterations_completed"], 10)
        self.assertTrue(record["outer_early_stopping_disabled"])

        with self.assertRaises(RuntimeError):
            run_fixed_harmony(pca, sections, runner=lambda *args, **kwargs: _HarmonyOutput(pca, 2))

    def test_histology_uses_array_grid_for_geometry_and_hires_only_for_pixels(self):
        fake = _FakeSpaGCN()
        settings = specification()["spagcn"]
        array_grid = np.asarray([[1, 2], [3, 4]], dtype=float)
        hires = np.asarray([[10.2, 20.4], [30.6, 40.8]], dtype=float)
        adjacency, record = _histology_adjacency("151507", array_grid, hires, fake, settings)
        self.assertEqual(adjacency.shape, (2, 2))
        self.assertEqual(fake.call["x"], [1.0, 3.0])
        self.assertEqual(fake.call["y"], [2.0, 4.0])
        self.assertEqual(fake.call["x_pixel"], [10, 31])
        self.assertEqual(fake.call["y_pixel"], [20, 41])
        self.assertEqual(record["spatial_geometry_coordinate_source"], ["array_row", "array_col"])
        self.assertEqual(record["histology_sampling_coordinate_source"],
                         ["rounded hires_pxl_row", "rounded hires_pxl_col"])

    def test_spatial_metric_is_six_nonself_neighbors(self):
        coordinates = np.column_stack((np.arange(12, dtype=float), np.zeros(12)))
        values = coordinates.copy()
        summary, arrays = spatial_metrics(values, coordinates, 6, 16)
        self.assertEqual(summary["k"], 6)
        self.assertTrue(summary["self_excluded_by_index"])
        self.assertAlmostEqual(summary["spatial_latent_knn_jaccard"], 1.0)
        self.assertEqual(arrays["latent_neighbors"].shape, (12, 6))
        self.assertEqual(arrays["spatial_neighbors"].shape, (12, 6))

    def test_training_binding_rejects_wrong_k_or_requested_manifests(self):
        harmony = {"sha256": "h", "size_bytes": 1}
        k_manifest = {"sha256": "k", "size_bytes": 2}
        decisions = {"pooled": {"n_clusters": 7}, "sections": {"151507": {"n_clusters": 4}}}
        valid = {"harmony_manifest": harmony, "k_selection_manifest": k_manifest,
                 "K_binding": decisions["pooled"]}
        validate_training_binding(valid, method="genorefine", harmony_manifest=harmony,
                                  k_selection_manifest=k_manifest, decisions=decisions)
        invalid = {**valid, "K_binding": {"n_clusters": 8}}
        with self.assertRaises(ValueError):
            validate_training_binding(invalid, method="genorefine", harmony_manifest=harmony,
                                      k_selection_manifest=k_manifest, decisions=decisions)
        spa = {"harmony_manifest": harmony, "k_selection_manifest": k_manifest,
               "K_binding": decisions["sections"]["151507"], "section": "151507"}
        with self.assertRaises(ValueError):
            validate_training_binding(spa, method="spagcn", harmony_manifest={"sha256": "other"},
                                      k_selection_manifest=k_manifest, decisions=decisions,
                                      section="151507")

    @staticmethod
    def _section_row(method, section, seed, value):
        donor = next(d for d, sections in specification()["donors"].items() if section in sections)
        return {"method": method, "section": section, "donor": donor,
                "algorithmic_seed": seed, **{metric: float(value) for metric in SECTION_METRICS}}

    def test_donor_first_aggregation_never_labels_donors_as_algorithmic_seeds(self):
        spec = specification()
        section_rows = []
        for section in spec["sections"]:
            section_rows.append(self._section_row("harmony_fixed", section, None, 1))
            section_rows.append(self._section_row("harmony_native_sensitivity", section, None, 2))
            for seed in range(5):
                section_rows.append(self._section_row("genorefine", section, seed, 3 + seed))
                section_rows.append(self._section_row("spagcn", section, seed, 4 + seed))
        section_summary = aggregate_sections(section_rows)
        self.assertEqual(section_summary["harmony_fixed"]["macro"]["ARI"]["n_algorithmic_seeds"], 0)
        self.assertEqual(section_summary["harmony_native_sensitivity"]["macro"]["ARI"]["n_algorithmic_seeds"], 0)
        self.assertEqual(section_summary["harmony_native_sensitivity"]["macro"]["ARI"]["n_donors"], 3)
        self.assertEqual(section_summary["genorefine"]["macro"]["ARI"]["n_algorithmic_seeds"], 5)

        donor_rows = []
        for donor in spec["donors"]:
            donor_rows.extend([
                {"method": "harmony_fixed", "donor": donor, "algorithmic_seed": None,
                 **{metric: 1.0 for metric in DONOR_METRICS}},
                {"method": "harmony_native_sensitivity", "donor": donor, "algorithmic_seed": None,
                 **{metric: 2.0 for metric in DONOR_METRICS}},
            ])
            donor_rows.extend(
                {"method": "genorefine", "donor": donor, "algorithmic_seed": seed,
                 **{metric: float(seed) for metric in DONOR_METRICS}}
                for seed in range(5)
            )
        donor_summary = aggregate_donors(donor_rows)
        self.assertEqual(donor_summary["harmony_fixed"]["macro"]["iLISI"]["n_algorithmic_seeds"], 0)
        self.assertEqual(donor_summary["harmony_native_sensitivity"]["macro"]["iLISI"]["n_donors"], 3)
        self.assertEqual(donor_summary["genorefine"]["macro"]["iLISI"]["n_algorithmic_seeds"], 5)

    def test_null_or_different_child_source_hash_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "revision_pipeline") as temporary:
            path = Path(temporary) / "child"; path.mkdir()
            (path / "run.json").write_text(json.dumps({"source_tree_sha256": None}), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_child_source_hashes([path], "wanted")
            (path / "run.json").write_text(json.dumps({"source_tree_sha256": "other"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_child_source_hashes([path], "wanted")
            (path / "run.json").write_text(json.dumps({"source_tree_sha256": "wanted"}), encoding="utf-8")
            validate_child_source_hashes([path], "wanted")

    def test_orchestrator_has_exclusive_prefix_lock_and_preserves_stale_runs(self):
        text = (ROOT / "revision_pipeline/spatial_multisection/run_panel.ps1").read_text(encoding="utf-8")
        self.assertIn("[IO.FileShare]::None", text)
        self.assertIn("Move-Item -LiteralPath $resolvedCandidate", text)
        self.assertIn("verify_completed", text)
        self.assertIn("finally {", text)


if __name__ == "__main__":
    unittest.main()
