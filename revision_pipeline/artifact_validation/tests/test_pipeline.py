# Purpose: Validate the end-to-end artifact-validation protocol and its provenance gates.
# Author: Ariana Rahman (Arizona State University)

import ast
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from revision_pipeline.data.readers import array_hash
from revision_pipeline.integrity import canonical_hash
from revision_pipeline.artifact_validation.metrics import (
    clean_neighbor_recovery,
    clean_neighbor_recovery_from_reference,
    preservation_metrics,
    preservation_metrics_from_neighbors,
)
from revision_pipeline.artifact_validation.consolidate import _validate_source_lineage
from revision_pipeline.artifact_validation.train import _save_inference_outputs
from revision_pipeline.artifact_validation import source_compatibility
from revision_pipeline.artifact_validation.run_panel import (
    build_plan,
    load_execution_config,
)
from revision_pipeline.independent_comparator.idec_compat import (
    _sequential_batch_bounds,
    _validate_dimensions,
)


class _TinyRun:
    def __init__(self, path):
        self.path = Path(path)

    def artifact_path(self, relative):
        path = self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        return path

    def write_json(self, relative, value):
        path = self.artifact_path(relative)
        path.write_text(json.dumps(value), encoding="utf-8")


class ArtifactPipelineHardeningTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(71)
        self.values = rng.normal(size=(24, 5))
        self.ids = tuple(f"cell_{index:03d}" for index in range(len(self.values)))
        self.labels = np.asarray(["a"] * 8 + ["b"] * 8 + ["rare"] * 8)

    def test_sequential_schedule_has_no_empty_exact_divisor_batch(self):
        bounds = _sequential_batch_bounds(128, 64, 4)
        self.assertEqual(bounds, [(0, 64), (64, 128), (0, 64), (64, 128)])
        visits = np.zeros(128, dtype=np.int64)
        for begin, stop in bounds:
            self.assertGreater(stop, begin)
            visits[begin:stop] += 1
        np.testing.assert_array_equal(visits, 2)

    def test_sequential_schedule_covers_partial_tail_twice(self):
        bounds = _sequential_batch_bounds(130, 64, 6)
        self.assertEqual(bounds, [(0, 64), (64, 128), (128, 130)] * 2)
        visits = np.zeros(130, dtype=np.int64)
        for begin, stop in bounds:
            visits[begin:stop] += 1
        np.testing.assert_array_equal(visits, 2)

    def test_dimensions_reject_lossy_integer_coercion(self):
        self.assertEqual(_validate_dimensions([5, 3, 2]), [5, 3, 2])
        for invalid in ([5, 2.5], [True, 2], np.asarray([5, 2])):
            with self.assertRaises(ValueError):
                _validate_dimensions(invalid)

    def test_cached_clean_neighbors_reproduce_direct_recovery(self):
        candidate = self.values.copy()
        candidate[:, 0] += np.linspace(-0.2, 0.2, len(candidate))
        direct = clean_neighbor_recovery(
            self.values,
            candidate,
            self.ids,
            self.ids,
            self.ids,
            k=5,
            working_memory_mb=1,
        )
        cached = clean_neighbor_recovery_from_reference(
            direct["clean_neighbors"],
            candidate,
            self.ids,
            self.ids,
            k=5,
            working_memory_mb=1,
        )
        self.assertEqual(
            cached["mean_clean_neighbor_jaccard"],
            direct["mean_clean_neighbor_jaccard"],
        )
        np.testing.assert_array_equal(
            cached["candidate_neighbors"], direct["candidate_neighbors"]
        )
        np.testing.assert_array_equal(
            cached["per_cell_clean_neighbor_jaccard"],
            direct["per_cell_clean_neighbor_jaccard"],
        )

    def test_one_neighbor_table_reproduces_preservation_metrics(self):
        direct = preservation_metrics(
            self.values,
            self.labels,
            self.ids,
            self.ids,
            rare_groups=["rare"],
            target_label="a",
            k=5,
            working_memory_mb=1,
        )
        cached = preservation_metrics_from_neighbors(
            direct["neighbors"],
            self.labels,
            self.ids,
            self.ids,
            rare_groups=["rare"],
            target_label="a",
            k=5,
        )
        self.assertEqual(direct["purity"]["mean_neighbor_purity"],
                         cached["purity"]["mean_neighbor_purity"])
        self.assertEqual(
            direct["rare_recall"]["groups"][0]["mean_same_class_recall_at_k"],
            cached["rare_recall"]["groups"][0]["mean_same_class_recall_at_k"],
        )
        self.assertEqual(
            direct["target_same_class_fraction"][
                "mean_target_same_class_fraction_at_k"
            ],
            cached["target_same_class_fraction"][
                "mean_target_same_class_fraction_at_k"
            ],
        )

    def test_score_preservation_call_matches_metric_signature(self):
        """Keep the production scorer aligned with the cached-neighbor API."""
        score_path = Path(__file__).resolve().parents[1] / "score.py"
        module_tree = ast.parse(score_path.read_text(encoding="utf-8"))
        run_score = next(
            node
            for node in module_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_score"
        )
        calls = [
            node
            for node in ast.walk(run_score)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "preservation_metrics_from_neighbors"
        ]
        self.assertEqual(len(calls), 1)
        supplied_keywords = {keyword.arg for keyword in calls[0].keywords}
        accepted_keywords = set(
            inspect.signature(preservation_metrics_from_neighbors).parameters
        )
        self.assertFalse(supplied_keywords - accepted_keywords)

    def test_inference_receipt_checks_expected_output_and_records_hashes(self):
        clean = self.values[:, :2]
        counterfactuals = np.stack([clean, clean + 0.1])
        with tempfile.TemporaryDirectory() as temporary:
            run = _TinyRun(temporary)
            _save_inference_outputs(run, clean, counterfactuals, clean, clean.copy())
            record = json.loads(
                (Path(temporary) / "inference/checks.json").read_text()
            )
            self.assertEqual(
                record["observed_reloaded_sha256"],
                record["expected_observed_sha256"],
            )
        invalid = clean.copy()
        invalid[0, 0] = np.inf
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                _save_inference_outputs(
                    _TinyRun(temporary), clean, counterfactuals, clean, invalid
                )

    def test_expected_observed_hash_uses_training_receipt_dtype(self):
        values = np.asarray([[0.1, 1.25], [2.5, -3.0]], dtype=np.float32)
        expected = array_hash(np.asarray(values, dtype=np.float64))
        self.assertEqual(source_compatibility.expected_observed_hash(values), expected)
        self.assertNotEqual(array_hash(values), expected)
        changed = values.copy()
        changed.view(np.uint32)[0, 0] ^= np.uint32(1)
        self.assertNotEqual(
            source_compatibility.expected_observed_hash(changed), expected
        )

    def test_scoring_recovery_allows_only_declared_source_surface(self):
        score_path = "revision_pipeline/artifact_validation/score.py"
        compatibility_path = (
            "revision_pipeline/artifact_validation/source_compatibility.py"
        )
        reference = {
            score_path: {"sha256": "a" * 64, "size_bytes": 10},
            "revision_pipeline/artifact_validation/metrics.py": {
                "sha256": "b" * 64,
                "size_bytes": 20,
            },
        }
        current = {
            **reference,
            score_path: {"sha256": "c" * 64, "size_bytes": 11},
            compatibility_path: {"sha256": "d" * 64, "size_bytes": 30},
        }
        reference_hash = canonical_hash(reference)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "source_manifest.json").write_text(
                json.dumps(reference), encoding="utf-8"
            )
            with mock.patch.object(
                source_compatibility,
                "REFERENCE_SOURCE_TREE_SHA256",
                reference_hash,
            ):
                receipt = source_compatibility.verify_scoring_source_transition(
                    path,
                    reference_hash,
                    current,
                    role="unit_test",
                )
                self.assertEqual(receipt["status"], "verified_scorer_only_recovery")
                invalid = {
                    **current,
                    "revision_pipeline/artifact_validation/metrics.py": {
                        "sha256": "e" * 64,
                        "size_bytes": 21,
                    },
                }
                with self.assertRaises(ValueError):
                    source_compatibility.verify_scoring_source_transition(
                        path,
                        reference_hash,
                        invalid,
                        role="unit_test",
                    )

    def test_consolidation_requires_exact_stage_source_lineage(self):
        stages = {
            "prepared": {"fit"},
            "baseline": {"fit"},
            "training": {"fit"},
            "candidate_score": {"score"},
        }
        self.assertEqual(
            _validate_source_lineage(stages, "score")["training"], "fit"
        )
        mixed = {**stages, "candidate_score": {"score", "legacy"}}
        with self.assertRaises(ValueError):
            _validate_source_lineage(mixed, "score")
        mismatched_fit = {**stages, "training": {"other"}}
        with self.assertRaises(ValueError):
            _validate_source_lineage(mismatched_fit, "score")

    def test_orchestration_plan_has_exact_cardinality_and_dependencies(self):
        plan = build_plan("unitpanel")
        self.assertEqual(
            {stage: len(jobs) for stage, jobs in plan.items()},
            {
                "prepare": 6,
                "baseline_score": 12,
                "training": 120,
                "candidate_score": 120,
                "consolidate": 1,
            },
        )
        ids = [job.run_id for jobs in plan.values() for job in jobs]
        self.assertEqual(len(ids), len(set(ids)))
        first_training = plan["training"][0]
        self.assertIn(
            "revision_pipeline/runs/unitpanel-prep-pan_scanorama",
            first_training.arguments,
        )
        self.assertIn(
            "revision_pipeline/runs/unitpanel-score-pan_scanorama-batch_simplex-baseline",
            first_training.arguments,
        )

    def test_execution_config_is_explicit_and_bounded(self):
        stages = {}
        for stage in (
            "prepare",
            "baseline_score",
            "training",
            "candidate_score",
            "consolidate",
        ):
            stages[stage] = {
                "prefix": ["python"],
                "workers": 1,
                "environment": {"OMP_NUM_THREADS": "1"},
            }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "execution.json"
            path.write_text(
                json.dumps({"schema_version": 1, "stages": stages}),
                encoding="utf-8",
            )
            parsed, receipt = load_execution_config(path)
            self.assertEqual(parsed["training"]["prefix"], ["python"])
            self.assertEqual(receipt["size_bytes"], path.stat().st_size)
            stages["candidate_score"]["workers"] = 33
            path.write_text(
                json.dumps({"schema_version": 1, "stages": stages}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_execution_config(path)


if __name__ == "__main__":
    unittest.main()
