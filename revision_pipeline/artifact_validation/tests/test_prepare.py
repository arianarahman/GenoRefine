import tempfile
import unittest
import gc
from pathlib import Path
from unittest.mock import patch

import numpy as np

from revision_pipeline.artifact_validation.common import (
    ARTIFACT_IDS,
    CASE_IDS,
    CaseInputs,
    case_spec,
    load_case_inputs,
    load_prepared,
    rare_group_registry,
    specification,
)
from revision_pipeline.artifact_validation.prepare import (
    _persist_prepared,
    generate_bundles,
)
from revision_pipeline.data.readers import array_hash
from revision_pipeline.integrity import canonical_hash
from revision_pipeline.pilot.common import completed


class FrozenProtocolTests(unittest.TestCase):
    def test_scope_is_exact_three_by_two_by_two_with_five_seeds(self):
        spec = specification()
        self.assertEqual(tuple(case["id"] for case in spec["cases"]), CASE_IDS)
        self.assertEqual(tuple(item["id"] for item in spec["artifacts"]), ARTIFACT_IDS)
        self.assertEqual(len(spec["cases"]) * len(spec["artifacts"]), 12)
        self.assertEqual(spec["replicate_seeds"], [0, 1, 2, 3, 4])
        self.assertFalse(spec["training_K_policy"]["reference_labels_used"])
        self.assertIn("store", case_spec("pan_harmony"))
        self.assertIn("fixed10-build0", case_spec("pan_harmony")["store"])

    def test_all_clean_inputs_match_their_frozen_parent_and_targets(self):
        expected = {
            "pan_scanorama": (14767, 100, "3"),
            "pan_harmony": (14767, 50, "3"),
            "hp_scanorama": (16382, 100, "alpha"),
            "hp_harmony": (16382, 50, "alpha"),
            "mouse_scanorama": (50000, 100, "B cell"),
            "mouse_harmony": (50000, 50, "B cell"),
        }
        for case_id, (rows, columns, target) in expected.items():
            inputs = load_case_inputs(case_id, verify_original_sources=False)
            self.assertEqual(inputs.values.shape, (rows, columns))
            self.assertEqual(inputs.case["target_label"], target)
            self.assertTrue(np.any(inputs.named_labels == target))
            self.assertEqual(
                set(inputs.batches[inputs.named_labels == target].tolist()),
                set(inputs.batches.tolist()),
            )
            self.assertEqual(inputs.parent_reference, inputs.case["parent_reference"])


class PreparationFixtureTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(118)
        n, d = 36, 5
        ids = tuple(f"fixture-{i:03d}" for i in range(n))
        values = rng.normal(size=(n, d))
        batches = np.asarray(["c", "a", "b"] * 12)
        labels = np.asarray(["target"] * 18 + ["common"] * 15 + ["rare"] * 3)
        # Each target/batch group is represented because batch labels cycle.
        lookup = {label: i for i, label in enumerate(sorted(set(labels.tolist())))}
        codes = np.asarray([lookup[label] for label in labels], dtype=np.int64)
        parent = {
            "dataset_fingerprint": "d" * 64,
            "embedding_id": "Fixture",
            "stored_values_file_sha256": "f" * 64,
            "selected_values_sha256": array_hash(values),
            "cell_order_sha256": canonical_hash(list(ids)),
            "shape": [n, d],
        }
        self.case = {
            "id": "fixture_case",
            "dataset": "fixture_dataset",
            "display_dataset": "Fixture",
            "backbone": "Fixture",
            "target_label": "target",
            "target_label_status": "test_fixture",
            "shape": [n, d],
            "case_run": "fixture/run",
            "case_manifest": {"sha256": "0" * 64, "size_bytes": 1},
            "parent_reference": parent,
        }
        self.spec = {
            "protocol_id": "artifact_validation_v2",
            "artifacts": [
                {
                    "id": "batch_simplex",
                    "implementation": "batch_regular_simplex_translation_v2",
                    "strength": 1.0,
                    "scale_k_nonself": 3,
                },
                {
                    "id": "target_local_warp",
                    "implementation": "target_local_nonlinear_batch_warp_v2",
                    "strength": 1.0,
                    "scale_k_same_class_nonself": 3,
                },
            ],
            "replicate_seeds": [0, 1, 2, 3, 4],
            "working_memory_mb": 1,
            "rare_fraction_max": 0.1,
            "training_K_policy": {
                "rule": "median_baseline_leiden_count",
                "resolution": 0.5,
                "leiden_seeds": [0, 1, 2],
                "reference_labels_used": False,
                "recompute_for_each": ["dataset", "backbone", "artifact"],
                "preparation_status": "metadata_only_not_computed_until_training_binding",
            },
        }
        self.inputs = CaseInputs(
            case=self.case,
            values=values,
            cell_ids=ids,
            coordinate_names=tuple(f"Fixture_coordinate_{i:03d}" for i in range(d)),
            named_labels=labels,
            reference_codes=codes,
            batches=batches,
            annotation_policy={"named_labels_allowed": True},
            parent_reference=parent,
            dataset_source_files={},
        )

    def test_rare_registry_uses_full_cohort_and_excludes_singletons(self):
        labels = np.asarray(["common"] * 97 + ["rare"] * 2 + ["singleton"])
        codes = np.asarray([0] * 97 + [1] * 2 + [2], dtype=np.int64)
        batches = np.asarray(["a", "b"] * 50)
        result = rare_group_registry(labels, codes, batches, fraction_max=0.02)
        self.assertEqual(
            [row["supplied_label"] for row in result["eligible_non_singleton_groups"]],
            ["rare"],
        )
        self.assertEqual(
            [row["supplied_label"] for row in result["excluded_singletons"]],
            ["singleton"],
        )

    def test_generation_persistence_and_loader_round_trip(self):
        bundles = generate_bundles(self.inputs, self.spec)
        self.assertEqual(set(bundles), set(ARTIFACT_IDS))
        with tempfile.TemporaryDirectory() as folder:
            path = _persist_prepared(
                output_root=Path(folder),
                run_id="fixture-prepared",
                spec=self.spec,
                case=self.case,
                inputs=self.inputs,
                bundles=bundles,
                sources={},
            )
            completed(path, "artifact_validation_v2_prepared_case")
            with patch(
                "revision_pipeline.artifact_validation.common.specification",
                return_value=self.spec,
            ), patch(
                "revision_pipeline.artifact_validation.common.case_spec",
                return_value=self.case,
            ):
                prepared = load_prepared(path)
            self.assertEqual(prepared.cell_ids, self.inputs.cell_ids)
            np.testing.assert_array_equal(prepared.named_labels, self.inputs.named_labels)
            self.assertEqual(prepared.record["rare_group_count"], 1)
            self.assertFalse(prepared.record["training_K_computed"])
            for artifact_id in ARTIFACT_IDS:
                view = prepared.artifact(artifact_id)
                np.testing.assert_array_equal(view.values, bundles[artifact_id].observed)
                np.testing.assert_array_equal(
                    view.counterfactuals[
                        view.batch_index, np.arange(len(view.cell_ids), dtype=np.int64)
                    ],
                    view.values,
                )
                self.assertEqual(
                    view.parent_reference()["selected_values_sha256"],
                    array_hash(view.values),
                )
                self.assertEqual(view.coordinate_names, self.inputs.coordinate_names)
            del view, prepared
            gc.collect()


if __name__ == "__main__":
    unittest.main()
