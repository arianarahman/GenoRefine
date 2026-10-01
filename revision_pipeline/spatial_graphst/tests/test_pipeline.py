# Purpose: Validate the GraphST spatial workflow, frozen inputs, and completion contracts.
# Author: Ariana Rahman (Arizona State University)

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from revision_pipeline.data.readers import array_hash
from revision_pipeline.integrity import canonical_hash, file_fingerprint
from revision_pipeline.spatial_graphst import common
from revision_pipeline.spatial_graphst import consolidate
from revision_pipeline.spatial_graphst.common import (
    ROOT, SPEC_PATH, load_metadata, raw_specification, section_labels, source_snapshot,
    specification,
)
from revision_pipeline.spatial_graphst.source_acquisition import source_locks, verify_locked_source


class GraphSTProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = raw_specification()

    def test_protocol_scope_and_label_policy_are_explicit(self) -> None:
        self.assertEqual(self.spec["protocol_id"], "spatial_graphst_panel_v1")
        self.assertEqual(self.spec["graphst"]["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(self.spec["donor_cluster_count_selection"]["leiden_seeds"], [0, 1, 2])
        self.assertFalse(self.spec["graphst"]["reference_labels_used"])
        self.assertFalse(self.spec["donor_cluster_count_selection"]["reference_labels_used"])
        self.assertFalse(self.spec["paste_alignment"]["reference_labels_used"])
        self.assertFalse(self.spec["aggregation"]["p_values"])
        self.assertIn("label-blind conditional on label availability", self.spec["label_policy"])
        self.assertEqual(self.spec["foundation"]["source_spots_before_complete_case_filter"], 23081)
        self.assertEqual(self.spec["foundation"]["spots_excluded_for_missing_manual_layer_label"], 113)
        self.assertEqual(
            self.spec["paste_alignment"]["coordinate_input"],
            ["pxl_col_in_fullres", "pxl_row_in_fullres"],
        )
        self.assertTrue(self.spec["paste_alignment"]["norm"])
        representation = self.spec["graphst"]["continuous_representation"]
        self.assertEqual(representation["official_train_output_dimensions"], 3000)
        self.assertEqual(representation["common_evaluator_dimensions"], 20)
        self.assertFalse(representation["hidden_bottleneck_reported_as_official_emb"])

    def test_donor_pairs_cover_each_section_exactly_once(self) -> None:
        flattened = [section for sections in self.spec["donors"].values() for section in sections]
        self.assertCountEqual(flattened, self.spec["sections"])
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertTrue(all(len(sections) == 2 for sections in self.spec["donors"].values()))

    def test_official_source_locks_are_exact(self) -> None:
        for lock in source_locks():
            receipt = verify_locked_source(ROOT / lock.destination, lock)
            self.assertEqual(receipt["commit"], lock.commit)
            self.assertEqual(receipt["selected_files"], lock.files)

    def test_runtime_build_inputs_match_declared_hashes(self) -> None:
        runtime = self.spec["runtime"]
        for path_key, hash_key in (
            ("dockerfile", "dockerfile_sha256"),
            ("requirements", "requirements_sha256"),
        ):
            self.assertEqual(
                file_fingerprint(ROOT / runtime[path_key])["sha256"],
                runtime[hash_key],
            )

    def test_frozen_specification_and_scoped_snapshot(self) -> None:
        frozen = specification()
        self.assertEqual(frozen, self.spec)
        snapshot = source_snapshot()
        self.assertIn(SPEC_PATH.relative_to(ROOT).as_posix(), snapshot)
        self.assertTrue(snapshot)
        self.assertFalse(any("__pycache__" in path or path.endswith(".pyc") for path in snapshot))
        self.assertFalse(any(path.startswith("Overleaf files/") for path in snapshot))

    def test_training_registry_omits_evaluation_labels(self) -> None:
        metadata = load_metadata(include_labels=False)
        self.assertNotIn("label", metadata.columns)
        self.assertEqual(len(metadata), self.spec["foundation"]["retained_spots"])

    def test_six_digit_section_ids_round_trip_through_alignment_loader(self) -> None:
        first = section_labels("151507", 2)
        second = section_labels("151508", 3)
        sections = np.concatenate([first, second])
        self.assertEqual(sections.dtype.kind, "U")
        self.assertGreaterEqual(sections.dtype.itemsize // 4, 6)
        self.assertEqual(sections.tolist(), ["151507", "151507", "151508", "151508", "151508"])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alignment = root / "alignment"
            foundation = root / "foundation"
            alignment.mkdir(); foundation.mkdir()
            (foundation / "run.json").write_text('{"fixture":true}\n', encoding="utf-8")
            (alignment / "run.json").write_text('{"fixture":true}\n', encoding="utf-8")
            ids = np.asarray([f"cell-{index}" for index in range(5)], dtype="U")
            coordinates = np.arange(10, dtype=np.float64).reshape(5, 2)
            transport = np.full((2, 3), 1.0 / 6.0, dtype=np.float64)
            np.savez_compressed(
                alignment / "alignment.npz", ids=ids, sections=sections,
                aligned_coordinates=coordinates, transport_plan=transport,
            )
            metadata = pd.DataFrame({
                "cell_id": ids, "section": sections,
                "donor": ["Br5292"] * 5,
                "pxl_col_in_fullres": np.arange(5, dtype=np.float64),
                "pxl_row_in_fullres": np.arange(5, dtype=np.float64) + 10,
            })
            settings = {"norm": True}
            config = {
                "protocol_id": "fixture", "donor": "Br5292",
                "sections": ["151507", "151508"],
                "foundation_manifest": file_fingerprint(foundation / "run.json"),
                "settings": settings, "reference_labels_loaded": False,
                "reference_labels_used": False,
            }
            record = {
                "cell_ids_sha256": canonical_hash(ids.tolist()),
                "source_fullres_pixel_coordinates_sha256": array_hash(metadata[
                    ["pxl_col_in_fullres", "pxl_row_in_fullres"]
                ].to_numpy(dtype=np.float64)),
                "aligned_coordinates_sha256": array_hash(coordinates),
                "transport_plan_sha256": array_hash(transport),
                "reference_labels_loaded": False, "reference_labels_used": False,
                "objective": 0.0,
            }
            (alignment / "config.json").write_text(
                json.dumps(config), encoding="utf-8",
            )
            (alignment / "alignment_record.json").write_text(
                json.dumps(record), encoding="utf-8",
            )
            spec = {
                "protocol_id": "fixture", "donors": {"Br5292": ["151507", "151508"]},
                "paste_alignment": settings,
            }
            with (
                patch.object(common, "specification", return_value=spec),
                patch.object(common, "require_run"),
                patch.object(common, "load_metadata", return_value=metadata),
                patch.object(common, "foundation_path", return_value=foundation),
            ):
                loaded = common.load_alignment(alignment, "Br5292")
            self.assertEqual(loaded["sections"].tolist(), sections.tolist())


class GraphSTAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = {
            "sections": ["a1", "a2", "b1", "b2", "c1", "c2"],
            "donors": {"A": ["a1", "a2"], "B": ["b1", "b2"], "C": ["c1", "c2"]},
            "graphst": {"seeds": [0, 1, 2, 3, 4]},
        }

    def _section_rows(self) -> list[dict]:
        rows = []
        for donor_index, (donor, sections) in enumerate(self.spec["donors"].items()):
            for section_index, section in enumerate(sections):
                for seed in self.spec["graphst"]["seeds"]:
                    value = float(donor_index * 100 + section_index * 10 + seed)
                    row = {
                        "section": section, "donor": donor, "algorithmic_seed": seed,
                        "native_graphst_partitions": {},
                    }
                    row.update({metric: value for metric in consolidate.SECTION_METRICS})
                    row["native_graphst_partitions"] = {
                        partition: {metric: value for metric in consolidate.NATIVE_METRICS}
                        for partition in consolidate.NATIVE_PARTITIONS
                    }
                    rows.append(row)
        return rows

    def test_expected_run_plan_is_complete_and_nonduplicated(self) -> None:
        with patch.object(consolidate, "specification", return_value=self.spec):
            plan = consolidate.expected_runs("fixture")
        self.assertEqual(len(plan["alignments"]), 3)
        self.assertEqual(len(plan["training"]), 15)
        self.assertEqual(len(plan["section_scores"]), 30)
        self.assertEqual(len(plan["donor_scores"]), 15)
        paths = consolidate._flatten_paths(plan)
        self.assertEqual(len(paths), 64)
        self.assertEqual(len({str(path) for path in paths}), 64)

    def test_section_aggregation_is_seed_aligned_donor_first(self) -> None:
        with patch.object(consolidate, "specification", return_value=self.spec):
            summary = consolidate.aggregate_sections(self._section_rows())
        # For each seed, donor means are 5+s, 105+s, and 205+s. Their macro
        # mean is 105+s; the five-seed descriptive mean is therefore 107.
        for metric in consolidate.SECTION_METRICS:
            self.assertEqual(summary["macro"][metric]["mean"], 107.0)
            self.assertEqual(summary["macro"][metric]["n_algorithmic_seeds"], 5)

    def test_native_aggregation_remains_separately_labeled(self) -> None:
        with patch.object(consolidate, "specification", return_value=self.spec):
            summary = consolidate.aggregate_native(self._section_rows())
        self.assertEqual(set(summary), set(consolidate.NATIVE_PARTITIONS))
        for partition in consolidate.NATIVE_PARTITIONS:
            self.assertIn("task-native", summary[partition]["role"])
            self.assertEqual(summary[partition]["macro"]["ARI"]["mean"], 107.0)

    def test_donor_mixing_aggregation_equal_weights_donors(self) -> None:
        rows = []
        for donor_index, donor in enumerate(self.spec["donors"]):
            for seed in self.spec["graphst"]["seeds"]:
                row = {"donor": donor, "algorithmic_seed": seed}
                row.update({metric: float(donor_index * 10 + seed) for metric in consolidate.DONOR_METRICS})
                rows.append(row)
        with patch.object(consolidate, "specification", return_value=self.spec):
            summary = consolidate.aggregate_donors(rows)
        for metric in consolidate.DONOR_METRICS:
            self.assertEqual(summary["macro"][metric]["mean"], 12.0)
            self.assertEqual(summary["macro"][metric]["n_algorithmic_seeds"], 5)


if __name__ == "__main__":
    unittest.main()
