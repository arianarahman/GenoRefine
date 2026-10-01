# Purpose: Validate native reporting behavior and invariants for the spatial multisection
#          workflow.
# Author: Ariana Rahman (Arizona State University)

from __future__ import annotations

import unittest

from revision_pipeline.spatial_multisection.report_native_spagcn import (
    aggregate_native_rows,
    extract_native_rows,
)


class NativeSpaGCNReportingTests(unittest.TestCase):
    sections = ["a1", "a2", "b1", "b2", "b3"]
    donors = {"A": ["a1", "a2"], "B": ["b1", "b2", "b3"]}

    @classmethod
    def _source_rows(cls):
        donor_for_section = {
            section: donor for donor, sections in cls.donors.items() for section in sections
        }
        rows = []
        offsets = {"a1": 0.0, "a2": 2.0, "b1": 10.0, "b2": 20.0, "b3": 30.0}
        for section in cls.sections:
            for seed in range(5):
                base = offsets[section] + seed
                rows.append({
                    "section": section,
                    "donor": donor_for_section[section],
                    "method": "spagcn",
                    "algorithmic_seed": seed,
                    "native_spagcn_partitions": {
                        "predicted": {
                            "ARI": base,
                            "NMI": base + 0.25,
                            "clusters": 5,
                            "role": "secondary task-native SpaGCN partition",
                        },
                        "refined": {
                            "ARI": base + 100.0,
                            "NMI": base + 100.25,
                            "clusters": 5,
                            "role": "secondary task-native SpaGCN partition",
                        },
                    },
                })
        return rows

    def test_extraction_requires_complete_unique_five_seed_panel(self):
        source = self._source_rows()
        rows = extract_native_rows(source, sections=self.sections, donors=self.donors)
        self.assertEqual(len(rows), 2 * len(self.sections) * 5)
        self.assertEqual(rows[0]["endpoint"], "predicted")
        with self.assertRaises(ValueError):
            extract_native_rows(source[:-1], sections=self.sections, donors=self.donors)
        with self.assertRaises(ValueError):
            extract_native_rows(source + [source[0]], sections=self.sections, donors=self.donors)

    def test_macro_is_seed_aligned_section_then_donor_not_pooled_section_mean(self):
        rows = extract_native_rows(self._source_rows(), sections=self.sections, donors=self.donors)
        summary, donor_rows, macro_rows = aggregate_native_rows(
            rows, sections=self.sections, donors=self.donors
        )
        # Seed 0: donor A=(0+2)/2=1; donor B=(10+20+30)/3=20; macro=(1+20)/2=10.5.
        predicted_seed0 = next(
            row for row in macro_rows
            if row["endpoint"] == "predicted" and row["algorithmic_seed"] == 0
        )
        self.assertEqual(predicted_seed0["ARI"], 10.5)
        donor_b_seed0 = next(
            row for row in donor_rows
            if row["donor"] == "B"
            and row["endpoint"] == "predicted"
            and row["algorithmic_seed"] == 0
        )
        self.assertEqual(donor_b_seed0["ARI"], 20.0)
        # Across seeds, the macro vector is 10.5, 11.5, 12.5, 13.5, 14.5.
        self.assertEqual(summary["macro"]["predicted"]["ARI"]["mean"], 12.5)
        self.assertEqual(summary["macro"]["predicted"]["ARI"]["n_algorithmic_seeds"], 5)
        self.assertEqual(summary["macro"]["refined"]["ARI"]["mean"], 112.5)

    def test_section_summary_retains_seed_values_and_cluster_granularity(self):
        rows = extract_native_rows(self._source_rows(), sections=self.sections, donors=self.donors)
        summary, _, _ = aggregate_native_rows(rows, sections=self.sections, donors=self.donors)
        ari = summary["sections"]["a1"]["predicted"]["ARI"]
        self.assertEqual([row["value"] for row in ari["seed_values"]], [0, 1, 2, 3, 4])
        clusters = summary["sections"]["a1"]["predicted"]["clusters"]
        self.assertEqual(clusters["unique_values"], [5])


if __name__ == "__main__":
    unittest.main()
