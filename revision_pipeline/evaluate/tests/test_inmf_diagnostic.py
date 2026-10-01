# Purpose: Validate inmf diagnostic behavior and invariants for the evaluate workflow.
# Author: Ariana Rahman (Arizona State University)

from copy import deepcopy
import csv
from pathlib import Path
import tempfile
import unittest

from revision_pipeline.evaluate.inmf_order_diagnostic import FILES, CELL_MANIFEST, load_pair, classify_crossover


class InmfDiagnosticTests(unittest.TestCase):
    def test_native_and_cross_matches_demonstrate_only_pipeline_effect(self):
        targets = {"primary": {"ARI": .4254, "RI": .84, "resolution": .5},
                   "robustness": {"ARI": .4598, "RI": .85, "resolution": .5}}
        rows = {(f, o): deepcopy(targets[o]) for f in FILES for o in FILES}
        r = classify_crossover(rows, targets)
        self.assertTrue(r["native_targets_reproduced"] and r["crossed_targets_reproduced"])
        self.assertIn("pipeline_effect_demonstrated", r["conclusion"])
        rows["primary", "robustness"]["ARI"] = .1
        self.assertEqual(classify_crossover(rows, targets)["conclusion"], "simple_historical_row_order_explanation_not_established")

    def test_incomplete_crossover_rejected(self):
        with self.assertRaises(ValueError):
            classify_crossover({}, {})

    def test_CSV_ID_value_and_recorded_order_gates(self):
        ids = ["a", "b", "c"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def write(name, order, perturb=False):
                path = root/FILES[name]
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow([""]+[f"Factor_{i}" for i in range(30)])
                    for cell in order:
                        writer.writerow([cell]+[float(ids.index(cell)+i)+(0.01 if perturb else 0) for i in range(30)])
            write("primary", ids)
            write("robustness", ids[::-1])
            (root/CELL_MANIFEST).write_text("cell_id\nc\nb\na\n", encoding="utf-8")
            loaded = load_pair(root, ids)
            self.assertEqual(loaded["robustness"]["ids"], tuple(ids[::-1]))
            write("robustness", ids[::-1], perturb=True)
            with self.assertRaises(ValueError):
                load_pair(root, ids)
            write("robustness", ids[::-1])
            (root/CELL_MANIFEST).write_text("cell_id\na\nb\nc\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_pair(root, ids)
