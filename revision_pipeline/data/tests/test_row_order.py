"""Historical-order adapters and scheduler-coverage safeguards; no training."""

import unittest
from unittest.mock import patch

import numpy as np

from revision_pipeline.data.store import Store, import_legacy, read_json
from revision_pipeline.integrity import file_fingerprint, iter_batches
from revision_pipeline.runs import write_json
from revision_pipeline.data.tests import test_data_store as fixtures


class HistoricalOrderTests(unittest.TestCase):
    def make_store(self, *, blocked=False, refined=False, graph=False, identity=False):
        fixture = fixtures.StoreTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.make_config(blocked=blocked, refined=refined, graph=graph)
        # A three-cycle, NOT a self-inverse permutation: detects wrong direction.
        rows = ([["001", 10, 20], ["1", 50, 60], ["cell-c", 30, 40]] if identity else
                [["cell-c", 30, 40], ["001", 10, 20], ["1", 50, 60]])
        fixture.csv_file(rows)
        lock_path = fixture.root / "revision_pipeline/configs/lock.json"
        lock = read_json(lock_path)
        lock["files"]["embedding.csv"] = file_fingerprint(fixture.root / "embedding.csv")
        write_json(lock_path, lock)
        return Store(import_legacy(fixture.root))

    def test_inverse_permutation_values_ids_annotations_and_reference(self):
        store = self.make_store()
        original_manifest = file_fingerprint(store.path / "run.json")
        canonical = store.embedding("example", "baseline")
        view = store.historical_input("example", "baseline")
        self.assertEqual(view.embedding.cell_ids, ("cell-c", "001", "1"))
        np.testing.assert_array_equal(view.embedding.values, [[30, 40], [10, 20], [50, 60]])
        self.assertEqual(view.dataset.cell_ids, view.embedding.cell_ids)
        self.assertEqual(view.dataset.batch_labels(), ("x", "x", "y"))
        self.assertEqual(view.dataset.named_labels(), ("A", "A", "B"))
        np.testing.assert_array_equal(view.dataset.reference_partition()[0], [0, 0, 1])
        self.assertNotEqual(view.parent_reference()["cell_order_sha256"], canonical.parent_reference()["cell_order_sha256"])
        self.assertEqual(view.parent_reference()["row_order_policy"], "historical_source_order_diagnostic")
        self.assertEqual(view.parent_reference()["stored_values_file_sha256"], canonical.parent_reference()["stored_values_file_sha256"])
        self.assertEqual(file_fingerprint(store.path / "run.json"), original_manifest)
        self.assertEqual(store.embedding("example", "baseline", view.embedding.cell_ids).cell_ids, tuple(fixtures.IDS))
        with self.assertRaises(ValueError):
            view.embedding.values[0, 0] = 99
        with self.assertRaises(ValueError):
            view.dataset.indices[0] = 0

    def test_canonical_output_round_trip_and_explicit_output_ids(self):
        store = self.make_store()
        view = store.historical_input("example", "baseline")
        output = view.embedding.values.astype(np.float32)
        for order in (np.arange(3), np.array([2, 0, 1])):
            values, ids = view.canonicalize_output(output[order], cell_ids=[view.embedding.cell_ids[i] for i in order])
            np.testing.assert_array_equal(values, store.embedding("example", "baseline").values)
            self.assertEqual(ids, tuple(fixtures.IDS))
            self.assertEqual(values.dtype, np.float32)
            self.assertFalse(values.flags.writeable)

    def test_bad_output_ids_shapes_and_values_fail(self):
        view = self.make_store().historical_input("example", "baseline")
        for ids in (("001", "001", "1"), ("001", "1"), ("001", "1", "wrong")):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                view.canonicalize_output(np.ones((len(ids), 2)), cell_ids=ids)
        for values in (np.ones((2, 2)), np.ones(3), np.ones((3, 0)), np.ones((3, 2), dtype=int),
                       np.full((3, 2), np.nan), np.full((3, 2), np.inf)):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                view.canonicalize_output(values, cell_ids=view.embedding.cell_ids)

    def test_annotation_gates_and_unverified_pairing_survive(self):
        view = self.make_store(blocked=True, refined=True).historical_input("example", "refined")
        with self.assertRaisesRegex(ValueError, "blocked"):
            view.dataset.named_labels()
        self.assertEqual(view.embedding.metadata["pairing_status"], "historical_input_pairing_unverified")
        self.assertIsNone(view.embedding.metadata["verified_parent_embedding"])

    def test_identity_order_remains_explicit_diagnostic(self):
        store = self.make_store(identity=True)
        view = store.historical_input("example", "baseline")
        self.assertEqual(view.embedding.cell_ids, tuple(fixtures.IDS))
        self.assertEqual(view.embedding.parent_reference(), store.embedding("example", "baseline").parent_reference())
        self.assertEqual(view.parent_reference()["row_order_policy"], "historical_source_order_diagnostic")

    def test_graph_proxy_and_new_baseline_rejected(self):
        store = self.make_store(graph=True)
        with self.assertRaisesRegex(ValueError, "graph is absent"):
            store.historical_input("example", "baseline")
        other = self.make_store()
        item = other.embedding("example", "baseline")
        item.metadata["kind"] = "recomputed_baseline"
        with patch.object(other, "embedding", return_value=item), self.assertRaisesRegex(ValueError, "imported historical"):
            other.historical_input("example", "baseline")

    def test_permutation_tampering_rejected(self):
        store = self.make_store()
        path = store.path / "embeddings/example/baseline.source_rows.npy"
        with path.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "integrity"):
            store.historical_input("example", "baseline")

    def test_malformed_and_wrong_direction_permutations_rejected_even_if_rehashed(self):
        store = self.make_store()
        relative = "embeddings/example/baseline.source_rows.npy"
        for bad in (np.array([0, 0, 2]), np.array([0, 1]), np.array([0., 1., 2.]),
                    np.array([0, 1, 3]), np.array([2, 0, 1])):
            # Fixture-only adversarial manifest edit: exercise semantic checks too.
            np.save(store.path / relative, bad, allow_pickle=False)
            store.manifest["artifacts"][relative] = file_fingerprint(store.path / relative)
            with self.subTest(permutation=bad), self.assertRaises(ValueError):
                store.historical_input("example", "baseline")

    def test_historical_api_does_not_accept_subset_membership(self):
        store = self.make_store()
        with self.assertRaises(TypeError):
            store.historical_input("example", "baseline", cell_ids=["001"])


class CoverageTests(unittest.TestCase):
    def test_hpcb_partial_batch_accounting_without_early_stop(self):
        batches = list(iter_batches(16382, 64, 300, shuffle=False, seed=3))
        visits = np.bincount(np.concatenate(batches), minlength=16382)
        self.assertEqual(sum(map(len, batches)), 19198)
        self.assertEqual([len(x) for x in batches if len(x) != 64], [62])
        np.testing.assert_array_equal(np.flatnonzero(visits == 2), np.arange(2816))
        self.assertTrue(np.all(visits[2816:] == 1))

    def test_mouse_prefix_bias_and_seeded_mixed_coverage(self):
        batches = np.array(["droplet"] * 25000 + ["facs"] * 25000)
        plain = np.concatenate(list(iter_batches(50000, 64, 300, shuffle=False, seed=3)))
        shuffled = np.concatenate(list(iter_batches(50000, 64, 300, shuffle=True, seed=3)))
        duplicate = np.concatenate(list(iter_batches(50000, 64, 300, shuffle=True, seed=3)))
        self.assertEqual(set(batches[plain]), {"droplet"})
        self.assertEqual(set(batches[shuffled]), {"droplet", "facs"})
        np.testing.assert_array_equal(shuffled, duplicate)
        self.assertEqual(len(np.unique(shuffled)), 19200)
        self.assertEqual(len(batches) - len(np.unique(shuffled)), 30800)

    def test_seeded_shuffling_changes_each_pass_not_only_initial_order(self):
        rows = list(iter_batches(10, 4, 6, shuffle=True, seed=7))
        first, second = np.concatenate(rows[:3]), np.concatenate(rows[3:])
        np.testing.assert_array_equal(np.sort(first), np.arange(10))
        np.testing.assert_array_equal(np.sort(second), np.arange(10))
        self.assertFalse(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
