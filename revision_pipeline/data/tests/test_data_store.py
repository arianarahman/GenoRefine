# Purpose: Validate data store behavior and invariants for the data workflow.
# Author: Ariana Rahman (Arizona State University)

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from scipy import sparse
from scipy.io import savemat

from revision_pipeline.integrity import canonical_hash, file_fingerprint
from revision_pipeline.runs import write_json
from revision_pipeline.data.embeddings import cache_key, canonical_selection, read_embedding_csv
from revision_pipeline.data.readers import categorical, h5_blocks, load_dataset, read_frame, scan_expression
from revision_pipeline.data.store import DatasetView, Store, check_sources, import_legacy, load_config


IDS = ["001", "1", "cell-c"]
POLICY = {"named_labels_allowed": True, "partition_description": "Supplied annotation partition",
          "restriction_reason": "Independent annotation provenance pending"}


def h5_fixture(path, *, legacy=False, csr=False):
    values = np.array([[1, 0], [-2, 3], [0, 0]], dtype=np.float32)
    strings = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as stream:
        if csr:
            x = sparse.csr_matrix(values)
            group = stream.create_group("X")
            group.attrs["encoding-type"] = "csr_matrix"
            group.attrs["shape"] = x.shape
            for name in ("data", "indices", "indptr"):
                group.create_dataset(name, data=getattr(x, name))
        else:
            stream.create_dataset("X", data=values)
        for name, ids in (("obs", IDS), ("var", ["gene-a", "gene-b"])):
            frame = stream.create_group(name)
            frame.attrs["encoding-type"] = "dataframe"
            frame.attrs["_index"] = "_index"
            frame.attrs["column-order"] = np.array(["label", "batch", "optional"] if name == "obs" else [], dtype=strings)
            frame.create_dataset("_index", data=np.array(ids, dtype=strings))
        obs = stream["obs"]
        obs.create_dataset("optional", data=[1.0, np.nan, 2.0])
        for name, codes, categories in (("label", [0, 1, 0], ["A", "B"]), ("batch", [0, 1, 0], ["x", "y"])):
            if legacy:
                cats = obs.require_group("__categories").create_dataset(name, data=np.array(categories, dtype=strings))
                obj = obs.create_dataset(name, data=np.array(codes, dtype=np.int8))
                obj.attrs["categories"] = cats.ref
            else:
                group = obs.create_group(name)
                group.attrs["encoding-type"] = "categorical"
                group.create_dataset("codes", data=np.array(codes, dtype=np.int8))
                group.create_dataset("categories", data=np.array(categories, dtype=strings))
    return values


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "input.h5ad"
        self.values = h5_fixture(self.path)
        self.spec = {"format": "h5ad", "path": "input.h5ad", "label_key": "label", "batch_key": "batch",
                     "annotation_policy": copy.deepcopy(POLICY)}
        self.registered = {"id": "example", "display_name": "Example", "cohort_group": "example",
                           "provenance_status": "pending", "observed_cells": 3, "batch_evaluation": True,
                           "spatial_evaluation": False, "inputs": [{"path": "input.h5ad", "role": "source"}], "notes": []}

    def csv_file(self, rows, header=None):
        path = self.root / "embedding.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header or ["", "x", "y"])
            writer.writerows(rows)
        return path


class ReaderTests(Fixture):
    def test_dense_values_order_dtype_and_missing_optional_metadata(self):
        record = load_dataset(self.root, self.registered, self.spec)
        self.assertEqual(record["cell_ids"], IDS)
        self.assertEqual(record["obs"]["label"], ["A", "B", "A"])
        self.assertEqual(record["obs"]["optional"], [1.0, None, 2.0])
        stats = record["report"]["expression"]
        self.assertEqual(stats["dtypes"], ["float32"])
        self.assertEqual(stats["zero_rows"], 1)
        self.assertEqual(stats["negative_values"], 1)

    def test_legacy_categories_match_modern(self):
        expected = load_dataset(self.root, self.registered, self.spec)["obs"]
        h5_fixture(self.path, legacy=True)
        self.assertEqual(load_dataset(self.root, self.registered, self.spec)["obs"], expected)

    def test_sparse_blocks_match_dense(self):
        h5_fixture(self.path, csr=True)
        blocks = list(h5_blocks(self.path, 2))
        np.testing.assert_array_equal(sparse.vstack([x for _, x in blocks]).toarray(), self.values)
        self.assertEqual([i for i, _ in blocks], [0, 2])
        self.assertEqual(scan_expression(iter(blocks), (3, 2))["zero_rows"], 1)

    def test_sparse_explicit_zero_and_empty_rows(self):
        block = sparse.csr_matrix((np.array([0., 3.]), np.array([0, 1]), np.array([0, 1, 2, 2])), shape=(3, 2))
        self.assertEqual(scan_expression(iter([(0, block)]), (3, 2))["zero_rows"], 2)

    def test_nonfinite_expression_rejected(self):
        with h5py.File(self.path, "r+") as stream:
            stream["X"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            load_dataset(self.root, self.registered, self.spec)

    def test_bad_sparse_pointer_rejected(self):
        h5_fixture(self.path, csr=True)
        with h5py.File(self.path, "r+") as stream:
            stream["X/indptr"][-1] = 0
        with self.assertRaisesRegex(ValueError, "CSR"):
            list(h5_blocks(self.path))

    def test_bad_sparse_column_rejected(self):
        h5_fixture(self.path, csr=True)
        with h5py.File(self.path, "r+") as stream:
            stream["X/indices"][0] = 4
        with self.assertRaisesRegex(ValueError, "bounds"):
            list(h5_blocks(self.path))

    def test_unsorted_sparse_indices_are_preserved(self):
        h5_fixture(self.path, csr=True)
        with h5py.File(self.path, "r+") as stream:
            stream["X/indices"][1:3] = [1, 0]
            stream["X/data"][1:3] = [3, -2]
        blocks = list(h5_blocks(self.path))
        np.testing.assert_array_equal(blocks[0][1].toarray(), self.values)
        np.testing.assert_array_equal(blocks[0][1].indices, [0, 1, 0])
        self.assertEqual(scan_expression(iter(blocks), (3, 2))["blocks_with_unsorted_sparse_indices"], 1)

    def test_duplicate_sparse_coordinates_rejected_without_summing(self):
        h5_fixture(self.path, csr=True)
        with h5py.File(self.path, "r+") as stream:
            stream["X/indices"][1:3] = [0, 0]
        with self.assertRaisesRegex(ValueError, "Duplicate CSR"):
            list(h5_blocks(self.path))

    def test_duplicate_obs_id_rejected(self):
        with h5py.File(self.path, "r+") as stream:
            stream["obs/_index"][1] = IDS[0]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_dataset(self.root, self.registered, self.spec)

    def test_duplicate_feature_id_rejected_not_silently_renamed(self):
        with h5py.File(self.path, "r+") as stream:
            stream["var/_index"][1] = "gene-a"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_dataset(self.root, self.registered, self.spec)

    def test_missing_required_label_rejected(self):
        with h5py.File(self.path, "r+") as stream:
            stream["obs/label/codes"][0] = -1
        with self.assertRaisesRegex(ValueError, "required annotation"):
            load_dataset(self.root, self.registered, self.spec)

    def test_bad_categorical_codes_and_missing_codes(self):
        self.assertEqual(categorical([-1, 0], [b"A"]), [None, "A"])
        for codes in ([2], [-2], [0.2]):
            with self.subTest(codes=codes), self.assertRaises(ValueError):
                categorical(codes, ["A"])

    def test_wrong_cell_count_rejected(self):
        self.registered["observed_cells"] = 4
        with self.assertRaisesRegex(ValueError, "cell count"):
            load_dataset(self.root, self.registered, self.spec)

    def test_matrix_annotation_shape_mismatch(self):
        with h5py.File(self.path, "r+") as stream:
            del stream["X"]
            stream.create_dataset("X", data=np.ones((2, 3)))
        with self.assertRaisesRegex(ValueError, "shape"):
            load_dataset(self.root, self.registered, self.spec)

    def test_unsupported_annotation_encoding_rejected(self):
        with h5py.File(self.path, "r+") as stream:
            stream["obs/label"].attrs["encoding-type"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            load_dataset(self.root, self.registered, self.spec)

    def test_single_batch_policy_mismatch(self):
        self.registered["batch_evaluation"] = False
        with self.assertRaisesRegex(ValueError, "Single-batch"):
            load_dataset(self.root, self.registered, self.spec)

    def test_mat_order_and_unknown_code_preserved(self):
        savemat(self.root / "part_a.mat", {"a": np.array([[1., 2.], [3., 4.]])})
        savemat(self.root / "part_b.mat", {"b": np.array([[5., 6.]])})
        savemat(self.root / "labels.mat", {"labels": np.array([[1], [15], [1]])})
        spec = {"format": "mat_parts", "parts": [{"path": "part_a.mat", "key": "a", "batch": "A"},
                {"path": "part_b.mat", "key": "b", "batch": "B"}], "label_path": "labels.mat",
                "label_variable": "labels", "label_key": "codes", "batch_key": "batch", "annotation_policy": POLICY}
        record = load_dataset(self.root, self.registered, spec)
        self.assertEqual(record["cell_ids"], ["Cell-1-Batch-A", "Cell-2-Batch-A", "Cell-1-Batch-B"])
        self.assertEqual(record["obs"]["codes"], ["1", "15", "1"])
        self.assertEqual(record["report"]["legacy_named_label_map_missing_codes"], ["15"])
        savemat(self.root / "labels.mat", {"labels": np.array([[1], [2]])})
        with self.assertRaisesRegex(ValueError, "concatenation"):
            load_dataset(self.root, self.registered, spec)


class EmbeddingTests(Fixture):
    def test_exact_alignment_preserves_leading_zero_ids_and_precision(self):
        path = self.csv_file([["cell-c", 3, 4], ["001", "1.1234567890123457", 2], ["1", 5, 6]])
        values, permutation, report = read_embedding_csv(path, IDS, dimensions=2)
        np.testing.assert_array_equal(permutation, [1, 2, 0])
        self.assertEqual(values[0, 0], float("1.1234567890123457"))
        self.assertEqual(values.dtype, np.float64)
        self.assertEqual(report["rows_reordered"], 3)

    def test_missing_extra_duplicate_and_suffix_modified_ids_rejected(self):
        for ids in (["001", "1"], ["001", "1", "cell-c", "extra"], ["001", "1", "1"], ["001", "1", "cell-c-1"]):
            with self.subTest(ids=ids):
                path = self.csv_file([[cell, 1, 2] for cell in ids])
                with self.assertRaises(ValueError):
                    read_embedding_csv(path, IDS, dimensions=2)

    def test_nonfinite_missing_and_text_embedding_values_rejected(self):
        for invalid in ("nan", "inf", "-inf", "", "text"):
            path = self.csv_file([[IDS[0], invalid, 2], [IDS[1], 1, 2], [IDS[2], 1, 2]])
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                read_embedding_csv(path, IDS, dimensions=2)

    def test_wrong_dimensions_duplicate_headers_and_ragged_rows_rejected(self):
        cases = [(["", "x"], [["001", 1]]), (["", "x", "x"], []), (["", "x", "y"], [["001", 1]])]
        for header, rows in cases:
            with self.subTest(header=header, rows=rows), self.assertRaises(ValueError):
                read_embedding_csv(self.csv_file(rows, header), IDS, dimensions=2)

    def test_subset_and_full_membership_use_canonical_order(self):
        np.testing.assert_array_equal(canonical_selection(IDS, IDS[::-1]), [0, 1, 2])
        np.testing.assert_array_equal(canonical_selection(IDS, ["cell-c", "001"]), [0, 2])

    def test_bad_subset_membership_rejected(self):
        for values in ([], ["001", "001"], ["outside"]):
            with self.subTest(ids=values), self.assertRaises(ValueError):
                canonical_selection(IDS, values)

    def test_cache_key_changes_for_every_design_input(self):
        args = dict(dataset_fingerprint="a" * 64, cell_ids=IDS, preprocessing={"hvg": 2000}, method="Harmony", backbone_seed=0)
        base = cache_key(**args)
        for key, value in (("dataset_fingerprint", "b" * 64), ("cell_ids", IDS[::-1]),
                           ("preprocessing", {"hvg": 1000}), ("method", "Scanorama"), ("backbone_seed", 1)):
            self.assertNotEqual(base, cache_key(**dict(args, **{key: value})))


class StoreTests(Fixture):
    def make_config(self, *, blocked=False, graph=False, refined=False):
        self.spec["annotation_policy"]["named_labels_allowed"] = not blocked
        self.csv_file([["cell-c", 30, 40], ["1", 50, 60], ["001", 10, 20]])
        cfg_dir = self.root / "revision_pipeline/configs"
        cfg_dir.mkdir(parents=True)
        write_json(cfg_dir / "datasets.json", {"schema_version": 1, "datasets": [self.registered], "excluded_inputs": []})
        embeddings = [{"dataset": "example", "id": "baseline", "path": "embedding.csv", "dimensions": 2,
                       "kind": "graph_proxy" if graph else "baseline", "claimed_baseline": None, "source_set": "test"}]
        files = {p: file_fingerprint(self.root / p) for p in ("input.h5ad", "embedding.csv")}
        if refined:
            (self.root / "refined.csv").write_bytes((self.root / "embedding.csv").read_bytes())
            embeddings.append(dict(embeddings[0], id="refined", path="refined.csv", kind="historical_refined", claimed_baseline="baseline"))
            files["refined.csv"] = file_fingerprint(self.root / "refined.csv")
        write_json(cfg_dir / "lock.json", {"schema_version": 1, "files": files})
        config = {"schema_version": 1, "registry": "revision_pipeline/configs/datasets.json",
                  "source_lock": "revision_pipeline/configs/lock.json", "loaders": {"example": self.spec},
                  "embeddings": embeddings, "evidence_files": [], "excluded_embedding_roots": [], "not_imported": []}
        write_json(cfg_dir / "data_store.json", config)
        return config

    def test_import_reload_subset_binding_and_read_only(self):
        self.make_config()
        store = Store(import_legacy(self.root))
        embedding = store.embedding("example", "baseline")
        self.assertEqual(embedding.cell_ids, tuple(IDS))
        np.testing.assert_array_equal(embedding.values, [[10, 20], [50, 60], [30, 40]])
        with self.assertRaises(ValueError):
            embedding.values[0, 0] = 0
        subset = store.embedding("example", "baseline", ["cell-c", "001"])
        self.assertEqual(subset.cell_ids, ("001", "cell-c"))
        np.testing.assert_array_equal(subset.values, [[10, 20], [30, 40]])
        self.assertNotEqual(subset.parent_reference(), embedding.parent_reference())
        self.assertEqual(store.embedding("example", "baseline", IDS[::-1]).parent_reference(), embedding.parent_reference())
        self.assertTrue(store.verify(self.root)["passed"])
        blocks = list(store.expression_blocks(self.root, "example", 2))
        np.testing.assert_array_equal(np.vstack([x for _, x in blocks]), self.values)

    def test_label_name_gate_and_anonymous_partition(self):
        self.make_config(blocked=True)
        ds = Store(import_legacy(self.root)).dataset("example")
        with self.assertRaisesRegex(ValueError, "blocked"):
            ds.named_labels()
        codes, description = ds.reference_partition()
        np.testing.assert_array_equal(codes, [0, 1, 0])
        self.assertEqual(description, POLICY["partition_description"])

    def test_allowed_names_and_single_batch_gate(self):
        record = load_dataset(self.root, self.registered, self.spec)
        ds = DatasetView(record, np.arange(3))
        self.assertEqual(ds.named_labels(), ("A", "B", "A"))
        record["registry"]["batch_evaluation"] = False
        with self.assertRaisesRegex(ValueError, "inapplicable"):
            ds.batch_labels(for_mixing_metric=True)

    def test_graph_proxy_requires_opt_in(self):
        self.make_config(graph=True)
        store = Store(import_legacy(self.root))
        with self.assertRaisesRegex(ValueError, "graph is absent"):
            store.embedding("example", "baseline")
        self.assertEqual(store.embedding("example", "baseline", allow_graph_proxy=True).values.shape, (3, 2))

    def test_legacy_refinement_not_declared_paired(self):
        self.make_config(refined=True)
        meta = Store(import_legacy(self.root)).embedding("example", "refined").metadata
        self.assertIsNone(meta["verified_parent_embedding"])
        self.assertEqual(meta["pairing_status"], "historical_input_pairing_unverified")

    def test_source_drift_refused_and_failed_run_retained(self):
        self.make_config()
        with (self.root / "embedding.csv").open("a") as stream:
            stream.write("extra,1,2\n")
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            import_legacy(self.root)
        runs = list((self.root / "revision_pipeline/runs").iterdir())
        self.assertEqual(len(runs), 1)
        self.assertTrue(runs[0].name.startswith(".incomplete-"))
        with self.assertRaisesRegex(ValueError, "completed"):
            Store(runs[0])

    def test_cached_embedding_tampering_detected(self):
        self.make_config()
        store = Store(import_legacy(self.root))
        path = store.path / "embeddings/example/baseline.npy"
        with path.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "integrity"):
            store.embedding("example", "baseline")

    def test_source_change_after_import_does_not_change_cache_but_blocks_raw_reader(self):
        self.make_config()
        store = Store(import_legacy(self.root))
        with h5py.File(self.path, "r+") as stream:
            stream["X"][0, 0] = 8
        self.assertEqual(store.embedding("example", "baseline").values[0, 0], 10)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            list(store.expression_blocks(self.root, "example"))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            store.verify(self.root)

    def test_stale_directory_excluded(self):
        config = self.make_config()
        config["excluded_embedding_roots"] = [{"path": "stale", "reason": "wrong export"}]
        config["embeddings"][0]["path"] = "stale/X.csv"
        write_json(self.root / "revision_pipeline/configs/data_store.json", config)
        with self.assertRaisesRegex(ValueError, "excluded"):
            load_config(self.root, "revision_pipeline/configs/data_store.json")

    def test_registry_path_mismatch_and_missing_lock_rejected(self):
        config = self.make_config()
        config["loaders"]["example"]["path"] = "other.h5ad"
        write_json(self.root / "revision_pipeline/configs/data_store.json", config)
        with self.assertRaisesRegex(ValueError, "registry"):
            load_config(self.root, "revision_pipeline/configs/data_store.json")

    def test_store_manifest_tamper_detected(self):
        self.make_config()
        path = import_legacy(self.root)
        write_json(path / "store.json", {"schema_version": 1})
        with self.assertRaisesRegex(ValueError, "integrity"):
            Store(path)


if __name__ == "__main__":
    unittest.main()
