from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from revision_pipeline.data.store import EmbeddingView
from revision_pipeline.integrity import canonical_hash, file_fingerprint
from revision_pipeline.pilot.common import read
from revision_pipeline.runs import RunDirectory, write_json
from revision_pipeline.step4.common import ROOT, anchor_partitions, bind_baseline_k, panel_spec, schedule_record
from revision_pipeline.step4.run import worker_count, verify_acceptance
from revision_pipeline.step4_policy import planned_training_config


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = read(ROOT/"revision_pipeline/configs/evaluation_primary_v1.json")
        self.ids = tuple("cell"+str(i) for i in range(8))
        self.embedding = EmbeddingView(np.arange(24, dtype=float).reshape(8,3), self.ids,
            {"id": "Scanorama", "dataset_fingerprint": "dataset", "stored_values_file_sha256": "values"})
        self.sources, self.store = {"source": "immutable"}, {"sha256": "store", "size_bytes": 1}

    def baseline(self, *, parent=None, cfg=None):
        settings = {"dataset": "hpcb", "embeddings": ["Scanorama"], "refined_bundle": None,
            "parent_order": "canonical", "store_manifest": self.store, "evaluation": cfg or self.config}
        with RunDirectory(self.root, kind="step3b_evaluation", config=settings) as run:
            run.manifest["source_tree_sha256"] = canonical_hash(self.sources)
            run.write_json("source_manifest.json", self.sources)
            run.write_json("embedding_0/input.json", {"reference": parent or self.embedding.parent_reference()})
            run.write_json("embedding_0/cell_ids.json", list(self.ids))
            parts = np.zeros((45,8), dtype=np.int32)
            for i, count in enumerate((2,3,4)):
                parts[i*15+3] = np.arange(8) % count
            np.save(run.artifact_path("embedding_0/partitions.npy"), parts, allow_pickle=False)
            # Dummy graph bytes are sufficient for hash binding tests, not scientific graph validation.
            np.savez(run.artifact_path("embedding_0/connectivities.npz"), data=np.ones(1))
            run.write_json("embedding_0/graph.json", {"test_fixture": True})
        return run.final_path

    def test_anchor_uses_frozen_positions_only(self):
        p = np.arange(45*8, dtype=np.int32).reshape(45,8)
        found = anchor_partitions(p, self.config, 8)
        for seed, position in ((0,3),(1,18),(2,33)):
            np.testing.assert_array_equal(found[seed], p[position])

    def test_incomplete_partition_array_rejected(self):
        for p in (np.zeros((44,8), dtype=int), np.zeros((45,8))):
            with self.assertRaises(ValueError):
                anchor_partitions(p, self.config, 8)

    def test_baseline_k_binding_never_needs_score_or_annotation_files(self):
        result = bind_baseline_k(self.baseline(), self.embedding, self.sources, self.store)
        self.assertEqual(result["n_clusters"], 3)
        self.assertFalse(result["baseline_partition_binding_required"])
        self.assertFalse(result["reference_labels_used"])
        self.assertIn("connectivities.npz", result["evidence"])

    def test_wrong_parent_rejected(self):
        with self.assertRaises(ValueError):
            bind_baseline_k(self.baseline(parent={"wrong": True}), self.embedding, self.sources, self.store)

    def test_wrong_sources_or_store_rejected(self):
        path = self.baseline()
        for source, store in (({}, self.store), (self.sources, {})):
            with self.assertRaises(ValueError):
                bind_baseline_k(path, self.embedding, source, store)

    def test_tampered_artifact_rejected(self):
        path = self.baseline()
        write_json(path/"embedding_0/graph.json", {"tampered": True})
        with self.assertRaises(ValueError):
            bind_baseline_k(path, self.embedding, self.sources, self.store)

    def test_changed_evaluation_rejected(self):
        with self.assertRaises(ValueError):
            bind_baseline_k(self.baseline(cfg=dict(self.config, fixed_resolution=.6)), self.embedding, self.sources, self.store)

    def test_paired_schedule_partial_batches_and_seed_binding(self):
        cfg = replace(planned_training_config(9, 2, 0), batch_size=4, max_updates=6)
        record, visits = schedule_record(9,cfg)
        self.assertEqual(record["batch_sizes"], [4,4,1,4,4,1])
        np.testing.assert_array_equal(visits, np.full(9,2))
        other, _ = schedule_record(9, replace(cfg, replicate_seed=None, cluster_seed=99))
        self.assertNotEqual(record["ordered_batch_indices_sha256"], other["ordered_batch_indices_sha256"])

    def test_scope_does_not_expand_to_other_data_or_claims(self):
        spec = panel_spec()
        self.assertEqual(spec["dataset"], "hpcb")
        self.assertIn("other_datasets", spec["no_automatic_expansion"])
        self.assertIn("scoring", spec["stop_boundary"])

    def test_resource_gate(self):
        spec = panel_spec()
        self.assertEqual(worker_count(spec,[2*1024**3]*2,24*1024**3),4)
        self.assertEqual(worker_count(spec,[2*1024**3]*2,8*1024**3),1)
        with self.assertRaises(ValueError):
            worker_count(spec,[5*1024**3]*2,24*1024**3)

    def test_acceptance_requires_current_sources_and_both_new_suites(self):
        with RunDirectory(self.root,kind="step3b_software_acceptance",config={}) as run:
            run.write_json("source_manifest.json", self.sources)
            run.write_json("checks.json", {"locked_environments_unchanged": True,
                "suites": [{"suite": s, "passed": True} for s in ("step4_policy","step4_contracts","step4_training")]})
        verify_acceptance(run.final_path, self.sources)
        with self.assertRaises(ValueError):
            verify_acceptance(run.final_path,{})


if __name__ == "__main__":
    unittest.main()
