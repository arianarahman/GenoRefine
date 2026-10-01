# Purpose: Validate preparation behavior and invariants for the pre step4 workflow.
# Author: Ariana Rahman (Arizona State University)

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.metrics import silhouette_samples

from revision_pipeline.pre_step4.common import config_for, selected_views, clocks, elapsed, numerical_recovery
from revision_pipeline.pre_step4.rare import full_population_rare
from revision_pipeline.pre_step4.worker import build_controls
from revision_pipeline.runs import write_json


ROOT = Path(__file__).resolve().parents[3]


class ProfileTests(unittest.TestCase):
    def test_named_factor_bridge_is_explicit(self):
        p,d,b,db,n,t,r = [config_for(ROOT,k) for k in ("P","D","B","DB","N","T","R")]
        self.assertEqual(p.n_neighbors,15)
        self.assertEqual(b.n_neighbors,16)
        self.assertEqual(d.dimensions,30)
        self.assertEqual(db.n_neighbors,16)
        self.assertEqual(n.n_neighbors,15)
        self.assertEqual(t.neighbor_threads,24)
        self.assertEqual(r.row_order,"historical_source")
        self.assertEqual(p.purpose,"post_pilot_diagnostic")
        self.assertTrue(all(c.precision=="preserve" for c in (p,d,b,db,n,t,r)))

    def test_primary_cannot_inherit_diagnostic_relaxation(self):
        with self.assertRaises(ValueError):
            replace(config_for(ROOT,"R"),purpose="primary_evaluation",protocol_id="pre3c_exact_v1")
        with self.assertRaises(ValueError):
            replace(config_for(ROOT,"P"),precision="float32")
        with self.assertRaises(ValueError):
            replace(config_for(ROOT,"P"),neighbor_threads=24)

    def test_selection_views_share_grid_and_never_pick_best_ARI(self):
        cfg = replace(config_for(ROOT,"P"),resolutions=(.4,.5,.6),leiden_seeds=(0,))
        rows = [dict(resolution=res,leiden_seed=0,n_clusters=k,ARI=ari,RI=.8) for res,k,ari in ((.4,14,.2),(.5,17,.9),(.6,14,.99))]
        view = selected_views(rows,cfg,14)
        self.assertEqual(view["fixed_resolution"]["mean_ARI"],.9)
        self.assertEqual(view["matched_reference_count"]["mean_ARI"],.2)
        self.assertTrue(view["matched_reference_count"]["rows"][0]["selection_label_informed"])
        self.assertFalse(view["fixed_resolution"]["rows"][0]["selection_label_informed"])
        self.assertNotIn("selection_label_informed",rows[0])

    def test_clocks_use_three_explicit_definitions(self):
        a,b=clocks(),clocks()
        d=elapsed(a,b)
        self.assertIn("boottime_seconds",d)
        self.assertGreaterEqual(d["monotonic_seconds"],0)

    def test_recovery_checks_graph_bytes_not_only_ARI(self):
        from scipy.sparse import csr_matrix,save_npz
        with tempfile.TemporaryDirectory() as directory:
            old,new=Path(directory)/"old",Path(directory)/"new"
            a,b=old/"embedding_0",new/"evaluation"
            a.mkdir(parents=True); b.mkdir(parents=True)
            row=dict(resolution=.5,leiden_seed=0,n_clusters=2,ARI=.5,RI=.8,n_cells=3,dimensions_used=2,distance="euclidean")
            for target in (a,b):
                write_json(target/"grid.json",[row])
                np.save(target/"partitions.npy",np.array([[0,0,1]],dtype=np.int32))
                for name in ("distances","connectivities"):
                    save_npz(target/(name+".npz"),csr_matrix(np.eye(3)))
            self.assertTrue(numerical_recovery(old,new)["passed"])
            save_npz(b/"distances.npz",csr_matrix(np.eye(3)*2))
            with self.assertRaises(ValueError):
                numerical_recovery(old,new)


class DimensionTests(unittest.TestCase):
    def test_controls_preserve_input_and_are_repeatable(self):
        x=np.random.default_rng(3).normal(size=(64,40)); original=x.copy()
        a,b,p=build_controls(x); c,d,q=build_controls(x)
        np.testing.assert_array_equal(x,original)
        np.testing.assert_array_equal(a,x[:,:32])
        np.testing.assert_array_equal(a,c); np.testing.assert_array_equal(b,d)
        np.testing.assert_allclose(b,p.transform(x),rtol=1e-12,atol=1e-12)
        self.assertFalse(p.whiten)
        self.assertEqual(p.svd_solver,"full")

    def test_controls_reject_too_few_dimensions(self):
        with self.assertRaises(ValueError):
            build_controls(np.ones((40,20)))


class OrchestratorTests(unittest.TestCase):
    def test_training_bridge_accepts_only_pinned_added_diagnostic(self):
        from revision_pipeline.pre_step4.run import ADDED_DIAGNOSTIC, training_source_compatibility
        before = {"revision_pipeline/refine/trainer.py": {"sha256": "unchanged", "size_bytes": 1}}
        result = training_source_compatibility(before, dict(before, **ADDED_DIAGNOSTIC))
        self.assertEqual(result["training_sources_preserved"], 1)
        self.assertEqual(result["permitted_added_diagnostics"], ADDED_DIAGNOSTIC)
        self.assertEqual(training_source_compatibility(before, before)["permitted_added_diagnostics"], {})

    def test_training_bridge_rejects_changed_deleted_or_unrecognized_source(self):
        from revision_pipeline.pre_step4.run import ADDED_DIAGNOSTIC, training_source_compatibility
        before = {"revision_pipeline/refine/trainer.py": {"sha256": "unchanged", "size_bytes": 1}}
        cases = [{}, {"revision_pipeline/refine/trainer.py": {"sha256": "changed", "size_bytes": 1}},
                 dict(before, **{"revision_pipeline/pilot/unrecognized.py": {"sha256": "added", "size_bytes": 1}}),
                 dict(before, **{next(iter(ADDED_DIAGNOSTIC)): {"sha256": "wrong", "size_bytes": 14619}})]
        for after in cases:
            with self.subTest(after=after), self.assertRaises(ValueError):
                training_source_compatibility(before, after)

    def test_existing_diagnostic_cannot_be_modified(self):
        from revision_pipeline.pre_step4.run import ADDED_DIAGNOSTIC, training_source_compatibility
        before = dict(ADDED_DIAGNOSTIC)
        with self.assertRaises(ValueError):
            training_source_compatibility(before, {next(iter(before)): {"sha256": "wrong", "size_bytes": 14619}})

    def resume_bridge_fixture(self):
        from revision_pipeline.integrity import canonical_hash
        from revision_pipeline.pre_step4.run import RESUME_BEFORE
        before = dict(RESUME_BEFORE)
        before["revision_pipeline/evaluate/graph.py"] = {"sha256": "numerical-code", "size_bytes": 1}
        after = dict(before)
        for k in RESUME_BEFORE:
            after[k] = {"sha256": "amended-"+k, "size_bytes": 2}
        bridge = {"protocol": "pre_step4_resume_verifier_amendment_20260917",
                  "before_source_tree_sha256": canonical_hash(before), "after_source_tree_sha256": canonical_hash(after),
                  "changes": {k:{"before":before[k], "after":after[k]} for k in RESUME_BEFORE}}
        return before, after, bridge

    def test_resume_bridge_accepts_exact_verifier_and_test_amendment(self):
        from revision_pipeline.pre_step4.run import validate_resume_sources
        before, after, bridge = self.resume_bridge_fixture()
        self.assertFalse(validate_resume_sources(before, after, bridge)["identical"])
        self.assertTrue(validate_resume_sources(before, before)["identical"])

    def test_resume_bridge_rejects_any_scientific_change_even_with_updated_hashes(self):
        from revision_pipeline.integrity import canonical_hash
        from revision_pipeline.pre_step4.run import validate_resume_sources
        before, after, bridge = self.resume_bridge_fixture()
        after["revision_pipeline/evaluate/graph.py"] = {"sha256": "changed", "size_bytes": 1}
        bridge["after_source_tree_sha256"] = canonical_hash(after)
        bridge["changes"]["revision_pipeline/evaluate/graph.py"] = {
            "before": before["revision_pipeline/evaluate/graph.py"], "after": after["revision_pipeline/evaluate/graph.py"]}
        with self.assertRaises(ValueError):
            validate_resume_sources(before, after, bridge)

    def test_resume_bridge_rejects_missing_or_tampered_pins(self):
        from copy import deepcopy
        from revision_pipeline.pre_step4.run import validate_resume_sources
        before, after, bridge = self.resume_bridge_fixture()
        cases = [None, {}, dict(bridge, after_source_tree_sha256="wrong"), dict(bridge, changes={})]
        bad = deepcopy(bridge)
        bad["changes"][next(iter(bad["changes"]))]["after"]["sha256"] = "wrong"
        cases.append(bad)
        for value in cases:
            with self.subTest(bridge=value), self.assertRaises(ValueError):
                validate_resume_sources(before, after, value)

    def test_resume_bridge_rejects_added_or_removed_files(self):
        from revision_pipeline.pre_step4.run import validate_resume_sources
        before, after, bridge = self.resume_bridge_fixture()
        with self.assertRaises(ValueError):
            validate_resume_sources(before, dict(after, added={}), bridge)
        after.pop("revision_pipeline/evaluate/graph.py")
        with self.assertRaises(ValueError):
            validate_resume_sources(before, after, bridge)

    def test_resource_gate_uses_WSL_memory(self):
        from revision_pipeline.pre_step4.run import resources
        result=resources()
        self.assertGreater(result["MemAvailable_bytes"],0)
        self.assertLessEqual(result["MemAvailable_bytes"],result["MemTotal_bytes"])

    def test_summary_keeps_dimension_control_provenance_distinct(self):
        from revision_pipeline.pre_step4.run import summarize
        from revision_pipeline.pre_step4.common import PROFILES
        cfg=config_for(ROOT,"P")
        rows=[dict(resolution=r,leiden_seed=s,n_clusters=14,ARI=.5,RI=.8) for s in cfg.leiden_seeds for r in cfg.resolutions]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); evaluations={}
            names=["baseline"]+["joint_"+str(i) for i in range(5)]
            for profile in PROFILES:
                for name in names+(["first32","pca32"] if profile=="P" else []):
                    path=root/(profile+"_"+name); (path/"evaluation").mkdir(parents=True)
                    write_json(path/"readouts.json",{"selection_views":selected_views(rows,cfg,14),"geometry":[]})
                    write_json(path/"evaluation/grid.json",rows)
                    evaluations[profile+"_"+name]=path
            write_json(root/"transform.json",{})
            result=summarize(ROOT,evaluations,root)
            self.assertEqual(len(result["profile_readouts"]),70)
            self.assertTrue(all(r["dimension_by_backend_interaction"]==0 for r in result["dimension_backend_interactions_and_path"]))
            pairs=result["primary_grid_and_matched_granularity"]
            self.assertEqual(pairs["baseline_vs_joint_0"]["pairing_status"],"exact_parent_reference_verified")
            self.assertIn("not_refiner_training_parent",pairs["pca32_vs_joint_0"]["pairing_status"])


class RareTests(unittest.TestCase):
    def setUp(self):
        self.x=np.array([[0.,0.],[0.,.1],[.1,0.],[3.,3.],[3.,3.1],[3.1,3.],[8.,8.],[8.1,8.]])
        self.y=np.array([0,0,0,1,1,1,2,2]); self.b=np.array([0,1,0,0,1,0,0,1])
        self.ids=["c"+str(i) for i in range(8)]

    def run_rare(self,**kwargs):
        return full_population_rare(self.x,self.y,self.b,self.ids,k=3,**kwargs)

    def test_full_population_ASW_agrees_with_direct_standard_score(self):
        result=self.run_rare(fraction=.4)
        actual=[r["ASW"] for r in result["cells"]]
        np.testing.assert_allclose(actual,silhouette_samples(self.x,self.y),rtol=1e-10,atol=1e-10)
        self.assertEqual(result["query_cells"],8)

    def test_isolation_uses_full_batches_not_rare_subset_or_sample(self):
        result=self.run_rare(fraction=.26)
        self.assertEqual(result["query_cells"],2)
        self.assertEqual(result["isolated_groups_full_population"],[])
        self.assertEqual(result["isolated_ASW"]["status"],"not_applicable")
        self.assertIsNone(result["isolated_ASW"]["value"])
        self.assertEqual(result["groups"][0]["full_batches"],2)

    def test_rare_purity_ceiling_and_recall(self):
        result=self.run_rare(fraction=.26)["groups"][0]
        self.assertEqual(result["mean_neighbor_purity"],1/3)
        self.assertEqual(result["purity_ceiling"],1/3)
        self.assertEqual(result["mean_same_class_recall_at_k"],1)

    def test_singleton_has_no_fabricated_silhouette(self):
        y=self.y.copy(); y[-1]=3
        result=full_population_rare(self.x,y,self.b,self.ids,k=3,fraction=.2)
        self.assertTrue(all(r["ASW"] is None for r in result["cells"]))
        self.assertTrue(all(r["same_class_recall_at_k"] is None for r in result["cells"]))

    def test_duplicate_coordinates_and_permutation_keep_identity_safe_neighbors(self):
        x=np.zeros_like(self.x); p=np.array([7,3,0,6,1,4,2,5])
        first=full_population_rare(x,self.y,self.b,self.ids,k=3,fraction=.4)
        second=full_population_rare(x[p],self.y[p],self.b[p],[self.ids[i] for i in p],k=3,fraction=.4)
        original={r["cell_index"]:r["same_class_neighbors"] for r in first["cells"]}
        self.assertEqual(original,{int(p[r["cell_index"]]):r["same_class_neighbors"] for r in second["cells"]})
        self.assertEqual(original[0],2) # Never counts self even when all distances tie.

    def test_rare_query_parameters_fail_closed(self):
        with self.assertRaises(ValueError): self.run_rare(fraction=1)
        with self.assertRaises(ValueError): self.run_rare(fraction=0)


if __name__=="__main__":
    unittest.main()
