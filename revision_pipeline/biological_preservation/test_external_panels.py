# Purpose: Validate external panels behavior and invariants for the biological preservation
#          workflow.
# Author: Ariana Rahman (Arizona State University)

import hashlib
import gzip
from pathlib import Path
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd

from revision_pipeline.biological_preservation.external_panels import (
    MarkerRecord, build_marker_panel, parse_cell_ontology,
    parse_cellmarker_mouse, parse_panglaodb_human)
from revision_pipeline.biological_preservation.mouse_official_labels import (
    recover_official_mouse_labels)
from revision_pipeline.biological_preservation.panel_io import read_panel, write_panel
from revision_pipeline.biological_preservation.source_acquisition import (
    SourceLock, acquire_locked_source, verify_locked_source)


OBO = """format-version: 1.2

[Term]
id: CL:0000001
name: alpha cell

[Term]
id: CL:0000002
name: B cell

[Term]
id: CL:0000003
name: obsolete thing
is_obsolete: true
"""


class ExternalPanelTests(unittest.TestCase):
    def test_locked_acquisition_and_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "origin.tsv"
            source.write_text("species\tgene\nHs\tINS\n", encoding="utf-8")
            payload = source.read_bytes()
            lock = SourceLock("tiny", source.as_uri(), "locked.tsv", len(payload),
                              hashlib.sha256(payload).hexdigest(),
                              {"kind": "delimited", "delimiter": "\t",
                               "required_columns": ["species", "gene"]})
            out = root / "sources"
            first = acquire_locked_source(lock, out)
            second = acquire_locked_source(lock, out)
            self.assertEqual(first["fingerprint"], second["fingerprint"])
            self.assertTrue(first["verified"])
            (out / "locked.tsv").write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_locked_source(out / "locked.tsv", lock)

    def test_locked_acquisition_preserves_compressed_schema_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "origin.tsv.gz"
            with gzip.open(source, "wt", encoding="utf-8") as stream:
                stream.write("species\tgene\nHs\tINS\n")
            payload = source.read_bytes()
            lock = SourceLock("tiny_gzip", source.as_uri(), "locked.tsv.gz", len(payload),
                              hashlib.sha256(payload).hexdigest(),
                              {"kind": "delimited", "delimiter": "\t",
                               "required_columns": ["species", "gene"]})
            record = acquire_locked_source(lock, root / "sources")
            self.assertTrue(record["verified"])
            self.assertEqual(record["observed_schema"]["columns"], ["species", "gene"])

    def test_panglao_human_canonical_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "panglao.tsv"
            path.write_text(
                "species\tofficial gene symbol\tcell type\tcanonical marker\n"
                "Hs\tINS\talpha cell\t1\nHs\tGCG\talpha cell\t0\n"
                "Mm\tIns1\talpha cell\t1\nMm Hs\tIAPP\talpha cell\t1\n"
                "Human\tMAFA\talpha cell\tTrue\n",
                encoding="utf-8")
            records = parse_panglaodb_human(path)
            self.assertEqual([r.official_gene_symbol for r in records], ["IAPP", "INS", "MAFA"])
            self.assertTrue(all(r.canonical for r in records))

    def test_cellmarker_filters_and_ontology(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obo = root / "cl.obo"; obo.write_text(OBO, encoding="utf-8")
            ontology = parse_cell_ontology(obo)
            workbook = root / "cellmarker.xlsx"
            table = pd.DataFrame([
                {"Species": "Mouse", "Cell name": "B cell", "Symbol": "Cd79a",
                 "GeneID": "12518", "Cell ontology ID": "CL:0000002", "Year": "2015"},
                {"Species": "Mouse", "Cell name": "B cell", "Symbol": "Cd79a",
                 "GeneID": "12518", "Cell ontology ID": "CL:0000002", "Year": "2015"},
                {"Species": "Mouse", "Cell name": "B cell", "Symbol": "Ms4a1",
                 "GeneID": "12482", "Cell ontology ID": "CL:0000002", "Year": "2019"},
                {"Species": "Human", "Cell name": "B cell", "Symbol": "MS4A1",
                 "GeneID": "931", "Cell ontology ID": "CL:0000002", "Year": "2014"},
                {"Species": "Mouse", "Cell name": "old", "Symbol": "Bad",
                 "GeneID": "1", "Cell ontology ID": "CL:0000003", "Year": "2010"},
            ])
            with pd.ExcelWriter(workbook) as writer:
                table.to_excel(writer, sheet_name="Normal cell", index=False)
            records = parse_cellmarker_mouse(workbook, ontology)
            self.assertEqual(len(records), 1)
            self.assertEqual((records[0].official_gene_symbol, records[0].gene_id),
                             ("Cd79a", "12518"))

    def test_cellmarker_current_all_sheet_filters_to_normal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obo = root / "cl.obo"; obo.write_text(OBO, encoding="utf-8")
            ontology = parse_cell_ontology(obo)
            workbook = root / "cellmarker-current.xlsx"
            table = pd.DataFrame([
                {"species": "Mouse", "cell_name": "B cell", "Symbol": "Cd79a",
                 "GeneID": "12518", "cellontology_id": "CL_0000002",
                 "year": "2015", "cancer_type": "Normal"},
                {"species": "Mouse", "cell_name": "B cell", "Symbol": "Ms4a1",
                 "GeneID": "12482", "cellontology_id": "CL_0000002",
                 "year": "2015", "cancer_type": "Lymphoma"},
            ])
            with pd.ExcelWriter(workbook) as writer:
                table.to_excel(writer, sheet_name="All", index=False)
            records = parse_cellmarker_mouse(workbook, ontology)
            self.assertEqual([(record.official_gene_symbol, record.gene_id)
                              for record in records], [("Cd79a", "12518")])

    def test_cellmarker_drops_conflicting_symbol_for_same_cl_and_gene_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obo = root / "cl.obo"; obo.write_text(OBO, encoding="utf-8")
            ontology = parse_cell_ontology(obo)
            workbook = root / "cellmarker-conflict.xlsx"
            table = pd.DataFrame([
                {"Species": "Mouse", "Cell name": "B cell", "Symbol": "Cd79a",
                 "GeneID": "12518", "Cell ontology ID": "CL:0000002", "Year": "2015"},
                {"Species": "Mouse", "Cell name": "B cell", "Symbol": "Cd79b",
                 "GeneID": "12518", "Cell ontology ID": "CL:0000002", "Year": "2015"},
                {"Species": "Mouse", "Cell name": "B cell", "Symbol": "Ms4a1",
                 "GeneID": "12482", "Cell ontology ID": "CL:0000002", "Year": "2015"},
            ])
            with pd.ExcelWriter(workbook) as writer:
                table.to_excel(writer, sheet_name="Normal cell", index=False)
            records = parse_cellmarker_mouse(workbook, ontology)
            self.assertEqual([(record.official_gene_symbol, record.gene_id)
                              for record in records], [("Ms4a1", "12482")])

    def test_panel_exact_matching_minimum_and_determinism(self):
        records = [MarkerRecord("", gene, "alpha", source="external")
                   for gene in ["G5", "G3", "G1", "G2", "G4", "g1"]]
        records += [MarkerRecord("", gene, "beta", source="external")
                    for gene in ["B1", "B2", "B3", "B4"]]
        kwargs = dict(available_genes=["G1", "G2", "G3", "G4", "G5", "B1", "B2", "B3", "B4"],
                      label_to_source_cell_types={"alpha endpoint": ["alpha"], "beta endpoint": ["beta"]},
                      dataset="test", source_provenance={"source": "frozen"}, minimum_genes=5)
        a = build_marker_panel(reversed(records), **kwargs)
        b = build_marker_panel(records, **kwargs)
        self.assertEqual(a, b)
        self.assertEqual(a["genes_by_label"]["alpha endpoint"], ["G1", "G2", "G3", "G4", "G5"])
        self.assertNotIn("g1", a["genes_by_label"]["alpha endpoint"])
        self.assertIn("beta endpoint", a["excluded_labels"])

    def test_panel_roundtrip_is_compact_and_refuses_overwrite(self):
        records = [MarkerRecord("", f"G{i}", "x") for i in range(5)]
        panel = build_marker_panel(records, [f"G{i}" for i in range(5)], {"x": ["x"]},
                                   dataset="d", source_provenance={}, minimum_genes=5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "panel.json.gz"
            write_panel(path, panel)
            self.assertEqual(read_panel(path), panel)
            self.assertLess(path.stat().st_size, len(str(panel).encode("utf-8")))
            with self.assertRaises(FileExistsError):
                write_panel(path, panel)

    def test_mouse_official_join_ignores_corrupt_local_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obo = root / "cl.obo"; obo.write_text(OBO, encoding="utf-8")
            ontology = parse_cell_ontology(obo)
            columns = {
                "cell": ["a", "b"], "method": ["droplet", "droplet"],
                "tissue": ["Liver", "Spleen"], "cell_ontology_class": ["alpha cell", "B cell"],
                "cell_ontology_id": ["CL:9999999", "NA"]}
            source = ad.AnnData(np.ones((2, 1)), obs=pd.DataFrame(columns, index=["s1", "s2"]))
            droplet = root / "droplet.h5ad"; source.write_h5ad(droplet)
            empty_columns = {key: [] for key in columns}
            facs_data = ad.AnnData(np.empty((0, 1)), obs=pd.DataFrame(empty_columns, index=[]))
            facs = root / "facs.h5ad"; facs_data.write_h5ad(facs)
            combined_columns = dict(columns)
            combined_columns["cell_ontology_class"] = ["wrong local", "B cell"]
            combined = ad.AnnData(np.ones((2, 1)),
                                  obs=pd.DataFrame(combined_columns, index=["c1", "c2"]))
            combined_path = root / "combined.h5ad"; combined.write_h5ad(combined_path)
            result = recover_official_mouse_labels(combined_path, droplet, facs, ontology=ontology)
            self.assertEqual(result.official_labels, ("alpha cell", "B cell"))
            self.assertEqual(result.official_ontology_ids, ("CL:0000001", "CL:0000002"))
            self.assertEqual(result.audit["local_label_mismatches_vs_official_source"], 1)
            self.assertFalse(result.audit["local_cell_ontology_id_audit"]["used_for_recovery"])


if __name__ == "__main__":
    unittest.main()
