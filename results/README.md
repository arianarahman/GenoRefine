# Aggregate result packages

This directory contains the machine-readable aggregate outputs and concise
reports used by the GenoRefine manuscript. The packages are organized by
analysis question so readers can inspect the included configurations, run
metadata, summary tables, and provenance records together. The release is a
curated aggregate layer; full run directories remain outside version control.

## Package index

| Directory | Contents |
|---|---|
| `package1_idec/` | Twelve-case comparison with the independently pinned IDEC implementation |
| `package2_external_markers/` | External-marker foundation records and representation-level marker summaries |
| `package3_artifacts/` | Injected-artifact panel configuration and aggregate condition/seed summaries |
| `package4_spatial/` | Six-section LIBD DLPFC common-evaluator and donor-level spatial summaries |
| `package4_spagcn_native/` | Provenance-linked SpaGCN native-partition summaries |
| `package4b_graphst/` | PASTE-aligned GraphST common-evaluator and native-partition summaries |

## File conventions

The packages use a shared set of artifact types:

- `report.md` provides a readable description of the package and its tables.
- `config.json` records the frozen analysis configuration associated with the
  released output.
- `run.json` records run identity, status, timing, and artifact references.
- `summary.json` and similarly named JSON files provide structured aggregates.
- Reviewed CSV files provide tabular values used in analysis and plotting.
- Source manifests and completion records connect the aggregate package to its
  executed workflow.

The exact files differ by package because the analysis units differ: a row may
represent a dataset/backbone case, algorithmic seed, tissue section, donor, or
evaluation endpoint. Each package's `report.md` defines its row and aggregation
units.

## Working with the packages

Inspect a package directly with any Markdown, JSON, or CSV reader. For example:

```bash
python -m json.tool results/package1_idec/summary.json
python -m json.tool results/package4_spatial/run.json
```

Publication-ready renderings are stored under [`../figures/`](../figures/).
The figure `number_provenance.json` and `asset_manifest.json` files connect
displayed values and rendered assets to these aggregate records.

Machine-local paths in selected public receipts have been normalized for
portable use. [`PATH_SANITIZATION.md`](PATH_SANITIZATION.md) records those
transformations and the corresponding fingerprints.

For the repository-wide provenance model, see
[`../docs/provenance.md`](../docs/provenance.md).
