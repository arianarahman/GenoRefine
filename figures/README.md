# Publication figures and audited tables

This directory contains derived figures and aggregate tables only. It contains no raw images, cell/spot-level data, per-cell embeddings, or model artifacts. Image and PDF files are permitted here as an explicit exception to the repository-wide image ignore rules.

## Package mapping

| Assets | Package/evidence |
|---|---|
| `main_independent_comparator_12case.{png,pdf}` and matching table | Package 1, 12-case GenoRefine/IDEC comparison |
| `supp_external_marker_8case.{png,pdf}` and matching table | Package 2, external-marker preservation panel |
| `package34_audited/package3_artifact_screen*` | Package 3, injected-artifact screens |
| `package34_audited/spatial_common_profiles*` | Packages 4/4b common-evaluator profiles, cluster granularity, and mixing |
| `package34_audited/spatial_native_partitions*` | Separately labeled SpaGCN and GraphST task-native partitions |
| `package34_audited/spatial_profiles_table*` | Machine-readable and rendered spatial aggregate tables |

## Audit metadata

- `asset_manifest.json` files record released-asset fingerprints and audit coverage.
- `number_provenance.json` files map displayed numbers to source fingerprints and aggregation contracts.
- CSV companions expose plotted aggregate values without requiring figure digitization.
- `package34_audited/README.md` records the detailed Package 3/4/4b source-run and interpretation boundaries.

Retain the complete manifest/provenance pair when copying or regenerating a figure. A visually identical image without its source and aggregation record is not equivalent evidence.

The Package 3/4/4b bundle README is preserved from its audited staging output.
Its phrase “this directory” refers to the builder's original default at
`revision_pipeline/manuscript_figures/staging/package34_audited`, not this
public `figures/` destination. Exact builders are included under
`revision_pipeline/manuscript_figures/`; rerunning them requires the excluded
full source-run directories.

The bundled assets should be treated as release outputs. This snapshot does not claim that every publication graphic can be rebuilt with one command from public accessions: raw/cell-level inputs and full run directories are excluded, and some processed primary-input lineage remains unresolved.
