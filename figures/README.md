# Publication figures and tables

This directory contains publication-ready figures, rendered tables, and their
machine-readable companions. Each asset group is accompanied by provenance
records that connect displayed values to aggregate sources and define the
aggregation used to produce them.

## Asset index

| Assets | Analysis package |
|---|---|
| `main_independent_comparator_12case.{png,pdf}` and table companions | Package 1: GenoRefine and IDEC comparison |
| `supp_external_marker_8case.{png,pdf}` and table companions | Package 2: external-marker preservation |
| `package34_audited/package3_artifact_screen*` | Package 3: injected-artifact evaluation |
| `package34_audited/spatial_common_profiles*` | Packages 4 and 4b: common spatial-evaluator profiles |
| `package34_audited/spatial_native_partitions*` | Packages 4 and 4b: method-native spatial partitions |
| `package34_audited/spatial_profiles_table*` | Packages 4 and 4b: rendered and machine-readable spatial tables |

## Provenance files

- `asset_manifest.json` records released-asset fingerprints and audit coverage.
- `number_provenance.json` maps displayed values to source fingerprints and
  aggregation contracts.
- CSV companions provide plotted aggregate values in a machine-readable form.
- `package34_audited/README.md` documents the detailed source-run bindings for
  the Package 3/4/4b asset bundle.

Keep each figure or table together with its matching manifest and number
provenance when copying it into another release or manuscript workspace.

## Build code

The exact downstream asset builders are located in
[`../revision_pipeline/manuscript_figures/`](../revision_pipeline/manuscript_figures/):

```bash
python revision_pipeline/manuscript_figures/build_package12_assets.py
python revision_pipeline/manuscript_figures/build_package34_assets.py
```

The builders validate their declared source-run artifacts and write staged
assets with updated manifests. Their prerequisites and output behavior are
documented in
[`../revision_pipeline/manuscript_figures/README.md`](../revision_pipeline/manuscript_figures/README.md).
