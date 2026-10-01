# Configuration registry

The JSON files in this directory define the datasets, inputs, algorithms,
hyperparameters, environments, evaluation rules, seeds, and reporting contracts
used by the GenoRefine workflows. Pipeline modules read these files directly so
the scientific settings remain separate from orchestration code.

## Configuration families

| Family | Files |
|---|---|
| Dataset and input registry | `datasets.json`, `data_store.json`, `step3a_source_lock.json`, `annotation_provenance_v1.json` |
| Shared evaluation | `evaluation_development.json`, `evaluation_primary_v1.json`, `step4_decisions_v1.json`, `step4a_scoring_v1.json` |
| Main benchmark | `main_benchmark_v1.json`, `pancreas_backbones.json`, `pancreas_*policy*.json` |
| Core staged validation | `step3c_pilot.json`, `step4a_hpcb_scanorama_v1.json`, `step4b_hpcb_scanorama_v1.json`, `step4c_synthetic_v1.json` |
| Independent comparators | `independent_idec_v1.json`, `independent_idec_panel_v2.json`, `step5a_genomoi_core_v1.json` |
| Biological preservation | `external_marker_sources_v1.json`, `independent_marker_foundation_v1.json`, `independent_marker_preservation_v1.json` |
| Additional validation | `artifact_validation_v2.json`, `broader_validation_v1.json`, `inductive_validation_v1.json` |
| Ablations | `focused_ablation_v1.json`, `objective_ablation_v1.json` |
| scVI comparisons | `scvi_comparator_v1.json`, `scvi_other_datasets_v1.json` |
| Spatial analyses | `spatial_multisection_v1.json`, `spatial_multisection_panel_v1.json`, `spatial_comparator_v1.json`, `spatial_graphst_panel_v1.json` |

## Working with configurations

- Treat each configuration as a versioned protocol. Create a new filename or
  schema version for a new scientific protocol.
- Keep repository paths relative to the project root. Runtime wrappers resolve
  the root before launching package modules.
- Record source URLs, commits, byte counts, and SHA-256 values in the relevant
  source-lock configuration.
- Validate required keys and value ranges in the consuming Python module before
  starting computation.
- Preserve the exact configuration with every completed run and aggregate
  result package.

The end-to-end mapping from released result package to protocol is documented
in [`../../docs/reproduction.md`](../../docs/reproduction.md).
