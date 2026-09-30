# Audited Package 3/4/4b manuscript assets

This directory is a downstream-only reporting bundle built from four completed, frozen runs. The builder validates every artifact fingerprint recorded by those run manifests, recomputes displayed seed-to-section-to-donor aggregates, and fails closed on coverage or numeric mismatch. It does not modify any scientific run artifact, configuration, model, or manuscript source.

## Package 3: injected-artifact validation

- Source: `revision_pipeline/runs/20260929T192600Z-artifactv2fix2-full-panel`.
- Preparation, baselines, and 120 training runs use frozen foundation hash `e83af7b23ae9a553d6b28e12dd1ed967f3613f713fa7f3e47e5acaaa3cab8c46`.
- Corrected scoring and consolidation use hash `4c61b60a8755122c8b5a13a49a61ca2dbbf4b1ab94d28d90b48c945de4c16304` under the recorded fail-closed expected/observed float64-hash bridge. Native embedding precision was unchanged.
- Outcome: 0/24 prespecified joint correction-and-preservation screens passed. GenoRefine met sensitivity in 5/12 conditions and IDEC in 8/12; neither method met clean-neighbor recovery or preservation in any condition.
- Boundary: this panel does not support a claim of general artifact correction. Algorithmic seeds are descriptive repeats, not biological replicates; no p-values are reported.

## Packages 4 and 4b: six-section LIBD DLPFC spatial panel

- Package 4 source: `revision_pipeline/runs/20260929-spatial-panel-v1-full-panel`, authoritative Package-4-specific hash `bb0f67f14ca3a460a9f708a920c2b64aa44f526994737d84b6e427b46f387b4f`.
- Package 4b source: `revision_pipeline/runs/20260930-graphst4b-v5-full-panel`, protocol hash `b20875a3c5ccb4f0ff7f164587e5836e1db715ae4e7838bf22e64040b02b11ae`, source hash `4d24edecd74a6814c4e09fc13c14b47bd2798d39fcbcf93a3fe2d4c46a853c79`.
- Six sections (two per donor) contain 22,968 retained spots. Layer identities were withheld from fitting and model selection; however, 113 spots lacking a reference layer were excluded when the benchmark cohort was frozen. Fits are therefore label-blind conditional on label availability.
- The common evaluator uses fixed Leiden resolution 0.5 and three Leiden seeds. Because realized common-evaluator cluster counts differ (Harmony 5.28, GenoRefine 3.20, SpaGCN 13.26, GraphST 6.50 donor-macro), common ARI/NMI are fixed-resolution profiles, not matched-K domain recovery.
- Task asymmetry remains explicit: Harmony and GenoRefine use a pooled six-section representation, SpaGCN is fit per section with histology, and GraphST uses label-free PASTE alignment followed by three donor-pair models. SpaGCN therefore has no donor-pair mixing endpoint.
- Spatial-neighborhood preservation is weak in absolute terms for Harmony, GenoRefine, and SpaGCN. GenoRefine modestly improves section mixing but does not improve the layer/local-spatial endpoint profile. Purpose-built GraphST performs substantially better on those common endpoints.
- SpaGCN and GraphST method-native partitions are reported separately from common-evaluator partitions. They must not be conflated.
- Five seeds are descriptive algorithmic repeats, not biological replicates. No p-values are reported.

## Files

- `package3_artifact_screen.{pdf,png}`: prespecified criterion counts and condition-level sensitivity.
- `package3_artifact_screen_table.{pdf,csv}`: compact summary and full condition-level machine-readable table.
- `spatial_common_profiles.{pdf,png}`: common fixed-resolution endpoints, realized cluster granularity, and donor-pair mixing.
- `spatial_native_partitions.{pdf,png}`: separately labeled SpaGCN and GraphST task-native ARI/NMI.
- `spatial_profiles_table.{pdf,csv}`: common-evaluator, mixing, cluster-granularity, and native-partition tables.
- `number_provenance.json`: one provenance entry per displayed number, with source fingerprints and aggregation contracts.
- `asset_manifest.json`: output fingerprints and audit coverage.

Regenerate with `python revision_pipeline/manuscript_figures/build_package34_assets.py`. The default destination is this directory.
