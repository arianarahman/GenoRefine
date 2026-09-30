# Aggregate result packages

This directory contains the public aggregate evidence bundle. It contains no raw, cell-level, or spot-level data and no per-cell embeddings or trained models. Some CSV files retain per-seed or per-section aggregate rows so published means and standard deviations can be audited.

This is a curated release subset, not a full copy of each scientific run. Package 4-family run manifests refer to seven detailed seed-row JSON files that are not shipped; the corresponding CSV records and summary JSON files are present. The included asset manifests report hash-matching released files and audited displayed values.

## Package 1: independent IDEC comparator

`package1_idec/` covers 12 dataset-backbone cases (HP-CB, five-study pancreas, and mouse; four upstream backbones each) with five algorithmic seeds for GenoRefine and IDEC. `long_table.csv` is the compact analysis table; `summary.json` preserves the full aggregation; `report.md` is the readable endpoint table.

The results are tradeoffs, not a universal ranking. In direct aggregate comparisons, GenoRefine exceeds IDEC on ARI in 10/12 cases, predicted-cluster silhouette in 9/12, and neighbor purity in 12/12, while IDEC exceeds GenoRefine on iLISI in 11/12. Label agreement and neighborhood preservation must therefore be read alongside mixing.

## Package 2: independent external markers

`package2_external_markers/` contains 88 completed prespecified representation evaluations across HP-CB and mouse: 8 upstream, 40 GenoRefine, and 40 IDEC evaluations. GenoRefine exceeds IDEC on marker AUROC in 7/8 case-level comparisons and on average precision and marker contrast in 8/8. The readable table reports all three endpoints. Marker panels and thresholds were fixed without reading evaluated embeddings or scores; reference labels entered only after marker scoring to define evaluation strata.

The `foundation/` subdirectory contains aggregate foundation metadata and public-source receipts, not marker scores per cell. The evaluated foundations cover 16,382 HP-CB cells across 12 marker classes and 50,000 mouse cells across 26 classes; observation-level scores are excluded.

## Package 3: injected artifacts

`package3_artifacts/` tests two injected artifacts over three datasets, two upstream backbones, two refiners, and five seeds. The prespecified joint correction-and-preservation criterion was unmet in all 24 refiner-condition screens. GenoRefine reduced the defined artifact sensitivity in 5/12 conditions and IDEC in 8/12; clean-neighbor recovery remained below the required threshold throughout.

This scoped result defines the evidence boundary for general artifact-correction claims; the injected conditions do not exhaust native biological or technical artifacts.

## Package 4: six-section LIBD DLPFC panel

`package4_spatial/` reports a three-donor, six-section panel on 22,968 retained spots. Donor-macro common-evaluator summaries include:

| Method | ARI | iLISI | spatial kNN Jaccard (k=6) |
|---|---:|---:|---:|
| Harmony fixed-10 | 0.1635 | 0.8626 | 0.0125 |
| Harmony + GenoRefine | 0.1711 ± 0.0233 | 0.8781 ± 0.0027 | 0.0102 ± 0.0004 |
| SpaGCN | 0.1457 ± 0.0157 | not defined | 0.0144 ± 0.0027 |

GenoRefine modestly increased donor-pair mixing, while the layer and local-spatial endpoint profile favored the upstream representation or purpose-built spatial comparators. SpaGCN is fit independently per section, so its cross-section mixing endpoint is undefined.

`package4_spatial/PUBLIC_NOTES.md` records the cohort-label and fixed-resolution
interpretation clarifications while preserving `report.md` byte-for-byte as a
frozen run artifact.

`package4_spagcn_native/` is a provenance-linked derivation from immutable Package 4 scores; it does not retrain or rescore a model. SpaGCN native predicted and hex-refined partitions have ARI 0.2222 ± 0.0105 and 0.2465 ± 0.0106, respectively. Keep these native partitions separate from fixed-resolution common-evaluator partitions.

## Package 4b: PASTE + GraphST

`package4b_graphst/` reports five label-free aligned algorithmic seeds. Common-evaluator donor-macro means are ARI 0.4206 ± 0.0086, NMI 0.5677 ± 0.0057, spatial kNN Jaccard 0.0723 ± 0.0007, and donor-pair iLISI 0.8356 ± 0.0008. The secondary native refined partition has ARI 0.4762 ± 0.0261 and NMI 0.6564 ± 0.0146.

GraphST uses label-free PASTE alignment and donor-pair fitting, so comparison with pooled Harmony/GenoRefine and per-section SpaGCN is descriptive and task-asymmetric.

## Shared boundaries

- Five seeds are algorithmic repeats, not biological replicates.
- No p-values are reported for Packages 3, 4, or 4b.
- The DLPFC foundation excluded 113 of 23,081 spots with missing manual layer labels. Fitting did not read label identities/values, but it is conditional on this complete-case cohort.
- Common spatial ARI/NMI use fixed resolution with unequal realized cluster counts; they are not matched-*K* estimates.

See each `report.md`, `docs/provenance.md`, and the matching figure provenance before quoting a number.
