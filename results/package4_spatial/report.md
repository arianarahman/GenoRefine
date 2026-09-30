# Six-section DLPFC spatial panel

Primary results use the fixed-10 pooled Harmony baseline. The native-stop Harmony result is a sensitivity only.
Five seeds are algorithmic repeats, not biological replicates; no p-values are reported.

## Donor-macro section endpoints

| Method | ARI | NMI | cluster SIL | reference SIL | purity | spatial kNN Jaccard (k=6) | physical/local ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Harmony fixed-10 | 0.1635 | 0.2300 | 0.0550 | -0.0177 | 0.4819 | 0.0125 | 17.2711 |
| Harmony + GenoRefine | 0.1711 ± 0.0233 | 0.2119 ± 0.0117 | 0.0427 ± 0.0068 | -0.0284 ± 0.0012 | 0.4469 ± 0.0038 | 0.0102 ± 0.0004 | 18.4093 ± 0.1589 |
| SpaGCN | 0.1457 ± 0.0157 | 0.3103 ± 0.0237 | 0.2774 ± 0.0080 | 0.0060 ± 0.0137 | 0.5650 ± 0.0229 | 0.0144 ± 0.0027 | 19.7433 ± 0.7122 |

## Donor-pair section mixing

| Method | iLISI | D_batch |
|---|---:|---:|
| Harmony fixed-10 | 0.8626 | 1.9323 |
| Harmony + GenoRefine | 0.8781 ± 0.0027 | 1.9477 ± 0.0025 |

SpaGCN is trained independently per section, so cross-section mixing is not a defined endpoint for that comparator.
Manual cortical-layer labels are evaluation-only. K is selected from fixed-resolution Harmony partitions without labels.
