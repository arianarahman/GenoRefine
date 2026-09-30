# Full independent IDEC comparator panel

All 12 dataset-backbone pairs were evaluated with five seeds, matched 32-dimensional outputs, label-free K, 100 pretraining epochs, a dataset-matched two-pass joint budget, and the frozen primary evaluator.

| Dataset | Backbone | Method | ARI | SIL | iLISI | Purity |
|---|---|---|---:|---:|---:|---:|
| HP-CB | Scanorama | upstream | 0.6717 | 0.1729 | 0.1250 | 0.9627 |
| HP-CB | Scanorama | GenoRefine | 0.4190 +/- 0.0355 | 0.6530 +/- 0.0222 | 0.1791 +/- 0.0062 | 0.9495 +/- 0.0079 |
| HP-CB | Scanorama | IDEC | 0.2610 +/- 0.0221 | 0.4025 +/- 0.0331 | 0.2523 +/- 0.0084 | 0.8469 +/- 0.0257 |
| HP-CB | Harmony | upstream | 0.8213 | 0.2426 | 0.1773 | 0.9534 |
| HP-CB | Harmony | GenoRefine | 0.8233 +/- 0.0020 | 0.1796 +/- 0.0108 | 0.1932 +/- 0.0011 | 0.9475 +/- 0.0012 |
| HP-CB | Harmony | IDEC | 0.8027 +/- 0.0047 | 0.0582 +/- 0.0143 | 0.2143 +/- 0.0024 | 0.8991 +/- 0.0032 |
| HP-CB | Seurat | upstream | 0.8304 | 0.3560 | 0.2510 | 0.9498 |
| HP-CB | Seurat | GenoRefine | 0.9154 +/- 0.0049 | 0.2696 +/- 0.0311 | 0.2678 +/- 0.0004 | 0.9489 +/- 0.0005 |
| HP-CB | Seurat | IDEC | 0.8509 +/- 0.0407 | 0.0585 +/- 0.0586 | 0.2699 +/- 0.0018 | 0.9348 +/- 0.0015 |
| HP-CB | Online iNMF | upstream | 0.5349 | 0.1371 | 0.1386 | 0.9498 |
| HP-CB | Online iNMF | GenoRefine | 0.5904 +/- 0.0124 | 0.2063 +/- 0.0249 | 0.1339 +/- 0.0009 | 0.9502 +/- 0.0008 |
| HP-CB | Online iNMF | IDEC | 0.3794 +/- 0.0181 | 0.2524 +/- 0.0050 | 0.1579 +/- 0.0078 | 0.9394 +/- 0.0027 |
| Pancreas | Scanorama | upstream | 0.3896 | 0.1621 | 0.0195 | 0.9531 |
| Pancreas | Scanorama | GenoRefine | 0.3797 +/- 0.0095 | 0.6576 +/- 0.0206 | 0.0533 +/- 0.0079 | 0.9533 +/- 0.0023 |
| Pancreas | Scanorama | IDEC | 0.2548 +/- 0.0169 | 0.3073 +/- 0.0196 | 0.1515 +/- 0.0142 | 0.8499 +/- 0.0232 |
| Pancreas | Harmony | upstream | 0.6998 | 0.1510 | 0.1211 | 0.9470 |
| Pancreas | Harmony | GenoRefine | 0.7194 +/- 0.0129 | 0.1027 +/- 0.0063 | 0.1203 +/- 0.0027 | 0.9387 +/- 0.0021 |
| Pancreas | Harmony | IDEC | 0.7600 +/- 0.0572 | 0.0327 +/- 0.0179 | 0.1801 +/- 0.0066 | 0.8726 +/- 0.0055 |
| Pancreas | Seurat | upstream | 0.6639 | 0.1962 | 0.2652 | 0.9502 |
| Pancreas | Seurat | GenoRefine | 0.7166 +/- 0.0229 | 0.1482 +/- 0.0043 | 0.2698 +/- 0.0017 | 0.9459 +/- 0.0006 |
| Pancreas | Seurat | IDEC | 0.8415 +/- 0.0433 | 0.0308 +/- 0.0142 | 0.2814 +/- 0.0026 | 0.8914 +/- 0.0037 |
| Pancreas | Online iNMF | upstream | 0.4275 | 0.1368 | 0.0184 | 0.9289 |
| Pancreas | Online iNMF | GenoRefine | 0.5138 +/- 0.0106 | 0.1345 +/- 0.0089 | 0.0175 +/- 0.0006 | 0.9324 +/- 0.0010 |
| Pancreas | Online iNMF | IDEC | 0.4886 +/- 0.0231 | 0.2721 +/- 0.0222 | 0.0412 +/- 0.0044 | 0.9262 +/- 0.0023 |
| Mouse | Scanorama | upstream | 0.5403 | 0.2359 | 0.3441 | 0.6992 |
| Mouse | Scanorama | GenoRefine | 0.5368 +/- 0.0223 | 0.7436 +/- 0.0346 | 0.4887 +/- 0.0088 | 0.6096 +/- 0.0028 |
| Mouse | Scanorama | IDEC | 0.4661 +/- 0.0027 | 0.4640 +/- 0.0642 | 0.5678 +/- 0.0135 | 0.5510 +/- 0.0057 |
| Mouse | Harmony | upstream | 0.5042 | 0.2723 | 0.3839 | 0.6615 |
| Mouse | Harmony | GenoRefine | 0.5067 +/- 0.0016 | 0.2449 +/- 0.0039 | 0.4022 +/- 0.0022 | 0.6593 +/- 0.0004 |
| Mouse | Harmony | IDEC | 0.4982 +/- 0.0058 | 0.1394 +/- 0.0064 | 0.4453 +/- 0.0057 | 0.6329 +/- 0.0011 |
| Mouse | Seurat | upstream | 0.5143 | 0.2797 | 0.4832 | 0.6659 |
| Mouse | Seurat | GenoRefine | 0.5076 +/- 0.0078 | 0.2561 +/- 0.0051 | 0.5065 +/- 0.0032 | 0.6632 +/- 0.0008 |
| Mouse | Seurat | IDEC | 0.4948 +/- 0.0065 | 0.1458 +/- 0.0077 | 0.5350 +/- 0.0047 | 0.6371 +/- 0.0008 |
| Mouse | Online iNMF | upstream | 0.5066 | 0.4187 | 0.2824 | 0.5916 |
| Mouse | Online iNMF | GenoRefine | 0.4607 +/- 0.0191 | 0.4985 +/- 0.0098 | 0.2197 +/- 0.0052 | 0.5875 +/- 0.0010 |
| Mouse | Online iNMF | IDEC | 0.4451 +/- 0.0102 | 0.6343 +/- 0.0818 | 0.1089 +/- 0.0126 | 0.5724 +/- 0.0021 |

Algorithmic seeds are reported descriptively; they are not biological replicates. Higher iLISI is interpreted only with the agreement and preservation endpoints.
