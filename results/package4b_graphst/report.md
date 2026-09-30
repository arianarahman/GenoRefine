# GraphST Package 4b: six-section LIBD DLPFC panel

GraphST was fit once per donor pair after label-free PASTE alignment. Five seeds are descriptive algorithmic repeats, not biological replicates; no p-values are reported.
Manual cortical-layer identities/values were introduced only during Package 4b evaluation. The inherited Package 4 foundation was already restricted to spots with nonmissing labels, so the fit is label-blind conditional on that complete-case cohort.

## Common-evaluator donor-macro endpoints

| Endpoint | Mean | SD across five aligned algorithmic seeds |
|---|---:|---:|
| ARI | 0.4206 | 0.0086 |
| NMI | 0.5677 | 0.0057 |
| predicted_cluster_silhouette | 0.1289 | 0.0011 |
| reference_label_silhouette | 0.0871 | 0.0004 |
| label_neighbor_purity | 0.7668 | 0.0007 |
| spatial_latent_knn_jaccard | 0.0723 | 0.0007 |
| physical_distance_among_latent_neighbors_mean_fullres_pixels | 1509.3995 | 3.2148 |
| physical_distance_ratio_to_local_spatial_knn_mean | 10.7452 | 0.0220 |

## Donor-pair section mixing

| Endpoint | Mean | SD |
|---|---:|---:|
| iLISI | 0.8356 | 0.0008 |
| D_batch | 1.9877 | 0.0003 |

## Secondary task-native GraphST partitions

These native endpoints are reported separately from common-evaluator clustering.

| Partition | Endpoint | Mean | SD |
|---|---|---:|---:|
| mclust | ARI | 0.4590 | 0.0255 |
| mclust | NMI | 0.6197 | 0.0175 |
| mclust | clusters | 6.3333 | 0.0000 |
| refined | ARI | 0.4762 | 0.0261 |
| refined | NMI | 0.6564 | 0.0146 |
| refined | clusters | 6.3333 | 0.0000 |
