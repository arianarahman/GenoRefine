# Package 4 SpaGCN task-native endpoints

These values are a provenance-linked derivation from the immutable completed Package 4 scores. No model was retrained and no embedding was rescored.

| Secondary SpaGCN endpoint | ARI | NMI |
|---|---:|---:|
| Native predicted partition | 0.2222 ± 0.0105 | 0.3471 ± 0.0187 |
| Native hex-refined partition | 0.2465 ± 0.0106 | 0.3785 ± 0.0166 |

Aggregation is performed within each seed: equal mean over two sections per donor, then an equal macro-average over the three donors. Mean ± sample SD is descriptive across seeds 0–4; the seeds are algorithmic repeats, not biological replicates.

These task-native partitions remain separate from the fixed-resolution common-evaluator ARI/NMI. See evidence_boundaries.json for the label-policy, cluster-granularity, and spatial-preservation limits.
