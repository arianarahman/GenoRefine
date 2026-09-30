# GraphST Package 4b execution contract

Package 4b is a separate, preregistered supporting comparator. It reuses the
completed six-section LIBD foundation and evaluation policy from Package 4; it
does not alter or rerun Package 4 outputs.

The full panel consists of three label-free PASTE donor-pair alignments, one
label-free donor-K selection, fifteen official GraphST fits (three donors by
five seeds), thirty section scores, fifteen donor-pair scores, and one
consolidation run. The preflight is a separate synthetic acceptance run and is
not part of the 65 scientific run directories.

Use exactly one orchestrator mode:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File revision_pipeline\spatial_graphst\run_panel.ps1 -Prefix <prefix> -PlanOnly
powershell -NoProfile -ExecutionPolicy Bypass -File revision_pipeline\spatial_graphst\run_panel.ps1 -Prefix <prefix> -PreflightOnly
powershell -NoProfile -ExecutionPolicy Bypass -File revision_pipeline\spatial_graphst\run_panel.ps1 -Prefix <prefix> -ExecutePanel
```

If the locked runtime tag must be reconstructed from its pinned inputs, disable
BuildKit provenance attestations so the local image identifier is the stable
content-manifest digest recorded by the protocol:

```powershell
docker build --provenance=false -f revision_pipeline\spatial_graphst\Dockerfile.graphst-gpu -t genorefine-graphst:1.1.1-panel-v1 .
```

`-ExecutePanel` requires the exact frozen image and a succeeded preflight under
the same prefix. Completed jobs are deeply revalidated before reuse. Stale
incomplete directories are preserved with a retired name; they are never
silently deleted or treated as evidence.

Manual cortical-layer identities and values are not read by PASTE alignment,
GraphST fitting, donor-K selection, mclust, or spatial refinement. They enter
Package 4b only in section scoring. The reused Package 4 foundation is,
however, an inherited complete-case cohort: it retained 22,968 spots with
nonmissing manual labels and excluded 113 of 23,081 spots. The fit is therefore
label-blind conditional on label availability, not independent of it. Five
seeds are descriptive algorithmic repeats, not biological replicates, and no
p-values are produced.

PASTE and GraphST use full-resolution Visium pixel coordinates. This preserves
the physical hex-lattice geometry for GraphST's official spatial-neighbor
construction; array row/column coordinates are not substituted for physical
coordinates. For PASTE's fused Gromov-Wasserstein objective, each slice's
distance matrix is normalized by its smallest nonzero distance (`norm=True`),
preventing the expression/spatial balance from depending on pixel units. The
rigid aligned full-resolution coordinates, rather than normalized distance
matrices, remain the GraphST spatial input.

The frozen PASTE API token is `dissimilarity="euclidean"`. In PASTE 1.4.0 this
branch delegates to POT `ot.dist` without a metric override, so the numerical
expression cost is squared Euclidean. Both the public API token and the
mathematical endpoint are recorded in the protocol.

GraphST 1.1.1 stores its reconstructed 3,000-HVG output as `emb`. The official
`clustering()` helper reduces that output to `emb_pca` with 20 principal
components (random state 42). Package 4b retains and hashes both; the frozen
common evaluator uses the official `emb_pca` representation. The 64-dimensional
encoder bottleneck is a training parameter and is not mislabeled as the
official GraphST embedding.

Common-evaluator rows and both task-native partitions (raw mclust and official
spatial refinement) are exported separately. Training receipts label the
reported objective components as last-training-step losses because the
official attributes are not recomputed after the final optimizer update.

The synthetic preflight validates source/runtime locks and same-seed
determinism, but it deliberately does not load scientific sections and is not
a full-size memory stress test. Official GraphST constructs dense spatial
matrices; therefore the first donor-pair training run remains the operational
resource gate and any out-of-memory attempt must be preserved rather than
reclassified as scientific evidence.
