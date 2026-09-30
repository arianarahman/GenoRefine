# Package 4: six-section DLPFC execution

This document preserves the preregistered execution plan and frozen protocol at
`revision_pipeline/configs/spatial_multisection_panel_v1.json`. The full panel
was subsequently completed as `20260929-spatial-panel-v1-full-panel`; its
audited aggregate outputs are released under `results/package4_spatial/`.

## Authoritative environments

- Fixed-10 Harmony: `<PYTHON_ENV>`, with the exact package versions locked in the protocol.
- Exact graph, K selection, scoring, and consolidation: WSL distribution `Ubuntu-24.04`, interpreter `<PYTHON_ENV>`.
- GenoRefine: `genorefine-gpu:tf25.02`, image ID `sha256:9f215e1ad258685d300d5162b70464363915020beca6d2584a799f60a51f2c8b`.
- SpaGCN: `genorefine-spagcn:1.2.7-panel-v1`, image ID `sha256:6eb8c07971d19ad54839f5cbb6fa7990ad15cec839d04e361797ee44778e79c6`.

Both container IDs and all relevant package versions are checked before training. The SpaGCN image recipe disables the NGC extra package index and uses the package versions pinned in `requirements-spagcn-gpu.txt`.

## Preflight commands

From `<PROJECT_ROOT>`:

```powershell
python -m pytest revision_pipeline\spatial_multisection\tests -q --basetemp revision_pipeline\tmp_pytest_spatial_panel_preflight

docker run --rm --gpus all --shm-size 4g `
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 `
  -e SPAGCN_IMAGE_ID=sha256:6eb8c07971d19ad54839f5cbb6fa7990ad15cec839d04e361797ee44778e79c6 `
  -v "${PWD}:/workspace" -w /workspace `
  genorefine-spagcn:1.2.7-panel-v1 `
  python -m pytest revision_pipeline/spatial_multisection/tests/test_spagcn_compat.py -q

& .\revision_pipeline\spatial_multisection\run_panel.ps1 `
  -Prefix 20260929-spatial-panel-v1 -PlanOnly
```

## Full run command

```powershell
& .\revision_pipeline\spatial_multisection\run_panel.ps1 `
  -Prefix 20260929-spatial-panel-v1
```

Use the same command and prefix to resume. Every completed child is deeply revalidated, including artifacts, config, and the non-null current source-tree hash. A stale incomplete directory is preserved under a `.retired-*` name before that job is retried. An exclusive prefix lock prevents two orchestrators from running the same panel concurrently.

## Expected work

The orchestration creates 131 completed run directories:

- 1 fixed-10 Harmony baseline;
- 1 label-free K-selection run;
- 5 pooled Harmony+GenoRefine training runs;
- 30 histology-aware SpaGCN training runs (6 sections x 5 seeds);
- 72 within-section score runs;
- 21 donor-pair mixing score runs;
- 1 consolidation run.

The final consolidated output is `revision_pipeline/runs/20260929-spatial-panel-v1-full-panel`. Five seeds are descriptive algorithmic repeats, not biological replicates, and the consolidation reports no p-values. Native-stop Harmony is retained only as a named sensitivity and is never counted as an algorithmic seed replicate.
