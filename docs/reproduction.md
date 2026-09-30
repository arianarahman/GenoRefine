# Reproduction guide

Reproduction has three distinct levels. The included aggregate artifacts can be inspected immediately; exact scientific reruns additionally require excluded inputs and frozen runtimes.

## 1. Inspect the released evidence

Use `results/README.md` as the package index. Each package contains a human-readable `report.md` plus structured JSON/CSV aggregates. Figure manifests and per-number provenance live under `figures/`.

The released tables are aggregates, including some per-seed or per-section records; they are not raw observations or per-cell/per-spot exports.

## 2. Acquire locked public dependencies

Follow `docs/data_sources.md` for marker and DLPFC acquisition. The download code verifies declared byte counts, SHA-256 digests, and schemas before publishing files.

Acquire the official GraphST and PASTE sources through the frozen archive locks:

```text
python -m revision_pipeline.spatial_graphst.source_acquisition --all --execute
```

The Package 1 IDEC and Package 4 SpaGCN compatibility paths require clean official checkouts at the repository-relative locations named by their configs. Use the exact commits in `docs/third_party.md`; do not use a moving branch or an unrecorded package release.

## 3. Recreate scientific runs

1. Create a clean environment from the relevant lock file under `revision_pipeline/environment/`, or build the package-specific Dockerfile.
2. Populate all excluded inputs at the repository-relative paths declared by `revision_pipeline/configs/data_store.json`.
3. Compare their sizes and SHA-256 hashes with `revision_pipeline/configs/step3a_source_lock.json`.
4. Run a non-training foundation audit before training (it writes a new ignored audit directory but does not modify the inputs):

   ```text
   python -m revision_pipeline audit --hash-inputs
   ```

5. Execute the package entry point with a new run prefix/ID. Generated runs belong under `revision_pipeline/runs/`, which is intentionally ignored.

Package-to-protocol mapping:

| Package | Frozen protocol/config | Principal entry points |
|---|---|---|
| 1 | `independent_idec_panel_v2.json` | `revision_pipeline.independent_comparator.run_idec`, scoring, then `consolidate_panel` |
| 2 | `independent_marker_foundation_v1.json` and `independent_marker_preservation_v1.json` | `build_marker_foundation`, then `analyze_all` |
| 3 | `artifact_validation_v2.json` | `revision_pipeline.artifact_validation.run_panel` |
| 4 | `spatial_multisection_v1.json` and `spatial_multisection_panel_v1.json` | `revision_pipeline/spatial_multisection/run_panel.ps1` |
| 4b | `spatial_graphst_panel_v1.json` | `revision_pipeline/spatial_graphst/run_panel.ps1` |

Plan the spatial orchestrators before execution:

```text
powershell -NoProfile -ExecutionPolicy Bypass -File revision_pipeline/spatial_multisection/run_panel.ps1 -Prefix <new-prefix> -PlanOnly
powershell -NoProfile -ExecutionPolicy Bypass -File revision_pipeline/spatial_graphst/run_panel.ps1 -Prefix <new-prefix> -PlanOnly
```

Read each spatial package's `EXECUTION.md` and run its preflight before a full panel. Preserve failed or incomplete attempts as evidence; do not relabel them as successful runs.

## Exactness limits

- The processed primary benchmark acquisition chain is incomplete; therefore this snapshot is not a turnkey end-to-end reproduction from public accessions.
- Frozen protocols may retain historical runtime identifiers and interpreter-location checks. Porting them silently would create a different protocol; record and review any portability change.
- Container image IDs, dependency locks, source commits, configs, inputs, seeds, and run manifests are all part of the computational identity.
- Successful execution reproduces a specified computation, not stronger biological provenance or a new biological replicate.
