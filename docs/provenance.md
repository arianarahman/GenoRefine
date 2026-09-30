# Provenance and evidence boundaries

GenoRefine uses layered provenance so that byte identity, computational lineage, and scientific interpretation are not conflated.

## Provenance layers

1. **Source locks** record URLs or repository commits, byte counts, SHA-256 digests, and selected-file/schema checks.
2. **Protocol configs** freeze inputs, algorithms, hyperparameters, evaluation rules, seeds, and environment expectations.
3. **Run manifests** identify completed jobs and bind outputs to configs and source-tree fingerprints.
4. **Release aggregates** under `results/` provide reports plus machine-readable summaries, per-seed/per-section aggregate rows, completion receipts, and source manifests.
5. **Figure provenance** uses `asset_manifest.json` and `number_provenance.json` to connect displayed values to aggregate sources and aggregation contracts.

The canonical code repository is <https://github.com/arianarahman/GenoRefine>.

## Public-source portability normalization

The completed-run source hashes identify the immutable code used for the
scientific executions. Before public release, workstation-specific project
prefixes and interpreter paths were replaced with `<PROJECT_ROOT>`,
`<PYTHON_ENV>`, environment variables, or the active interpreter. These
portability-only edits necessarily give the public tree a different source
hash; they do not change scientific algorithms, hyperparameters, measurements,
or aggregate results. See `results/PATH_SANITIZATION.md`.

## What is included

The release includes aggregate JSON/CSV/Markdown outputs and derived PNG/PDF figures. It does not include raw data, cell/spot metadata, per-cell embeddings, model checkpoints, or full scientific run directories. A structured row for a seed, section, donor, method, or endpoint is an aggregate record, not an observation-level record.

This is a curated subset rather than a byte-for-byte copy of every completed run directory. The Package 4-family `run.json` records refer to seven detailed seed-row JSON artifacts that are not shipped; corresponding CSV rows and aggregate summaries are included. Released files and displayed values are covered by the included asset and number-provenance manifests, but a run manifest should not be read as a claim that every original run artifact is present here.

## Interpretation rules

- **Hashes are not provenance by themselves.** `step3a_source_lock.json` proves identity of inspected local bytes; it does not resolve accessions, preprocessing, annotations, or cohort lineage.
- **Seeds are descriptive.** Five-seed means and sample standard deviations describe algorithmic repeats, not biological replication. The release does not turn seed-level variation into biological inference.
- **No favorable selection.** Package reports retain prespecified cases and endpoints, including negative results.
- **Package 3 is scoped.** Its injected artifacts are fully observed test conditions. Zero of 24 joint correction-and-preservation screens passed; the panel cannot support a general artifact-correction claim.
- **Spatial labels have a conditional boundary.** Layer identities/values were withheld from fitting and label-free model selection, but 113 of 23,081 spots were excluded upstream because labels were missing. The analyzed 22,968-spot cohort is label-blind conditional on label availability.
- **Spatial tasks are asymmetric.** Harmony/GenoRefine use a pooled six-section representation, SpaGCN is fit per section with histology, and GraphST follows label-free PASTE alignment with donor-pair fits. SpaGCN has no cross-section mixing endpoint.
- **Common and native partitions differ.** Fixed-resolution common-evaluator ARI/NMI are not matched-*K* domain recovery. SpaGCN and GraphST native partitions are exported and labeled separately.
- **Scope is not universality.** These datasets, backbones, comparators, artifacts, and spatial sections do not establish performance for every integration method, tissue, or failure mode.

When reporting a value, cite the package report and retain its cohort, endpoint, seed, and aggregation qualifier. Prefer the number-provenance record when a figure value is involved.
