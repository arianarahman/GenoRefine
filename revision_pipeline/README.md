# GenoRefine revision pipeline

`revision_pipeline` contains the reusable GenoRefine implementation and the
provenance-aware workflows used to prepare, train, evaluate, compare, and
report the manuscript analyses.

Run module commands from the repository root so all configuration and artifact
paths resolve consistently.

## Pipeline organization

| Area | Modules | Purpose |
|---|---|---|
| Core refinement | `refine/` | Structured layout, autoencoder training, joint refinement, transformation, and model bundles |
| Data foundation | `data/`, `audit.py`, `integrity.py`, `runs.py` | Registered inputs, fingerprints, validation, and atomic run publication |
| Evaluation | `evaluate/` | Shared clustering, mixing, neighborhood, and contrast evaluation |
| Main benchmark | `pilot/`, `pre_step4/`, `step4/`, `main_benchmark/` | Acceptance checks and coordinated benchmark execution |
| Comparator analyses | `independent_comparator/`, `matched_comparators/`, `scvi_backbones/`, `scvi_comparator/` | IDEC, GenoM-01, and scVI comparison workflows |
| Biological evaluation | `biological_preservation/` | Marker, expression, purity, and geometry analyses |
| Validation panels | `artifact_validation/`, `broader_validation/`, `inductive_validation/` | Additional prespecified evaluation workflows |
| Ablations | `focused_ablation/`, `objective_ablation/` | Model-component and objective analyses |
| Spatial workflows | `spatial_multisection/`, `spatial_graphst/` | LIBD DLPFC preparation, fitting, scoring, and consolidation |
| Reporting | `manuscript_figures/` | Audited manuscript figure and table builders |
| Protocols | `configs/` | Versioned dataset, method, runtime, and evaluation contracts |
| Environments | `environment/` | Pinned Python inventories and container recipes |
| Third-party snapshot | `vendor/` | Fingerprinted GenoMap source snapshot and upstream license |

## Common entry points

Display the foundation command help:

```bash
python -m revision_pipeline --help
```

Run the generated-data core acceptance workflow:

```bash
python -m revision_pipeline.refine smoke
```

Inspect and fingerprint all inputs registered by the dataset registry:

```bash
python -m revision_pipeline audit --hash-inputs
```

Verify an existing imported data store:

```bash
python -m revision_pipeline.data verify path/to/store
```

Acquire and verify the public spatial inputs:

```bash
python -m revision_pipeline.spatial_multisection.source_acquisition --mode acquire --execute
python -m revision_pipeline.spatial_multisection.source_acquisition --mode verify
```

The full study contains multiple environment-specific packages. Start with
[`../docs/reproduction.md`](../docs/reproduction.md), then use the matching
configuration and execution guide for the package being run.

## Configurations and environments

Scientific settings are read from explicit JSON files under
[`configs/`](configs/). Dependency locks and container definitions are under
[`environment/`](environment/), with additional comparator-specific container
files alongside the spatial modules.

Use a separate environment for each lock family. This keeps the core TensorFlow
runtime, integration backbones, evaluation stack, and spatial comparators
aligned with their recorded protocols.

## Run directories

Generated work is stored beneath `revision_pipeline/runs/`. Pipeline modules
write a run manifest, configuration, source fingerprints, logs, and declared
artifacts into each run directory. Orchestrated workflows use unique run IDs
and validate completed children before reuse.

The `runs/` tree is intentionally ignored by Git. Reviewed aggregate outputs
selected for the public repository are copied to [`../results/`](../results/).

## Tests and preflights

Unit tests are colocated with the modules they exercise. Run an individual
suite from the repository root, for example:

```bash
python -m pytest revision_pipeline/refine/tests -q
python -m pytest revision_pipeline/evaluate/tests -q
python -m pytest revision_pipeline/spatial_multisection/tests -q
```

Use the environment required by the selected module. Spatial execution details
are documented in:

- [`spatial_multisection/EXECUTION.md`](spatial_multisection/EXECUTION.md)
- [`spatial_graphst/EXECUTION.md`](spatial_graphst/EXECUTION.md)
