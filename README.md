# GenoRefine

GenoRefine is a post-integration refinement framework for single-cell
RNA-sequencing embedding geometry. It accepts an existing integrated cell
representation, maps its dimensions to a structured two-dimensional layout,
and learns a compact representation for clustering and downstream evaluation.

**Code author:** Ariana Rahman (Arizona State University)

## Project goals

- Refine embeddings produced by established single-cell integration methods.
- Support cluster-aware representation learning with explicit evaluation of
  batch mixing, neighborhood structure, biological-marker preservation,
  robustness, and spatial organization.
- Make benchmark workflows reproducible through versioned configurations,
  pinned environments, run manifests, integrity checks, and aggregate outputs.
- Connect publication figures and displayed values to machine-readable source
  records.

## How GenoRefine works

1. An upstream integration method supplies a cells-by-dimensions embedding.
2. GenoMap-based layout code organizes embedding dimensions on a structured
   two-dimensional grid.
3. A convolutional autoencoder learns a compact latent representation.
4. Joint reconstruction and clustering optimization produces the refined
   representation.
5. Paired evaluation compares the upstream and refined representations on the
   endpoints selected for the study.

The reusable staged implementation is in
[`revision_pipeline/refine/`](revision_pipeline/refine/). The repository also
contains the configurations and workflows used for the associated benchmark
analyses.

## Repository organization

| Path | Purpose |
|---|---|
| `revision_pipeline/refine/` | Core layout, model, training, transformation, and model-bundle code |
| `revision_pipeline/configs/` | Dataset, benchmark, comparator, and evaluation specifications |
| `revision_pipeline/environment/` | Python dependency locks and container recipes |
| `HPCBDataset/` | HP-CB pancreas preprocessing and benchmark workflow |
| `MouseDataset/` | Mouse atlas preprocessing and benchmark workflow |
| `PancreasDataset/` | Five-study pancreas preprocessing and benchmark workflow |
| `PBMCSingeBatchDataset/` | Single-batch PBMC validation and sensitivity workflows |
| `results/` | Published aggregate result packages and run metadata |
| `figures/` | Publication figures, tables, and provenance manifests |
| `docs/` | Data acquisition, reproduction, provenance, and third-party guidance |
| `scripts/` | Repository integrity and release-manifest utilities |

## Requirements

The pinned CPU workflow targets Python 3.12 in Linux or Windows Subsystem for
Linux (WSL). Dataset-specific comparison workflows also use R. GPU protocols
use Docker with an NVIDIA-compatible runtime. Exact environment choices are
documented in
[`revision_pipeline/environment/README.md`](revision_pipeline/environment/README.md).

## Quick start

From a Linux or WSL shell:

```bash
git clone https://github.com/arianarahman/GenoRefine.git
cd GenoRefine
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r revision_pipeline/environment/requirements-wsl-cpu.lock.txt
python -m revision_pipeline.refine smoke
```

The smoke command uses generated data and writes a provenance-bearing run
directory beneath `revision_pipeline/runs/`.

For a code example that refines an existing embedding, see
[`revision_pipeline/refine/README.md`](revision_pipeline/refine/README.md).

## Verify a clean release checkout

The following checks use the repository root as the working directory. Run the
release guard in a clean checkout without separately acquired datasets or local
run artifacts; it intentionally examines ignored and untracked files too.

```bash
python scripts/check_release.py
python scripts/write_manifest.py --check
python -m compileall -q revision_pipeline HPCBDataset MouseDataset PancreasDataset PBMCSingeBatchDataset
```

## Run the research workflows

1. Follow [`docs/data_sources.md`](docs/data_sources.md) to acquire the required
   inputs.
2. Select the matching dependency lock or container recipe from
   [`revision_pipeline/environment/`](revision_pipeline/environment/).
3. Place inputs at the repository-relative locations declared in
   `revision_pipeline/configs/data_store.json` or in the applicable
   dataset-specific `Dataset/` directory.
4. Inspect the registered inputs with:

   ```bash
   python -m revision_pipeline audit --hash-inputs
   ```

5. Follow [`docs/reproduction.md`](docs/reproduction.md) and the README in the
   relevant dataset or pipeline directory for the exact command sequence.

New pipeline runs are written beneath `revision_pipeline/runs/`. Published
aggregate outputs are indexed in [`results/README.md`](results/README.md), and
publication assets are indexed in [`figures/README.md`](figures/README.md).

## Data

Research inputs are acquired separately and kept outside version control.
[`data/README.md`](data/README.md) describes the expected layout, while
[`docs/data_sources.md`](docs/data_sources.md) records source acquisition and
verification procedures.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). GitHub also
exposes these metadata through **Cite this repository**.

## License

See [`LICENSE_PENDING.md`](LICENSE_PENDING.md) for the current repository-wide
licensing notice. Bundled third-party material remains under the terms
identified in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md),
[`docs/third_party.md`](docs/third_party.md), and the preserved upstream
license texts under [`third_party/`](third_party/README.md).
