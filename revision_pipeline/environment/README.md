# Reproducible environments

This directory contains the pinned Python inventories and container recipes
used by the GenoRefine workflows. Select the environment that matches the
specific pipeline stage instead of combining every dependency into one
environment.

## Environment index

| File | Intended use |
|---|---|
| `requirements-wsl-cpu.lock.txt` | Core CPU GenoRefine refinement and generated-data smoke workflow |
| `requirements-wsl-step3a-backbones.lock.txt` | Upstream integration-backbone generation |
| `requirements-wsl-step3b-historical.lock.txt` | Historical evaluation compatibility |
| `requirements-wsl-step3b-evaluation.lock.txt` | Current evaluation and metric workflows |
| `requirements-step2.in` | Direct core-refiner requirements used to define the CPU environment |
| `requirements-step3a-backbones.in` | Direct backbone-generation requirements |
| `requirements-step3b-historical.in` | Direct historical-evaluation requirements |
| `requirements-step3b-evaluation.in` | Direct current-evaluation requirements |
| `Dockerfile.gpu` | NVIDIA TensorFlow runtime for GPU GenoRefine training |
| `Dockerfile.scvi` | NVIDIA PyTorch runtime for scVI workflows |

The spatial comparators have dedicated files beside their implementations:

- `revision_pipeline/spatial_multisection/Dockerfile.spagcn-gpu`
- `revision_pipeline/spatial_multisection/requirements-spagcn-gpu.txt`
- `revision_pipeline/spatial_graphst/Dockerfile.graphst-gpu`
- `revision_pipeline/spatial_graphst/requirements-graphst-gpu.txt`

## CPU setup

The validated lock files target Ubuntu/WSL and Python 3.12. For the core
refiner:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r revision_pipeline/environment/requirements-wsl-cpu.lock.txt
python -m revision_pipeline.refine smoke
```

Create a separate virtual environment when switching to a different lock file.
The pipeline records package inventories in run metadata so the active
environment can be compared with its protocol.

## GPU and comparator environments

Build containers from the repository root so the Docker build context includes
the required source tree. The package execution guides provide the required
image tags, runtime arguments, and preflight commands:

- [`../spatial_multisection/EXECUTION.md`](../spatial_multisection/EXECUTION.md)
- [`../spatial_graphst/EXECUTION.md`](../spatial_graphst/EXECUTION.md)

The main benchmark PowerShell wrappers use environment variables such as
`GENOREFINE_WSL_DISTRO`, `GENOREFINE_PRIMARY_PYTHON`,
`GENOREFINE_EVALUATION_PYTHON`, and `GENOREFINE_HARMONY_PYTHON` to select the
recorded interpreters without embedding machine-specific paths in source code.
