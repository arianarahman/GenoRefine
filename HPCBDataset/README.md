# HP-CB pancreas benchmark workflow

This directory contains the preprocessing, integration, refinement, evaluation,
and robustness scripts for the HP-CB pancreas benchmark.

Run the commands below with `HPCBDataset` as the working directory. The Python
environment must provide the packages imported by the selected scripts,
including NumPy, pandas, SciPy, AnnData, Scanpy, Scanorama, scikit-learn,
TensorFlow/Keras, and GenoMap. The R comparison scripts use Seurat, Matrix,
`zellkonverter`, `SingleCellExperiment`, and `rliger`.

## Input

Place the registered AnnData file at:

```text
Dataset/human_pancreas_norm_complexBatch.h5ad
```

The dataset registry and data-handling guidance are in
[`../revision_pipeline/configs/datasets.json`](../revision_pipeline/configs/datasets.json)
and [`../docs/data_sources.md`](../docs/data_sources.md).

## Main workflow

Generate the canonical highly variable gene set:

```bash
python make_hie_canonical.py
```

Generate the Seurat and Online iNMF comparison embeddings:

```bash
Rscript seurat_integration_hie.R
Rscript rliger_online_inmf_hie.R
```

Run the complete benchmark and the supporting focus analyses:

```bash
python bench_hie_master.py
python focus_tests_hie.py
```

The scripts write their primary outputs to `Benchmark_Hie_Out/`.

## Robustness workflow

Create the shared cell manifests, generate R comparison embeddings for those
manifests, and run the Python benchmark:

```bash
python make_hie_robustness.py
Rscript seurat_integration_hie_robustness.R
Rscript rliger_online_inmf_hie_robustness.R
python bench_hie_robustness.py
```

Robustness artifacts are organized beneath
`Benchmark_Hie_Out/Robustness_R_Embeddings/` and the output directory declared
by `bench_hie_robustness.py`.

## Custom location

Python preparation and robustness scripts accept `--base-dir` or the
`HIE_BASE_DIR` environment variable. The R scripts accept
`GENOREFINE_DATASET_ROOT`. When neither is set, the current working directory
is used.

Display available Python options with, for example:

```bash
python make_hie_canonical.py --help
python bench_hie_robustness.py --help
```
