# Mouse atlas benchmark workflow

This directory contains the canonical feature-selection, integration,
refinement, and evaluation workflow for the mouse atlas benchmark.

Run the commands below with `MouseDataset` as the working directory. The Python
environment must provide the packages imported by the selected scripts,
including NumPy, pandas, SciPy, AnnData, Scanpy, Scanorama, scikit-learn,
TensorFlow/Keras, and GenoMap. The R comparison scripts use Seurat, Matrix,
`zellkonverter`, `SingleCellExperiment`, and `rliger`.

## Input

Place the registered AnnData file at:

```text
Dataset/tabula-muris_sub50k_combined.h5ad
```

The dataset registry and data-handling guidance are in
[`../revision_pipeline/configs/datasets.json`](../revision_pipeline/configs/datasets.json)
and [`../docs/data_sources.md`](../docs/data_sources.md).

## Run the workflow

Generate the canonical highly variable gene set:

```bash
python make_mouse_canonical.py
```

Generate the Seurat and Online iNMF comparison embeddings:

```bash
Rscript seurat_integration_mouse.R
Rscript rliger_online_inmf_mouse.R
```

Run the benchmark:

```bash
python bench_mouse_master.py
```

The workflow writes canonical feature records, comparison embeddings, benchmark
tables, plots, and logs to `Benchmark_Mouse_Out/`.

## Custom location

The R scripts read `GENOREFINE_DATASET_ROOT` when it is set; otherwise they use
the current working directory. The Python scripts use repository-relative
defaults defined in their configuration dataclasses.
