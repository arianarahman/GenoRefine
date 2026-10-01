# Single-batch PBMC workflows

This directory contains biological-validation and parameter-sensitivity
workflows for a single-batch PBMC dataset. The directory name is retained as
`PBMCSingeBatchDataset` so existing repository-relative paths continue to
resolve.

Run the commands below with `PBMCSingeBatchDataset` as the working directory.
The Python environment must provide the packages imported by the selected
scripts, including NumPy, pandas, SciPy, AnnData, Scanpy, Scanorama,
scikit-learn, TensorFlow/Keras, and GenoMap.

## Input

Place the registered AnnData file at:

```text
Dataset/pbmcs_ctrl_labeled.h5ad
```

The dataset registry and data-handling guidance are in
[`../revision_pipeline/configs/datasets.json`](../revision_pipeline/configs/datasets.json)
and [`../docs/data_sources.md`](../docs/data_sources.md).

## Biological-validation workflow

```bash
python bench_pbmc_biological_validation_genorefine.py
```

This script writes marker-audit tables, differential-expression consistency
tables, and figures to `Biological_Validation_Results_PBMC_SingleBatch/`.

## Sensitivity workflow

```bash
python bench_pbmc_sensitivity.py
```

This script writes the parameter grid, aggregate sensitivity table, and plot to
`Sensitivity_Results/`.

Both scripts use repository-relative defaults declared near the beginning of
their source files. Run them from this directory so the input and output paths
resolve as documented.
