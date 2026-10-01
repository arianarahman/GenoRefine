# Five-study pancreas benchmark workflow

This directory contains the canonical feature-selection, integration,
refinement, evaluation, robustness, focus-test, and scalability workflows for
the five-study pancreas benchmark.

Run the commands below with `PancreasDataset` as the working directory. The
Python environment must provide the packages imported by the selected scripts,
including NumPy, pandas, SciPy, AnnData, Scanpy, Scanorama, scikit-learn,
TensorFlow/Keras, and GenoMap. The R comparison scripts use Seurat, Matrix,
`R.matlab`, and `rliger`.

## Inputs

Place these registered MATLAB files in `Dataset/`:

```text
Dataset/dataBaronX.mat
Dataset/dataMuraroX.mat
Dataset/dataScapleX.mat
Dataset/dataWangX.mat
Dataset/dataXinX.mat
Dataset/classLabel.mat
```

The dataset registry and data-handling guidance are in
[`../revision_pipeline/configs/datasets.json`](../revision_pipeline/configs/datasets.json)
and [`../docs/data_sources.md`](../docs/data_sources.md).

## Main workflow

Generate the canonical highly variable gene set:

```bash
python make_pancreas_master_hvg.py
```

Generate the Seurat and Online iNMF comparison embeddings:

```bash
Rscript seurat_integration_pancreas_master.R
Rscript rliger_online_inmf_pancreas_master.R
```

Run the main benchmark:

```bash
python bench_pancreas_multi_master.py
```

The workflow writes canonical feature records, comparison embeddings, benchmark
tables, plots, and logs to `Benchmark_Out/`.

## Robustness workflow

Create the shared cell manifests, generate the corresponding R embeddings, and
run the robustness benchmark:

```bash
python make_pancreas_robustness_hvg.py
Rscript seurat_integration_pancreas_robustness.R
Rscript rliger_online_inmf_pancreas_robustness.R
python bench_pancreas_multi_robustness_genorefine_core.py
```

The shared manifests and R embeddings are stored beneath
`Benchmark_Out/Robustness_R_Embeddings/`.

## Supporting analyses

After completing the main benchmark, run the focus and scalability analyses as
needed:

```bash
python focus_tests_pancreas_genorefine_core.py
python bench_pancreas_scalability.py
```

## Custom location

The R scripts read `GENOREFINE_DATASET_ROOT` when it is set; otherwise they use
the current working directory. The Python scripts use repository-relative
defaults defined in their configuration dataclasses.
