# Core GenoRefine implementation

This package provides the staged refinement interface used throughout the
repository. It separates layout fitting, autoencoder pretraining, joint
refinement, transformation, and model serialization so each stage can be
recorded and inspected.

The public staged class is `StagedGenoDR`; GenoRefine is the project and method
name used by the surrounding workflows.

## Install the CPU environment

The pinned CPU environment targets Python 3.12 in Linux or WSL. From the
repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r revision_pipeline/environment/requirements-wsl-cpu.lock.txt
```

Run the generated-data acceptance workflow:

```bash
python -m revision_pipeline.refine smoke
```

## Input contract

The staged API expects:

- a finite numeric array with cells in rows and upstream embedding dimensions
  in columns;
- one stable, unique cell identifier per row;
- one stable feature identifier per embedding dimension; and
- an explicit cluster count together with a declaration of how it was chosen.

Training stages verify the input values, row order, cell identifiers, and
feature order against the fitted layout.

## Refine an embedding

The runtime profile must be configured before importing the TensorFlow-backed
staged model.

```python
from pathlib import Path

import numpy as np

from revision_pipeline.refine.config import (
    LayoutConfig,
    RefinerConfig,
    TrainingConfig,
)
from revision_pipeline.refine.runtime import configure_cpu

configure_cpu()

from revision_pipeline.refine.staged import StagedGenoDR


embedding = np.load("embedding.npy").astype(np.float64, copy=False)
cell_ids = [f"cell_{index}" for index in range(embedding.shape[0])]
feature_ids = [f"dimension_{index}" for index in range(embedding.shape[1])]

config = RefinerConfig(
    layout=LayoutConfig(
        requested_side=33,
        scaling="none",
        transport_iterations=200,
    ),
    training=TrainingConfig.for_replicate(
        0,
        n_clusters=10,
        cluster_count_source="label_free_external_rule",
        latent_dim=32,
        batch_size=64,
        pretrain_epochs=100,
        max_updates=300,
        target_update_interval=50,
    ),
)

output = Path("genorefine_run")
output.mkdir(parents=True, exist_ok=False)

model = StagedGenoDR(config)
model.fit_layout(embedding, cell_ids=cell_ids, feature_ids=feature_ids)
model.pretrain(
    embedding,
    cell_ids=cell_ids,
    feature_ids=feature_ids,
    directory=output / "pretrain",
)
refined_embedding = model.cluster(
    embedding,
    cell_ids=cell_ids,
    feature_ids=feature_ids,
    directory=output / "cluster",
)

np.save(output / "refined_embedding.npy", refined_embedding)
model.save(output / "model")
```

Choose `n_clusters` using the rule appropriate for the analysis and record that
choice in `cluster_count_source`. The accepted source labels are defined by
`TrainingConfig` in [`config.py`](config.py).

## Transform additional cells

A saved model bundle contains the fitted layout, model weights, configuration,
training identifiers, summaries, and artifact fingerprints. Load it and apply
the same ordered feature space to new cells:

```python
from revision_pipeline.refine.staged import StagedGenoDR

loaded = StagedGenoDR.load("genorefine_run/model")
new_refined_embedding = loaded.transform(
    new_embedding,
    cell_ids=new_cell_ids,
    feature_ids=feature_ids,
)
```

Configure the CPU or GPU runtime before importing `StagedGenoDR` in the new
process.

## Generated artifacts

The pretraining and clustering directories contain weights, training summaries,
visit counts, loss/target logs, refined features, and assignment probabilities.
`model.save()` writes a hash-checked inference bundle that can be loaded by
`StagedGenoDR.load()`.

Use a new output directory for each run; the staged API does not overwrite an
existing stage directory.
