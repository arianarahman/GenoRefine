# Data setup

GenoRefine keeps research inputs outside version control. This directory is a
documented location for data setup; downloaded inputs, generated arrays,
embeddings, model files, and local run directories are ignored by Git.

## Public source acquisition

Source URLs, version locks, expected byte counts, checksums, and verification
procedures are documented in [`../docs/data_sources.md`](../docs/data_sources.md).

From the repository root, acquire and verify the six-section LIBD DLPFC
materials with:

```bash
python -m revision_pipeline.spatial_multisection.source_acquisition --mode acquire --execute
python -m revision_pipeline.spatial_multisection.source_acquisition --mode verify
```

Acquire the pinned GraphST and PASTE source snapshots with:

```bash
python -m revision_pipeline.spatial_graphst.source_acquisition --all --execute
```

The external-marker acquisition code and its checksum registry are located at:

- `revision_pipeline/biological_preservation/source_acquisition.py`
- `revision_pipeline/configs/external_marker_sources_v1.json`

## Dataset-specific inputs

The benchmark workflows use these repository-relative input locations:

| Workflow | Input location |
|---|---|
| HP-CB pancreas | `HPCBDataset/Dataset/` |
| Mouse atlas | `MouseDataset/Dataset/` |
| Five-study pancreas | `PancreasDataset/Dataset/` |
| Single-batch PBMC | `PBMCSingeBatchDataset/Dataset/` |

Exact filenames are listed in each workflow's README and in
`revision_pipeline/configs/datasets.json` and `data_store.json`.

## Verify registered inputs

After the files are in place, inspect and hash the registered sources from the
repository root:

```bash
python -m revision_pipeline audit --hash-inputs
```

Generated audit records and scientific runs are written beneath
`revision_pipeline/runs/` and remain outside version control.

Before publishing repository changes, run `python scripts/check_release.py`
from a separate clean release checkout. The guard intentionally scans ignored
and untracked files as well as tracked files, so it will reject a working copy
that currently contains the separately acquired inputs described above.
