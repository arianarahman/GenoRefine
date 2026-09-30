# Data are not distributed

This directory is intentionally empty except for this notice. **The GenoRefine release contains no raw, cell-level, or spot-level data.** In particular, it excludes count matrices, expression objects, cell/spot annotations, per-cell embeddings, spatial images, learned weights, and scientific run directories.

Public ancillary sources and the LIBD DLPFC inputs can be acquired with checksum-enforcing code described in [`../docs/data_sources.md`](../docs/data_sources.md). Several processed primary benchmark inputs do not yet have a complete public acquisition chain in the frozen registry; their recorded SHA-256 values establish byte identity only and must not be presented as proof of biological provenance.

Keep all downloaded and generated material outside version control. The repository `.gitignore` blocks common genomics formats, arrays, models, runs, images outside `figures/`, caches, local environments, logs, secrets, and machine-specific path files. Before committing, still inspect `git status` manually; ignore rules are a guardrail, not a data-disclosure control.
