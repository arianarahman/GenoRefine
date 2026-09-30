# Data sources and acquisition

No raw or cell/spot-level data are shipped with this release. Source locks separate two questions:

1. **Are these the expected bytes?** Size and SHA-256 checks answer this.
2. **Is the biological provenance complete?** A checksum cannot answer this; that status is recorded separately in the provenance configs.

## Fully locked public sources

| Use | Public source | Authoritative lock | Verification code |
|---|---|---|---|
| Package 2 marker panel | PanglaoDB markers (2020-03-27), CellMarker 2.0 workbook, and Cell Ontology release 2022-09-15 | `revision_pipeline/configs/external_marker_sources_v1.json` | `revision_pipeline/biological_preservation/source_acquisition.py` |
| Packages 4/4b spatial foundation | LIBD Human DLPFC Visium sections 151507, 151508, 151669, 151670, 151673, and 151674; study DOI `10.1038/s41593-020-00787-0` | `revision_pipeline/configs/spatial_multisection_v1.json` | `revision_pipeline/spatial_multisection/source_acquisition.py` |

The marker lock records an HTTPS URL, filename, exact byte count, SHA-256 digest, and a schema check for every file. The DLPFC lock records those fields for counts, positions, histology images, scale factors, manual labels, and the source manifest. HumanPilot-hosted files are pinned to repository commit `044446d6bd8fc154aa74f7be62ec67effb1ec376`.

From the repository root, acquire and verify the marker sources:

```text
python -c "from pathlib import Path; from revision_pipeline.biological_preservation.source_acquisition import load_source_locks, acquire_locked_source; locks=load_source_locks('revision_pipeline/configs/external_marker_sources_v1.json'); destination=Path('revision_pipeline/data/external/marker_sources_v1'); [acquire_locked_source(lock, destination) for lock in locks]"
```

Acquire or verify the spatial files:

```text
python -m revision_pipeline.spatial_multisection.source_acquisition --mode acquire --execute
python -m revision_pipeline.spatial_multisection.source_acquisition --mode verify
```

Downloads are published only after byte-count and SHA-256 checks pass. Marker sources also undergo their declared schema checks; DLPFC structure is validated during foundation preparation. Existing mismatching files fail closed, and replacement requires the acquisition module's explicit repair option.

## Processed primary benchmark inputs

The registry names four processed inputs but does not distribute them:

| Registry ID | Processed input family | Frozen provenance status |
|---|---|---|
| `pancreas_five_study` | Five-study pancreas MAT files and numeric class partition | Accession, feature/cell identity mapping, preprocessing, and the complete class-name map remain pending. |
| `hpcb` | HP-CB pancreas H5AD | Download/accession and exact lineage remain pending; contributing studies overlap the five-study pancreas cohort. |
| `mouse_senis` | 50,000-cell Tabula Muris Senis-derived H5AD | Original releases and executed subsampling lineage remain pending. |
| `pbmc_control` | Single-batch PBMC H5AD | Original counts and authoritative annotation-generation lineage remain pending. |

`revision_pipeline/configs/datasets.json` and `annotation_provenance_v1.json` define those boundaries. `step3a_source_lock.json` records exact local byte hashes, but explicitly states that hashes do not establish biological provenance. Do not substitute a similarly named public file and call it an exact reproduction.

Package 2 also requires the official Tabula Muris droplet/FACS label sources named in `independent_marker_foundation_v1.json`. Those processed files are not distributed and currently lack an automated public download lock in this snapshot.
