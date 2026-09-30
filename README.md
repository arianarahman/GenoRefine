# GenoRefine

GenoRefine is a post-integration refinement framework for single-cell RNA-sequencing embedding geometry. This repository is the public research-artifact snapshot for the GenoRefine analyses and is organized around frozen protocols, provenance-bearing aggregate results, and publication figures.

**No raw, cell-level, or spot-level data are distributed in this repository.** Count matrices, annotations, per-cell embeddings, trained models, run directories, and source images must be acquired or generated outside version control. The tracked `results/` and `figures/` trees contain aggregate outputs and derived visualizations only.

## Release contents

| Package | Question | Included aggregate evidence | Headline boundary/result |
|---|---|---|---|
| 1 | How does GenoRefine compare with an independently implemented, commit-pinned IDEC comparator? | `results/package1_idec/` and `figures/main_independent_comparator_12case*` | Twelve dataset-backbone cases, five algorithmic seeds per refiner; results show endpoint tradeoffs rather than a universal winner. |
| 2 | Are external-marker signals preserved? | `results/package2_external_markers/` and `figures/supp_external_marker_8case*` | All 88 prespecified representation evaluations completed; marker panels were fixed without reading evaluated embeddings or scores. |
| 3 | Does refinement meet a correction-and-preservation screen under injected artifacts? | `results/package3_artifacts/` and `figures/package34_audited/package3*` | The joint criterion was unmet in all 24 refiner-condition screens, defining a boundary for general artifact-correction claims. |
| 4 | What happens in a six-section, three-donor LIBD DLPFC spatial panel? | `results/package4_spatial/`, `results/package4_spagcn_native/`, and spatial assets under `figures/package34_audited/` | GenoRefine modestly increased donor-pair mixing, while layer and local-spatial endpoints favored the upstream representation or purpose-built spatial comparators; native partitions are reported separately. |
| 4b | How does the supporting PASTE + GraphST comparator behave? | `results/package4b_graphst/` and spatial assets under `figures/package34_audited/` | GraphST provided the stronger reported spatial/layer profile under an asymmetric donor-pair design and a complete-case cohort boundary. |

Detailed package inventories and key numbers are in [`results/README.md`](results/README.md). Figure-to-number provenance is described in [`figures/README.md`](figures/README.md).

## Evidence boundary

- Algorithmic seeds quantify run-to-run variation; they are not biological replicates.
- Package 3 is an exploratory injected-artifact panel, not proof of general correction.
- Packages 4 and 4b use manual layer values only for evaluation/model assessment, but the inherited cohort was restricted to spots with nonmissing layer labels. Fits are therefore label-blind conditional on label availability, not independent of it.
- Common spatial ARI/NMI use a fixed-resolution evaluator with different realized cluster counts across methods; they are not matched-*K* domain-recovery estimates.
- Several processed primary benchmark inputs have unresolved accession/preprocessing lineage in the frozen registry. Exact file hashes establish byte identity, not biological provenance.

See [`docs/provenance.md`](docs/provenance.md) for the full interpretation rules.

## Reproduction and data

Start with:

- [`data/README.md`](data/README.md) for the no-data policy;
- [`docs/data_sources.md`](docs/data_sources.md) for public source locks and acquisition;
- [`docs/reproduction.md`](docs/reproduction.md) for verification and rerun levels;
- [`docs/third_party.md`](docs/third_party.md) for pinned external methods and the bundled GenoMap notice.

The canonical repository is <https://github.com/arianarahman/GenoRefine>.
The final publication steps are listed in
[`docs/release_checklist.md`](docs/release_checklist.md).

## Citation and licensing

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). A repository-wide software license has **not** yet been selected. [`LICENSE_PENDING.md`](LICENSE_PENDING.md) records the action required before public release; it is not a software license. Bundled third-party material remains under its own terms.
