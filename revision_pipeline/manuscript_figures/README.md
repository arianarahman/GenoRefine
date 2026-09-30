# Audited manuscript-asset builders

`build_package12_assets.py` and `build_package34_assets.py` are the exact
downstream builders recorded by the released figure manifests. They validate
completed scientific-run artifacts, recompute displayed aggregates from saved
rows, and write manuscript-ready figures, tables, and provenance records. They
do not train models or edit LaTeX.

The public release deliberately omits full run directories and cell/spot-level
artifacts. Consequently, the builders are provided for code transparency and
can be rerun only after the source runs have been reconstructed at the run IDs
declared in the scripts and provenance manifests. The already audited outputs
are distributed under the repository-level `figures/` directory.

With the complete source runs restored, invoke the builders from the project
root:

```text
python revision_pipeline/manuscript_figures/build_package12_assets.py
python revision_pipeline/manuscript_figures/build_package34_assets.py
```

Their default destinations are ignored local staging directories under this
folder. Copy only reviewed outputs, together with their manifest and
number-provenance files, into a release bundle.
