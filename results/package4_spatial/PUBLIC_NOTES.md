# Public interpretation notes

`report.md` is retained byte-for-byte as a frozen scientific-run artifact. Two
clarifications apply when interpreting the public release:

- Layer identities were withheld from fitting, model selection, and the
  label-free K selection based on fixed-resolution Harmony partitions. The
  benchmark cohort had already excluded 113 spots without a reference-layer
  value, so the fits are label-blind conditional on label availability.
- The common evaluator uses a fixed Leiden resolution rather than a matched
  cluster count. Realized cluster granularity differs across methods, so the
  reported common ARI/NMI values are fixed-resolution profiles rather than
  matched-K domain-recovery estimates.

These clarifications do not change the frozen run, metric values, or aggregate
tables. The separate `preflight/source_hash_scope_clarification.json` explains
the base/global versus Package-4-specific source-hash scope.
