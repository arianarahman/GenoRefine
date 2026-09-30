# Path sanitization

Machine-local project prefixes and Python interpreter locations in the public
result/config receipts were normalized to `<PROJECT_ROOT>` and `<PYTHON_ENV>`
placeholders. Release-source launch defaults were likewise changed from one
workstation's absolute paths to environment variables or the active Python
interpreter. Scientific algorithms, parameters, reported measurements, and
result values were not changed.

Consequently, source-tree hashes recorded by completed runs identify the
immutable execution source, whereas this public source tree has a distinct
hash because of these portability-only edits. Original execution hashes remain
in the package reports and source manifests.

The following retained artifacts were sanitized after execution. Their
original hashes remain in the immutable `run.json` records; the public hashes
below identify the released, path-normalized copies.

| Public artifact | Original execution SHA-256 | Public SHA-256 |
|---|---|---|
| `package2_external_markers/foundation/source_receipts.json` | `9aaae2ce7e3c4ae713375831d4a86b371cb486f0709d46df2abdce4ce47d48a5` | `455d122487c80dc6db29d0046a3ee4db41ecdb4b23af13b0870bdd25fbb9566b` |
| `package4_spatial/config.json` | `e086bc58edd6aef6397e9af23dc860ba30ab50c8c0204e8662300d096b556f0f` | `03d324aa3e530495255d66da3c70a5a443397f622aef264d5b84124f9a9d5969` |
| `package4b_graphst/config.json` | `59b26b76483628653b82ad4a3c34257842d099e8cea4dcd1bc70716aa11f8df6` | `9b7bfb4ff8bb839c1afe50caee465109b4b8dc211403490f9a2b9eb325d9315d` |

These are the only retained result/config files whose byte-level artifact hash
differs from the recorded execution artifact because of path sanitization.
Metric arrays, summary values, scientific parameters, and the native
embedding precision were not changed. Verification of the public snapshot
should use the repository-level `MANIFEST.sha256`; verification against the
execution manifests requires the original unsanitized artifacts.
