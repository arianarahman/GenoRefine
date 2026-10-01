# Repository utility scripts

These scripts maintain the public repository's content policy and integrity
manifest. Run them from the repository root.

## Check release contents

```bash
python scripts/check_release.py
```

`check_release.py` scans every file in the checkout except `.git`, including
untracked and ignored files. It checks for unexpected research-data formats,
local paths, credential patterns, unreviewed CSV files, oversized files, and
publication images outside the designated `figures/` directory. Run it in a
clean release checkout that does not contain separately acquired inputs or
local run artifacts.

## Maintain the SHA-256 manifest

After all reviewed source and documentation changes are complete, regenerate
the manifest:

```bash
python scripts/write_manifest.py
```

Verify an existing manifest without changing it:

```bash
python scripts/write_manifest.py --check
```

`MANIFEST.sha256` covers the released files and is also checked by the GitHub
Actions workflow in `.github/workflows/release-guard.yml`.

Run the content check before regenerating the manifest so unreviewed files are
not added to a release inventory.
