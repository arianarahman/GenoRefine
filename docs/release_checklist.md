# Public-release checklist

The repository tree is prepared as a data-free release candidate. Complete
these governance steps before publishing it as `main`:

1. Select a repository-wide license with all authors and institutions, replace
   `LICENSE_PENDING.md` with `LICENSE`, and add the matching SPDX identifier to
   `CITATION.cff`.
2. Review the bundled, unmodified GenoMap snapshot and its CC BY-NC-ND 2.0
   terms for compatibility with the chosen repository license.
3. Decide how to replace the existing GitHub history. A normal cleanup commit
   is insufficient because earlier commits retain large derived outputs. Use a
   reviewed orphan-history replacement or recreate the repository; archive the
   old history privately if institutional policy requires it.
4. Run `python scripts/check_release.py`, regenerate `MANIFEST.sha256` with
   `python scripts/write_manifest.py`, and verify it with
   `python scripts/write_manifest.py --check` from a clean checkout.
5. Confirm that the GitHub Actions release guard passes, then create an
   immutable version tag. Update `CITATION.cff` with that version and release
   date.
6. If a Zenodo archive is created, add its DOI to `CITATION.cff`, the README,
   and the manuscript's Data and Code Availability statement.

No force-push, tag, release, or DOI has been created by preparation of this
folder.
