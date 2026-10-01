# Public-release maintenance checklist

Use this checklist whenever the public GenoRefine repository or a versioned
release is updated.

1. Review the repository-wide licensing notice. When a license is selected by
   the authors and institutions, replace `LICENSE_PENDING.md` with `LICENSE`,
   add the corresponding SPDX identifier to `CITATION.cff`, and retain all
   third-party notices.
2. Review the bundled GenoMap snapshot and the other pinned components listed
   in `docs/third_party.md` whenever their contents or versions change.
3. Confirm that tracked files contain source code, documentation, reviewed
   aggregate outputs, and publication assets only. Keep research inputs,
   observation-level exports, trained weights, caches, and local run
   directories outside version control.
4. Audit the history of every public branch and tag, not only the candidate
   working tree. The current data-free release candidate is an orphan snapshot,
   while the existing remote history contains earlier derived outputs. A normal
   cleanup commit will not remove those historical objects. Before publication,
   choose and review either an orphan-history replacement or a newly created
   repository; coordinate any history replacement with collaborators, preserve
   the prior history privately when policy requires it, and verify the remote
   branch and tag graph after publication.
5. Run the release-content guard from the repository root:

   ```text
   python scripts/check_release.py
   ```

6. After all reviewed file changes are complete, regenerate and verify the
   repository manifest:

   ```text
   python scripts/write_manifest.py
   python scripts/write_manifest.py --check
   ```

7. Compile the Python source tree and run the applicable unit and preflight
   tests for each changed workflow.
8. Confirm that the GitHub Actions release guard passes on the target commit.
9. For a tagged release, synchronize the version and release date in
   `CITATION.cff`, create an immutable tag, and record any archive DOI in
   `CITATION.cff`, the root README, and the manuscript's code-availability
   statement.
10. Recheck all links and commands from a clean checkout before announcing the
   release.

Preparation of this folder does not replace remote history, push a branch,
create a tag, or publish an archive.
