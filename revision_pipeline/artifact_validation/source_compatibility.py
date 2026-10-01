# Purpose: Narrow provenance bridge for the artifact scorer-only recovery.
# Author: Ariana Rahman (Arizona State University)

"""Narrow provenance bridge for the artifact scorer-only recovery.

The artifact panel completed all model fits under one frozen source tree before
candidate scoring exposed a dtype mismatch in an integrity check.  Repeating
the fits would not change the learned artifacts.  This module permits those
immutable fits to be scored only when the earlier and current source manifests
differ in the small, explicit recovery surface below.  Any other difference
fails closed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..data.readers import array_hash
from ..integrity import canonical_hash, file_fingerprint


RECOVERY_ID = "artifact_validation_expected_observed_float64_hash_v1"
REFERENCE_SOURCE_TREE_SHA256 = (
    "e83af7b23ae9a553d6b28e12dd1ed967f3613f713fa7f3e47e5acaaa3cab8c46"
)
ALLOWED_CHANGED_PATHS = frozenset(
    {
        "revision_pipeline/artifact_validation/score.py",
        "revision_pipeline/artifact_validation/consolidate.py",
        "revision_pipeline/artifact_validation/source_compatibility.py",
        "revision_pipeline/artifact_validation/tests/test_pipeline.py",
    }
)


def expected_observed_hash(values: np.ndarray) -> str:
    """Hash in the float64 validation domain used by training receipts."""
    return array_hash(np.asarray(values, dtype=np.float64))


def _manifest_difference(
    reference: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    reference_paths, current_paths = set(reference), set(current)
    added = sorted(current_paths - reference_paths)
    removed = sorted(reference_paths - current_paths)
    changed = sorted(
        path
        for path in reference_paths & current_paths
        if reference[path] != current[path]
    )
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "fingerprints": {
            path: {
                "reference": reference.get(path),
                "current": current.get(path),
            }
            for path in sorted(set(added) | set(removed) | set(changed))
        },
    }


def verify_scoring_source_transition(
    reference_run: Path | str,
    reference_source_tree_sha256: str,
    current_sources: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    """Verify and describe an exact or narrowly allowed scoring transition."""
    reference_run = Path(reference_run).resolve()
    manifest_path = reference_run / "source_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Reference run has no source manifest")
    reference_sources = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_reference_hash = canonical_hash(reference_sources)
    if actual_reference_hash != reference_source_tree_sha256:
        raise ValueError("Reference source manifest does not match its run manifest")

    current_sources = dict(current_sources)
    current_hash = canonical_hash(current_sources)
    difference = _manifest_difference(reference_sources, current_sources)
    different_paths = set(difference["added"]) | set(difference["removed"]) | set(
        difference["changed"]
    )
    if actual_reference_hash == current_hash:
        if different_paths:
            raise RuntimeError("Equal source-tree hashes unexpectedly contain differences")
        status = "exact_source_match"
    else:
        if actual_reference_hash != REFERENCE_SOURCE_TREE_SHA256:
            raise ValueError("Source transition is not from the frozen recovery snapshot")
        unauthorized = sorted(different_paths - ALLOWED_CHANGED_PATHS)
        if unauthorized:
            raise ValueError(
                "Source transition includes changes outside the scoring recovery: "
                + ", ".join(unauthorized)
            )
        if "revision_pipeline/artifact_validation/score.py" not in different_paths:
            raise ValueError("Recovery transition does not include the declared scorer fix")
        if difference["removed"]:
            raise ValueError("Recovery transition may not remove frozen source files")
        status = "verified_scorer_only_recovery"

    return {
        "recovery_id": RECOVERY_ID,
        "status": status,
        "role": role,
        "reference_source_tree_sha256": actual_reference_hash,
        "current_source_tree_sha256": current_hash,
        "reference_source_manifest": file_fingerprint(manifest_path),
        "difference": difference,
        "allowed_changed_paths": sorted(ALLOWED_CHANGED_PATHS),
        "upstream_artifacts_reused_without_mutation": True,
        "scientific_boundary": (
            "Previously completed model outputs are immutable. The transition only "
            "canonicalizes the observed-output integrity hash to the float64 dtype "
            "already used when the training receipt was written and adds provenance, "
            "tests, orchestration bookkeeping, and consolidation checks."
        ),
    }
