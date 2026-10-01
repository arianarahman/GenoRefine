# Purpose: Deterministic real-embedding artifact construction and validation metrics.
# Author: Ariana Rahman (Arizona State University)

"""Deterministic real-embedding artifact construction and validation metrics.

This package contains pure numerical primitives.  It does not select datasets,
fit a refiner, launch scientific jobs, or change manuscript claims.
"""

from .artifacts import (
    ArtifactBundle,
    batch_simplex_artifact,
    regular_simplex_directions,
    target_local_warp_artifact,
)
from .metrics import (
    clean_neighbor_recovery,
    counterfactual_batch_sensitivity,
    evaluate_prespecified_pass_rule,
    preservation_metrics,
)

__all__ = [
    "ArtifactBundle",
    "batch_simplex_artifact",
    "regular_simplex_directions",
    "target_local_warp_artifact",
    "clean_neighbor_recovery",
    "counterfactual_batch_sensitivity",
    "evaluate_prespecified_pass_rule",
    "preservation_metrics",
]
