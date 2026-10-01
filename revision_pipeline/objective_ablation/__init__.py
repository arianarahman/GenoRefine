# Purpose: Matched loss-weight ablations from the saved Step 4A pretraining boundary.
# Author: Ariana Rahman (Arizona State University)

"""Matched loss-weight ablations from the saved Step 4A pretraining boundary."""

VARIANTS = ("cluster_weight_0p01", "cluster_weight_1p0", "no_reconstruction")

WEIGHTS = {
    "cluster_weight_0p01": (1.0, 0.01),
    "cluster_weight_1p0": (1.0, 1.0),
    "no_reconstruction": (0.0, 0.1),
}
