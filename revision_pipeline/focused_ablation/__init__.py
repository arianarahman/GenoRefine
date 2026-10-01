# Purpose: Focused GenoRefine architecture and objective ablations.
# Author: Ariana Rahman (Arizona State University)

"""Focused GenoRefine architecture and objective ablations."""

# Kept outside the TensorFlow-dependent model module so that scoring and
# result consolidation can run in the lightweight evaluation environment.
VARIANTS = ("no_pretraining", "kernel5", "half_capacity", "map12_fixed36", "map24_fixed36")
