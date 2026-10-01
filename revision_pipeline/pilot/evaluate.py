# Purpose: Score a fixed pilot output in one specified environment/profile.
# Author: Ariana Rahman (Arizona State University)

"""Score a fixed pilot output in one specified environment/profile."""

import argparse
from pathlib import Path

from .common import completed, read, specification
from ..evaluate.config import EvaluationConfig, historical_profile
from ..evaluate.runner import evaluate_store


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("primary", "historical"), required=True)
    p.add_argument("--training-run", type=Path)
    p.add_argument("--baseline-evaluation", type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    spec = specification(root)
    cfg = (EvaluationConfig.from_dict(read(root/spec["primary_evaluation_config"])) if args.profile == "primary"
           else historical_profile(spec["historical_profile"]))
    bundle = None
    if args.training_run:
        completed(args.training_run, "step3c_refiner_training")
        if read(args.training_run/"config.json")["pilot"] != spec:
            raise ValueError("Training used a different pilot specification")
        if read(args.training_run/"output_checks.json")["dtype"] != "float32":
            raise ValueError("Historical training-to-evaluator path requires native float32 output")
        bundle = args.training_run/"refined_bundle"
    path = evaluate_store(root, root/spec["store"], spec["dataset"], spec["embedding"], cfg,
                          refined_bundle=bundle, parent_order="historical_source",
                          baseline_evaluation=args.baseline_evaluation)
    print(path, flush=True)


if __name__ == "__main__":
    main()
