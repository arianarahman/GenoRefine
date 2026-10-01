# Purpose: Fresh frozen baseline evaluation, before outcome-independent K binding.
# Author: Ariana Rahman (Arizona State University)

"""Fresh frozen baseline evaluation, before outcome-independent K binding."""

from ..evaluate.config import EvaluationConfig
from ..evaluate.runner import evaluate_store
from ..pilot.common import read
from ..step4_policy import load_policy
from .common import ROOT, panel_spec


def main():
    spec, policy = panel_spec(), load_policy()
    config = EvaluationConfig.from_dict(read(ROOT/policy["primary_evaluation_config"]))
    path = evaluate_store(ROOT, ROOT/spec["store"], spec["dataset"], spec["embedding"], config)
    print(path, flush=True)


if __name__ == "__main__":
    main()
