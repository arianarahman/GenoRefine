"""Evaluate a validated store; all scientific settings come from explicit JSON."""

import argparse
from pathlib import Path

from ..data.store import read_json
from .config import EvaluationConfig
from .runner import evaluate_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--compare-to")
    parser.add_argument("--refined-bundle", type=Path)
    parser.add_argument("--baseline-evaluation", type=Path, help="Verified standalone baseline evaluation to reuse")
    parser.add_argument("--parent-order", choices=("canonical", "historical_source"), default="canonical")
    args = parser.parse_args()
    config = EvaluationConfig.from_dict(read_json(args.config))
    root = Path(__file__).resolve().parents[2]
    print(evaluate_store(root, args.store, args.dataset, args.embedding, config,
                         compare_to=args.compare_to, refined_bundle=args.refined_bundle, parent_order=args.parent_order,
                         baseline_evaluation=args.baseline_evaluation))


if __name__ == "__main__":
    main()
