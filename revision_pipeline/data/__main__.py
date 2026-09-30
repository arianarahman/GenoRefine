"""Run only Step 3A. No implicit evaluator, integration or training calls."""

import argparse
import json
from pathlib import Path

from .store import Store, import_legacy


def main():
    parser = argparse.ArgumentParser(description="Step 3A dataset/embedding store")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("import-legacy", help="Validate/import only explicitly registered saved sources")
    create.add_argument("--config", default="revision_pipeline/configs/data_store.json")
    verify = sub.add_parser("verify", help="Verify saved artifacts and all original source hashes")
    verify.add_argument("store", type=Path)
    args = parser.parse_args()
    if args.command == "import-legacy":
        print(f"Step 3A store saved: {import_legacy(args.project_root, args.config)}", flush=True)
    else:
        print(json.dumps(Store(args.store).verify(args.project_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
