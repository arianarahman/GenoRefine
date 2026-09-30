"""Run from the project root with: python -m revision_pipeline --help."""

import argparse
from pathlib import Path
import sys

from .audit import run_audit, verify_audit


def main(argv=None):
    parser = argparse.ArgumentParser(description="GenoRefine revision foundation (no training yet)")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Snapshot sources and inspect registered inputs without training")
    audit.add_argument("--registry", default="revision_pipeline/configs/datasets.json")
    audit.add_argument("--hash-inputs", action="store_true", help="Stream SHA256 over every registered input")
    verify = commands.add_parser("verify", help="Read-only comparison against a completed full-hash audit")
    verify.add_argument("reference", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            path, report = run_audit(args.project_root, args.registry, hash_inputs=args.hash_inputs)
            print(f"Audit saved: {path}")
            print(f"Sources: {report['source_files_snapshotted']}; inputs: {report['registered_input_files']}")
            print("Scientific runs are NOT ready. See audit_report.json and summary.md.")
            return 2 if report["missing_inputs"] else 0
        discrepancies = verify_audit(args.project_root, args.reference)
        if discrepancies:
            print("Integrity check failed:\n" + "\n".join(discrepancies))
            return 2
        print("Recorded source files, input bytes and saved audit artifacts match.")
        print("This does not validate biological provenance, the runtime or numerical results.")
        return 0
    except (OSError, ValueError, RuntimeError, SyntaxError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
