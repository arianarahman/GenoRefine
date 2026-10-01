# Purpose: Read-only project inspection; generated evidence goes into a new run directory.
# Author: Ariana Rahman (Arizona State University)

"""Read-only project inspection; generated evidence goes into a new run directory."""

import ast
import base64
import csv
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys

from .integrity import canonical_hash, file_fingerprint, load_registry, project_path
from .runs import RunDirectory


DATASET_FOLDERS = ("PancreasDataset", "HPCBDataset", "MouseDataset", "PBMCSingeBatchDataset")
SITE_PACKAGES = "venv_pancreas/Lib/site-packages"


def source_paths(root):
    """Bounded inventory: scientific scripts, GenoMap source and revision code/docs."""
    root = Path(root).resolve()
    found = set()
    for folder in DATASET_FOLDERS:
        for directory in (root / folder, root / folder / "Dataset"):
            found.update(path for path in directory.glob("*")
                         if path.is_file() and path.suffix.lower() in {".py", ".r"})
    package = root / SITE_PACKAGES / "genomap"
    found.update(package.rglob("*.py"))
    for directory, folders, filenames in os.walk(root / "revision_pipeline"):
        # Do not descend into growing run/snapshot trees or any local environment.
        folders[:] = [name for name in folders if name not in {"runs", "__pycache__", ".venv"}]
        for name in filenames:
            path = Path(directory) / name
            if path.suffix in {".py", ".json", ".md", ".txt", ".in", ".patch"} or name == ".gitignore":
                found.add(path)
    for name in ("constraints_pancreas.txt", "venv_pancreas/pyvenv.cfg"):
        path = root / name
        if path.is_file():
            found.add(path)
    for dist in (root / SITE_PACKAGES).glob("genomap-*.dist-info"):
        for name in ("RECORD", "METADATA", "WHEEL", "LICENSE.txt"):
            if (dist / name).is_file():
                found.add(dist / name)
    for path in found:
        project_path(root, path.relative_to(root).as_posix())
    return sorted(found)


def environment_inventory(root):
    site = project_path(root, SITE_PACKAGES)
    packages = sorted(
        ({"name": dist.metadata.get("Name", "unknown"), "version": dist.version}
         for dist in metadata.distributions(path=[str(site)])),
        key=lambda item: (item["name"].lower(), item["version"]),
    )
    return {
        "audit_interpreter": {"executable": sys.executable, "python": platform.python_version(),
                              "platform": platform.platform()},
        "copied_environment": {
            "path": "venv_pancreas", "site_packages_present": site.is_dir(),
            "windows_interpreter_present": (Path(root) / "venv_pancreas/Scripts/python.exe").is_file(),
            "packages": packages,
            "status": "inventory_only_not_a_validated_environment_lock",
        },
        "limitations": [
            "No TensorFlow, Keras, Scanpy or GenoMap import or training was performed.",
            "Metadata describes files present now, not proof of the environment for every old result.",
            "Dependency resolution, native libraries, R packages, GPU and WSL compatibility remain untested.",
            "Rebuild an isolated runtime and pass import/numerical smoke tests before scientific runs.",
        ],
    }


def genomap_record_audit(root):
    """Check local Python files against the copied installation's RECORD, not PyPI."""
    root = Path(root)
    records = list((root / SITE_PACKAGES).glob("genomap-*.dist-info/RECORD"))
    if len(records) != 1:
        return {"status": "unavailable", "reason": "Expected one GenoMap RECORD file"}
    results = []
    with records[0].open(encoding="utf-8", newline="") as stream:
        for relative, recorded_hash, size in csv.reader(stream):
            if not relative.startswith("genomap/") or not relative.endswith(".py"):
                continue
            local_relative = SITE_PACKAGES + "/" + relative
            path = project_path(root, local_relative)
            item = {"path": local_relative, "record_hash": recorded_hash}
            if not path.is_file():
                item["status"] = "missing"
            elif not recorded_hash.startswith("sha256="):
                item["status"] = "not_sha256_checkable"
            else:
                fingerprint = file_fingerprint(path)
                encoded = base64.urlsafe_b64encode(bytes.fromhex(fingerprint["sha256"])).decode().rstrip("=")
                item.update(fingerprint)
                item["status"] = "matches_record" if recorded_hash == "sha256=" + encoded else "differs_from_record"
            results.append(item)
    return {"status": "checked", "reference": records[0].relative_to(root).as_posix(),
            "reference_limit": "Local installation RECORD is not an independently authenticated release checksum.",
            "files": results}


def detect_legacy_signatures(source, rule):
    """Narrow diagnostics for known issues; absence is NOT proof of correctness."""
    tree = ast.parse(source)
    lines = []
    for node in ast.walk(tree):
        if rule == "last_cell_omitted" and isinstance(node, ast.For):
            call = node.iter
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == "range" and len(call.args) == 2
                    and ast.unparse(call.args[1]).replace(" ", "") == "numCell-1"):
                lines.append(node.lineno)
        if rule == "neighbor_double_exclusion" and isinstance(node, ast.Subscript):
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "kneighbors" and not call.args
                    and not any(k.arg == "X" for k in call.keywords)
                    and ast.unparse(node.slice).replace(" ", "") == "(:,1:)"):
                lines.append(node.lineno)
        if rule == "empty_batch_wrap" and isinstance(node, ast.Assign):
            if re.search(r"index = index \+ 1 if \(index \+ 1\) \* batch_size <= x.shape\[0\] else 0",
                         ast.unparse(node)):
                lines.append(node.lineno)
    return sorted(lines)


def known_issue_scan(root):
    checks = (
        ("last_cell_omitted", SITE_PACKAGES + "/genomap/genomap.py"),
        ("empty_batch_wrap", SITE_PACKAGES + "/genomap/utils/FcDEC.py"),
        ("neighbor_double_exclusion", "HPCBDataset/focus_tests_hie.py"),
    )
    findings = []
    for rule, relative in checks:
        path = project_path(root, relative)
        if not path.is_file():
            findings.append({"rule": rule, "path": relative, "status": "source_missing", "lines": []})
            continue
        lines = detect_legacy_signatures(path.read_text(encoding="utf-8-sig"), rule)
        findings.append({"rule": rule, "path": relative, "lines": lines,
                         "status": "known_signature_present" if lines else "signature_not_found_not_a_pass"})
    return findings


def run_audit(root, registry_relative, *, hash_inputs=False):
    root = Path(root).resolve()
    registry_path = project_path(root, registry_relative)
    registry = load_registry(root, registry_path)
    config = {"operation": "foundation_audit", "hash_inputs": hash_inputs,
              "registry_path": registry_relative, "registry": registry}
    output = project_path(root, "revision_pipeline/runs")
    with RunDirectory(output, kind="foundation_audit", config=config) as run:
        source_manifest = {}
        for path in source_paths(root):
            relative = path.relative_to(root).as_posix()
            before = file_fingerprint(path)
            target = run.artifact_path("source_snapshot/" + relative)
            shutil.copy2(path, target)
            if file_fingerprint(target) != before:
                raise RuntimeError(f"Source changed during snapshot: {relative}")
            source_manifest[relative] = before
        run.write_json("source_manifest.json", source_manifest)
        run.manifest["source_tree_sha256"] = canonical_hash(source_manifest)
        environment = environment_inventory(root)
        run.write_json("environment_inventory.json", environment)
        run.manifest["environment_inventory_sha256"] = canonical_hash(environment)
        run.manifest["random_seeds"] = {"status": "not_applicable_no_stochastic_experiment"}
        run.manifest["git"] = {"status": "not_queried_source_snapshot_is_authoritative_for_this_audit"}
        inputs, missing = [], []
        for dataset in registry["datasets"]:
            for item in dataset["inputs"]:
                path = project_path(root, item["path"])
                record = {"dataset": dataset["id"], **item, "exists": path.is_file()}
                if path.is_file():
                    record.update(file_fingerprint(path) if hash_inputs else {"size_bytes": path.stat().st_size})
                    record["hash_status"] = "full_file_sha256" if hash_inputs else "not_requested"
                else:
                    missing.append(item["path"])
                    record["hash_status"] = "missing_file"
                inputs.append(record)
        input_report = {"status": "current_files_not_validated_scientific_provenance", "files": inputs,
                        "contents_schema_validation": "pending_dataset_loaders",
                        "excluded_inputs": registry["excluded_inputs"]}
        run.write_json("input_inventory.json", input_report)
        run.manifest["input_inventory_sha256"] = canonical_hash(input_report)
        patches = genomap_record_audit(root)
        run.write_json("genomap_record_audit.json", patches)
        findings = known_issue_scan(root)
        report = {
            "scientific_runs_ready": False,
            "source_files_snapshotted": len(source_manifest), "registered_input_files": len(inputs),
            "input_hashes_complete": hash_inputs and not missing,
            "missing_inputs": missing, "known_code_findings": findings,
            "locally_changed_genomap_files": [x["path"] for x in patches.get("files", [])
                                              if x["status"] == "differs_from_record"],
            "blocking_next_steps": [
                "Verify the separate staged-refiner runtime with its acceptance suite; the copied venv remains historical evidence only.",
                "Use the tested staged refiner, not the preserved legacy constructors; integrate the shared evaluator and neighbor helper next.",
                "Validate dataset contents, cell IDs, annotation provenance and cached embedding pairing.",
                "Finalize analysis decisions before decisive experiments; do not infer independent cohorts from file count.",
            ],
            "scope": "File integrity/source audit only. No model fits, metric recalculation or manuscript edits.",
        }
        run.write_json("audit_report.json", report)
        summary = ["# Foundation audit", "", "Scientific runs ready: **No**.", "",
                   f"Source files snapshotted: {len(source_manifest)}.",
                   f"Registered input files: {len(inputs)}; missing: {len(missing)}.",
                   f"Full input hashes: {'yes' if hash_inputs else 'not requested'}.", "",
                   "## Known code signatures", ""]
        summary.extend(f"- {x['rule']}: {x['status']}; {x['path']}; lines {x['lines']}." for x in findings)
        summary += ["", "## Next steps", ""] + ["- " + item for item in report["blocking_next_steps"]]
        summary += ["", "An audit success is not scientific validation. Original files remain unchanged.", ""]
        run.artifact_path("summary.md").write_text("\n".join(summary), encoding="utf-8")
    return run.final_path, report


def verify_audit(root, reference):
    """Compare current full input/source bytes and saved evidence; never refresh pins."""
    root, reference = Path(root).resolve(), Path(reference).resolve()
    manifest = json.loads((reference / "run.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "succeeded" or manifest.get("kind") != "foundation_audit":
        raise ValueError("Reference is not a completed foundation audit")
    discrepancies = []
    for relative, expected in manifest["artifacts"].items():
        path = project_path(reference, relative)
        if not path.is_file() or file_fingerprint(path) != expected:
            discrepancies.append("reference_artifact_changed: " + relative)
    # Do not trust input/source expectations from a damaged reference artifact.
    if discrepancies:
        return discrepancies
    sources = json.loads((reference / "source_manifest.json").read_text(encoding="utf-8"))
    current_paths = {path.relative_to(root).as_posix() for path in source_paths(root)}
    for relative in sorted(current_paths - set(sources)):
        discrepancies.append("new_source: " + relative)
    for relative, expected in sources.items():
        path = project_path(root, relative)
        if not path.is_file() or file_fingerprint(path) != expected:
            discrepancies.append("source_changed_or_missing: " + relative)
    inventory = json.loads((reference / "input_inventory.json").read_text(encoding="utf-8"))
    for item in inventory["files"]:
        if "sha256" not in item:
            discrepancies.append("reference_input_not_hashed: " + item["path"])
            continue
        path = project_path(root, item["path"])
        expected = {key: item[key] for key in ("sha256", "size_bytes")}
        if not path.is_file() or file_fingerprint(path) != expected:
            discrepancies.append("input_changed_or_missing: " + item["path"])
    return discrepancies
