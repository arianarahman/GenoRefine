# Purpose: Acquire and verify the commit-pinned six-section LIBD DLPFC sources.
# Author: Ariana Rahman (Arizona State University)

"""Acquire and verify the commit-pinned six-section LIBD DLPFC sources.

The acquisition command never accepts an unpinned URL.  Every downloaded byte
is checked against the size and SHA-256 lock before an atomic publication.
Existing corrupt files are rejected unless ``--repair`` is explicitly used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
from typing import Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import uuid

from ..integrity import file_fingerprint, project_path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "revision_pipeline/configs/spatial_multisection_v1.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_ROLES = {
    "source_manifest", "labels", "counts", "positions",
    "histology_hires", "scalefactors",
}


def load_spec(path: Path | str = DEFAULT_CONFIG) -> dict:
    """Load and strictly validate the source/preprocessing contract."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        spec = json.load(stream)
    required = {"schema_version", "protocol_id", "dataset", "sources", "preprocessing", "scope"}
    if set(spec) != required or spec["schema_version"] != 1:
        raise ValueError("Unexpected spatial multi-section configuration schema")
    dataset = spec["dataset"]
    dataset_required = {
        "name", "study_doi", "repository", "repository_commit", "source_root",
        "sections", "expected_total_spots", "expected_labeled_spots",
        "expected_missing_labels", "label_classes", "label_policy",
    }
    if set(dataset) != dataset_required:
        raise ValueError("Unexpected dataset configuration fields")
    commit = dataset["repository_commit"]
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ValueError("repository_commit must be a full lowercase Git commit")
    sections = dataset["sections"]
    if not isinstance(sections, list) or len(sections) != 6:
        raise ValueError("Exactly six sections are required")
    section_ids: list[str] = []
    donors: set[str] = set()
    totals = {"expected_spots": 0, "expected_labeled_spots": 0, "expected_missing_labels": 0}
    for row in sections:
        if set(row) != {"id", "donor", *totals}:
            raise ValueError("Unexpected section fields")
        if not isinstance(row["id"], str) or not row["id"].isdigit() or row["id"] in section_ids:
            raise ValueError("Section IDs must be unique numeric strings")
        if not isinstance(row["donor"], str) or not row["donor"].strip():
            raise ValueError("Every section requires a donor")
        section_ids.append(row["id"])
        donors.add(row["donor"])
        for key in totals:
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError(f"Invalid {key}")
            totals[key] += row[key]
        if row["expected_spots"] != row["expected_labeled_spots"] + row["expected_missing_labels"]:
            raise ValueError(f"Section {row['id']} spot counts do not reconcile")
    if len(donors) != 3:
        raise ValueError("The six sections must represent exactly three donors")
    for key, total in totals.items():
        dataset_key = "expected_total_spots" if key == "expected_spots" else key
        if dataset[dataset_key] != total:
            raise ValueError(f"Dataset {dataset_key} does not equal the section sum")
    labels = dataset["label_classes"]
    if not isinstance(labels, list) or not labels or len(labels) != len(set(labels)):
        raise ValueError("label_classes must be a nonempty unique list")
    project_path(ROOT, dataset["source_root"])

    files = spec["sources"]
    if not isinstance(files, list) or len(files) != 26:
        raise ValueError("Expected one manifest, one label file and four files per section")
    seen_paths: set[str] = set()
    by_role: dict[str, list[dict]] = {role: [] for role in _ROLES}
    for item in files:
        if set(item) != {"role", "section", "relative_path", "url", "sha256", "size_bytes"}:
            raise ValueError("Unexpected source-file fields")
        role = item["role"]
        if role not in _ROLES:
            raise ValueError(f"Unknown source role: {role!r}")
        section = item["section"]
        if role in {"counts", "positions", "histology_hires", "scalefactors"}:
            if section not in section_ids:
                raise ValueError(f"Invalid section for {role}")
        elif section is not None:
            raise ValueError(f"{role} must have a null section")
        relative = item["relative_path"]
        if not isinstance(relative, str) or Path(relative).name != relative or relative in seen_paths:
            raise ValueError("Source relative paths must be unique filenames")
        seen_paths.add(relative)
        parsed = urlparse(item["url"])
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Every source URL must use HTTPS")
        if role in {"source_manifest", "labels", "positions", "scalefactors"} and commit not in item["url"]:
            raise ValueError(f"{role} URL is not pinned to the declared commit")
        if not isinstance(item["sha256"], str) or not _SHA256.fullmatch(item["sha256"]):
            raise ValueError("Every source requires a lowercase SHA-256 lock")
        if type(item["size_bytes"]) is not int or item["size_bytes"] <= 0:
            raise ValueError("Every source requires a positive byte-size lock")
        by_role[role].append(item)
    if len(by_role["source_manifest"]) != 1 or len(by_role["labels"]) != 1:
        raise ValueError("Exactly one source manifest and one label map are required")
    for role in ("counts", "positions", "histology_hires", "scalefactors"):
        if sorted(row["section"] for row in by_role[role]) != sorted(section_ids):
            raise ValueError(f"{role} does not cover every section exactly once")

    prep = spec["preprocessing"]
    if set(prep) != {
        "pooled_gene_min_cells", "normalize_total_target_sum", "log1p", "hvg",
        "scale", "pca", "harmony",
    }:
        raise ValueError("Unexpected preprocessing fields")
    if prep["hvg"] != {"n_top_genes": 2000, "flavor": "seurat", "batch_key": "section"}:
        raise ValueError("The frozen batch-aware 2,000-gene Seurat HVG policy changed")
    if prep["pca"] != {"n_components": 50, "svd_solver": "arpack", "random_state": 0}:
        raise ValueError("The frozen PCA policy changed")
    if (
        prep["harmony"]["batch_key"] != "section"
        or prep["harmony"]["max_iter_harmony"] != 50
        or prep["harmony"].get("thread_limit") != 1
    ):
        raise ValueError("The frozen Harmony section-batch policy changed")
    return spec


def source_root(spec: dict, project_root: Path | str = ROOT) -> Path:
    return project_path(Path(project_root), spec["dataset"]["source_root"])


def source_index(spec: dict) -> dict[tuple[str, str | None], dict]:
    return {(row["role"], row["section"]): row for row in spec["sources"]}


def verify_locked_file(path: Path | str, item: dict) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing locked source: {path}")
    observed = file_fingerprint(path)
    expected = {"sha256": item["sha256"], "size_bytes": item["size_bytes"]}
    if observed != expected:
        raise ValueError(f"Locked source mismatch for {path.name}: expected {expected}, observed {observed}")
    return observed


def verify_sources(spec: dict, project_root: Path | str = ROOT) -> dict:
    root = source_root(spec, project_root)
    records = []
    for item in spec["sources"]:
        path = root / item["relative_path"]
        observed = verify_locked_file(path, item)
        records.append({
            "role": item["role"], "section": item["section"],
            "relative_path": item["relative_path"], "url": item["url"], **observed,
        })
    return {
        "protocol_id": spec["protocol_id"],
        "repository": spec["dataset"]["repository"],
        "repository_commit": spec["dataset"]["repository_commit"],
        "source_root": spec["dataset"]["source_root"],
        "verified_files": records,
    }


def _download_https(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "GenoRefine-spatial-foundation/1"})
    with urlopen(request, timeout=120) as response, destination.open("xb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)


def acquire_locked_file(
    item: dict,
    destination: Path | str,
    *,
    repair: bool = False,
    downloader: Callable[[str, Path], None] = _download_https,
) -> dict:
    """Acquire one file and atomically publish it only after lock verification."""
    destination = Path(destination)
    if destination.exists():
        try:
            observed = verify_locked_file(destination, item)
            return {"status": "verified_existing", **observed}
        except ValueError:
            if not repair:
                raise
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.part")
    try:
        downloader(item["url"], temporary)
        observed = verify_locked_file(temporary, item)
        os.replace(temporary, destination)
        return {"status": "downloaded" if not repair else "downloaded_or_repaired", **observed}
    finally:
        if temporary.exists():
            temporary.unlink()


def acquire_sources(
    spec: dict,
    project_root: Path | str = ROOT,
    *,
    repair: bool = False,
    downloader: Callable[[str, Path], None] = _download_https,
) -> dict:
    root = source_root(spec, project_root)
    results = []
    for item in spec["sources"]:
        result = acquire_locked_file(
            item, root / item["relative_path"], repair=repair, downloader=downloader
        )
        results.append({"relative_path": item["relative_path"], **result})
    return {"source_root": str(root), "files": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--mode", choices=("verify", "acquire"), default="verify")
    parser.add_argument("--repair", action="store_true", help="Replace an existing corrupt file after lock verification")
    parser.add_argument("--execute", action="store_true", help="Required for network acquisition")
    args = parser.parse_args()
    spec = load_spec(args.config)
    if args.mode == "verify":
        if args.repair:
            parser.error("--repair applies only to --mode acquire")
        result = verify_sources(spec, args.project_root)
    else:
        if not args.execute:
            parser.error("--mode acquire requires explicit --execute")
        result = acquire_sources(spec, args.project_root, repair=args.repair)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
