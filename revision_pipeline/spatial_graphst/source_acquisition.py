# Purpose: Acquire the exact official GraphST and PASTE source archives, fail closed.
# Author: Ariana Rahman (Arizona State University)

"""Acquire the exact official GraphST and PASTE source archives, fail closed."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import urllib.request
import uuid

from ..integrity import file_fingerprint, project_path
from ..runs import write_json


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "revision_pipeline/configs/spatial_graphst_panel_v1.json"


@dataclass(frozen=True)
class SourceLock:
    name: str
    version: str
    repository: str
    commit: str
    archive_url: str
    archive_sha256: str
    archive_size_bytes: int
    archive_root: str
    destination: str
    files: dict[str, str]


def _read_spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def source_locks() -> tuple[SourceLock, ...]:
    sources = _read_spec()["sources"]
    return tuple(SourceLock(**sources[key]) for key in ("graphst", "paste"))


def _safe_member_name(lock: SourceLock, relative: str) -> str:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"Unsafe source member: {relative!r}")
    return f"{lock.archive_root}/{pure.as_posix()}"


def verify_locked_source(destination: Path, lock: SourceLock) -> dict:
    destination = Path(destination).resolve()
    expected_files = set(lock.files) | {"SOURCE_LOCK.json"}
    observed_files = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*") if path.is_file()
    }
    if observed_files != expected_files:
        raise ValueError(f"Locked {lock.name} tree has missing/extra files")
    for relative, digest in lock.files.items():
        fingerprint = file_fingerprint(project_path(destination, relative))
        if fingerprint["sha256"] != digest:
            raise ValueError(f"Locked {lock.name} file changed: {relative}")
    record = json.loads((destination / "SOURCE_LOCK.json").read_text(encoding="utf-8"))
    expected_record = {
        "name": lock.name,
        "version": lock.version,
        "repository": lock.repository,
        "commit": lock.commit,
        "archive_url": lock.archive_url,
        "archive_sha256": lock.archive_sha256,
        "archive_size_bytes": lock.archive_size_bytes,
        "selected_files": lock.files,
    }
    if record != expected_record:
        raise ValueError(f"Locked {lock.name} source receipt changed")
    return expected_record


def _download(lock: SourceLock) -> bytes:
    request = urllib.request.Request(lock.archive_url, headers={"User-Agent": "GenoRefine-source-lock/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read(lock.archive_size_bytes + 1)
        if response.read(1):
            raise ValueError(f"{lock.name} archive exceeds the frozen byte count")
    if len(payload) != lock.archive_size_bytes:
        raise ValueError(f"{lock.name} archive size changed")
    if hashlib.sha256(payload).hexdigest() != lock.archive_sha256:
        raise ValueError(f"{lock.name} archive digest changed")
    return payload


def acquire_locked_source(lock: SourceLock) -> Path:
    destination = project_path(ROOT, lock.destination)
    if destination.exists():
        verify_locked_source(destination, lock)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".incomplete-source-{destination.name}-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        payload = _download(lock)
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            members = {member.name: member for member in archive.getmembers() if member.isfile()}
            for relative, digest in lock.files.items():
                member_name = _safe_member_name(lock, relative)
                member = members.get(member_name)
                if member is None or member.size < 0:
                    raise ValueError(f"Frozen archive lacks {member_name}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Cannot read {member_name}")
                data = stream.read()
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError(f"Archive member digest changed: {member_name}")
                target = project_path(temporary, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        write_json(temporary / "SOURCE_LOCK.json", {
            "name": lock.name,
            "version": lock.version,
            "repository": lock.repository,
            "commit": lock.commit,
            "archive_url": lock.archive_url,
            "archive_sha256": lock.archive_sha256,
            "archive_size_bytes": lock.archive_size_bytes,
            "selected_files": lock.files,
        })
        verify_locked_source(temporary, lock)
        temporary.rename(destination)
    except BaseException:
        # Preserve the failed acquisition for diagnosis; never publish it as the lock.
        raise
    verify_locked_source(destination, lock)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.all or not args.execute:
        parser.error("Explicit --all --execute is required")
    for lock in source_locks():
        print(acquire_locked_source(lock), flush=True)


if __name__ == "__main__":
    main()
