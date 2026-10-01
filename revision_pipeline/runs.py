# Purpose: Isolated run directories with an atomic success publication step.
# Author: Ariana Rahman (Arizona State University)

"""Isolated run directories with an atomic success publication step."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import traceback
import uuid

from .integrity import canonical_hash, file_fingerprint, project_path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    """Atomically replace one generated JSON artifact; never used on legacy files."""
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


class RunDirectory:
    """A failed/interrupted run never appears as a completed run.

    Failures retain their .incomplete-* directory, traceback and status. A killed
    process may leave status='running'; only the published directory is complete.
    The root is a dedicated output directory, not a legacy results directory.
    """

    def __init__(self, root, *, kind, config, run_id=None):
        self.root = Path(root).resolve()
        self.run_id = run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                 + "-" + uuid.uuid4().hex[:12])
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", self.run_id):
            raise ValueError("Invalid run ID")
        if self.run_id.upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)],
                                   *[f"LPT{i}" for i in range(1, 10)]}:
            raise ValueError("Reserved run ID")
        self.path = self.root / (".incomplete-" + self.run_id)
        self.final_path = self.root / self.run_id
        # Validate serializability before creating anything.
        self.config = config
        self.manifest = {"schema_version": 1, "run_id": self.run_id, "kind": kind,
                         "status": "created", "created_at_utc": utc_now(),
                         "config_sha256": canonical_hash(config),
                         "wall_seconds": None, "peak_memory_bytes": None,
                         "peak_memory_status": "not_measured",
                         "scientific_experiment": False}

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if self.final_path.exists():
            raise FileExistsError(f"Refusing to overwrite completed run: {self.final_path}")
        self.path.mkdir()  # Exclusive reservation also protects concurrent runs.
        self.started = time.perf_counter()
        self.manifest.update(status="running", started_at_utc=utc_now())
        try:
            write_json(self.path / "config.json", self.config)
            write_json(self.path / "run.json", self.manifest)
        except BaseException as failure:
            self._fail(failure)
            raise
        return self

    def artifact_path(self, relative):
        path = project_path(self.path, relative)
        if path.name == "run.json":
            raise ValueError("run.json is reserved")
        if path.exists():
            raise FileExistsError(f"Artifact already exists: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, relative, value):
        write_json(self.artifact_path(relative), value)

    def _fail(self, error):
        self.manifest.update(status="failed", finished_at_utc=utc_now(),
                             wall_seconds=time.perf_counter() - self.started,
                             error_type=type(error).__name__, error=str(error))
        (self.path / "failure.txt").write_text(
            "".join(traceback.format_exception(error)), encoding="utf-8")
        write_json(self.path / "run.json", self.manifest)

    def __exit__(self, exc_type, error, tb):
        if error is not None:
            self._fail(error)
            return False
        try:
            artifacts = {}
            for path in sorted(self.path.rglob("*")):
                if path.is_file() and path != self.path / "run.json":
                    artifacts[path.relative_to(self.path).as_posix()] = file_fingerprint(path)
            self.manifest.update(status="succeeded", finished_at_utc=utc_now(),
                                 wall_seconds=time.perf_counter() - self.started,
                                 artifacts=artifacts)
            write_json(self.path / "run.json", self.manifest)
            if self.final_path.exists():
                raise FileExistsError(f"Refusing to overwrite: {self.final_path}")
            self.path.rename(self.final_path)
        except BaseException as failure:
            self._fail(failure)
            raise
        return False
