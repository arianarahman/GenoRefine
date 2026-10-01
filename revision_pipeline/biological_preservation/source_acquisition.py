# Purpose: Immutable acquisition of externally curated biological-reference files.
# Author: Ariana Rahman (Arizona State University)

"""Immutable acquisition of externally curated biological-reference files.

Every source is identified by a URL, exact byte count, SHA-256 digest, and a
minimal schema contract.  Publication uses a same-filesystem hard link, so a
concurrent process cannot replace an already published source.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ..integrity import canonical_hash, file_fingerprint


@dataclass(frozen=True)
class SourceLock:
    """Complete immutable identity and schema contract for one source."""

    source_id: str
    url: str
    filename: str
    size_bytes: int
    sha256: str
    schema: Mapping[str, Any]

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,79}", self.source_id):
            raise ValueError("source_id must be a short lowercase identifier")
        parsed = urlparse(self.url)
        if parsed.scheme not in {"https", "file"}:
            raise ValueError("Only immutable HTTPS or local file URLs are accepted")
        if Path(self.filename).name != self.filename or not self.filename:
            raise ValueError("filename must be a basename")
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise ValueError("size_bytes must be positive")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("sha256 must contain 64 lowercase hexadecimal digits")
        if not isinstance(self.schema, Mapping) or "kind" not in self.schema:
            raise ValueError("schema must declare its kind")

    def record(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "url": self.url,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "schema": dict(self.schema),
            "schema_lock_sha256": canonical_hash(dict(self.schema)),
        }


def _open_text(path: Path):
    return (gzip.open(path, "rt", encoding="utf-8-sig", newline="")
            if path.suffix.lower() == ".gz" else
            path.open("r", encoding="utf-8-sig", newline=""))


def _tabular_columns(path: Path, delimiter: str) -> list[str]:
    with _open_text(path) as stream:
        row = next(csv.reader(stream, delimiter=delimiter), None)
    if not row or any(not value.strip() for value in row):
        raise ValueError(f"Missing or blank tabular header in {path}")
    if len(row) != len(set(row)):
        raise ValueError(f"Duplicate tabular columns in {path}")
    return row


def validate_locked_schema(path: str | Path, schema: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a source's declared schema without interpreting its biology."""

    path = Path(path)
    kind = schema.get("kind")
    if kind == "delimited":
        delimiter = schema.get("delimiter", "\t")
        if not isinstance(delimiter, str) or len(delimiter) != 1:
            raise ValueError("Delimited schema requires a one-character delimiter")
        columns = _tabular_columns(path, delimiter)
        required = schema.get("required_columns", [])
        if not isinstance(required, list) or any(not isinstance(x, str) for x in required):
            raise ValueError("required_columns must be a list of strings")
        missing = [column for column in required if column not in columns]
        if missing:
            raise ValueError(f"Missing locked columns: {missing}")
        exact = schema.get("exact_columns")
        if exact is not None and columns != exact:
            raise ValueError("Exact column order differs from the source lock")
        return {"kind": kind, "columns": columns, "columns_sha256": canonical_hash(columns)}
    if kind == "xlsx":
        import pandas as pd
        sheet = schema.get("sheet")
        with pd.ExcelFile(path) as workbook:
            if sheet not in workbook.sheet_names:
                raise ValueError(f"Locked worksheet is absent: {sheet!r}")
            columns = [str(value) for value in
                       pd.read_excel(workbook, sheet_name=sheet, nrows=0).columns]
        required = schema.get("required_columns", [])
        missing = [column for column in required if column not in columns]
        if missing:
            raise ValueError(f"Missing locked columns: {missing}")
        return {"kind": kind, "sheet": sheet, "columns": columns,
                "columns_sha256": canonical_hash(columns)}
    if kind == "obo":
        text = path.read_text(encoding="utf-8")
        term_count = text.count("\n[Term]\n") + int(text.startswith("[Term]\n"))
        minimum = int(schema.get("minimum_term_count", 1))
        if term_count < minimum or "format-version:" not in text:
            raise ValueError("OBO ontology does not satisfy the locked schema")
        return {"kind": kind, "term_count": term_count}
    if kind == "owl":
        from xml.etree import ElementTree
        root = ElementTree.parse(path).getroot()
        class_count = sum(1 for node in root.iter() if node.tag.endswith("}Class"))
        minimum = int(schema.get("minimum_class_count", 1))
        if class_count < minimum:
            raise ValueError("OWL ontology does not satisfy the locked schema")
        return {"kind": kind, "class_count": class_count}
    raise ValueError(f"Unsupported locked schema kind: {kind!r}")


def verify_locked_source(path: str | Path, lock: SourceLock) -> dict[str, Any]:
    """Verify bytes and schema, returning a serializable provenance record."""

    path = Path(path)
    fingerprint = file_fingerprint(path)
    expected = {"sha256": lock.sha256, "size_bytes": lock.size_bytes}
    if fingerprint != expected:
        raise ValueError(f"Source fingerprint differs for {lock.source_id}: "
                         f"expected {expected}, observed {fingerprint}")
    schema_record = validate_locked_schema(path, lock.schema)
    return {**lock.record(), "path": str(path.resolve()), "fingerprint": fingerprint,
            "observed_schema": schema_record, "verified": True}


def acquire_locked_source(lock: SourceLock, destination: str | Path, *, timeout: float = 120.0,
                          user_agent: str = "GenoRefine-revision-source-lock/1") -> dict[str, Any]:
    """Acquire, verify, and atomically publish one immutable external source.

    Existing files are never replaced.  A mismatching existing file is an error.
    """

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / lock.filename
    if target.exists():
        return verify_locked_source(target, lock)
    # Preserve the source extension on the temporary file because schema
    # readers use it to select gzip, XLSX, OBO, or OWL decoding.  The random
    # file remains unpublished until byte and schema verification succeed.
    source_suffix = "".join(Path(lock.filename).suffixes)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{lock.filename}.", suffix=f".part{source_suffix}", dir=destination)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    total = 0
    try:
        request = Request(lock.url, headers={"User-Agent": user_agent})
        with os.fdopen(descriptor, "wb") as output, urlopen(request, timeout=timeout) as response:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > lock.size_bytes:
                    raise ValueError(f"Downloaded source exceeds locked size for {lock.source_id}")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if total != lock.size_bytes or digest.hexdigest() != lock.sha256:
            raise ValueError(f"Downloaded bytes differ from lock for {lock.source_id}")
        validate_locked_schema(temporary, lock.schema)
        try:
            os.link(temporary, target)
        except FileExistsError:
            pass
        return verify_locked_source(target, lock)
    finally:
        temporary.unlink(missing_ok=True)


def load_source_locks(path: str | Path) -> tuple[SourceLock, ...]:
    """Load a small versioned JSON lock file with duplicate-ID rejection."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(value) != {"schema_version", "sources"} or value["schema_version"] != 1:
        raise ValueError("Unsupported source-lock document")
    locks = tuple(SourceLock(**item) for item in value["sources"])
    if len({item.source_id for item in locks}) != len(locks):
        raise ValueError("Duplicate source_id in source-lock document")
    return locks
