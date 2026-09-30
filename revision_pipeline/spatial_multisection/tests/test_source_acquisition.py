from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ..source_acquisition import (
    DEFAULT_CONFIG,
    ROOT,
    acquire_locked_file,
    load_spec,
    verify_sources,
)


def _item(content: bytes) -> dict:
    return {
        "role": "labels",
        "section": None,
        "relative_path": "locked.bin",
        "url": "https://example.invalid/locked.bin",
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def test_frozen_configuration_and_downloaded_sources_match() -> None:
    spec = load_spec(DEFAULT_CONFIG)
    assert [row["id"] for row in spec["dataset"]["sections"]] == [
        "151507", "151508", "151669", "151670", "151673", "151674",
    ]
    assert [row["donor"] for row in spec["dataset"]["sections"]] == [
        "Br5292", "Br5292", "Br5595", "Br5595", "Br8100", "Br8100",
    ]
    assert spec["dataset"]["expected_labeled_spots"] == 22968
    assert spec["dataset"]["expected_missing_labels"] == 113
    inventory = verify_sources(spec, ROOT)
    assert len(inventory["verified_files"]) == 26
    assert len([row for row in inventory["verified_files"] if row["role"] == "histology_hires"]) == 6
    assert len([row for row in inventory["verified_files"] if row["role"] == "scalefactors"]) == 6
    assert inventory["repository_commit"] == "044446d6bd8fc154aa74f7be62ec67effb1ec376"


def test_locked_acquisition_is_atomic_and_repair_is_explicit(tmp_path: Path) -> None:
    content = b"locked spatial source\n"
    item = _item(content)
    destination = tmp_path / item["relative_path"]

    def downloader(url: str, path: Path) -> None:
        assert url == item["url"]
        path.write_bytes(content)

    first = acquire_locked_file(item, destination, downloader=downloader)
    assert first["status"] == "downloaded"
    second = acquire_locked_file(item, destination, downloader=downloader)
    assert second["status"] == "verified_existing"
    destination.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Locked source mismatch"):
        acquire_locked_file(item, destination, downloader=downloader)
    repaired = acquire_locked_file(item, destination, repair=True, downloader=downloader)
    assert repaired["status"] == "downloaded_or_repaired"
    assert destination.read_bytes() == content


def test_failed_locked_download_is_not_published(tmp_path: Path) -> None:
    item = _item(b"expected")
    destination = tmp_path / item["relative_path"]

    def bad_downloader(_url: str, path: Path) -> None:
        path.write_bytes(b"wrong")

    with pytest.raises(ValueError, match="Locked source mismatch"):
        acquire_locked_file(item, destination, downloader=bad_downloader)
    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))
