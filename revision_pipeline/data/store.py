"""Immutable-run dataset/embedding store with explicit provenance and policy gates."""

from collections import Counter
from dataclasses import dataclass
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys
import sysconfig

import numpy as np

from ..audit import source_paths
from ..integrity import alignment_indices, canonical_hash, file_fingerprint, load_registry, project_path, validate_cell_ids
from ..runs import RunDirectory
from .embeddings import canonical_selection, matrix_diagnostics, read_embedding_csv
from .readers import array_hash, expression_blocks, load_dataset


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def dataset_sources(spec):
    if spec["format"] == "h5ad":
        return [spec["path"]]
    return [part["path"] for part in spec["parts"]] + [spec["label_path"]]


def load_config(root, relative):
    config = read_json(project_path(root, relative))
    if set(config) != {"schema_version", "registry", "source_lock", "loaders", "embeddings",
                       "evidence_files", "excluded_embedding_roots", "not_imported"} or config["schema_version"] != 1:
        raise ValueError("Unsupported data-store configuration")
    registry = load_registry(root, project_path(root, config["registry"]))
    registered = {d["id"]: d for d in registry["datasets"]}
    if set(config["loaders"]) != set(registered):
        raise ValueError("Exactly one explicit loader per registered dataset is required")
    paths = set()
    for identifier, spec in config["loaders"].items():
        common = {"format", "label_key", "batch_key", "annotation_policy"}
        if spec["format"] == "h5ad":
            expected = common | {"path"}
        elif spec["format"] == "mat_parts":
            expected = common | {"parts", "label_path", "label_variable"}
            if not spec["parts"]:
                raise ValueError("MAT parts must not be empty")
            for part in spec["parts"]:
                if set(part) != {"path", "key", "batch"} or not all(isinstance(v, str) and v for v in part.values()):
                    raise ValueError("Invalid explicit MAT part")
        else:
            raise ValueError("Unsupported loader format")
        if set(spec) != expected or not all(isinstance(spec[k], str) and spec[k] for k in ("label_key", "batch_key")):
            raise ValueError("Unexpected loader fields")
        policy = spec["annotation_policy"]
        if (set(policy) != {"named_labels_allowed", "partition_description", "restriction_reason"}
                or type(policy["named_labels_allowed"]) is not bool
                or not all(isinstance(policy[k], str) and policy[k] for k in ("partition_description", "restriction_reason"))):
            raise ValueError("Explicit annotation policy required")
        inputs = dataset_sources(spec)
        if len(inputs) != len(set(inputs)) or set(inputs) != {i["path"] for i in registered[identifier]["inputs"]}:
            raise ValueError("Loader inputs differ from the dataset registry")
        paths.update(inputs)
    identifiers, embedding_paths = set(), set()
    excluded = [project_path(root, x["path"]) for x in config["excluded_embedding_roots"]]
    for item in config["embeddings"]:
        if set(item) != {"dataset", "id", "path", "dimensions", "kind", "claimed_baseline", "source_set"}:
            raise ValueError("Unexpected embedding fields")
        key = (item["dataset"], item["id"])
        if key in identifiers or item["dataset"] not in registered or not re.fullmatch(r"[A-Za-z0-9_-]+", item["id"]):
            raise ValueError("Invalid or duplicate embedding ID")
        identifiers.add(key)
        if item["kind"] not in {"baseline", "historical_refined", "graph_proxy"}:
            raise ValueError("Invalid embedding kind")
        if type(item["dimensions"]) is not int or item["dimensions"] < 1:
            raise ValueError("Invalid embedding dimension")
        if (item["kind"] == "historical_refined") != isinstance(item["claimed_baseline"], str):
            raise ValueError("Historical refinement must declare a candidate baseline, not verified pairing")
        path = project_path(root, item["path"])
        if item["path"] in embedding_paths or any(path == p or path.is_relative_to(p) for p in excluded):
            raise ValueError("Duplicate or explicitly excluded embedding path")
        embedding_paths.add(item["path"])
        paths.add(item["path"])
    for item in config["embeddings"]:
        if item["claimed_baseline"] is not None and (item["dataset"], item["claimed_baseline"]) not in identifiers:
            raise ValueError("Unknown candidate baseline")
    for item in config["evidence_files"]:
        if set(item) != {"dataset", "path", "kind"} or item["dataset"] not in registered or item["kind"] != "historical_removed_cells":
            raise ValueError("Unknown evidence file")
        paths.add(item["path"])
    lock = read_json(project_path(root, config["source_lock"]))
    if lock["schema_version"] != 1 or set(lock["files"]) != paths:
        raise ValueError("Source lock must cover exactly the selected data, embeddings and evidence")
    for relative_path, fingerprint in lock["files"].items():
        project_path(root, relative_path)
        if (set(fingerprint) != {"sha256", "size_bytes"}
                or not re.fullmatch(r"[0-9a-f]{64}", fingerprint["sha256"])
                or type(fingerprint["size_bytes"]) is not int or fingerprint["size_bytes"] < 1):
            raise ValueError("Invalid pinned source fingerprint")
    return config, registered, lock["files"]


def check_sources(root, sources):
    for relative, expected in sources.items():
        if file_fingerprint(project_path(root, relative)) != expected:
            raise ValueError(f"Source fingerprint mismatch (do not silently refresh lock): {relative}")


def runtime_inventory():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "executable": sys.executable,
            "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions(
                path=sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")}))),
            "training": False, "scanpy_or_leiden_required": False}


def import_legacy(root, config_relative="revision_pipeline/configs/data_store.json"):
    root = Path(root).resolve()
    config, registered, lock = load_config(root, config_relative)
    with RunDirectory(root / "revision_pipeline/runs", kind="step3a_data_store",
                      config={"operation": "validated_legacy_import", "settings": config}) as run:
        sources = {p.relative_to(root).as_posix(): file_fingerprint(p) for p in source_paths(root)}
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.write_json("runtime.json", runtime_inventory())
        run.write_json("input_manifest.json", lock)
        index = {"schema_version": 1, "datasets": {}, "embeddings": {},
                 "source_files": lock, "not_imported": config["not_imported"]}
        datasets, summaries = {}, []
        for identifier, spec in config["loaders"].items():
            print(f"Validating dataset: {identifier}", flush=True)
            dataset_lock = {p: lock[p] for p in dataset_sources(spec)}
            check_sources(root, dataset_lock)
            record = load_dataset(root, registered[identifier], spec)
            check_sources(root, dataset_lock)
            record["source_files"] = dataset_lock
            record["dataset_fingerprint"] = canonical_hash({"sources": dataset_lock, "loader": spec,
                                                           "cell_order": record["report"]["cell_order_sha256"]})
            path = f"datasets/{identifier}.json"
            run.write_json(path, record)
            index["datasets"][identifier] = path
            datasets[identifier] = record
            summaries.append(record["report"])
        embedding_summaries = []
        for item in config["embeddings"]:
            identifier, name, relative = item["dataset"], item["id"], item["path"]
            print(f"Importing embedding: {identifier}/{name}", flush=True)
            check_sources(root, {relative: lock[relative]})
            dataset = datasets[identifier]
            values, permutation, diagnostics = read_embedding_csv(
                project_path(root, relative), dataset["cell_ids"], dimensions=item["dimensions"])
            check_sources(root, {relative: lock[relative]})
            prefix = f"embeddings/{identifier}/{name}"
            values_path, order_path = prefix + ".npy", prefix + ".source_rows.npy"
            np.save(run.artifact_path(values_path), values, allow_pickle=False)
            np.save(run.artifact_path(order_path), permutation, allow_pickle=False)
            reloaded = np.load(run.path / values_path, mmap_mode="r", allow_pickle=False)
            np.testing.assert_array_equal(values, reloaded)
            meta = {**item, **diagnostics, "source_fingerprint": lock[relative],
                    "dataset_fingerprint": dataset["dataset_fingerprint"],
                    "values_path": values_path, "source_rows_path": order_path,
                    "reload_exact": True, "verified_parent_embedding": None,
                    "pairing_status": "historical_input_pairing_unverified" if item["kind"] == "historical_refined" else "not_a_refinement",
                    "graph_status": "coordinates_only; BBKNN graph not saved/imported" if item["kind"] == "graph_proxy" else "not_applicable",
                    "historical_preprocessing_and_seed": "not established by the saved coordinates"}
            meta_path = prefix + ".json"
            run.write_json(meta_path, meta)
            index["embeddings"].setdefault(identifier, {})[name] = meta_path
            embedding_summaries.append(meta)
            del reloaded, values
        evidence = []
        for item in config["evidence_files"]:
            check_sources(root, {item["path"]: lock[item["path"]]})
            with project_path(root, item["path"]).open(encoding="utf-8-sig") as stream:
                removed = validate_cell_ids([line.rstrip("\r\n") for line in stream])
            ids = set(datasets[item["dataset"]]["cell_ids"])
            evidence.append({**item, "listed_ids": len(removed), "ids_present_in_current_dataset": len(ids & set(removed)),
                             "ids_outside_current_dataset": len(set(removed) - ids),
                             "status": "unresolved historical lineage; ID/value checks do not explain this log"})
            check_sources(root, {item["path"]: lock[item["path"]]})
        run.write_json("store.json", index)
        run.write_json("validation.json", {"passed": True, "scope": "Data import/content integrity only; no metrics or training",
                                          "datasets": summaries, "embeddings": embedding_summaries,
                                          "evidence": evidence, "not_imported": config["not_imported"]})
        lines = ["# Step 3A data import", "", "Data integrity checks passed. No integration, scoring or training was run.", "",
                 f"Datasets: {len(datasets)}. Saved coordinate matrices: {len(embedding_summaries)}.", "",
                 "Canonical order is the source dataset order, including for subsets. Decimal CSV values are stored as float64.", "",
                 "Historical refined inputs are NOT proven paired. BBKNN coordinate files do NOT contain the BBKNN graph.", "",
                 "PBMC biological label names and pancreas biological label names are blocked pending annotation verification.", "",
                 "See validation.json for expression/embedding diagnostics, raw annotation provenance and unresolved issues."]
        run.artifact_path("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        after = {p.relative_to(root).as_posix(): file_fingerprint(p) for p in source_paths(root)}
        if after != sources:
            raise RuntimeError("Pipeline source changed during import; rerun against a stable source snapshot")
    return run.final_path


@dataclass
class DatasetView:
    record: dict
    indices: np.ndarray

    @property
    def cell_ids(self):
        return tuple(self.record["cell_ids"][i] for i in self.indices)

    @property
    def annotation_policy(self):
        return dict(self.record["loader"]["annotation_policy"])

    def reference_partition(self):
        """Anonymous partition codes, with an obligatory interpretation string."""
        values = self.record["obs"][self.record["loader"]["label_key"]]
        lookup = {label: i for i, label in enumerate(sorted(set(values)))}
        return (np.asarray([lookup[values[i]] for i in self.indices], dtype=np.int64),
                self.annotation_policy["partition_description"])

    def named_labels(self):
        policy = self.annotation_policy
        if not policy["named_labels_allowed"]:
            raise ValueError("Biological label-name output blocked: " + policy["restriction_reason"])
        values = self.record["obs"][self.record["loader"]["label_key"]]
        return tuple(values[i] for i in self.indices)

    def batch_labels(self, *, for_mixing_metric=False):
        if for_mixing_metric and not self.record["registry"]["batch_evaluation"]:
            raise ValueError("Batch-mixing evaluation is inapplicable to this single-batch dataset")
        values = self.record["obs"][self.record["loader"]["batch_key"]]
        return tuple(values[i] for i in self.indices)


@dataclass
class EmbeddingView:
    values: np.ndarray
    cell_ids: tuple
    metadata: dict

    def parent_reference(self):
        """Bind future refinement to the exact loaded values AND selected row order."""
        return {"dataset_fingerprint": self.metadata["dataset_fingerprint"],
                "embedding_id": self.metadata["id"],
                "stored_values_file_sha256": self.metadata["stored_values_file_sha256"],
                "selected_values_sha256": array_hash(self.values),
                "cell_order_sha256": canonical_hash(list(self.cell_ids)),
                "shape": list(self.values.shape)}


@dataclass
class HistoricalOrderView:
    """Full imported input in source-file order, explicitly for diagnostics only.

    Feed this embedding/order to the whole refiner, not just its joint stage.
    The dataset view uses the same order and preserves existing annotation gates.
    This recovers saved CSV order, not proof of the historical training inputs.
    """

    embedding: EmbeddingView
    dataset: DatasetView
    canonical_cell_ids: tuple
    permutation_file_sha256: str

    def parent_reference(self):
        return {**self.embedding.parent_reference(),
                "row_order_policy": "historical_source_order_diagnostic",
                "source_row_permutation_file_sha256": self.permutation_file_sha256}

    def canonicalize_output(self, values, *, cell_ids):
        """Realign explicitly ID-tagged output; never assume its incoming order."""
        ids = validate_cell_ids(cell_ids)
        values = np.asarray(values)
        if (values.ndim != 2 or values.shape[0] != len(ids) or values.shape[1] < 1
                or values.dtype.kind != "f" or not np.isfinite(values).all()):
            raise ValueError("Output must be a finite floating-point matrix with one row per ID")
        indices = alignment_indices(self.canonical_cell_ids, ids)
        aligned = np.ascontiguousarray(values[indices])
        aligned.flags.writeable = False
        return aligned, self.canonical_cell_ids


class Store:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.manifest = read_json(self.path / "run.json")
        if self.manifest.get("status") != "succeeded" or self.manifest.get("kind") != "step3a_data_store":
            raise ValueError("Only a completed Step 3A store can be loaded")
        self.index = self._json("store.json")
        if self.index.get("schema_version") != 1:
            raise ValueError("Unsupported store schema")

    def _artifact(self, relative):
        path = project_path(self.path, relative)
        expected = self.manifest["artifacts"].get(relative)
        if expected is None or file_fingerprint(path) != expected:
            raise ValueError(f"Store artifact integrity failure: {relative}")
        return path

    def _json(self, relative):
        return read_json(self._artifact(relative))

    def dataset(self, identifier, cell_ids=None):
        record = self._json(self.index["datasets"][identifier])
        validate_cell_ids(record["cell_ids"])
        if canonical_hash(record["cell_ids"]) != record["report"]["cell_order_sha256"]:
            raise ValueError("Dataset cell-order hash mismatch")
        return DatasetView(record, canonical_selection(record["cell_ids"], cell_ids))

    def embedding(self, dataset_id, name, cell_ids=None, *, allow_graph_proxy=False):
        dataset = self.dataset(dataset_id, cell_ids)
        meta = self._json(self.index["embeddings"][dataset_id][name])
        if meta["kind"] == "graph_proxy" and not allow_graph_proxy:
            raise ValueError("BBKNN graph is absent; coordinates may only be loaded explicitly as a graph proxy")
        if meta["dataset_fingerprint"] != dataset.record["dataset_fingerprint"]:
            raise ValueError("Embedding belongs to a different dataset fingerprint")
        if meta["canonical_cell_order_sha256"] != dataset.record["report"]["cell_order_sha256"]:
            raise ValueError("Embedding row order differs from dataset")
        path = self._artifact(meta["values_path"])
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if (list(values.shape) != meta["shape"] or values.dtype != np.dtype(meta["dtype"])
                or values.dtype.kind != "f" or array_hash(values) != meta["values_sha256"]):
            raise ValueError("Stored embedding shape, precision or numeric hash mismatch")
        if cell_ids is not None:
            values = values[dataset.indices]
        values.flags.writeable = False
        meta["stored_values_file_sha256"] = self.manifest["artifacts"][meta["values_path"]]["sha256"]
        return EmbeddingView(values, dataset.cell_ids, meta)

    def historical_input(self, dataset_id, name):
        """Restore FULL imported source order, never change canonical loading.

        This opt-in path is for historical-settings/metric diagnostics. Subsets
        and newly recomputed baselines have no historical-order replay claim.
        """
        canonical = self.embedding(dataset_id, name)
        if canonical.metadata["kind"] not in {"baseline", "historical_refined"}:
            raise ValueError("Historical order requires an imported historical coordinate file")
        relative = canonical.metadata["source_rows_path"]
        permutation = np.load(self._artifact(relative), allow_pickle=False)
        n = len(canonical.cell_ids)
        if (permutation.dtype.kind not in "iu" or permutation.shape != (n,)
                or not np.array_equal(np.sort(permutation), np.arange(n))):
            raise ValueError("Invalid saved source-row permutation")
        # Imported canonical[i] = source[permutation[i]], so invert to restore.
        source_indices = np.argsort(permutation)
        ids = tuple(canonical.cell_ids[i] for i in source_indices)
        if canonical_hash(list(ids)) != canonical.metadata["source_cell_order_sha256"]:
            raise ValueError("Reconstructed original cell order does not match its saved hash")
        values = np.ascontiguousarray(canonical.values[source_indices])
        values.flags.writeable = False
        source_indices.flags.writeable = False
        meta = dict(canonical.metadata, view_row_order="historical_source_order_diagnostic",
                    view_cell_order_sha256=canonical_hash(list(ids)))
        dataset = DatasetView(self.dataset(dataset_id).record, source_indices)
        return HistoricalOrderView(EmbeddingView(values, ids, meta), dataset, canonical.cell_ids,
                                   self.manifest["artifacts"][relative]["sha256"])

    def expression_blocks(self, project_root, dataset_id, rows_per_block=1024):
        record = self.dataset(dataset_id).record
        check_sources(project_root, record["source_files"])
        yield from expression_blocks(project_root, record["loader"], rows_per_block)
        check_sources(project_root, record["source_files"])

    def verify(self, project_root=None):
        for relative in self.manifest["artifacts"]:
            self._artifact(relative)
        if project_root is not None:
            check_sources(project_root, self.index["source_files"])
        for identifier, embeddings in self.index["embeddings"].items():
            n = len(self.dataset(identifier).cell_ids)
            for name in embeddings:
                embedding = self.embedding(identifier, name, allow_graph_proxy=True)
                permutation = np.load(self._artifact(embedding.metadata["source_rows_path"]), allow_pickle=False)
                if permutation.dtype.kind not in "iu" or permutation.shape != (n,) or not np.array_equal(np.sort(permutation), np.arange(n)):
                    raise ValueError("Invalid saved source-row permutation")
        return {"passed": True, "dataset_count": len(self.index["datasets"]),
                "embedding_count": sum(len(x) for x in self.index["embeddings"].values()),
                "original_sources_checked": project_root is not None}
