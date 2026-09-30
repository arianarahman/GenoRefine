"""Deterministic external marker-panel parsing and construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import gzip
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence
from xml.etree import ElementTree

from ..integrity import canonical_hash, file_fingerprint


@dataclass(frozen=True, order=True)
class OntologyTerm:
    ontology_id: str
    name: str
    synonyms: tuple[str, ...] = ()
    obsolete: bool = False


@dataclass(frozen=True)
class OntologyIndex:
    terms: tuple[OntologyTerm, ...]

    def __post_init__(self):
        ids = [term.ontology_id for term in self.terms]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate ontology IDs")

    @property
    def by_id(self) -> dict[str, OntologyTerm]:
        return {term.ontology_id: term for term in self.terms}

    @property
    def exact_name_to_id(self) -> dict[str, str]:
        candidates: dict[str, list[str]] = {}
        for term in self.terms:
            if not term.obsolete:
                candidates.setdefault(term.name, []).append(term.ontology_id)
        return {name: values[0] for name, values in candidates.items() if len(values) == 1}

    def resolve_exact(self, label: str) -> str | None:
        return self.exact_name_to_id.get(label)


@dataclass(frozen=True, order=True)
class MarkerRecord:
    ontology_id: str
    official_gene_symbol: str
    source_cell_type: str
    gene_id: str = ""
    species: str = ""
    source: str = ""
    year: int | None = None
    canonical: bool = True


def _open_text(path: Path):
    return (gzip.open(path, "rt", encoding="utf-8-sig", newline="")
            if path.suffix.lower() == ".gz" else
            path.open("r", encoding="utf-8-sig", newline=""))


def _header_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _column(columns: Sequence[object], *aliases: str, required: bool = True) -> object | None:
    lookup: dict[str, object] = {}
    for value in columns:
        key = _header_key(value)
        if key in lookup:
            raise ValueError(f"Ambiguous normalized column name: {value!r}")
        lookup[key] = value
    for alias in aliases:
        if _header_key(alias) in lookup:
            return lookup[_header_key(alias)]
    if required:
        raise ValueError(f"Missing required column; accepted aliases: {aliases}")
    return None


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "canonical marker"}


def parse_panglaodb_human(path: str | Path) -> tuple[MarkerRecord, ...]:
    """Parse canonical human PanglaoDB markers without using benchmark results."""

    path = Path(path)
    with _open_text(path) as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("PanglaoDB table has no header")
        species_col = _column(reader.fieldnames, "species")
        gene_col = _column(reader.fieldnames, "official gene symbol", "gene symbol", "symbol")
        cell_col = _column(reader.fieldnames, "cell type", "celltype")
        canonical_col = _column(reader.fieldnames, "canonical marker", "canonical")
        records = set()
        for row in reader:
            species = str(row[species_col]).strip()
            normalized_species = species.lower()
            species_tokens = set(re.findall(r"[a-z]+", normalized_species))
            if (normalized_species not in {"human", "homo sapiens", "homo sapiens sapiens"}
                    and "hs" not in species_tokens):
                continue
            if not _truthy(row[canonical_col]):
                continue
            gene = str(row[gene_col]).strip()
            cell_type = str(row[cell_col]).strip()
            if not gene or not cell_type or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", gene):
                continue
            records.add(MarkerRecord("", gene, cell_type, species=species,
                                     source="PanglaoDB-2020", canonical=True))
    if not records:
        raise ValueError("No canonical human PanglaoDB markers passed the locked filters")
    return tuple(sorted(records))


def _parse_obo(path: Path) -> tuple[OntologyTerm, ...]:
    terms: list[OntologyTerm] = []
    current: dict[str, object] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines() + ["[Term]"]:
        line = raw.strip()
        if line == "[Term]":
            if current and current.get("id") and current.get("name"):
                terms.append(OntologyTerm(str(current["id"]), str(current["name"]),
                                          tuple(sorted(set(current.get("synonyms", [])))),
                                          bool(current.get("obsolete", False))))
            current = {"synonyms": []}
        elif line.startswith("["):
            if current and current.get("id") and current.get("name"):
                terms.append(OntologyTerm(str(current["id"]), str(current["name"]),
                                          tuple(sorted(set(current.get("synonyms", [])))),
                                          bool(current.get("obsolete", False))))
            current = None
        elif current is not None and line.startswith("id: CL:"):
            current["id"] = line.split("id:", 1)[1].strip()
        elif current is not None and line.startswith("name:"):
            current["name"] = line.split("name:", 1)[1].strip()
        elif current is not None and line.startswith("synonym:"):
            match = re.match(r'synonym:\s+"([^"]+)"', line)
            if match:
                current["synonyms"].append(match.group(1))
        elif current is not None and line == "is_obsolete: true":
            current["obsolete"] = True
    return tuple(terms)


def _parse_owl(path: Path) -> tuple[OntologyTerm, ...]:
    root = ElementTree.parse(path).getroot()
    rdf_about = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"
    labels = "{http://www.w3.org/2000/01/rdf-schema#}label"
    deprecated = "{http://www.w3.org/2002/07/owl#}deprecated"
    terms = []
    for node in root.iter():
        if not node.tag.endswith("}Class"):
            continue
        about = node.attrib.get(rdf_about, "")
        match = re.search(r"CL[_:](\d{7})$", about)
        label = next((child.text for child in node if child.tag == labels and child.text), None)
        if not match or not label:
            continue
        is_obsolete = any(child.tag == deprecated and str(child.text).lower() == "true" for child in node)
        terms.append(OntologyTerm(f"CL:{match.group(1)}", label.strip(), (), is_obsolete))
    return tuple(terms)


def parse_cell_ontology(path: str | Path) -> OntologyIndex:
    """Parse a pinned Cell Ontology OBO or OWL snapshot."""

    path = Path(path)
    terms = _parse_obo(path) if path.suffix.lower() == ".obo" else _parse_owl(path)
    terms = tuple(sorted(term for term in terms if re.fullmatch(r"CL:\d{7}", term.ontology_id)))
    if not terms:
        raise ValueError("No Cell Ontology terms found")
    return OntologyIndex(terms)


def _split(value: object) -> list[str]:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na"}:
        return []
    return [part.strip() for part in re.split(r"[;,|]", text) if part.strip()]


def parse_cellmarker_mouse(path: str | Path, ontology: OntologyIndex, *,
                           maximum_year: int = 2016) -> tuple[MarkerRecord, ...]:
    """Parse pre-2017 normal-mouse CellMarker records with strict identifiers."""

    import pandas as pd
    path = Path(path)
    with pd.ExcelFile(path) as workbook:
        normal_sheets = [name for name in workbook.sheet_names if "normal" in name.lower()]
        all_sheets = [name for name in workbook.sheet_names if name.strip().lower() == "all"]
        if len(normal_sheets) == 1:
            sheet = normal_sheets[0]
            normal_filter_required = False
        elif not normal_sheets and len(all_sheets) == 1:
            # The current CellMarker 2.0 bulk workbook stores normal and cancer
            # records together in one ``All`` worksheet.  Normal status is then
            # an explicit row field rather than a worksheet-level contract.
            sheet = all_sheets[0]
            normal_filter_required = True
        else:
            raise ValueError(
                "CellMarker workbook must contain one unambiguous Normal or All worksheet")
        table = pd.read_excel(workbook, sheet_name=sheet, dtype=str).fillna("")
    species_col = _column(table.columns, "species")
    cell_col = _column(table.columns, "cell name", "cell type", "celltype")
    symbol_col = _column(table.columns, "symbol", "gene symbol", "official gene symbol", "cell marker")
    gene_col = _column(table.columns, "geneid", "gene id", "entrez gene id")
    ontology_col = _column(table.columns, "cell ontology id", "cell ontology", "cl id")
    year_col = _column(table.columns, "year")
    cancer_col = (_column(table.columns, "cancer type")
                  if normal_filter_required else None)
    valid_ids = ontology.by_id
    deduplicated: dict[tuple[str, str], MarkerRecord | None] = {}
    for _, row in table.iterrows():
        species = str(row[species_col]).strip()
        if species.lower() not in {"mouse", "mus musculus", "mm"}:
            continue
        if normal_filter_required and str(row[cancer_col]).strip().lower() != "normal":
            continue
        year_match = re.search(r"\d{4}", str(row[year_col]))
        if not year_match or int(year_match.group()) > maximum_year:
            continue
        ontology_id = str(row[ontology_col]).strip().replace("CL_", "CL:")
        if not re.fullmatch(r"CL:\d{7}", ontology_id):
            continue
        term = valid_ids.get(ontology_id)
        if term is None or term.obsolete:
            continue
        symbols, gene_ids = _split(row[symbol_col]), _split(row[gene_col])
        if len(symbols) != len(gene_ids):
            continue
        for symbol, gene_id in zip(symbols, gene_ids):
            if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", symbol)
                    or not re.fullmatch(r"\d+", gene_id)):
                continue
            key = (ontology_id, gene_id)
            record = MarkerRecord(ontology_id, symbol, str(row[cell_col]).strip() or term.name,
                                  gene_id=gene_id, species=species, source="CellMarker-Normal",
                                  year=int(year_match.group()), canonical=True)
            if key in deduplicated:
                previous = deduplicated[key]
                if previous is None:
                    continue
                if previous.official_gene_symbol != symbol:
                    # A GeneID with competing symbols is not an exact official-
                    # symbol identity.  Exclude the pair rather than choosing a
                    # row based on workbook order.
                    deduplicated[key] = None
                continue
            deduplicated[key] = record
    retained = tuple(sorted(record for record in deduplicated.values()
                            if record is not None))
    if not retained:
        raise ValueError("No mouse CellMarker records passed the locked filters")
    return retained


def build_marker_panel(records: Iterable[MarkerRecord], available_genes: Sequence[str],
                       label_to_source_cell_types: Mapping[str, Sequence[str]], *,
                       dataset: str, source_provenance: Mapping[str, object],
                       minimum_genes: int = 5) -> dict[str, object]:
    """Build a deterministic, result-independent panel by exact gene identity.

    ``label_to_source_cell_types`` is a prespecified biological crosswalk.  No
    embedding, clustering, or downstream score is accepted by this interface.
    """

    if type(minimum_genes) is not int or minimum_genes < 1:
        raise ValueError("minimum_genes must be positive")
    genes = list(available_genes)
    if not genes or any(not isinstance(gene, str) or not gene for gene in genes):
        raise ValueError("available_genes must contain nonempty strings")
    if len(genes) != len(set(genes)):
        raise ValueError("available_genes must be unique for exact matching")
    available = set(genes)
    records = tuple(sorted(set(records)))
    output: dict[str, list[str]] = {}
    excluded: dict[str, dict[str, object]] = {}
    source_map: dict[str, list[str]] = {}
    for label in sorted(label_to_source_cell_types):
        source_types = sorted(set(label_to_source_cell_types[label]))
        if not source_types:
            raise ValueError(f"Label {label!r} has no prespecified source cell type")
        matched = sorted({record.official_gene_symbol for record in records
                          if record.canonical and record.source_cell_type in source_types
                          and record.official_gene_symbol in available})
        source_map[label] = source_types
        if len(matched) >= minimum_genes:
            output[label] = matched
        else:
            excluded[label] = {"reason": "fewer_than_minimum_exact_gene_matches",
                               "exact_gene_matches": len(matched)}
    if not output:
        raise ValueError("No endpoint satisfies the prespecified minimum-gene rule")
    payload = {
        "schema": "genorefine.external_marker_panel.v1",
        "dataset": dataset,
        "selection_policy": {
            "result_independent": True,
            "evaluated_embedding_or_score_used": False,
            "gene_matching": "case-sensitive exact official-symbol match",
            "minimum_genes": minimum_genes,
            "endpoint_crosswalk_frozen_before_scoring": True,
        },
        "source_provenance": dict(source_provenance),
        "label_to_source_cell_types": source_map,
        "genes_by_label": output,
        "excluded_labels": excluded,
        "available_gene_order_sha256": canonical_hash(genes),
    }
    payload["panel_content_sha256"] = canonical_hash(payload)
    return payload


def external_source_record(path: str | Path) -> dict[str, object]:
    path = Path(path)
    return {"path": str(path.resolve()), "fingerprint": file_fingerprint(path)}
