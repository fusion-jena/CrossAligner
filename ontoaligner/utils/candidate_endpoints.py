"""Utilities for separating alignment endpoints from ontology dependencies."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


def load_candidate_iris(path: str) -> Set[str]:
    """Load IRIs from a ROBOT-style term file.

    Blank lines and lines beginning with ``#`` are ignored. Angle-bracketed
    IRIs and tab-separated rows are accepted, so both plain term files and
    simple TSV exports can be used.
    """
    candidate_iris: Set[str] = set()
    with Path(path).expanduser().open(encoding="utf-8") as term_file:
        for line in term_file:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            value = value.split("\t", 1)[0].strip().strip("<>")
            if value.lower() in {"iri", "id", "term"}:
                continue
            candidate_iris.add(value)
    if not candidate_iris:
        raise ValueError(f"Candidate endpoint file contains no IRIs: {path}")
    return candidate_iris


def filter_candidate_endpoints(
    concepts: Iterable[Dict[str, Any]], candidate_iris: Set[str]
) -> List[Dict[str, Any]]:
    """Return candidate concepts in ontology order and validate the term set."""
    concepts = list(concepts)
    available = {str(concept.get("iri")) for concept in concepts}
    missing = candidate_iris - available
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(
            f"{len(missing)} candidate endpoint IRI(s) are absent from the ontology: "
            f"{preview}"
        )
    return [concept for concept in concepts if str(concept.get("iri")) in candidate_iris]


def descendant_candidate_iris(
    concepts: Iterable[Dict[str, Any]], root_iris: Iterable[str]
) -> Set[str]:
    """Return roots and all descendants in the parsed asserted hierarchy.

    This follows the parser's direct ``childrens`` links transitively. It does
    not invoke an OWL reasoner, so descendants that exist only by inference are
    outside its scope.
    """
    concepts = list(concepts)
    children_by_iri = {
        str(concept.get("iri")): {
            str(child.get("iri"))
            for child in concept.get("childrens", [])
            if child.get("iri")
        }
        for concept in concepts
        if concept.get("iri")
    }
    roots = {str(iri) for iri in root_iris}
    if not roots:
        raise ValueError("At least one candidate root IRI is required.")
    missing_roots = roots - children_by_iri.keys()
    if missing_roots:
        preview = ", ".join(sorted(missing_roots)[:5])
        raise ValueError(f"Candidate root IRI(s) are absent from the ontology: {preview}")

    selected: Set[str] = set()
    pending = list(roots)
    while pending:
        current = pending.pop()
        if current in selected:
            continue
        selected.add(current)
        pending.extend(children_by_iri.get(current, set()) - selected)
    return selected