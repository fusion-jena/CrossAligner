from rdflib import Graph, URIRef, BNode, Literal
from rdflib.namespace import OWL, RDF, RDFS, SKOS, DCTERMS
from rdflib.collection import Collection
from tqdm import tqdm
from typing import Any, Dict, List, Optional
import re

from ..base import BaseOntologyParser, OMDataset

track = "Generic"


class GenericOntology(BaseOntologyParser):
    """
    Generic OWL ontology parser.

    Extracts:
      - ontology metadata
      - class info
      - properties (object/datatype + optional rdf:Property)
      - individuals
    """

    def __init__(
        self,
        language: str = "en",
        include_rdf_properties: bool = False,
        normalize_property_labels: bool = True,
        render_complex_domain_range: bool = True,
    ):
        self.language = language
        self.include_rdf_properties = include_rdf_properties
        self.normalize_property_labels = normalize_property_labels
        self.render_complex_domain_range = render_complex_domain_range
        self.graph: Graph = None

    # ---------------------------
    # Basic label helpers
    # ---------------------------

    def is_valid_label(self, label: str) -> Any:
        invalids = ["root", "thing"]
        if label is None:
            return None
        if label.lower() in invalids:
            return None
        return label

    def normalize_label(self, text: str) -> str:
        """
        Normalizes ontology labels / local names:

        - Strips surrounding whitespace
        - Replaces underscores with spaces
        - Splits CamelCase: 'WaterObservationParameter' -> 'Water Observation Parameter'
        - Collapses multiple spaces
        """
        if text is None:
            return None

        s = text.strip()

        # Separate base and fragment for URI-like strings
        if "#" in s:
            _, frag = s.rsplit("#", 1)
        elif "/" in s:
            _, frag = s.rsplit("/", 1)
        else:
            frag = s

        frag_clean = frag.replace("_", " ")
        frag_clean = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", frag_clean)
        frag_clean = re.sub(r"\s+", " ", frag_clean).strip()
        return frag_clean

    def expand_hy_label(self, label: str) -> str:
        if not label or not isinstance(label, str):
            return label

        if label.startswith("HY "):
            core = label[3:]
            return f"Hydrology {core}".strip()

        return label

    '''def expand_hy_label(self, label: str) -> str:

        if not label or not isinstance(label, str):
            return label

        stripped = re.sub(r"^\s*HY[\s_-]+", "", label, flags=re.IGNORECASE).strip()
        if stripped:
            return stripped

        return label'''

    def _finalize_label(self, raw: Optional[str]) -> Optional[str]:
        if raw is None:
            return None
        clean = self.normalize_label(raw)
        clean = self.expand_hy_label(clean)
        return self.is_valid_label(clean)

    def get_label(self, owl_class: str) -> Any:
        """
        Extract label for a given URI in the specified language from the RDF graph.
    
        Preference order:
        1. rdfs:label in configured language    
        2. skos:prefLabel in configured language
        3. first available rdfs:label
        4. first available skos:prefLabel
        5. local name from IRI
        """
        entity = URIRef(owl_class)
        label_predicates = [RDFS.label, SKOS.prefLabel]
    
        # 1) Prefer labels in the chosen language, with predicate priority.
        for predicate in label_predicates:
            labels = list(self.graph.objects(subject=entity, predicate=predicate))
            for lab in labels:
                if getattr(lab, "language", None) == self.language:
                    return self._finalize_label(str(lab))
    
        # 2) Fallback: first non-URI label, with predicate priority.
        for predicate in label_predicates:
            labels = list(self.graph.objects(subject=entity, predicate=predicate))
            for lab in labels:
                raw = str(lab)
                if not raw.startswith("http"):
                    return self._finalize_label(raw)
    
        # 3) Fallback: local name from the IRI.
        if "#" in owl_class:
            local_name = owl_class.split("#")[-1]
        elif "/" in owl_class:
            local_name = owl_class.rstrip("/").split("/")[-1]
        else:
            local_name = owl_class
    
        return self._finalize_label(local_name)
    

    def get_labels_multilang(self, entity: URIRef) -> List[Dict[str, str]]:
        """
        Return all rdfs:label and skos:prefLabel values with language tags.
        """
        labels = []
        for predicate in [RDFS.label, SKOS.prefLabel]:
            for lab in self.graph.objects(entity, predicate):
                labels.append({
                    "value": str(lab),
                    "lang": getattr(lab, "language", None),
                    "predicate": str(predicate),
                })
        return labels

    def get_preferred_label(self, entity: URIRef) -> Optional[str]:
        """
        Like get_label() but for any RDF entity (URIRef/BNode),
        returns a single preferred label in the configured language.
    
        Preference order:
        1. rdfs:label in configured language
        2. skos:prefLabel in configured language
        3. first available rdfs:label
        4. first available skos:prefLabel
        5. local name for URIRef
        """
        label_predicates = [RDFS.label, SKOS.prefLabel]
        chosen = None
    
        for predicate in label_predicates:
            labels = list(self.graph.objects(entity, predicate))
            for lab in labels:
                if getattr(lab, "language", None) == self.language:
                    chosen = str(lab)
                    break
            if chosen is not None:
                break
    
        if chosen is None:
            for predicate in label_predicates:
                labels = list(self.graph.objects(entity, predicate))
                if labels:
                    chosen = str(labels[0])
                    break
    
        if chosen is None:
            if isinstance(entity, URIRef):
                s = str(entity)
                if "#" in s:
                    chosen = s.split("#")[-1]
                elif "/" in s:
                    chosen = s.rstrip("/").split("/")[-1]
                else:
                    chosen = s
            else:
                return None
    
        if not self.normalize_property_labels:
            return self.is_valid_label(chosen)
    
        return self._finalize_label(chosen)

    # ---------------------------
    # Ontology loading & metadata
    # ---------------------------

    def load_ontology(self, input_file_path: str) -> Graph:
        graph = Graph()
        graph.parse(input_file_path, format="xml")
        return graph

    # ---------------------------
    # Class / TBox helpers
    # ---------------------------

    def is_class(self, owl_class: URIRef) -> bool:
        return (self.graph.value(owl_class, RDFS.subClassOf) is not None
                or (owl_class, RDF.type, OWL.Class) in self.graph)

    def get_synonyms(self, owl_class: URIRef) -> List[Dict[str, str]]:
        synonyms = []
        for alt_label in self.graph.objects(owl_class, SKOS.altLabel):
            synonyms.append({"iri": str(owl_class), "label": str(alt_label), "name": str(alt_label)})
        return synonyms

    def get_name(self, owl_class: URIRef) -> str:
        return self.get_label(owl_class)

    def get_iri(self, owl_class: URIRef) -> str:
        return str(owl_class)

    def clean_text(self, text: str) -> str:
        text = text.replace("\n", " ").replace("\t", " ")
        text = text.replace("_", " ")
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = re.sub(r"(\w)\s*-\s*(\w)", r"\1-\2", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\bHY\b", "Hydrology", text)
        return text.strip()

    def get_comments(self, owl_class: URIRef) -> List[str]:
        comments = []
        definition_props = [RDFS.comment, SKOS.definition, DCTERMS.description]
        for defi in definition_props:
            for value in self.graph.objects(owl_class, defi):
                text = self.clean_text(str(value))
                if text not in comments:
                    comments.append(text)
        return comments

    def get_parents(self, owl_class: URIRef) -> List[Dict[str, str]]:
        parents = []
        for parent in self.graph.objects(owl_class, RDFS.subClassOf):
            if isinstance(parent, BNode):
                continue
            if self.is_class(parent):
                label = self.get_label(parent)
                iri = self.get_iri(parent)
                if label and iri:
                    parents.append({"iri": iri, "label": label, "name": label})
        return parents

    def get_childrens(self, owl_class: URIRef) -> List[Dict[str, str]]:
        childrens = []
        for child in self.graph.subjects(RDFS.subClassOf, owl_class):
            if isinstance(child, BNode):
                continue
            if self.is_class(child):
                label = self.get_label(child)
                iri = self.get_iri(child)
                if label and iri:
                    childrens.append({"iri": iri, "label": label, "name": label})
        return childrens

    def get_class_restrictions(self, cls: URIRef) -> List[Dict[str, Any]]:
        restrictions = []
        for sup in self.graph.objects(cls, RDFS.subClassOf):
            if isinstance(sup, BNode) and (sup, RDF.type, OWL.Restriction) in self.graph:
                prop = self.graph.value(sup, OWL.onProperty)
                some = self.graph.value(sup, OWL.someValuesFrom)
                allv = self.graph.value(sup, OWL.allValuesFrom)
                exact = self.graph.value(sup, OWL.cardinality)
                minc = self.graph.value(sup, OWL.minCardinality)
                maxc = self.graph.value(sup, OWL.maxCardinality)

                restrictions.append({
                    "on_property": str(prop) if prop else None,
                    "someValuesFrom": str(some) if some else None,
                    "allValuesFrom": str(allv) if allv else None,
                    "cardinality": int(exact) if exact else None,
                    "minCardinality": int(minc) if minc else None,
                    "maxCardinality": int(maxc) if maxc else None,
                })
        return restrictions

    def get_class_info(self, owl_class: URIRef) -> Any:
        label = self.get_label(owl_class)
        iri = self.get_iri(owl_class)

        if not label or not iri:
            return None

        return {
            "name": label,
            "iri": iri,
            "label": label,
            "childrens": self.get_childrens(owl_class),
            "parents": self.get_parents(owl_class),
            "synonyms": self.get_synonyms(owl_class),
            "comment": self.get_comments(owl_class),
            "restrictions": self.get_class_restrictions(owl_class),
        }

    def extract_data(self, graph: Any) -> List[Dict[str, Any]]:
        self.graph = graph
        parsed_ontology = []

        for owl_class in tqdm(self.graph.subjects(RDF.type, OWL.Class)):
            if isinstance(owl_class, BNode):
                continue
            class_info = self.get_class_info(owl_class)
            if class_info:
                parsed_ontology.append(class_info)

        return parsed_ontology

    # ---------------------------
    # Properties
    # ---------------------------

    # This comment is with URI
    """def get_property_comments(self, prop: URIRef) -> List[Dict[str, str]]:
        comments = []
        for c in self.graph.objects(prop, RDFS.comment):
            comments.append({"value": self.clean_text(str(c))})
        return comments"""

    def get_property_comments(self, prop: URIRef) -> List[str]:
        comments = []
        definition_props = [RDFS.comment, SKOS.definition, DCTERMS.description]
        for defi in definition_props:
            for value in self.graph.objects(prop, defi):
                if not isinstance(value, Literal):
                    continue # skip URIs/blank nodes
                text = self.clean_text(str(value))
                if text not in comments:
                    comments.append(text)
        return comments

    def _local_name(self, uri: URIRef) -> str:
        s = str(uri)
        if "#" in s:
            return s.split("#")[-1]
        if "/" in s:
            return s.rstrip("/").split("/")[-1]
        return s

    def _render_union_of(self, list_node: BNode) -> List[Dict[str, Any]]:
        """
        Render owl:unionOf list into a list of readable items.
        """
        items = []
        try:
            for member in Collection(self.graph, list_node):
                items.append(self._render_node(member))
        except Exception:
            items.append({"type": "bnode", "value": str(list_node)})
        return items

    def _render_restriction(self, r: BNode) -> Dict[str, Any]:
        """
        Render owl:Restriction into a readable dict.
        """
        on_prop = self.graph.value(r, OWL.onProperty)
        some = self.graph.value(r, OWL.someValuesFrom)
        allv = self.graph.value(r, OWL.allValuesFrom)
        exact = self.graph.value(r, OWL.cardinality)
        minc = self.graph.value(r, OWL.minCardinality)
        maxc = self.graph.value(r, OWL.maxCardinality)

        out = {
            "type": "restriction",
            "onProperty": str(on_prop) if on_prop else None,
            "onProperty_label": self.get_preferred_label(on_prop) if on_prop else None,
            "someValuesFrom": self._render_node(some) if some else None,
            "allValuesFrom": self._render_node(allv) if allv else None,
            "cardinality": int(exact) if exact else None,
            "minCardinality": int(minc) if minc else None,
            "maxCardinality": int(maxc) if maxc else None,
        }
        return out

    def _rendered_labels_only(self, rendered_nodes: List[Dict[str, Any]]) -> List[str]:
        """
        Take output of _render_node(...) and return only label strings.
        For complex nodes (restriction/unionOf/bnode), fall back to a readable string.
        """
        labels = []
        for n in rendered_nodes or []:
            if isinstance(n, dict):
                if n.get("label"):
                    labels.append(n["label"])
                else:
                    labels.append(str(n))
            else:
                labels.append(str(n))
        return labels


    def _render_node(self, node: Any) -> Dict[str, Any]:
        """
        Turn a domain/range node into something readable:
          - URIRef -> iri + label
          - BNode restriction -> readable restriction dict
          - BNode unionOf -> list of members
          - other BNode -> bnode id
        """
        if node is None:
            return {"type": "none", "value": None}

        if isinstance(node, URIRef):
            return {
                "type": "uri",
                "iri": str(node),
                "label": self.get_preferred_label(node),
            }

        if isinstance(node, BNode):
            # restriction?
            if (node, RDF.type, OWL.Restriction) in self.graph:
                return self._render_restriction(node)

            # unionOf?
            union_list = self.graph.value(node, OWL.unionOf)
            if union_list is not None:
                return {
                    "type": "unionOf",
                    "members": self._render_union_of(union_list),
                }

            # sometimes unionOf is directly attached (rare), handle too
            if (node, OWL.unionOf, None) in self.graph:
                ul = self.graph.value(node, OWL.unionOf)
                return {
                    "type": "unionOf",
                    "members": self._render_union_of(ul) if ul else [{"type": "bnode", "value": str(node)}],
                }

            return {"type": "bnode", "value": str(node)}

        # literal or other
        return {"type": "literal", "value": str(node)}

    
    def _extract_domain_range_raw(self, prop: URIRef) -> Dict[str, List[str]]:

        domains_raw = list(self.graph.objects(prop, RDFS.domain))
        ranges_raw = list(self.graph.objects(prop, RDFS.range))

        return {
            "domains": [str(d) for d in domains_raw],
            "ranges": [str(r) for r in ranges_raw],
        }
        
    # This is addition fuction required to keep extract_domain_range_raw and extract_domain_range_labels run smoothly
    def _rendered_labels_only(self, rendered_nodes: List[Dict[str, Any]]) -> List[str]:
        labels = []
        for n in rendered_nodes or []:
            if isinstance(n, dict) and n.get("label"):
                labels.append(n["label"])
            else:
                labels.append(str(n))
        return labels


    def _extract_domain_range_labels(self, prop: URIRef) -> Dict[str, List[str]]:
        """
        Task 2: Extract human-readable domain/range labels only.
        Uses _render_node + _rendered_labels_only (your existing helpers).
        Returns:
          {"domain_label": [...], "range_label": [...]}
        """
        if not getattr(self, "render_complex_domain_range", False):
            return {"domain_label": [], "range_label": []}

        domains_raw = list(self.graph.objects(prop, RDFS.domain))
        ranges_raw = list(self.graph.objects(prop, RDFS.range))

        domains_rendered = [self._render_node(d) for d in domains_raw]
        ranges_rendered = [self._render_node(r) for r in ranges_raw]

        return {
            "domain_label": self._rendered_labels_only(domains_rendered),
            "range_label": self._rendered_labels_only(ranges_rendered),
        }

    def _inverse_of(self, prop: URIRef) -> List[Dict[str, Any]]:
        """
        Extract owl:inverseOf (both directions, because some ontologies only declare one side).
        """
        inverses = set()

        # prop owl:inverseOf inv
        for inv in self.graph.objects(prop, OWL.inverseOf):
            if isinstance(inv, (URIRef, BNode)):
                inverses.add(inv)

        # inv owl:inverseOf prop
        for inv in self.graph.subjects(OWL.inverseOf, prop):
            if isinstance(inv, (URIRef, BNode)):
                inverses.add(inv)

        rendered = []
        for inv in inverses:
            if isinstance(inv, URIRef):
                rendered.append({
                    "iri": str(inv),
                    "label": self.get_preferred_label(inv),
                })
            else:
                rendered.append({"iri": str(inv), "label": None})
        return rendered

    def _property_types(self, prop: URIRef) -> List[str]:
        """
        Collect RDF/OWL type(s) attached to the property.
        """
        return [str(t) for t in self.graph.objects(prop, RDF.type)]

    def _iter_all_properties(self) -> List[URIRef]:
        """
        Decide which nodes count as properties based on options.
        """
        props = set()

        # Always include OWL-typed properties
        for p in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            if isinstance(p, URIRef):
                props.add(p)
        for p in self.graph.subjects(RDF.type, OWL.DatatypeProperty):
            if isinstance(p, URIRef):
                props.add(p)

        # Optionally include rdf:Property
        if self.include_rdf_properties:
            for p in self.graph.subjects(RDF.type, RDF.Property):
                if isinstance(p, URIRef):
                    props.add(p)

        return sorted(props, key=lambda x: str(x))

    def get_properties(self) -> List[Dict[str, Any]]:
        
        results = []

        for prop in self._iter_all_properties():
            labels_multilang = self.get_labels_multilang(prop)
            preferred_label = self.get_preferred_label(prop)

            
            dr_raw = self._extract_domain_range_raw(prop)         # {"domains": [...], "ranges": [...]}
            dr_labels = self._extract_domain_range_labels(prop)   # {"domain_label": [...], "range_label": [...]}

            info = {
                "iri": str(prop),
                "property_label": preferred_label,
                "labels": labels_multilang,
                "comments": self.get_property_comments(prop),
                "types": self._property_types(prop),
                "inverse_of": self._inverse_of(prop),
                
                "domains": dr_raw["domains"],
                "ranges": dr_raw["ranges"],
                "domain_label": dr_labels["domain_label"],
                "range_label": dr_labels["range_label"],
                
                "subproperties": [str(sp) for sp in self.graph.subjects(RDFS.subPropertyOf, prop)],
                "superproperties": [str(sp) for sp in self.graph.objects(prop, RDFS.subPropertyOf)],
            }
            results.append(info)

        return results

    # ---------------------------
    # Individuals 
    # ---------------------------

    def get_individuals(self) -> List[Dict[str, Any]]:
        inds = []
        for ind in self.graph.subjects(RDF.type, OWL.NamedIndividual):
            if isinstance(ind, BNode):
                continue
            entry = {
                "iri": str(ind),
                "labels": self.get_labels_multilang(ind),
                "comments": self.get_property_comments(ind),
                "types": [str(t) for t in self.graph.objects(ind, RDF.type) if t != OWL.NamedIndividual],
            }
            inds.append(entry)
        return inds

    # ---------------------------
    # High-level entry point
    # ---------------------------

    def parse_ontology_file(self, input_file_path: str) -> Dict[str, Any]:
        graph = self.load_ontology(input_file_path)
        self.graph = graph

        # New: "properties" is the unified extraction (object + datatype + optional rdf:Property)
        properties = self.get_properties()

        return {
            "classes": self.extract_data(graph),
            "properties": properties,
            "individuals": self.get_individuals(),
        }


class GenericOMDataset(OMDataset):
    track = track
    ontology_name = "Source-Target"
    source_ontology = GenericOntology()
    target_ontology = GenericOntology()
