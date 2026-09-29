# Copyright 2025 Scientific Knowledge Organization (SciKnowOrg) Research Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script defines three encoder classes that inherit from the LightweightEncoder class.
These encoders are used to process and transform OWL (Web Ontology Language) items into a format suitable for downstream tasks.
Each encoder is specialized for different types of OWL items: Concept, Concept with Children, and Concept with Parent.

Classes:
    - ConceptLightweightEncoder: Encodes OWL items representing concepts.
    - ConceptChildrenLightweightEncoder: Encodes OWL items representing concepts and their children.
    - ConceptParentLightweightEncoder: Encodes OWL items representing concepts and their parents.
"""
from typing import Any, Dict, List

from ..base import BaseEncoder
from .concept_formatting import format_concept_context, format_concept_definition_context

class LightweightEncoder(BaseEncoder):
    """
    A lightweight encoder for parsing ontology data and preprocessing it.

    This class provides methods for parsing ontological data, applying text preprocessing,
    and formatting the data into a structure suitable for further processing.
    """
    def parse(self, **kwargs) -> Any:
        """
        Parses the source and target ontologies, applying preprocessing.

        This method extracts ontology items (IRI and label) from the source and target ontologies,
        applies text preprocessing to the labels, and returns the encoded data.

        Parameters:
            **kwargs: Contains the source and target ontologies as keyword arguments.

        Returns:
            list: A list containing two elements, the processed source and target ontologies.
        """
        source_onto, target_onto = kwargs["source"], kwargs["target"]
        source_ontos = []
        for source in source_onto:
            encoded_source = self.get_owl_items(owl=source)
            #encoded_source["text"] = self.preprocess(encoded_source["Property"])
            encoded_source["text"] = self.preprocess(encoded_source["text"])
            source_ontos.append(encoded_source)
        target_ontos = []
        for target in target_onto:
            encoded_target = self.get_owl_items(owl=target)
            #encoded_target["text"] = self.preprocess(encoded_target["Property"])
            encoded_target["text"] = self.preprocess(encoded_target["text"])
            target_ontos.append(encoded_target)
        return [source_ontos, target_ontos]

    def __str__(self):
        """
        Returns a string representation of the encoder.

        Returns:
            dict: A dictionary with the class name as key and items_in_owl as value.
        """
        return {"LightweightEncoder": self.items_in_owl}

    def get_owl_items(self, owl: Dict) -> Any:
        """
        Abstract method for extracting ontology data.

        This method should be implemented by subclasses to extract specific ontology data
        (e.g., IRI and label) from the provided ontology item.

        Parameters:
            owl (Dict): A dictionary representing an ontology item.

        Returns:
            Any: The extracted ontology data.
        """
        pass

    def get_encoder_info(self):
        """
        Provides information about the encoder.

        Returns:
            str: A description of the encoder's function in the overall pipeline.
        """
        return "INPUT CONSIST OF COMBINED INFORMATION TO FUZZY STRING MATCHING"


# This only for labels




class PropertyLightweightEncoder(LightweightEncoder):
    items_in_owl: str = "(Property)"

    def get_owl_items(self, owl: Dict) -> Any:
        text = owl.get("property_label") or owl.get("iri")
        return {"iri": owl["iri"], "Property": text}

'''class PropertyDomainRangeLightweightEncoder(LightweightEncoder):
    items_in_owl: str = "(Property+DomainRange)"

    def get_owl_items(self, owl: Dict) -> Any:
        label = owl.get("property_label") or ""

        domain_labels = owl.get("domain_label", [])
        range_labels = owl.get("range_label", [])

        # fallback: use local names from IRIs if labels are empty
        if not domain_labels:
            domain_labels = [d.rstrip("/").split("/")[-1].split("#")[-1] for d in owl.get("domains", [])]
        if not range_labels:
            range_labels = [r.rstrip("/").split("/")[-1].split("/")[-1].split("#")[-1] for r in owl.get("ranges", [])]

        dom = " ".join(domain_labels)
        rng = " ".join(range_labels)

        text = " ".join([label, dom, rng]).strip() or owl.get("iri")
        return {"iri": owl["iri"], "Property": text}'''
        
class PropertyDomainRangeLightweightEncoder(LightweightEncoder):        
    items_in_owl: str = "(Property+DomainRange)"

    def __init__(self, require_domain_and_range: bool = False, *args, **kwargs):
        """
        require_domain_and_range:
          - False: always return text (even if domain/range missing)
          - True: return Domain/Property/Range format ONLY if both domain and range exist,
                  otherwise fall back to property label only
        """
        super().__init__(*args, **kwargs)
        self.require_domain_and_range = require_domain_and_range

    def _local_name(self, iri: str) -> str:
        return iri.rstrip("/").split("/")[-1].split("#")[-1]

    def _format_dpr(self, dom_labels: List[str], prop_text: str, rng_labels: List[str]) -> str:
        dom_text = "; ".join(dom_labels) if dom_labels else "(none)"
        rng_text = "; ".join(rng_labels) if rng_labels else "(none)"
        return f"Domain: {dom_text}\nProperty: {prop_text}\nRange: {rng_text}"

    def get_owl_items(self, owl: Dict) -> Any:
        prop_label = (owl.get("property_label") or "").strip()
        prop_text = prop_label if prop_label else (self._local_name(owl.get("iri", "")) or "(none)")

        domain_labels = list(owl.get("domain_label") or [])
        range_labels = list(owl.get("range_label") or [])

        # fallback: use local names from IRIs if labels are empty
        if not domain_labels:
            domain_labels = [self._local_name(d) for d in (owl.get("domains") or [])]
        if not range_labels:
            range_labels = [self._local_name(r) for r in (owl.get("ranges") or [])]

        has_domain = len(domain_labels) > 0
        has_range = len(range_labels) > 0

        # Toggle behavior:
        # If require_domain_and_range is True, only show DPR when both are present.
        if self.require_domain_and_range and not (has_domain and has_range):
            # fall back to property only (no DPR block)
            return {"iri": owl["iri"], "text": prop_text} #Return only Property label

        # Otherwise show DPR block (even if one side is missing, it will show "(none)")
        text = self._format_dpr(domain_labels, prop_text, range_labels)
        return {"iri": owl["iri"], "Property": text}



'''class ConceptLightweightEncoder(LightweightEncoder):
    """
    Encodes OWL class concepts using ALL fields from GenericOntology.get_class_info().
    """
    items_in_owl: str = "(Concept)"

    def get_owl_items(self, owl: Dict) -> Any:
        """
        Uses: label, comment, synonyms, restrictions, name from generic.py
        """
        # Core fields - all directly from GenericOntology.get_class_info()
        label = owl.get("label", "")
        name = owl.get("name", "")  # ← NEW: from get_name()
        comments = owl.get("comment", []) or []
        synonyms = owl.get("synonyms", []) or []
        restrictions = owl.get("restrictions", []) or []

        # 1. Main comment (list → first item)
        main_comment = comments[0] if comments else ""

        # 2. Synonyms (NEW from get_synonyms())
        syn_labels = [s.get("label", "") for s in synonyms if s.get("label")]
        syn_text = f"Synonyms: {', '.join(syn_labels)}." if syn_labels else ""

        # 3. Restrictions (NEW from get_class_restrictions())
        restr_parts = []
        for r in restrictions:
            prop = r.get("on_property")
            card = r.get("cardinality")
            if prop and card is not None:
                restr_parts.append(f"exactly {card} {prop}")
            elif prop:
                restr_parts.append(str(prop))
        
        restr_text = f"Constraints: {' '.join(restr_parts)}." if restr_parts else ""

        # 4. Build single text (same output format as before)
        parts = []
        if label:
            parts.append(label)
        if name and name != label:  # avoid duplication
            parts.append(f"({name})")
        if main_comment:
            parts.append(main_comment)
        if syn_text:
            parts.append(syn_text)
        if restr_text:
            parts.append(restr_text)

        text = " ".join(parts).strip()

        return {
            "iri": owl["iri"],
            "text": text  # ← enriched but still single string
        }'''


'''class ConceptLightweightEncoder(LightweightEncoder):
    """
    Encodes OWL items that represent concepts.

    This class inherits from the `LightweightEncoder` class and is designed to encode OWL items that consist of
    concepts. The `get_owl_items` method retrieves the IRI and label of the concept.

    Attributes:
        items_in_owl (str): Specifies the type of OWL items being encoded, in this case, a Concept.
    """
    items_in_owl: str = """(Concept)"""

    def _join_labels(self, items: List[Dict[str, Any]]) -> str:
        labels = [item.get("label") or item.get("name") or item.get("iri") for item in items or []]
        labels = [label for label in labels if label]
        return "; ".join(labels) if labels else ""

    def _join_comments(self, comments: List[str]) -> str:
        cleaned_comments = [comment.strip() for comment in (comments or []) if isinstance(comment, str) and comment.strip()]
        return " ".join(cleaned_comments) if cleaned_comments else ""

    def _formastriction in restrictions or []:
            on_property = restriction.get("on_property")
            some_values_from = restriction.get("someValuesFrom")
            all_values_from = restriction.get("allValuesFrom")
            cardinality = restriction.get("cardinality")
            min_cardinality = restriction.get("minCardinality")
            max_cardinality = restriction.get("maxCardinality")

            if on_property and some_values_from:
                rendered_restrictions.append(f"{on_property} some {some_values_from}")
            elif on_property and all_values_from:
                rendered_restrictions.append(f"{on_property} all {all_values_from}")
            elif on_property and cardinality is not None:
                rendered_restrictions.append(f"{on_property} exactly {cardinality}")
            elif on_property and min_cardinality is not None:
                rendered_restrictions.append(f"{on_property} min {min_cardinality}")
            elif on_property and max_cardinality is not None:
                rendered_restrictions.append(f"{on_property} max {max_cardinality}")
            elif on_property:
                rendered_restrictions.append(str(on_property))
        return "; ".join(rendered_restrictions) if rendered_restrictions else ""

    def get_owl_items(self, owl: Dict) -> Any:
        """
        Extracts the IRI and label of a concept from the given OWL item.

        Parameters:
            owl (Dict): A dictionary representing an OWL item, expected to contain 'iri' and 'label' keys.

        Returns:
            Dict: A dictionary containing the IRI and label of the concept.
        """
        
        concept_label = owl.get("label") or owl.get("name") or owl.get("iri", "")
        parents = self._join_labels(owl.get("parents", []))
        childrens = self._join_labels(owl.get("childrens", []))
        synonyms = self._join_labels(owl.get("synonyms", []))
        definition = self._join_comments(owl.get("comment", []))
        restrictions = self._format_restrictions(owl.get("restrictions", []))

        text = (
            f"Concept: {concept_label}\n"
            f"Parents: {parents}\n"
            f"Children: {childrens}\n"
            f"Synonyms: {synonyms}\n"
            f"Definition: {definition}\n"
            f"Restrictions: {restrictions}"
        )
        return {"iri": owl["iri"], "text": text}

        #return {"iri": owl["iri"], "text": owl["label"]}t_restrictions(self, restrictions: List[Dict[str, Any]]) -> str:
        rendered_restrictions = []
        for restriction in restrictions or []:
            on_property = restriction.get("on_property")
            some_values_from = restriction.get("someValuesFrom")
            all_values_from = restriction.get("allValuesFrom")
            cardinality = restriction.get("cardinality")
            min_cardinality = restriction.get("minCardinality")
            max_cardinality = restriction.get("maxCardinality")

            if on_property and some_values_from:
                rendered_restrictions.append(f"{on_property} some {some_values_from}")
            elif on_property and all_values_from:
                rendered_restrictions.append(f"{on_property} all {all_values_from}")
            elif on_property and cardinality is not None:
                rendered_restrictions.append(f"{on_property} exactly {cardinality}")
            elif on_property and min_cardinality is not None:
                rendered_restrictions.append(f"{on_property} min {min_cardinality}")
            elif on_property and max_cardinality is not None:
                rendered_restrictions.append(f"{on_property} max {max_cardinality}")
            elif on_property:
                rendered_restrictions.append(str(on_property))
        return "; ".join(rendered_restrictions) if rendered_restrictions else ""

    def get_owl_items(self, owl: Dict) -> Any:
        """
        Extracts the IRI and label of a concept from the given OWL item.

        Parameters:
            owl (Dict): A dictionary representing an OWL item, expected to contain 'iri' and 'label' keys.

        Returns:
            Dict: A dictionary containing the IRI and label of the concept.
        """
        
        concept_label = owl.get("label") or owl.get("name") or owl.get("iri", "")
        parents = self._join_labels(owl.get("parents", []))
        childrens = self._join_labels(owl.get("childrens", []))
        synonyms = self._join_labels(owl.get("synonyms", []))
        definition = self._join_comments(owl.get("comment", []))
        restrictions = self._format_restrictions(owl.get("restrictions", []))

        text = (
            f"Concept: {concept_label}\n"
            f"Parents: {parents}\n"
            f"Children: {childrens}\n"
            f"Synonyms: {synonyms}\n"
            f"Definition: {definition}\n"
            f"Restrictions: {restrictions}"
        )
        return {"iri": owl["iri"], "text": text}

        #return {"iri": owl["iri"], "text": owl["label"]}'''


class ConceptLightweightEncoder(LightweightEncoder):
    """
    Encodes OWL items that represent concepts.

    This class inherits from the `LightweightEncoder` class and is designed to encode OWL items that consist of
    concepts. The `get_owl_items` method retrieves the IRI and label of the concept.

    Attributes:
        items_in_owl (str): Specifies the type of OWL items being encoded, in this case, a Concept.
    """
    items_in_owl: str = """(Concept)"""

    def get_owl_items(self, owl: Dict) -> Any:
        """
        Extracts the IRI and label of a concept from the given OWL item.

        Parameters:
            owl (Dict): A dictionary representing an OWL item, expected to contain 'iri' and 'label' keys.

        Returns:
            Dict: A dictionary containing the IRI and label of the concept.
        """
        #return {"iri": owl["iri"], "text": owl["label"]}
        return {"iri": owl["iri"], "text": format_concept_context(owl)}


## This only for retrival (Candidate serach using SBERT in RAG aligner)
class ConceptDefinitionLightweightEncoder(LightweightEncoder):
    """
    Encodes concepts for retrieval using only concept and definition fields.

    This encoder is intended for retriever-side text where compact context tends to
    work better than fully expanded hierarchical context.
    """
    items_in_owl: str = """(Concept+Definition)"""

    def get_owl_items(self, owl: Dict) -> Any:
        return {"iri": owl["iri"], "text": format_concept_definition_context(owl)}

        
class ConceptChildrenLightweightEncoder(LightweightEncoder):
    """
    Encodes OWL items that represent concepts and their children.

    This class inherits from the `LightweightEncoder` class and is designed to encode OWL items that consist of
    concepts and their children. The `get_owl_items` method retrieves the IRI, label of the concept, and the labels of its children.

    Attributes:
        items_in_owl (str): Specifies the type of OWL items being encoded, in this case, a Concept with Children.
    """
    items_in_owl: str = "(Concept, Children)"

    def get_owl_items(self, owl: Dict) -> Any:
        """
        Extracts the IRI and label of a concept, along with the labels of its children, from the given OWL item.

        Parameters:
            owl (Dict): A dictionary representing an OWL item, expected to contain 'iri', 'label',
                        and 'childrens' keys where 'childrens' is a list of children with 'label' attributes.

        Returns:
            Dict: A dictionary containing the IRI, label of the concept, and the concatenated labels of its children.
        """
        childrens = ", ".join([children["label"] for children in owl["childrens"]])
        #return {"iri": owl["iri"], "text": owl["label"] + " " + str(childrens)}
        return {"iri": owl["iri"], "text": owl["label"], "comment": owl["comment"]}


class ConceptParentLightweightEncoder(LightweightEncoder):
    """
    Encodes OWL items that represent concepts and their parents.

    This class inherits from the `LightweightEncoder` class and is designed to encode OWL items that consist of
    concepts and their parents. The `get_owl_items` method retrieves the IRI, label of the concept, and the labels of its parents.

    Attributes:
        items_in_owl (str): Specifies the type of OWL items being encoded, in this case, a Concept with Parent.
    """
    items_in_owl: str = "(Concept, Parent)"

    def get_owl_items(self, owl: Dict) -> Any:
        """
        Extracts the IRI and label of a concept, along with the labels of its parents, from the given OWL item.

        Parameters:
            owl (Dict): A dictionary representing an OWL item, expected to contain 'iri', 'label',
                        and 'parents' keys where 'parents' is a list of parents with 'label' attributes.

        Returns:
            Dict: A dictionary containing the IRI, label of the concept, and the concatenated labels of its parents.
        """
        parents = ", ".join([parent["label"] for parent in owl["parents"]])
        #return {"iri": owl["iri"], "text": owl["label"] + " " + str(parents)}
        return {"iri": owl["iri"], "text": owl["label"], "comment": owl["comment"]}



class ConceptChildrenParentLightweightEncoder(LightweightEncoder):
    """
    Encodes OWL items that represent concepts along with both their parents and their children.

    This class inherits from the `LightweightEncoder` class and is designed to encode OWL items that
    consist of concepts, their parents, and their children. The `get_owl_items` method retrieves
    the IRI, label of the concept, and the labels of both its parents and its children.

    Attributes:
        items_in_owl (str): Specifies the type of OWL items being encoded, in this case,
                            a Concept with Parents and Children.
    """
    items_in_owl: str = "(Concept, Parents, Children)"

    def get_owl_items(self, owl: Dict) -> Any:
        """
        Extracts the IRI and label of a concept, along with the labels of its parents and children,
        from the given OWL item.

        Parameters:
            owl (Dict): A dictionary representing an OWL item, expected to contain 'iri', 'label',
                        'parents' and 'childrens', where 'parents' and 'childrens' are lists with
                        dictionaries containing 'label' attributes.

        Returns:
            Dict: A dictionary containing the IRI, label of the concept, and the concatenated labels
                  of its parents and children.
        """
        parents_list = owl.get("parents", [])
        children_list = owl.get("childrens", [])
        
        if not parents_list:                                # If NO parents → return empty result
            return {"iri": owl["iri"], "text": ""}

        if not children_list:                               # If NO childrens → return empty result
            return {"iri": owl["iri"], "text": ""}
            
        # Extract labels
        parents = ", ".join([parent["label"] for parent in parents_list])
        childrens = ", ".join([child["label"] for child in children_list])

        # Always concatenate parent → concept → children
        text = f"{parents} {owl['label']} {childrens}"

        return {"iri": owl["iri"], "text": text}



## With definations
'''# This is label + defination
class ConceptLightweightEncoder(LightweightEncoder):
    items_in_owl: str = "(Concept)"

    def get_owl_items(self, owl: Dict) -> Any:
        label = owl.get("label", "").strip()
        comments = owl.get("comment", [])

        # Convert list of comments → single clean string
        if isinstance(comments, list):
            comment_text = " ".join(comments).strip()
        else:
            comment_text = str(comments).strip()

        # Build final text
        if comment_text:
            text = f"{label} : {comment_text}"
        else:
            text = label

        return {"iri": owl["iri"], "text": text}'''

   