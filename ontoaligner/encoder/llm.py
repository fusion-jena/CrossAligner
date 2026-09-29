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
from typing import Any, Dict, List

from ..base import BaseEncoder
from .concept_formatting import format_concept_context
from .concept_formatting import format_candidate_concept_context

class LLMEncoder(BaseEncoder):
    """
    A naive encoder for ontology alignment.
    """
    def parse(self, **kwargs) -> Any:
        """
        Processes the source and target ontologies into a prompt for ontology alignment.

        This method formats the source and target ontologies into a string representation,
        filling in a pre-defined template that includes ontology items (IRI and label).

        Parameters:
            **kwargs: Contains the source and target ontologies as keyword arguments.

        Returns:
            list: A list containing the formatted prompt string for ontology matching.
        """
        source_onto, target_onto = kwargs["source"], kwargs["target"]
        source_ontos = []
        for source in source_onto:
            encoded_source = self.get_owl_items(owl=source)
            #encoded_source["concept"] = self.preprocess(encoded_source["text"])
            source_ontos.append(encoded_source)
        target_ontos = []
        for target in target_onto:
            encoded_target = self.get_owl_items(owl=target)
            #encoded_target["concept"] = self.preprocess(encoded_target["text"])
            target_ontos.append(encoded_target)
        return [source_ontos, target_ontos]

    def __str__(self):
        """
        Returns a string representation of the encoder.

        Returns:
            dict: A dictionary with the template and items_in_owl values.
        """
        return {"LLMEncoder": self.items_in_owl}

    def get_owl_items(self, owl: Dict) -> str:
        """
        Abstract method to extract ontology data as a string.

        This method should be implemented by subclasses to extract specific ontology data
        (e.g., IRI and label) from the provided ontology item.

        Parameters:
            owl (Dict): A dictionary representing an ontology item.

        Returns:
            str: The extracted ontology data as a string.
        """
        pass

    def get_encoder_info(self) -> str:
        """
        Provides information about the encoder and its prompt template.

        Returns:
            str: A description of the encoder's components.
        """
        return "INPUT CONSIST OF A DICTIONARY THAT CONSIST OF INFORMATION FOR THE GIVEN SOURCE-TARGET ONTOLOGIES."


class PropertyLLMEncoder(LLMEncoder):
    items_in_owl: str = "(Property)"

    def get_owl_items(self, owl: Dict) -> Any:
        text = owl.get("property_label") or owl.get("iri")
        return {"iri": owl["iri"], "Property": text}

# Direct full text
'''class PropertyDomainRangeLLMEncoder(LLMEncoder):
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
        return {"iri": owl["iri"], "text": text}'''

# domain, property and range
'''class PropertyDomainRangeLLMEncoder(LLMEncoder):
    items_in_owl: str = "(Property+DomainRange)"

    def _local_name(self, iri: str) -> str:
        return iri.rstrip("/").split("/")[-1].split("#")[-1]

    def get_owl_items(self, owl: Dict) -> Any:
        prop_label = owl.get("property_label") or ""

        domain_labels = owl.get("domain_label", []) or []
        range_labels = owl.get("range_label", []) or []

        # fallback: use local names from IRIs if labels are empty
        if not domain_labels:
            domain_labels = [self._local_name(d) for d in owl.get("domains", [])]
        if not range_labels:
            range_labels = [self._local_name(r) for r in owl.get("ranges", [])]

        dom_text = "; ".join(domain_labels) if domain_labels else "(none)"
        rng_text = "; ".join(range_labels) if range_labels else "(none)"
        prop_text = prop_label.strip() if prop_label.strip() else self._local_name(owl.get("iri", "")) or "(none)"

        text = f"Domain: {dom_text}\nProperty: {prop_text}\nRange: {rng_text}"
        return {"iri": owl["iri"], "text": text}'''

# Toggle both source and target should have domain, range and property
class PropertyDomainRangeLLMEncoder(LLMEncoder):
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


# First Stage
class ConceptCandidateLLMEncoder(LLMEncoder):
    """Encode the metadata used by first-stage candidate selection.

    The candidate stage receives concept, synonyms, definition, and
    verbalized axioms. Parents and children remain exclusive to the regular
    rich :class:`ConceptLLMEncoder` used for second-stage reasoning.
    """

    items_in_owl: str = "(Concept, Synonyms, Definition, Verbalized Axioms)"

    def get_owl_items(self, owl: Dict) -> Any:
        return {
            "iri": owl["iri"],
            "concept": format_candidate_concept_context(owl),
        }


# Stage 2
class ConceptLLMEncoder(LLMEncoder):
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
        #return {"iri": owl["iri"], "concept": owl["label"]}
        return {"iri": owl["iri"], "concept": format_concept_context(owl)}




        

class ConceptChildrenLLMEncoder(LLMEncoder):
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
        return {"iri": owl["iri"], "concept": owl["label"], "childrens": str(childrens)}


class ConceptParentLLMEncoder(LLMEncoder):
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
        return {"iri": owl["iri"], "concept": owl["label"], "parents": str(parents)}
