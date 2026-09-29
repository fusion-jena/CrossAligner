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
This script defines a set of custom dataset classes for handling various types of data used in a real-world entity classification task.
These datasets preprocess and format the input data to create structured prompts for a classification model, with variations
on how the relationship between concepts is represented (e.g., with or without parent/children context).

Classes:
    - RAGDataset: The base class for creating datasets for real-world entity classification tasks.
    - LabelRAGDataset: A subclass of RAGDataset for creating classification tasks that compare two concepts for similarity.
    - LabelParentRAGDataset: A subclass of RAGDataset that compares two concepts, considering the parent concepts of each.
    - LabelChildrenRAGDataset: A subclass of RAGDataset that compares two concepts, considering the children concepts of each.
"""
from typing import Any, Dict

from torch.utils.data import Dataset


class RAGDataset(Dataset):
    """
    A base dataset class for handling real-world entity classification tasks. This class preprocesses data and formats it into
    a suitable structure for the model's input, including creating prompts from the given concepts.

    Attributes:
        prompt (str): The template prompt used for generating classification tasks. Default is None.
        data (Any): The raw dataset provided at initialization.
        len (int): The length of the dataset.
    """
    prompt: str = None

    def __init__(self, data: Any) -> None:
        """
        Initializes the dataset with the provided data and computes the dataset length.

        Parameters:
            data (Any): The dataset to be used for classification tasks.
        """
        self.data = data
        self.len = len(data)

    def preprocess(self, text: str) -> str:
        """
        Preprocesses the input text by replacing underscores with spaces and converting it to lowercase.

        Parameters:
            text (str): The raw text to be preprocessed.

        Returns:
            str: The preprocessed text.
        """
        text = text.replace("_", " ")
        text = text.lower()
        return text

    def __getitem__(self, index: int) -> Dict:
        """
        Retrieves the data sample at the specified index and formats it into a dictionary with text and IRIs.

        Parameters:
            index (int): The index of the data sample to retrieve.

        Returns:
            dict: A dictionary containing the processed text and IRIs for the sample.
        """
        return {
            "prompts": self.fill_one_sample(self.data[index]),
            "iris": [self.data[index]["source"]["iri"], self.data[index]["target"]["iri"]]
        }

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.len

    def fill_one_sample(self, input_data: Any) -> str:
        """
        Placeholder method for filling a single sample. This method should be overridden by subclasses.

        Parameters:
            input_data (Any): The data sample to format.

        Returns:
            str: The formatted sample for model input.
        """
        pass

    def collate_fn(self, batchs):
        """
        Prepares a batch of data by collecting the processed texts and IRIs.

        Parameters:
            batchs (List[Dict]): A list of dictionaries containing texts and IRIs for the batch.

        Returns:
            dict: A dictionary containing lists of texts and IRIs for the batch.
        """
        batchs_clear = {"prompts": [], "iris": []}
        for batch in batchs:
            batchs_clear["prompts"].append(batch["prompts"])
            batchs_clear["iris"].append(batch["iris"])
        return batchs_clear

from typing import Any, Dict, List
from torch.utils.data import Dataset


class PropertyRAGDataset(Dataset):

    prompt = """Determine whether the following two properties refer to the same real-world relation.
Consider label and domain/range hints if present. Respond with "yes" or "no" only.

### Property 1:
{source}

### Property 2:
{target}

### Your Answer:"""

    def __init__(self, data: List[Dict[str, Any]]) -> None:
        """
        data is produced by rag.py build_llm_inputs(...).
        Each item in data should contain:
          - source: dict with iri + property text fields
          - target: dict with iri + property text fields
        """
        self.data = data
        self.len = len(self.data)

    def preprocess(self, text: str) -> str:
        if text is None:
            return ""
        text = text.replace("_", " ")
        text = text.lower()
        return text

    def _get_prop_text(self, item: Dict[str, Any]) -> str:
        # Works for different encoders:
        return (
            item.get("property")
            or item.get("text")
            or item.get("preferred_label")
            or item.get("label")
            or item.get("iri", "")
        )

    def __len__(self):
        return self.len

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.data[index]

        # be defensive: allow different key names
        s = row.get("source") or row.get("source_item") or row.get("s")
        t = row.get("target") or row.get("target_item") or row.get("t")

        if s is None or t is None:
            raise KeyError(
                "PropertyRAGDataset expected each row to contain keys "
                "'source' and 'target' (or aliases). Got keys: "
                + str(list(row.keys()))
            )

        source_text = self.preprocess(self._get_prop_text(s))
        target_text = self.preprocess(self._get_prop_text(t))

        prompt = self.prompt.replace("{source}", source_text).replace("{target}", target_text)

        return {
            "prompts": prompt,
            "iris": [s.get("iri"), t.get("iri")],
        }

    def collate_fn(self, batchs):
        batchs_clear = {"prompts": [], "iris": []}
        for batch in batchs:
            batchs_clear["prompts"].append(batch["prompts"])
            batchs_clear["iris"].append(batch["iris"])
        return batchs_clear


class PropertyDomainRangeRAGDataset(Dataset):
    prompt = """
You are an expert in ontology alignment.

Task: Classify the mapping between two ontology properties using ONLY:

- equivalent_property
- subproperty_of
- superproperty_of
- inverse_of
- related_property
- no_relation

Rules:
- If same meaning -> equivalent_property
- If same meaning but opposite direction -> inverse_of
- If one is more specific -> subproperty_of
- If one is more general -> superproperty_of
- If related but not exact -> related_property
- If unrelated in meaning -> no_relation


Here are some examples
[
  {
    "Property 1": "has Birth Date",
    "Domain 1": "Person",
    "Range 1": "date",
    "Property 2": "date Of Birth",
    "Domain 2": "Human",
    "Range 2": "date",
    "Mapping": "equivalent_property",
    "Reason": "Both properties represent the same concept (birth date) with compatible domain and range (if present)."
  },
  {
    "Property 1": "has Biological Mother",
    "Domain 1": "Person",
    "Range 1": "Woman",
    "Property 2": "has Mother",
    "Domain 2": "Person",
    "Range 2": "Woman",
    "Mapping": "subproperty_of",
    "Reason": "Every biological mother is a mother, but not every mother relation is biological (e.g., adoptive mother). Domains and ranges remain compatible."
  },
  {
    "Property 1": "has Part",
    "Domain 1": "Vehicle",
    "Range 1": "Component",
    "Property 2": "has Engine Part",
    "Domain 2": "Car",
    "Range 2": "Engine Component",
    "Mapping": "superproperty_of",
    "Reason": "has Part is more general; hasEnginePart is a specific type of part relation. The second has a narrower domain and range."
  },
  {
    "Property 1": "has Part",
    "Domain 1": "Car",
    "Range 1": "Wheel",
    "Property 2": "part Of",
    "Domain 2": "Wheel",
    "Range 2": "Car",
    "Mapping": "inverse_of",
    "Reason": "They express the same part-whole relation in opposite directions, with domain and range swapped."
  },
  {
    "Property 1": "affects Water Quality",
    "Domain 1": "Pollutant",
    "Range 1": "Water Body",
    "Property 2": "influences Water Condition",
    "Domain 2": "Contaminant",
    "Range 2": "Aquatic System",
    "Mapping": "related_property",
    "Reason": "The meanings are close, and domains/ranges are broadly compatible, but they are not clearly identical or in a strict sub/superproperty relation."
  },
  {
    "Property 1": "hasAuthor",
    "Domain 1": "Document",
    "Range 1": "Person",
    "Property 2": "published In Year",
    "Domain 2": "Article",
    "Range 2": "gYear",
    "Mapping": "no_relation",
    "Reason": "They represent different semantics (authorship vs publication year) and the ranges are incompatible (Person vs year)."
  }
]


### Property 1:
{source}
### Property 2:
{target}

Return ONLY valid JSON.

### Your Answer:
{
  "Property 1": "...",
  "Domain 1": "...",
  "Range 1": "...",
  "Property 2": "...",
  "Domain 2": "...",
  "Range 2": "...",
  "mapping": "...",
  "reason": "..."
}

"""

    def __init__(self, data: List[Dict[str, Any]]) -> None:
        """
        data is produced by rag.py build_llm_inputs(...).
        Each item in data should contain:
          - source: dict with iri + property text fields
          - target: dict with iri + property text fields
        """
        self.data = data
        self.len = len(self.data)

    def preprocess(self, text: str) -> str:
        if text is None:
            return ""
        text = text.replace("_", " ")
        text = text.lower()
        return text

    def _get_prop_text(self, item: Dict[str, Any]) -> str:
        # Works for different encoders:
        return (
            item.get("property")
            or item.get("text")
            or item.get("preferred_label")
            or item.get("label")
            or item.get("iri", "")
        )

    def __len__(self):
        return self.len

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.data[index]

        # be defensive: allow different key names
        s = row.get("source") or row.get("source_item") or row.get("s")
        t = row.get("target") or row.get("target_item") or row.get("t")

        if s is None or t is None:
            raise KeyError(
                "PropertyRAGDataset expected each row to contain keys "
                "'source' and 'target' (or aliases). Got keys: "
                + str(list(row.keys()))
            )

        source_text = self.preprocess(self._get_prop_text(s))
        target_text = self.preprocess(self._get_prop_text(t))

        prompt = self.prompt.replace("{source}", source_text).replace("{target}", target_text)

        return {
            "prompts": prompt,
            "iris": [s.get("iri"), t.get("iri")],
        }

    def collate_fn(self, batchs):
        batchs_clear = {"prompts": [], "iris": []}
        for batch in batchs:
            batchs_clear["prompts"].append(batch["prompts"])
            batchs_clear["iris"].append(batch["iris"])
        return batchs_clear
    

class ConceptRAGDataset(RAGDataset):
    """
    A subclass of RAGDataset used for real-world entity classification tasks comparing two concepts
    for similarity. It formats the input data into a classification prompt with the question of whether
    two concepts refer to the same real-world entity.

    Attributes:
        prompt (str): The template prompt used for generating the classification task.
    """
    prompt = """Classify if two concepts refer to the same real world entity or not (answer only yes or no).
### First concept:
{source}
### Second concept:
{target}
### Answer:"""

    def fill_one_sample(self, input_data: Any) -> str:
        """
        Formats the input data into a classification prompt comparing two concepts for similarity.

        Parameters:
            input_data (Any): The data sample containing source and target concepts to compare.

        Returns:
            str: The formatted classification prompt.
        """
        source = self.preprocess(input_data["source"]["label"])
        target = self.preprocess(input_data["target"]["label"])
        return self.prompt.replace("{source}", source).replace("{target}", target)


class ConceptParentRAGDataset(RAGDataset):
    """
    A subclass of RAGDataset used for real-world entity classification tasks comparing two concepts,
    considering the parent concepts of each.

    Attributes:
        prompt (str): The template prompt used for generating the classification task.
    """
    prompt = """Classify if two concepts refer to the same real world entity or not (answer only yes or no).
### First concept:
{source}
Parents: {source_parents}
### Second concept:
{target}
Parents: {target_parents}
### Answer:"""

    def fill_one_sample(self, input_data: Any) -> str:
        """
        Formats the input data into a classification prompt comparing two concepts for similarity,
        with additional context from their parent concepts.

        Parameters:
            input_data (Any): The data sample containing source and target concepts with parent information.

        Returns:
            str: The formatted classification prompt.
        """
        template = self.prompt
        source = self.preprocess(input_data["source"]["label"])
        target = self.preprocess(input_data["target"]["label"])
        source_parents = ", ".join([self.preprocess(parent["label"]) for parent in input_data["source"]["parents"]])
        target_parents = ", ".join([self.preprocess(parent["label"]) for parent in input_data["target"]["parents"]])
        template = (
            template.replace("{source}", source)
            .replace("{target}", target)
            .replace("{source_parents}", source_parents)
            .replace("{target_parents}", target_parents)
        )
        return template


class ConceptChildrenRAGDataset(RAGDataset):
    """
    A subclass of RAGDataset used for real-world entity classification tasks comparing two concepts,
    considering the children concepts of each.

    Attributes:
        prompt (str): The template prompt used for generating the classification task.
    """
    prompt = """Classify if two concepts refer to the same real world entity or not (answer only yes or no).
### First concept:
{source}
Children: {source_children}
### Second concept:
{target}
Children: {target_children}
### Answer:"""

    def fill_one_sample(self, input_data: Any) -> str:
        """
        Formats the input data into a classification prompt comparing two concepts for similarity,
        with additional context from their children concepts.

        Parameters:
            input_data (Any): The data sample containing source and target concepts with children information.

        Returns:
            str: The formatted classification prompt.
        """
        template = self.prompt
        source = self.preprocess(input_data["source"]["label"])
        target = self.preprocess(input_data["target"]["label"])
        source_children = ", ".join([self.preprocess(children["label"]) for children in input_data["source"]["childrens"]])
        target_children = ", ".join([self.preprocess(children["label"]) for children in input_data["target"]["childrens"]])
        template = (
            template.replace("{source}", source)
            .replace("{target}", target)
            .replace("{source_children}", source_children)
            .replace("{target_children}", target_children)
        )
        return template


class ConceptRichRAGDataset(RAGDataset):
    """
    RAG prompt dataset for rich concept context.

    Intended to pair with `ConceptDefinitionPromptRAGEncoder`, where retrieval uses
    compact concept+definition text while prompt generation uses richer fields.
    """
    prompt = """You are an expert in ontology alignment. Determine the semantic relation between Concept 1 and Concept 2 using only:
concept, parents, children, synonyms, definition.

Allowed relations:
- equivalent_class
- sub_class_of
- super_class_of
- related_to
- no_relation

Rules:
- Use only the provided fields.
- If a field is missing/empty/none, ignore it.
- Do not guess.
- Return exactly ONE JSON object.
- No markdown/code fences/extra text.

### Example 1
Concept 1:
concept: lake
parents: water body
children: alpine lake
synonyms: inland water body
definition: standing body of inland surface water

Concept 2:
concept: inland lake
parents: water body
children: glacial inland lake
synonyms: lake
definition: inland standing surface water body

Output:
{"Concept 1":"lake","Concept 2":"inland lake","relation":"equivalent_class","reason":"same core meaning and matching parent/synonyms"}

### Example 2
Concept 1:
concept: water indicator
parents: indicator
children: nutrient indicator
synonyms: (none)
definition: indicator describing water conditions

Concept 2:
concept: indicator
parents: (none)
children: water indicator
synonyms: (none)
definition: generic indicator concept

Output:
{"Concept 1":"water indicator","Concept 2":"indicator","relation":"sub_class_of","reason":"concept 1 is explicitly a narrower child of concept 2"}

### Example 3
Concept 1:
concept: monitoring facility
parents: feature of interest
children: (none)
synonyms: (none)
definition: fixed installation used for monitoring

Concept 2:
concept: day of week
parents: temporal unit
children: monday
synonyms: weekday
definition: unit of calendar time

Output:
{"Concept 1":"monitoring facility","Concept 2":"day of week","relation":"no_relation","reason":"domains are unrelated (physical facility vs time unit)"}

### Concept 1:
{source}
### Concept 2:
{target}
### Your Answer:
{"Concept 1":"...", "Concept 2":"...", "relation":"equivalent_class | sub_class_of | super_class_of | related_to | no_relation", "reason":"..."}
"""

    def _concept_prompt_text(self, concept: Dict[str, Any]) -> str:
        if "concept" in concept and isinstance(concept["concept"], str):
            return self.preprocess(concept["concept"])

        # Late import to avoid circular imports during module initialization.
        from ...encoder.concept_formatting import format_concept_prompt_context
        return self.preprocess(format_concept_prompt_context(concept))

    def fill_one_sample(self, input_data: Any) -> str:
        source = self._concept_prompt_text(input_data["source"])
        target = self._concept_prompt_text(input_data["target"])
        return self.prompt.replace("{source}", source).replace("{target}", target)




'''class ConceptRichRAGDataset(LLMDataset):
    """
    RAG prompt dataset for rich concept context.

    Intended to pair with `ConceptDefinitionPromptRAGEncoder`, where retrieval uses
    compact concept+definition text while prompt generation uses richer fields.
    """
    prompt = """Determine whether Concept 1 and Concept 2 refer to the same real-world meaning.
Use only the provided concept context.

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
Respond with only \"yes\" or \"no\".
"""

    def _concept_prompt_text(self, concept: Dict[str, Any]) -> str:
        if "concept" in concept and isinstance(concept["concept"], str):
            return self.preprocess(concept["concept"])

        # Late import to avoid circular imports during module initialization.
        from ...encoder.concept_formatting import format_concept_prompt_context
        return self.preprocess(format_concept_prompt_context(concept))

    def fill_one_sample(self, input_data: Any) -> str:
        source = self._concept_prompt_text(input_data["source"])
        target = self._concept_prompt_text(input_data["target"])
        return self.prompt.replace("{source}", source).replace("{target}", target)'''