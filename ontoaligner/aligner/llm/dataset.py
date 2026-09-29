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
from typing import Any, Dict

from torch.utils.data import Dataset


class LLMDataset(Dataset):
    prompt: str = None

    def __init__(self, source_onto: Any, target_onto: Any) -> None:
        self.data = []
        for source in source_onto:
            for target in target_onto:
                self.data.append({
                    "source": source,
                    "target": target
                })

        self.len = len(self.data)

    def preprocess(self, text: str) -> str:
        text = text.replace("_", " ")
        text = text.lower()
        return text

    def __getitem__(self, index: int) -> Dict:
        return {
            "prompts": self.fill_one_sample(self.data[index]),
            "iris": [self.data[index]["source"]["iri"], self.data[index]["target"]["iri"]]
        }

    def __len__(self):
        return self.len

    def fill_one_sample(self, input_data: Any) -> str:
        pass

    def collate_fn(self, batchs):
        batchs_clear = {"prompts": [], "iris": []}
        for batch in batchs:
            batchs_clear["prompts"].append(batch["prompts"])
            batchs_clear["iris"].append(batch["iris"])
        return batchs_clear


# Stage 1
class ConceptLLMDataset(LLMDataset):
    prompt = """Determine whether the following two concepts are semantically related or not. Respond with "yes" or "no".
### Concept 1:
{source}
### Concept 2:
{target}
### Your Answer:"""

    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["concept"])
        target = self.preprocess(input_data["target"]["concept"])
        return self.prompt.replace("{source}", source).replace("{target}", target)


###################################
           # Stage 2 #
###################################


class CandidateConceptLLMDataset(ConceptLLMDataset):

    prompt = None

    def __init__(self, source_onto, target_onto, candidate_pairs):
        if not self.prompt:
            raise ValueError(f"{self.__class__.__name__} must define a prompt.")

        source_by_iri = {item["iri"]: item for item in source_onto}
        target_by_iri = {item["iri"]: item for item in target_onto}
        self.data = []

        for pair in candidate_pairs:
            source = source_by_iri.get(pair["source"])
            target = target_by_iri.get(pair["target"])
            if source is not None and target is not None:
                self.data.append({
                    "source": source,
                    "target": target,
                })

        self.len = len(self.data)


#############################
        # MeSH ONS #
#############################

# Concept1-2 without_example

class MeshOnsSourceToTargetZeroShotDataset(CandidateConceptLLMDataset):

    prompt = """You are an expert in ontology alignment and semantic relationship identification.

Determine whether Concept 1 has a direct and meaningful semantic relationship with Concept 2 using their concept, parents, childrens, synonyms definitions, verbalized axioms.

If a relationship exists:
- Express the relationship as one short sentence with Concept 1 as the subject and Concept 2 as the object.
- Select the relation that best represents how Concept 1 is connected to Concept 2.
- Do not merely state that the concepts are related.
- Do not classify the relationship as equivalence, subclass, or superclass unless that is the clearest relationship.

If no direct and meaningful relationship can be identified, set "related" to "no" and return an empty relationship.
Do not invent a relationship that cannot be reasonably inferred from the provided concept labels, parents, children, synonyms, definitions and Verbalized Axioms.

Rules:
- Ignore missing, null, empty, or none fields.
- Keep the relationship sentence short and direct.
- Preserve the direction from Concept 1 to Concept 2.
- Return exactly one valid JSON object.
- Do not include markdown or additional text.

Required JSON format:
{"related": "yes or no", "relationship": "short directional relationship sentence"}

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
"""


# Concept 2-1 without_example

class MeshOnsTargetToSourceZeroShotDataset(CandidateConceptLLMDataset):
    prompt = """You are an expert in ontology alignment and semantic relationship identification.
Determine whether Concept 2 has a direct and meaningful semantic relationship with Concept 1 using their concept, parents, childrens, synonyms, definitions, verbalized axioms.

If a relationship exists:
- Express the relationship as one short sentence with Concept 2 as the subject and Concept 1 as the object.
- Select the relation that best represents how Concept 2 is connected to Concept 1.
- Do not merely state that the concepts are related.
- Do not classify the relationship as equivalence, subclass, or superclass unless that is the clearest relationship.

If no direct and meaningful relationship can be identified, set "related" to "no" and return an empty relationship.
Do not invent a relationship that cannot be reasonably inferred from the provided concept labels, parents, children, synonyms, definitions and Verbalized Axioms.

Rules:
- Ignore missing, null, empty, or none fields.
- Keep the relationship sentence short and direct.
- Preserve the direction from Concept 2 to Concept 1.
- Return exactly one valid JSON object.
- Do not include markdown or additional text.

Required JSON format:
{"related": "yes or no", "relationship": "short directional relationship sentence"}

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
"""

        

# Concept1-2 with_example

class MeshOnsSourceToTargetFewShotDataset(CandidateConceptLLMDataset):
    prompt = """You are an expert in ontology alignment and semantic relationship identification.

Determine whether Concept 1 has a direct and meaningful semantic relationship with Concept 2 using their concept, parents, childrens, synonyms definitions, verbalized axioms.

If a relationship exists:
- Express the relationship as one short sentence with Concept 1 as the subject and Concept 2 as the object.
- Select the relation that best represents how Concept 1 is connected to Concept 2.
- Do not merely state that the concepts are related.
- Do not classify the relationship as equivalence, subclass, or superclass unless that is the clearest relationship.

If no direct and meaningful relationship can be identified, set "related" to "no" and return an empty relationship.
Do not invent a relationship that cannot be reasonably inferred from the provided concept labels, parents, children, synonyms, definitions and Verbalized Axioms.

Rules:
- Ignore missing, null, empty, or none fields.
- Keep the relationship sentence short and direct.
- Preserve the direction from Concept 1 to Concept 2.
- Return exactly one valid JSON object.
- Do not include markdown or additional text.

Required JSON format:
{"related": "yes or no", "relationship": "short directional relationship sentence"}

Examples:

Concept 1: 
concept: food contamination
parents: public health; food quality
childrens: (none)
synonyms: adulteration, food; adulterations, food; contamination, food; contaminations, food; food adulteration; food adulterations; food contaminations
definitions: the presence in food of harmful, unpalatable, or otherwise objectionable foreign substances, e.g. chemicals, microorganisms or diluents, before, during, or after processing or storage.
verbalized axioms: (none)

Concept 2: 
concept: food product
parents: food material
childrens: plant food product; fungus food product; animal food product
synonyms: (none)
definitions: (none)
verbalized axioms: (none)

Answer:
{"related": "yes", "relationship": "food contamination affects food product."}


Concept 1: 
concept: food industry
parents: industry
childrens: food labeling; food technology
synonyms: food industries; industries, food; industry, food
definitions: the industry concerned with processing, preparing, preserving, distributing, and serving of foods and beverages.
verbalized axioms: (none)

Concept 2: 
concept: food material
parents: material entity
childrens: food product; vegetarian food material; diet
synonyms: (none)
definitions: (none)
verbalized axioms: (none)

Answer:
{"related": "yes", "relationship": "food industry processes food material."}

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
"""

        

# Concept 2-1 with_example

class MeshOnsTargetToSourceFewShotDataset(CandidateConceptLLMDataset):
    prompt = """You are an expert in ontology alignment and semantic relationship identification.

Determine whether Concept 2 has a direct and meaningful semantic relationship with Concept 1 using their concept, parents, childrens, synonyms definitions, verbalized axioms.

If a relationship exists:
- Express the relationship as one short sentence with Concept 2 as the subject and Concept 1 as the object.
- Select the relation that best represents how Concept 2 is connected to Concept 1.
- Do not merely state that the concepts are related.
- Do not classify the relationship as equivalence, subclass, or superclass unless that is the clearest relationship.

If no direct and meaningful relationship can be identified, set "related" to "no" and return an empty relationship.
Do not invent a relationship that cannot be reasonably inferred from the provided concept labels, parents, children, synonyms, definitions and Verbalized Axioms.

Rules:
- Ignore missing, null, empty, or none fields.
- Keep the relationship sentence short and direct.
- Preserve the direction from Concept 2 to Concept 1.
- Return exactly one valid JSON object.
- Do not include markdown or additional text.

Required JSON format:
{"related": "yes or no", "relationship": "short directional relationship sentence"}

Examples:

Concept 1: 
concept: food contamination
parents: public health; food quality
childrens: (none)
synonyms: adulteration, food; adulterations, food; contamination, food; contaminations, food; food adulteration; food adulterations; food contaminations
definitions: the presence in food of harmful, unpalatable, or otherwise objectionable foreign substances, e.g. chemicals, microorganisms or diluents, before, during, or after processing or storage.
verbalized axioms: (none)

Concept 2: 
concept: food product
parents: food material
childrens: plant food product; fungus food product; animal food product
synonyms: (none)
definitions: (none)
verbalized axioms: (none)

Answer:
{"related": "yes", "relationship": "food product are affects by food contamination."}


Concept 1: 
concept: food industry
parents: industry
childrens: food labeling; food technology
synonyms: food industries; industries, food; industry, food
definitions: the industry concerned with processing, preparing, preserving, distributing, and serving of foods and beverages.
verbalized axioms: (none)

Concept 2: 
concept: food material
parents: material entity
childrens: food product; vegetarian food material; diet
synonyms: (none)
definitions: (none)
verbalized axioms: (none)

Answer:
{"related": "yes", "relationship": "food material is processed by the food industry."}

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
"""
        


##################
#### OccO-ONS ####
##################



# Concept1-2 without_example

class OccoOnsSourceToTargetZeroShotDataset(CandidateConceptLLMDataset):
    prompt = """You are an expert in ontology alignment and semantic relationship identification.

Determine whether Concept 1 has a direct and meaningful semantic relationship with Concept 2 using their concept, parents, childrens, synonyms definitions, verbalized axioms.

If a relationship exists:
- Express the relationship as one short sentence with Concept 1 as the subject and Concept 2 as the object.
- Select the relation that best represents how Concept 1 is connected to Concept 2.
- Do not merely state that the concepts are related.
- Do not classify the relationship as equivalence, subclass, or superclass unless that is the clearest relationship.

If no direct and meaningful relationship can be identified, set "related" to "no" and return an empty relationship.
Do not invent a relationship that cannot be reasonably inferred from the provided concept labels, parents, children, synonyms, definitions and Verbalized Axioms.

Rules:
- Ignore missing, null, empty, or none fields.
- Keep the relationship sentence short and direct.
- Preserve the direction from Concept 1 to Concept 2.
- Return exactly one valid JSON object.
- Do not include markdown or additional text.

Required JSON format:
{"related": "yes or no", "relationship": "short directional relationship sentence"}

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
"""


# Concept 2-1 without_example

class OccoOnsTargetToSourceZeroShotDataset(CandidateConceptLLMDataset):
    prompt = """You are an expert in ontology alignment and semantic relationship identification.
Determine whether Concept 2 has a direct and meaningful semantic relationship with Concept 1 using their concept, parents, childrens, synonyms, definitions, verbalized axioms.

If a relationship exists:
- Express the relationship as one short sentence with Concept 2 as the subject and Concept 1 as the object.
- Select the relation that best represents how Concept 2 is connected to Concept 1.
- Do not merely state that the concepts are related.
- Do not classify the relationship as equivalence, subclass, or superclass unless that is the clearest relationship.

If no direct and meaningful relationship can be identified, set "related" to "no" and return an empty relationship.
Do not invent a relationship that cannot be reasonably inferred from the provided concept labels, parents, children, synonyms, definitions and Verbalized Axioms.

Rules:
- Ignore missing, null, empty, or none fields.
- Keep the relationship sentence short and direct.
- Preserve the direction from Concept 2 to Concept 1.
- Return exactly one valid JSON object.
- Do not include markdown or additional text.

Required JSON format:
{"related": "yes or no", "relationship": "short directional relationship sentence"}

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
"""

        

# Concept1-2 with_example

class OccoOnsSourceToTargetFewShotDataset(CandidateConceptLLMDataset):
    prompt = """You are an expert in ontology alignment and semantic relationship identification.

Determine whether Concept 1 has a direct and meaningful semantic relationship with Concept 2 using their concept, parents, childrens, synonyms definitions, verbalized axioms.

If a relationship exists:
- Express the relationship as one short sentence with Concept 1 as the subject and Concept 2 as the object.
- Select the relation that best represents how Concept 1 is connected to Concept 2.
- Do not merely state that the concepts are related.
- Do not classify the relationship as equivalence, subclass, or superclass unless that is the clearest relationship.

If no direct and meaningful relationship can be identified, set "related" to "no" and return an empty relationship.
Do not invent a relationship that cannot be reasonably inferred from the provided concept labels, parents, children, synonyms, definitions and Verbalized Axioms.

Rules:
- Ignore missing, null, empty, or none fields.
- Keep the relationship sentence short and direct.
- Preserve the direction from Concept 1 to Concept 2.
- Return exactly one valid JSON object.
- Do not include markdown or additional text.

Required JSON format:
{"related": "yes or no", "relationship": "short directional relationship sentence"}

Examples:

Concept 1: 
concept: chef or head cook
parents: occupation holder; food preparation or serving related occupation holder
childrens: (none)
synonyms: chefs and head cooks
definitions: (none)
verbalized axioms: chef or head cook is a subclass of something that has ability problem sensitivity; chef or head cook is a subclass of something that has ability speech recognition; chef or head cook is a subclass of something that has ability oral comprehension; chef or head cook is a subclass of something that has ability speech clarity; chef or head cook is a subclass of something that has ability oral expression; chef or head cook is a subclass of something that has ability deductive reasoning; chef or head cook is a subclass of something that has skill coordination; chef or head cook is a subclass of something that has skill monitoring

Concept 2: 
concept: food product
parents: food material
childrens: plant food product; fungus food product; animal food product
synonyms: (none)
definitions: (none)
verbalized axioms: (none)

Answer:
{"related": "yes", "relationship": "chef or head cook prepares food product."}


Concept 1: 
concept: cook
parents: food preparation or serving related occupation holder
childrens: cook, institution or cafeteria; cook, restaurant
synonyms: (none)
definitions: (none)
verbalized axioms: (none)

Concept 2: 
concept: food product
parents: food material
childrens: plant food product; fungus food product; animal food product
synonyms: (none)
definitions: (none)
verbalized axioms: (none)

Answer:
{"related": "yes", "relationship": "cook prepares food product."}


### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
"""
        

# Concept 2-1 with_example

class OccoOnsTargetToSourceFewShotDataset(CandidateConceptLLMDataset):
    prompt = """You are an expert in ontology alignment and semantic relationship identification.

Determine whether Concept 2 has a direct and meaningful semantic relationship with Concept 1 using their concept, parents, childrens, synonyms definitions, verbalized axioms.

If a relationship exists:
- Express the relationship as one short sentence with Concept 2 as the subject and Concept 1 as the object.
- Select the relation that best represents how Concept 2 is connected to Concept 1.
- Do not merely state that the concepts are related.
- Do not classify the relationship as equivalence, subclass, or superclass unless that is the clearest relationship.

If no direct and meaningful relationship can be identified, set "related" to "no" and return an empty relationship.
Do not invent a relationship that cannot be reasonably inferred from the provided concept labels, parents, children, synonyms, definitions and Verbalized Axioms.

Rules:
- Ignore missing, null, empty, or none fields.
- Keep the relationship sentence short and direct.
- Preserve the direction from Concept 2 to Concept 1.
- Return exactly one valid JSON object.
- Do not include markdown or additional text.

Required JSON format:
{"related": "yes or no", "relationship": "short directional relationship sentence"}

Examples:

Concept 1: 
concept: chef or head cook
parents: occupation holder; food preparation or serving related occupation holder
childrens: (none)
synonyms: chefs and head cooks
definitions: (none)
verbalized axioms: chef or head cook is a subclass of something that has ability problem sensitivity; chef or head cook is a subclass of something that has ability speech recognition; chef or head cook is a subclass of something that has ability oral comprehension; chef or head cook is a subclass of something that has ability speech clarity; chef or head cook is a subclass of something that has ability oral expression; chef or head cook is a subclass of something that has ability deductive reasoning; chef or head cook is a subclass of something that has skill coordination; chef or head cook is a subclass of something that has skill monitoring

Concept 2: 
concept: food product
parents: food material
childrens: plant food product; fungus food product; animal food product
synonyms: (none)
definitions: (none)
verbalized axioms: (none)

Answer:
{"related": "yes", "relationship": "food product prepared by chef or head cook."}


Concept 1: 
concept: cook
parents: food preparation or serving related occupation holder
childrens: cook, institution or cafeteria; cook, restaurant
synonyms: (none)
definitions: (none)
verbalized axioms: (none)

Concept 2: 
concept: food product
parents: food material
childrens: plant food product; fungus food product; animal food product
synonyms: (none)
definitions: (none)
verbalized axioms: (none)

Answer:
{"related": "yes", "relationship": "food product prepared by cook."}

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
"""



##################
#### OccO-Mesh ####
##################



# Concept1-2 without_example

class OccoMeshSourceToTargetZeroShotDataset(CandidateConceptLLMDataset):
    prompt = """You are an expert in ontology alignment and semantic relationship identification.

Determine whether Concept 1 has a direct and meaningful semantic relationship with Concept 2 using their concept, parents, childrens, synonyms definitions, verbalized axioms.

If a relationship exists:
- Express the relationship as one short sentence with Concept 1 as the subject and Concept 2 as the object.
- Select the relation that best represents how Concept 1 is connected to Concept 2.
- Do not merely state that the concepts are related.
- Do not classify the relationship as equivalence, subclass, or superclass unless that is the clearest relationship.

If no direct and meaningful relationship can be identified, set "related" to "no" and return an empty relationship.
Do not invent a relationship that cannot be reasonably inferred from the provided concept labels, parents, children, synonyms, definitions and Verbalized Axioms.

Rules:
- Ignore missing, null, empty, or none fields.
- Keep the relationship sentence short and direct.
- Preserve the direction from Concept 1 to Concept 2.
- Return exactly one valid JSON object.
- Do not include markdown or additional text.

Required JSON format:
{"related": "yes or no", "relationship": "short directional relationship sentence"}

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
"""


# Concept 2-1 without_example

class OccoMeshTargetToSourceZeroShotDataset(CandidateConceptLLMDataset):
    prompt = """You are an expert in ontology alignment and semantic relationship identification.
Determine whether Concept 2 has a direct and meaningful semantic relationship with Concept 1 using their concept, parents, childrens, synonyms, definitions, verbalized axioms.

If a relationship exists:
- Express the relationship as one short sentence with Concept 2 as the subject and Concept 1 as the object.
- Select the relation that best represents how Concept 2 is connected to Concept 1.
- Do not merely state that the concepts are related.
- Do not classify the relationship as equivalence, subclass, or superclass unless that is the clearest relationship.

If no direct and meaningful relationship can be identified, set "related" to "no" and return an empty relationship.
Do not invent a relationship that cannot be reasonably inferred from the provided concept labels, parents, children, synonyms, definitions and Verbalized Axioms.

Rules:
- Ignore missing, null, empty, or none fields.
- Keep the relationship sentence short and direct.
- Preserve the direction from Concept 2 to Concept 1.
- Return exactly one valid JSON object.
- Do not include markdown or additional text.

Required JSON format:
{"related": "yes or no", "relationship": "short directional relationship sentence"}

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
"""
        

# Concept1-2 with_example

class OccoMeshSourceToTargetFewShotDataset(CandidateConceptLLMDataset):
    prompt = """You are an expert in ontology alignment and semantic relationship identification.

Determine whether Concept 1 has a direct and meaningful semantic relationship with Concept 2 using their concept, parents, childrens, synonyms definitions, verbalized axioms.

If a relationship exists:
- Express the relationship as one short sentence with Concept 1 as the subject and Concept 2 as the object.
- Select the relation that best represents how Concept 1 is connected to Concept 2.
- Do not merely state that the concepts are related.
- Do not classify the relationship as equivalence, subclass, or superclass unless that is the clearest relationship.

If no direct and meaningful relationship can be identified, set "related" to "no" and return an empty relationship.
Do not invent a relationship that cannot be reasonably inferred from the provided concept labels, parents, children, synonyms, definitions and Verbalized Axioms.

Rules:
- Ignore missing, null, empty, or none fields.
- Keep the relationship sentence short and direct.
- Preserve the direction from Concept 1 to Concept 2.
- Return exactly one valid JSON object.
- Do not include markdown or additional text.

Required JSON format:
{"related": "yes or no", "relationship": "short directional relationship sentence"}

Examples:

Concept 1: 
concept: food preparation or serving related occupation holder
parents: occupation holder
childrens: chef or head cook; cook; food or beverage serving worker
synonyms: (none)
definitions: (none)
verbalized axioms: (none)

Concept 2: 
concept: food industry
parents: industry
childrens: food labeling; food technology
synonyms: food industries; industries, food; industry, food
definitions: the industry concerned with processing, preparing, preserving, distributing, and serving of foods and beverages.
verbalized axioms: (none)

Answer:
{"related": "yes", "relationship": "food preparation or serving related occupation holder works in the food industry."}

Concept 1: 
concept: cook, restaurant
parents: occupation holder with job zone 2; cook
childrens: (none)
synonyms: (none)
definitions: (none)
verbalized axioms: cook, restaurant is a subclass of something that has skill judgment and decision making; cook, restaurant is a subclass of something that has skill active listening; cook, restaurant is a subclass of something that has skill quality control analysis; cook, restaurant is a subclass of something that has ability visual color discrimination; cook, restaurant is a subclass of something that has skill speaking; cook, restaurant is a subclass of something that has skill coordination; cook, restaurant is a subclass of something that has skill monitoring; cook, restaurant is a subclass of something that has ability speech recognition

Concept 2: 
concept: food quality
parents: food technology; public health
childrens: food contamination; nutritive value
synonyms: food qualities; qualities, food; quality, food
definitions: ratings of the characteristics of food including flavor, appearance, nutritional content, and the amount of microbial and chemical contamination.
verbalized axioms: (none)

Answer:
{"related": "yes", "relationship": "cook, restaurant check food quality."}

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
"""
        

# Concept 2-1 with_example

class OccoMeshTargetToSourceFewShotDataset(CandidateConceptLLMDataset):
    prompt = """You are an expert in ontology alignment and semantic relationship identification.

Determine whether Concept 2 has a direct and meaningful semantic relationship with Concept 1 using their concept, parents, childrens, synonyms definitions, verbalized axioms.

If a relationship exists:
- Express the relationship as one short sentence with Concept 2 as the subject and Concept 1 as the object.
- Select the relation that best represents how Concept 2 is connected to Concept 1.
- Do not merely state that the concepts are related.
- Do not classify the relationship as equivalence, subclass, or superclass unless that is the clearest relationship.

If no direct and meaningful relationship can be identified, set "related" to "no" and return an empty relationship.
Do not invent a relationship that cannot be reasonably inferred from the provided concept labels, parents, children, synonyms, definitions and Verbalized Axioms.

Rules:
- Ignore missing, null, empty, or none fields.
- Keep the relationship sentence short and direct.
- Preserve the direction from Concept 2 to Concept 1.
- Return exactly one valid JSON object.
- Do not include markdown or additional text.

Required JSON format:
{"related": "yes or no", "relationship": "short directional relationship sentence"}

Examples:

Concept 1: 
concept: food preparation or serving related occupation holder
parents: occupation holder
childrens: chef or head cook; cook; food or beverage serving worker
synonyms: (none)
definitions: (none)
verbalized axioms: (none)

Concept 2: 
concept: food industry
parents: industry
childrens: food labeling; food technology
synonyms: food industries; industries, food; industry, food
definitions: the industry concerned with processing, preparing, preserving, distributing, and serving of foods and beverages.
verbalized axioms: (none)

Answer:
{"related": "yes", "relationship": "food industry employs food preparation or serving related occupation holders."}

Concept 1: 
concept: cook, restaurant
parents: occupation holder with job zone 2; cook
childrens: (none)
synonyms: (none)
definitions: (none)
verbalized axioms: cook, restaurant is a subclass of something that has skill judgment and decision making; cook, restaurant is a subclass of something that has skill active listening; cook, restaurant is a subclass of something that has skill quality control analysis; cook, restaurant is a subclass of something that has ability visual color discrimination; cook, restaurant is a subclass of something that has skill speaking; cook, restaurant is a subclass of something that has skill coordination; cook, restaurant is a subclass of something that has skill monitoring; cook, restaurant is a subclass of something that has ability speech recognition

Concept 2: 
concept: food quality
parents: food technology; public health
childrens: food contamination; nutritive value
synonyms: food qualities; qualities, food; quality, food
definitions: ratings of the characteristics of food including flavor, appearance, nutritional content, and the amount of microbial and chemical contamination.
verbalized axioms: (none)

Answer:
{"related": "yes", "relationship": "food quality check by cook, restaurant."}

### Concept 1:
{source}

### Concept 2:
{target}

### Your Answer:
"""