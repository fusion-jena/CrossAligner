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
Set of helper functions for post-processing methods.

- eval_preprocess_ir_outputs: Processes and filters IR outputs based on confidence score.
- threshold_finder: Determines the threshold value for a given set of scores from a dictionary.
- build_outputdict: Constructs a dictionary mapping sources to their respective predicted targets and scores.
- confidence_score_ratio_based_filtering: Filters predictions based on confidence ratios and a given threshold.
- confidence_score_based_filtering: Filters predictions based on LLM confidence scores and IR scores.
"""
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import re


def eval_preprocess_ir_outputs(predicts: List) -> List:
    """
    Filters out redundant IR predictions based on the source-target pair and their respective scores.

    Parameters:
        predicts (List): List of dictionaries containing source, target candidates, and score candidates.

    Returns:
        List: A filtered list of predictions with unique source-target pairs and positive scores.
    """
    predicts_temp = []
    predict_map = {}
    for predict in tqdm(predicts):
        source = predict["source"]
        target_cands = predict["target-cands"]
        score_cands = predict["score-cands"]
        for target, score in zip(target_cands, score_cands):
            if score > 0:
                adjusted = False
                if predict_map.get(f"{source}-{target}", "NA") != "NA":
                    adjusted = True
                    break
                if not adjusted:
                    predicts_temp.append({"source": source, "target": target, "score": score})
                    predict_map[f"{source}-{target}"] = f"{source}-{target}"
    return predicts_temp


def threshold_finder(dictionary: dict, index: int, use_lst: bool = False) -> float:
    """
    Finds the threshold value based on the given index of a score in a dictionary.

    Parameters:
        dictionary (dict): Dictionary containing predictions with scores.
        index (int): The index of the score in the prediction output to be thresholded.
        use_lst (bool, optional): Whether to use the list of values or the dictionary. Defaults to False.

    Returns:
        float: The computed threshold value.
    """
    scores_dict = {}
    scores_list = []
    for outputs in dictionary.values():
        for output in outputs:
            scores_list.append(output[index])
            if scores_dict.get(output[0], 0) != 0:
                if scores_dict.get(output[0], 0) < output[index]:
                    scores_dict[output[0]] = output[index]
            else:
                scores_dict[output[0]] = output[index]
    if not use_lst:
        scores_list = list(scores_dict.values())
    threshold = sum(scores_list) / len(scores_list) if len(scores_list) != 0 else 0
    return threshold


def build_outputdict(llm_outputs: List, ir_outputs: List) -> Dict:
    """
    Builds a dictionary mapping source IRIs to target predictions with their scores
    from both IR and LLM outputs.

    Parameters:
        llm_outputs (List): List of LLM prediction outputs.
        ir_outputs (List): List of IR prediction outputs.

    Returns:
        Dict: A dictionary where each source is mapped to a list of target predictions and their associated scores.
    """
    outputdict = {}
    for llm_output in tqdm(llm_outputs):
        for ir_output in ir_outputs:
            if llm_output["source"] == ir_output["source"] and llm_output["target"] == ir_output["target"]:
                confidence_ratio = llm_output["score"] * ir_output["score"]
                predicts_list = [llm_output["target"], ir_output["score"], llm_output["score"], confidence_ratio]
                if llm_output["source"] not in list(outputdict.keys()):
                    outputdict[llm_output["source"]] = [predicts_list]
                else:
                    outputdict[llm_output["source"]].append(predicts_list)
    return outputdict


def confidence_score_ratio_based_filtering(outputdict: Dict, topk_confidence_ratio: int, cr_threshold: float) -> Dict:
    """
    Filters the predictions based on confidence ratio values, selecting the top-k predictions
    that exceed the specified confidence ratio threshold.

    Parameters:
        outputdict (Dict): Dictionary containing source-target predictions with scores and confidence ratios.
        topk_confidence_ratio (int): Number of top predictions to keep based on confidence ratio.
        cr_threshold (float): The threshold for confidence ratio to filter predictions.

    Returns:
        Dict: Filtered predictions with the top-k items exceeding the confidence ratio threshold.
    """
    outputdict_confidence_ratios = {}
    for source_iri, target_cands in outputdict.items():
        top_k_items = sorted(target_cands, key=lambda X: X[3] >= cr_threshold, reverse=True)[:topk_confidence_ratio]
        outputdict_confidence_ratios[source_iri] = top_k_items
    return outputdict_confidence_ratios


def confidence_score_based_filtering(outputdict_confidence_ratios: Dict, topk_confidence_score: int, llm_confidence_threshold: float, ir_score_threshold: float) -> List:
    """
    Filters the predictions based on LLM confidence score and IR score, selecting the top-k
    predictions that exceed the given thresholds.

    Parameters:
        outputdict_confidence_ratios (Dict): Dictionary with source-target predictions filtered by confidence ratio.
        topk_confidence_score (int): Number of top predictions to keep based on LLM confidence score.
        llm_confidence_threshold (float): The threshold for LLM confidence score to filter predictions.
        ir_score_threshold (float): The threshold for IR score to filter predictions.

    Returns:
        List: Filtered predictions based on LLM confidence score and IR score thresholds.
    """
    filtered_predicts = []
    for source_iri, target_cands in outputdict_confidence_ratios.items():
        top_k_items = sorted(target_cands, key=lambda X: (X[2] >= llm_confidence_threshold), reverse=True)[:topk_confidence_score]
        for target, ir_score, llm_confidence, confidence_ratio in top_k_items:
            if ir_score >= ir_score_threshold:
                filtered_predicts.append(
                    {
                        "source": source_iri,
                        "target": target,
                        "score": ir_score,
                        "confidence": llm_confidence,
                        "confidence-ratio": confidence_ratio,
                    }
                )
    return filtered_predicts

def truncate_generation_noise(text: str) -> str:
    """
    Truncate common trailing prompt/regeneration artifacts from model output.

    This keeps the primary answer region while dropping repeated prompt sections
    such as "### Concept 1" / "### Concept 2" / "### Output".
    """
    if not isinstance(text, str):
        return ""
    markers = ["\n### Concept 1", "\n### Concept 2", "\n### Output", "\n### Your Answer", "\n### Explanation"]
    cut_positions = [text.find(marker) for marker in markers if text.find(marker) != -1]
    if cut_positions:
        text = text[:min(cut_positions)]
    return text.strip()

def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse the first balanced JSON object from a generated text.
    Returns None if parsing fails.
    """
    if not isinstance(text, str):
        return None

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.replace("“", "\"").replace("”", "\"")

    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    end = None
    for idx in range(start, len(cleaned)):
        char = cleaned[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "\"":
                in_string = False
            continue
        if char == "\"":
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break

    if end is None:
        return None

    candidate = cleaned[start:end]
    try:
        parsed = json.loads(candidate)
    except Exception:
        # Fallback 1: remove trailing commas before } or ]
        relaxed = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            parsed = json.loads(relaxed)
        except Exception:
            # Fallback 2: Python-literal style dict (single quotes, etc.)
            try:
                parsed = ast.literal_eval(candidate)
            except Exception:
                # Fallback 3: try first regex-captured {...} block with DOTALL
                match = re.search(r"\{[\s\S]*\}", cleaned)
                if not match:
                    return None
                blob = match.group(0)
                blob = re.sub(r",\s*([}\]])", r"\1", blob)
                try:
                    parsed = json.loads(blob)
                except Exception:
                    try:
                        parsed = ast.literal_eval(blob)
                    except Exception:
                        return None

    return parsed if isinstance(parsed, dict) else None
    

def normalize_relation_label(relation: str) -> str:
    """
    Normalize relation labels produced by LLMs into canonical taxonomy labels.
    """
    if not isinstance(relation, str):
        return "no_relation"
    norm = relation.strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "equivalent": "equivalent_class",
        "equivalent_class": "equivalent_class",
        "sub_class": "sub_class_of",
        "sub_class_of": "sub_class_of",
        "subclass_of": "sub_class_of",
        "super_class": "super_class_of",
        "super_class_of": "super_class_of",
        "superclass_of": "super_class_of",
        "related": "related_to",
        "related_to": "related_to",
        "no_relation": "no_Relation",
        "nomrelation": "no_relation",
        "no_relationn": "no_relation",
    }
    return mapping.get(norm, "no_relation") 
# Optionally add a strict mode: if JSON parse fails, mark as relation="uncertain" with reason "parse_failed".

