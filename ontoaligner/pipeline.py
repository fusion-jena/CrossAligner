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
Ontology Alignment Pipeline. Various methods such as lightweight matching, retriever-based matching, LLM-based matching,
and RAG (Retriever-Augmented Generation) techniques has been applied.
"""
import json
from pathlib import Path
from tqdm import tqdm
import warnings

from .utils.candidate_endpoints import (
    descendant_candidate_iris,
    filter_candidate_endpoints,
    load_candidate_iris,
)
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import Any, Dict
from sklearn.linear_model import LogisticRegression

from .utils.prompt_preview import (
    resolve_prompt_preview_count,
    validate_prompt_preview_count,
)

from .base import BaseEncoder, BaseOMModel, OMDataset
from .encoder import ConceptLightweightEncoder, ConceptLLMEncoder, ConceptRAGEncoder, ConceptParentFewShotEncoder
from .utils import metrics, xmlify
from .aligner import SimpleFuzzySMLightweight, SBERTRetrieval, AutoModelDecoderLLM, ConceptLLMDataset
#from .postprocess import retriever_postprocessor, llm_postprocessor, rag_hybrid_postprocessor, TFIDFLabelMapper, LabelMapper
from .postprocess import (
    retriever_postprocessor,
    llm_postprocessor,
    llm_relation_postprocessor,
    rag_hybrid_postprocessor,
    TFIDFLabelMapper,
    LabelMapper,
)

class OntoAlignerPipeline:
    """
    A pipeline for performing ontology alignment tasks using various methods and models.
    """
    '''def __init__(self, task_class: OMDataset, source_ontology_path: str, target_ontology_path: str,
                 reference_matching_path: str, output_dir: str ="results", output_format: str ="xml"):'''
    def __init__(self, task_class: OMDataset, source_ontology_path: str, target_ontology_path: str,
                 #output_dir: str ="results", output_format: str ="xml"):
                 output_dir: str ="results", output_format: str ="xml",
                 source_candidate_iris_path: str = None,
                 target_candidate_iris_path: str = None,
                 source_candidate_root_iris=None,
                 target_candidate_root_iris=None):
        """
        Initializes the OntoAlignerPipeline.

        Parameters:
            task_class (OMDataset): Class responsible for handling ontology matching tasks.
            source_ontology_path (str): Path to the source ontology file.
            target_ontology_path (str): Path to the target ontology file.
            reference_matching_path (str): Path to the reference alignments.
            output_dir (str, optional): Directory to save results. Defaults to "results".
            output_format (str, optional): Format of output files. Defaults to "xml".
            source_candidate_iris_path (str, optional): ROBOT-style term file
                containing source classes that may be matching endpoints.
            target_candidate_iris_path (str, optional): ROBOT-style term file
                containing target classes that may be matching endpoints.
            source_candidate_root_iris (str or iterable, optional): Source root
                IRI or IRIs whose asserted descendants are matching endpoints.
            target_candidate_root_iris (str or iterable, optional): Target root
                IRI or IRIs whose asserted descendants are matching endpoints.
        """
        self.task_class = task_class
        self.source_ontology_path = source_ontology_path
        self.target_ontology_path = target_ontology_path
        #self.reference_matching_path = reference_matching_path
        self.output_dir = Path(output_dir)
        self.output_format = output_format.lower()
        self.source_candidate_iris_path = source_candidate_iris_path
        self.target_candidate_iris_path = target_candidate_iris_path
        self.source_candidate_root_iris = source_candidate_root_iris
        self.target_candidate_root_iris = target_candidate_root_iris
        self.task = self._initialize_task()
        self.dataset = self._collect_dataset()
        self._filter_candidate_endpoints()

    def _initialize_task(self):
        """
        Initializes the ontology matching task.

        Returns:
            OMDataset: Initialized task object.
        """
        return self.task_class()

    def _collect_dataset(self):
        """
        Collects the dataset required for ontology matching.

        Returns:
            dict: A dictionary containing source, target, and reference datasets.
        """
        return self.task.collect(
            source_ontology_path=self.source_ontology_path,
            target_ontology_path=self.target_ontology_path,
            #reference_matching_path=self.reference_matching_path
        )

    def _filter_candidate_endpoints(self):
        """Restrict matching endpoints without modifying either OWL module."""
        endpoint_configs = {
            "source": (
                self.source_candidate_iris_path,
                self.source_candidate_root_iris,
            ),
            "target": (
                self.target_candidate_iris_path,
                self.target_candidate_root_iris,
            ),
        }
        for side, (path, roots) in endpoint_configs.items():
            if path is not None and roots is not None:
                raise ValueError(
                    f"Configure either {side}_candidate_iris_path or "
                    f"{side}_candidate_root_iris, not both."
                )
            if path is None and roots is None:
                continue
            if path is not None:
                candidate_iris = load_candidate_iris(path)
                selector = f"term file {path}"
            else:
                root_iris = [roots] if isinstance(roots, str) else list(roots)
                candidate_iris = descendant_candidate_iris(
                    self.dataset[side], root_iris
                )
                selector = f"root descendant closure ({len(root_iris)} root(s))"
            before = len(self.dataset[side])
            self.dataset[side] = filter_candidate_endpoints(
                self.dataset[side], candidate_iris
            )
            print(
                f"Candidate endpoint filter ({side}, {selector}): "
                f"{before} ontology classes -> {len(self.dataset[side])} endpoints"
            )

    def __call__(self, method: str, encoder_model: BaseEncoder = None, model_class: BaseOMModel = None, dataset_class: Dataset = None, postprocessor: Any = None,
                 llm_path: str = None, retriever_path: str = None, device: str = "cuda", batch_size: int = 2048, max_length: int = 300, max_new_tokens: int = 10,
                 top_k: int = 10, fuzzy_sm_threshold: float = 0.2, evaluate: bool = False, return_matching: bool = True, output_file_name: str = "matchings",
                 save_matchings: bool = False, ir_threshold: float = 0.5, ir_rag_threshold: float = 0.7, llm_threshold: float = 0.5, llm_mapper: LabelMapper = None,
                 llm_mapper_interested_class: str = 'yes', answer_set: Dict = {"yes": ["yes", "true"], "no": ["no", "false"]}, huggingface_access_token: str = "",
                # openai_key: str = "", device_map: str = "auto", positive_ratio: float = 0.7, n_shots: int = 5) -> [Any, Any]:
                 openai_key: str = "", device_map: str = "auto", positive_ratio: float = 0.7, n_shots: int = 5,
                 candidate_matching_path: str = None, candidate_pairs: Any = None,
                 llm_output_mode: str = "binary", llm_kwargs: Dict[str, Any] = None, apply_top_k_in_llm: bool = False, llm_prompt_preview_count: int = 0,
                 rag_mode: str = "hybrid") -> [Any, Any]:
        """
        Executes the ontology alignment process using the specified method.

        Parameters:
            method (str): The method to use, e.g., "lightweight", "retriever", or "llm".
            encoder_model (BaseEncoder, optional): Encoder model to encode ontologies. Defaults to None.
            model_class (BaseOMModel, optional): Model class for matching. Defaults to None.
            dataset_class (Dataset, optional): Dataset class for LLM-based methods. Defaults to None.
            postprocessor (Any, optional): Post-processing function. Defaults to None.
            llm_path (str, optional): Path to the LLM model. Defaults to None.
            retriever_path (str, optional): Path to the retriever model. Defaults to None.
            device (str, optional): Device to use for computation. Defaults to "cuda".
            batch_size (int, optional): Batch size for LLM-based methods. Defaults to 2048.
            max_length (int, optional): Maximum input length for LLM-based methods. Defaults to 300.
            max_new_tokens (int, optional): Maximum tokens to generate for LLM-based methods. Defaults to 10.
            top_k (int, optional): Number of top matches to retrieve in the retriever method. Defaults to 10.
            fuzzy_sm_threshold (float, optional): Threshold for fuzzy matching in lightweight methods. Defaults to 0.2.
            evaluate (bool, optional): Whether to evaluate the matching results. Defaults to False.
            return_matching (bool, optional): Whether to return the matching results. Defaults to True.
            output_file_name (str, optional): Output file name without file type. Defaults to "matchings".
            save_matchings (bool, optional): Whether to save the matching results. Defaults to False.
            ir_threshold (float, optional): Retrieval postprocessor threshold.
            ir_rag_threshold (float, optional): Retrieval postprocessor threshold in RAG module.
            llm_threshold (float, optional): LLM postprocessor threshold.
            llm_mapper (LabelMapper, optional): Mapper for LLM outputs.
            llm_mapper_interested_class (str, optional): Class to filter output pairs in LLM postprocessing.
            answer_set (dict, optional): Mapping of yes/no answers. Defaults to {"yes": ["yes", "true"], "no": ["no", "false"]}.
            huggingface_access_token (str, optional): Access token for Hugging Face models. Defaults to "".
            openai_key (str, optional): API key for OpenAI models. Defaults to "".
            device_map (str, optional): Device map for model allocation. Defaults to "auto".
            positive_ratio (float, optional): Ratio of positive examples in few-shot methods. Defaults to 0.7.
            n_shots (int, optional): Number of shots for few-shot learning. Defaults to 5.
            #llm_output_mode (str, optional): "binary" for yes/no mapper flow, "relation" for JSON relation parsing flow.
            llm_output_mode (str, optional): "binary" for yes/no mapper flow, "relation" for JSON relation parsing flow,
                or "raw" to skip postprocessing and return generated raw text per source-target pair.
            llm_kwargs (Dict[str, Any], optional): Additional keyword arguments passed to the LLM model class
                (e.g., temperature, top_p, num_beams, do_sample, quantization_config).
            apply_top_k_in_llm (bool, optional): If True, limit LLM target candidates to first `top_k` items
                before building source-target pairs. Defaults to False.
            candidate_matching_path (str, optional): JSON file containing
                Stage-1 ``source``/``target`` pairs for a Stage-2 dataset.
            candidate_pairs (list, optional): Already-loaded Stage-1 pair
                dictionaries. Mutually exclusive with candidate_matching_path.
            llm_prompt_preview_count (int, optional): Print up to this many fully
                rendered prompts before inference. Defaults to 0 (disabled).
            rag_mode (str, optional): RAG output mode. "hybrid" applies the configured RAG postprocessor
                (default behavior). "raw" skips postprocessing and returns raw model outputs from RAG.

        Returns:
            dict or None: Evaluation report if `evaluate` is True. Matching results if `return_matching` is True.
        """
        if not (0 <= fuzzy_sm_threshold <= 1):
            raise ValueError(f"fuzzy_sm_threshold must be between 0 and 1. Got {fuzzy_sm_threshold}")

        validate_prompt_preview_count(llm_prompt_preview_count)
        if candidate_matching_path is not None and candidate_pairs is not None:
            raise ValueError(
                "Configure either candidate_matching_path or candidate_pairs, not both."
            )
        candidate_source = None
        if candidate_matching_path is not None:
            with open(candidate_matching_path, encoding="utf-8") as candidate_file:
                candidate_pairs = json.load(candidate_file)
            candidate_source = candidate_matching_path
        elif candidate_pairs is not None:
            candidate_source = "the candidate_pairs argument"

        if method not in ["lightweight", "retrieval", "llm"] and "rag" not in method:
            raise ValueError(f"Unknown method: {method}")

        if method == "lightweight":
            matchings = self._run_lightweight(encoder_model or ConceptLightweightEncoder(), model_class or SimpleFuzzySMLightweight,
                                              postprocessor, fuzzy_sm_threshold)
        if method == "retrieval":
            matchings = self._run_retriever(encoder_model or ConceptLightweightEncoder(), model_class or SBERTRetrieval, postprocessor or retriever_postprocessor,
                                             retriever_path, device, top_k, ir_threshold)

        if method == "llm":
            default_llm_postprocessor = None if llm_output_mode == "raw" else (
                llm_relation_postprocessor if llm_output_mode == "relation" else llm_postprocessor
            )
            '''matchings = self._run_llm(encoder_model or ConceptLLMEncoder(), model_class or AutoModelDecoderLLM, dataset_class or ConceptLLMDataset,
                           postprocessor or default_llm_postprocessor, llm_mapper or TFIDFLabelMapper(classifier=LogisticRegression(), ngram_range=(1, 1)),
                           llm_mapper_interested_class, llm_path, device, batch_size, max_length, max_new_tokens,
                           llm_threshold, llm_output_mode, huggingface_access_token, device_map, llm_kwargs,
                           top_k, apply_top_k_in_llm, candidate_pairs, candidate_source, llm_prompt_preview_count)'''


            matchings = self._run_llm(
                encoder_model=encoder_model or ConceptLLMEncoder(),
                model_class=model_class or AutoModelDecoderLLM,
                dataset_class=dataset_class or ConceptLLMDataset,
                postprocessor=postprocessor or default_llm_postprocessor,
                llm_mapper=llm_mapper or TFIDFLabelMapper(
                    classifier=LogisticRegression(), ngram_range=(1, 1)
                ),
                llm_mapper_interested_class=llm_mapper_interested_class,
                llm_path=llm_path,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                llm_threshold=llm_threshold,
                llm_output_mode=llm_output_mode,
                huggingface_access_token=huggingface_access_token,
                device_map=device_map,
                llm_kwargs=llm_kwargs,
                top_k=top_k,
                apply_top_k_in_llm=apply_top_k_in_llm,
                candidate_pairs=candidate_pairs,
                candidate_source=candidate_source,
                llm_prompt_preview_count=llm_prompt_preview_count,
            )
            
            '''matchings = self._run_llm(encoder_model or ConceptLLMEncoder(), model_class or AutoModelDecoderLLM, dataset_class or ConceptLLMDataset,
                                       postprocessor or default_llm_postprocessor, llm_mapper or TFIDFLabelMapper(classifier=LogisticRegression(), ngram_range=(1, 1)),
                                       llm_mapper_interested_class, llm_path, device, batch_size, max_length, max_new_tokens,
                                       llm_threshold, llm_output_mode, huggingface_access_token, device_map, llm_kwargs)'''

        if 'rag' in method:
            retriever_config = {"device": device, "top_k": top_k, "openai_key": openai_key}
            llm_config = {"device": device, "batch_size": batch_size, "answer_set": answer_set, "huggingface_access_token": huggingface_access_token,
                          "max_length": max_length, "max_new_tokens": max_new_tokens, "openai_key": openai_key, "device_map": device_map}
            rag_config = {"retriever_config": retriever_config, "llm_config": llm_config}
            if method == 'fewshot-rag':
                rag_config['n_shots'] = n_shots
                rag_config['positive_ratio'] = positive_ratio
                encoder_model = encoder_model or ConceptParentFewShotEncoder()
            else:
                encoder_model = encoder_model or ConceptRAGEncoder()
            matchings = self._run_rag(method, encoder_model, model_class, postprocessor or rag_hybrid_postprocessor,
                                      llm_threshold,  ir_rag_threshold, retriever_path, llm_path, rag_config, rag_mode)
                                     #llm_threshold,  ir_rag_threshold, retriever_path, llm_path, rag_config)
        return self._process_results(matchings, method, evaluate, return_matching, output_file_name, save_matchings)

    def _run_lightweight(self, encoder_model, model_class, postprocessor, fuzzy_sm_threshold):
        """
        Executes the lightweight ontology alignment method.

        This method uses a lightweight matching model to generate ontology matchings
        based on encoded source and target ontologies. Optionally, a postprocessor
        is applied to refine the matching results.

        Parameters:
            encoder_model (BaseEncoder): Encoder model to encode the source and target ontologies.
            model_class (BaseOMModel): A class implementing the lightweight matching logic.
            postprocessor (callable or None): A function to refine the matching results. Optional.
            fuzzy_sm_threshold (float): Threshold value for fuzzy similarity matching in the model.

        Returns:
            dict: The resulting matchings after encoding and processing.
        """
        encoder_output = encoder_model(source=self.dataset['source'], target=self.dataset['target'])
        model = model_class(fuzzy_sm_threshold=fuzzy_sm_threshold)
        matchings = model.generate(input_data=encoder_output)
        if postprocessor:
            matchings = postprocessor(matchings)
        return matchings

    def _run_retriever(self, encoder_model, model_class, postprocessor, retriever_path, device, top_k, ir_threshold):
        """
        Executes the retriever-based ontology alignment method.

        This method leverages a retriever model to identify top-k potential matches
        between ontologies based on their encoded representations. The results are
        refined using a postprocessor.

        Parameters:
            encoder_model (BaseEncoder): Encoder model to encode the source and target ontologies.
            model_class (BaseOMModel): A class implementing the retriever-based matching logic.
            postprocessor (callable): A function to refine the matching results.
            retriever_path (str): File path to the pretrained retriever model.
            device (str): The computational device (e.g., 'cpu' or 'cuda').
            top_k (int): Number of top candidate matches to retrieve.
            ir_threshold (float): Threshold for the postprocessor to filter results.

        Returns:
            dict: The resulting matchings after encoding, retrieval, and processing.
        """
        encoder_output = encoder_model(source=self.dataset['source'], target=self.dataset['target'])
        model = model_class(device=device, top_k=top_k)
        model.load(path=retriever_path)
        matchings = model.generate(input_data=encoder_output)
        matchings = postprocessor(matchings, threshold=ir_threshold)
        return matchings

    def _run_llm(self, encoder_model, model_class, dataset_class, postprocessor, llm_mapper,
                 llm_mapper_interested_class,
                 llm_path, device, batch_size, max_length, max_new_tokens, llm_threshold, llm_output_mode,
                 huggingface_access_token, device_map, llm_kwargs, top_k, apply_top_k_in_llm,
                 candidate_pairs, candidate_source, llm_prompt_preview_count):
        """

        Executes the LLM-based ontology alignment method.

        This method uses a large language model (LLM) to generate matching predictions
        based on a dataset constructed from the encoded ontologies. The results are
        postprocessed to align with the desired format and filtering criteria.

        Parameters:
            encoder_model (BaseEncoder): Encoder model to encode the source and target ontologies.
            model_class (BaseOMModel): A class implementing LLM-based matching logic.
            dataset_class (Dataset): A class to construct datasets for LLM-based methods.
            postprocessor (callable): A function to refine the matching results.
            llm_mapper (LabelMapper): A mapper to process LLM outputs into matchings.
            llm_mapper_interested_class (str): Specific class of matchings to filter in the results.
            llm_path (str): File path to the pretrained LLM model.
            device (str): The computational device (e.g., 'cpu' or 'cuda').
            batch_size (int): Number of samples to process in each batch.
            max_length (int): Maximum input sequence length for the LLM.
            max_new_tokens (int): Maximum tokens to generate for each sequence.
            llm_threshold (float): Threshold for the postprocessor to filter results.

        Returns:
            dict: The resulting matchings after encoding, LLM generation, and processing.
        """
        encoder_output = encoder_model(source=self.dataset['source'], target=self.dataset['target'])
        #llm_dataset = dataset_class(source_onto=encoder_output[0], target_onto=encoder_output[1])
        source_onto, target_onto = encoder_output[0], encoder_output[1]
        if apply_top_k_in_llm:
            if top_k < 1:
                raise ValueError("top_k must be at least 1 when apply_top_k_in_llm=True")
            warnings.warn(
                "apply_top_k_in_llm selects the first targets in ontology parser "
                "order; it does not select the most similar targets per source. "
                "Use a retrieval/RAG stage for ranked per-source candidates.",
                UserWarning,
                stacklevel=2,
            )
            target_onto = target_onto[:top_k]
        #llm_dataset = dataset_class(source_onto=source_onto, target_onto=target_onto)

        if candidate_pairs is not None:
            llm_dataset = dataset_class(
                source_onto=source_onto,
                target_onto=target_onto,
                candidate_pairs=candidate_pairs,
            )
            print(
                f"Stage-2 candidate dataset: {len(llm_dataset)} valid pairs "
                f"loaded from {candidate_source}"
            )
        else:
            print(
                "LLM candidate Cartesian product: "
                f"{len(source_onto)} source endpoints x "
                f"{len(target_onto)} target endpoints = "
                f"{len(source_onto) * len(target_onto)} pairs"
            )
            llm_dataset = dataset_class(source_onto=source_onto, target_onto=target_onto)
        preview_count = resolve_prompt_preview_count(
            llm_prompt_preview_count,
            len(llm_dataset),
        )
        for index in range(preview_count):
            preview = llm_dataset[index]
            print(
                f"\n===== LLM PROMPT PREVIEW {index + 1}/{preview_count} "
                f"| IRIs: {preview['iris']} =====\n"
                f"{preview['prompts']}\n"
                "===== END LLM PROMPT PREVIEW =====\n"
            )

        dataloader = DataLoader(llm_dataset, batch_size=batch_size, shuffle=False, collate_fn=llm_dataset.collate_fn)
        #model = model_class(device=device, max_length=max_length, max_new_tokens=max_new_tokens)
        model = model_class(
            device=device,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            huggingface_access_token=huggingface_access_token,
            device_map=device_map,
            **(llm_kwargs or {}),
        )
        model.load(path=llm_path)

        matchings = []
        for batch in tqdm(dataloader):
            sequences = model.generate(batch["prompts"])
            matchings.extend(sequences)

        if llm_output_mode == "raw":
            source_by_iri = {
                concept["iri"]: concept
                for concept in self.dataset["source"]
            }
        
            target_by_iri = {
                concept["iri"]: concept
                for concept in self.dataset["target"]
            }
        
            results = []
        
            for predict, data in zip(matchings, llm_dataset):
                source_iri = data["iris"][0]
                target_iri = data["iris"][1]
        
                source_concept = source_by_iri.get(source_iri, {})
                target_concept = target_by_iri.get(target_iri, {})
        
                source_label = (
                    source_concept.get("label")
                    or source_concept.get("name")
                    or source_iri
                )
        
                target_label = (
                    target_concept.get("label")
                    or target_concept.get("name")
                    or target_iri
                )
        
                results.append(
                    {
                        "source": source_iri,
                        "source_label": source_label,
                        "target": target_iri,
                        "target_label": target_label,
                        "raw_output": predict,
                    }
                )
        
            return results

        if llm_output_mode == "relation":
            relation_postprocessor = (
                postprocessor
                or llm_relation_postprocessor
            )
        
            matchings = relation_postprocessor(
                predicts=matchings,
                dataset=llm_dataset,
            )
        
        else:
            # Binary mode
            generated_count = len(matchings)
            matchings = postprocessor(
                predicts=matchings,
                mapper=llm_mapper,
                dataset=llm_dataset,
                interested_class=llm_mapper_interested_class,
            )

            print(
                "LLM binary screening result: "
                f"{len(matchings)} retained {llm_mapper_interested_class!r} "
                f"pairs from {generated_count} generated decisions"
            )
            if generated_count and not matchings:
                warnings.warn(
                    f"The LLM generated {generated_count} binary decisions, but "
                    f"none mapped to {llm_mapper_interested_class!r}; the saved "
                    "matching file will therefore be empty. Use "
                    "llm_output_mode='raw' to inspect the model responses.",
                    UserWarning,
                    stacklevel=2,
                )
        
        # Add labels to binary and relation outputs.
        source_by_iri = {
            concept["iri"]: concept
            for concept in self.dataset["source"]
        }
        
        target_by_iri = {
            concept["iri"]: concept
            for concept in self.dataset["target"]
        }
        
        for matching in matchings:
            source_iri = matching["source"]
            target_iri = matching["target"]
        
            source_concept = source_by_iri.get(
                source_iri,
                {},
            )
        
            target_concept = target_by_iri.get(
                target_iri,
                {},
            )
        
            matching["source_label"] = (
                source_concept.get("label")
                or source_concept.get("name")
                or source_iri
            )
        
            matching["target_label"] = (
                target_concept.get("label")
                or target_concept.get("name")
                or target_iri
            )
        
        return matchings
        

    def _run_rag(self, method, encoder_model, model_class, postprocessor, llm_threshold, ir_threshold, retriever_path,
                 llm_path, rag_config, rag_mode: str = "hybrid"):
                
        """
        Executes the RAG (Retriever-Augmented Generation) ontology alignment method.

        This method combines retriever-based and LLM-based techniques to generate
        ontology matchings. A retriever identifies candidate matches, and an LLM
        refines the results. The final matchings are postprocessed to meet thresholds.

        Parameters:
            method (str): Specific RAG method to use (e.g., 'icv-rag', 'fewshot-rag').
            encoder_model (BaseEncoder): Encoder model to encode the ontologies.
            model_class (BaseOMModel): A class implementing RAG-based matching logic.
            postprocessor (callable): A function to refine the matching results.
            llm_threshold (float): Confidence threshold for the LLM-based predictions.
            ir_threshold (float): Score threshold for the retriever results.
            retriever_path (str): File path to the retriever model.
            llm_path (str): File path to the LLM model.
            rag_config (dict): Configuration parameters for the RAG model.
            rag_mode (str): "hybrid" to apply postprocessing, "raw" to return raw model outputs.

        Returns:
            dict: The resulting matchings after encoding, RAG generation, and processing.
        """
        encoder_output = encoder_model(source=self.dataset['source'],
                                       target=self.dataset['target'],
                                       reference=self.dataset[
                                           'reference'] if method == 'icv-rag' or method == 'fewshot-rag' else None)
        model = model_class(**rag_config)
        model.load(llm_path=llm_path, ir_path=retriever_path)
        matchings = model.generate(input_data=encoder_output)
        if rag_mode == "raw":
            return matchings
        matchings, _ = postprocessor(matchings, ir_score_threshold=ir_threshold, llm_confidence_th=llm_threshold)
        return matchings

    def _process_results(self, matchings, method, evaluate, return_matching, output_file_name, save_matchings):
        """
        Processes and evaluates the matching results.

        Parameters:
            matchings (list): List of matching results.
            method (str): The method used for alignment.
            evaluate (bool): Whether to evaluate the results.
            return_matching (bool): Whether to return the matching results.
            output_file_name (str): Output file name.
            save_matchings (bool, optional):  Whether to save the matching in results or not.

        Returns:
            dict or None: Evaluation report if `evaluate` is True. Matching results if `return_matching` is True.
        """
        output_matches = None
        if self.output_format == "xml":
            xml_str = xmlify.xml_alignment_generator(matchings=matchings)
            output_matches = xml_str
        elif self.output_format == "json":
            output_matches = matchings
        else:
            raise ValueError("Unsupported output format")

        if save_matchings:
            output_dir = self.output_dir / method
            output_dir.mkdir(parents=True, exist_ok=True)
            if self.output_format == "xml":
                with open(output_dir / f"{output_file_name}.xml", "w", encoding="utf-8") as xml_file:
                    xml_file.write(output_matches)
            elif self.output_format == "json":
                with open(output_dir / f"{output_file_name}.json", "w", encoding="utf-8") as json_file:
                    json.dump(output_matches, json_file, indent=4, ensure_ascii=False)

        evaluation = None
        if evaluate:
            evaluation = metrics.evaluation_report(predicts=matchings, references=self.dataset['reference'])

        if return_matching:
            if evaluate:
                return matchings, evaluation
            return output_matches
        if evaluate:
            return evaluation
