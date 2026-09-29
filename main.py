import ontoaligner
from ontoaligner.ontology.generic import GenericOMDataset
from ontoaligner.utils import metrics, xmlify
from ontoaligner.aligner import MistralLLMBERTRetrieverRAG
from ontoaligner.encoder import ConceptParentRAGEncoder
from ontoaligner.postprocess import rag_hybrid_postprocessor

task = GenericOMDataset()
dataset = task.collect(
    source_ontology_path="/vast/ve83rur/OntoAligner/assets/AberOWL/envo.fixed.owl",
    target_ontology_path="/vast/ve83rur/OntoAligner/assets/AberOWL/foodon.fixed.owl"
)

encoder_model = ontoaligner.encoder.ConceptRAGEncoder()
encoded_ontology = encoder_model(source=dataset['source'], target=dataset['target'])

# Step 4: Define configuration for retriever and LLM
retriever_config = {"device": 'cuda', "top_k": 5,}
llm_config = {"device": "cuda", "max_length": 300, "max_new_tokens": 10, "batch_size": 15}

# Step 5: Initialize Generate predictions using RAG-based ontology matcher
model = MistralLLMBERTRetrieverRAG(retriever_config=retriever_config, llm_config=llm_config)
model.load(llm_path = "mistralai/Mistral-7B-v0.3", ir_path="all-MiniLM-L6-v2")
predicts = model.generate(input_data=encoded_ontology)

# Step 6: Apply hybrid postprocessing
hybrid_matchings, hybrid_configs = rag_hybrid_postprocessor(predicts=predicts,
                                                            ir_score_threshold=0.1,
                                                            llm_confidence_th=0.8)

evaluation = metrics.evaluation_report(predicts=hybrid_matchings, references=dataset['reference'])
print("Hybrid Matching Evaluation Report:", evaluation)

# Step 7: Convert matchings to XML format and save the XML representation
xml_str = xmlify.xml_alignment_generator(matchings=hybrid_matchings)
open("matchings.xml", "w", encoding="utf-8").write(xml_str)