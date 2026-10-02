#!/usr/bin/env bash
#SBATCH --job-name=ontoalign-Qwen3-32B-reason
#SBATCH --nodes=1
#SBATCH --partition=gpu,gpu-test
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --exclude=gpu013
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-user=""
#SBATCH --mail-type=ALL

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

: "${HF_TOKEN:?HF_TOKEN is not set. Export it before submitting the job.}"
mkdir -p logs




echo "# Job $SLURM_JOB_NAME started at $(date +%F-%T)"
module purge;
 
export http_proxy="http://internet4nzm.rz.uni-jena.de:3128"
export https_proxy="http://internet4nzm.rz.uni-jena.de:3128"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"



srun --cpu-bind=none python3 - <<'PY'
import json
import os
import pandas as pd
    
from transformers import BitsAndBytesConfig
from ontoaligner.encoder import ConceptLLMEncoder
from ontoaligner.ontology.generic import GenericOMDataset
from ontoaligner.pipeline import OntoAlignerPipeline
from ontoaligner.ontology.deeponto_verbalizer import DeepOntoAxiomEnricher
from ontoaligner.aligner.llm.dataset import (
    MeshOnsSourceToTargetZeroShotDataset,
    MeshOnsTargetToSourceZeroShotDataset,
    MeshOnsSourceToTargetFewShotDataset,
    MeshOnsTargetToSourceFewShotDataset,
)

# Change only this value when selecting another experiment.
experiment_name = "source_to_target_zero_shot"           # or experiment_name = "target_to_source_zero_shot" /"source_to_target_few_shot" /"target_to_source_few_shot"

experiment_configs = {
    "source_to_target_zero_shot": {
        "dataset_class": MeshOnsSourceToTargetZeroShotDataset,
        "direction": "1-2",
        "prompt_mode": "without_eg",
    },
    "target_to_source_zero_shot": {
        "dataset_class": MeshOnsTargetToSourceZeroShotDataset,
        "direction": "2-1",
        "prompt_mode": "without_eg",
    },
    "source_to_target_few_shot": {
        "dataset_class": MeshOnsSourceToTargetFewShotDataset,
        "direction": "1-2",
        "prompt_mode": "with_eg",
    },
    "target_to_source_few_shot": {
        "dataset_class": MeshOnsTargetToSourceFewShotDataset,
        "direction": "2-1",
        "prompt_mode": "with_eg",
    },
}

experiment = experiment_configs[experiment_name]

dataset_class = experiment["dataset_class"]
direction = experiment["direction"]
prompt_mode = experiment["prompt_mode"]


stage1_run = "qwen"  # Available options: "qwen", "llama"

stage1_configs = {
    "qwen": {
        "model_name": "Qwen",
        "candidate_file": (
            "results/Stage_1/Qwen/mesh_ons/"
            "Qwen3-32B_raw_s_f_mesh_ons_deeponto_stage1_case4.json"
        ),
    },
    "llama": {
        "model_name": "Llama",
        "candidate_file": (
            "results/Stage_1/Llama/mesh_ons/"
            "Llama-3.3-70B-Instruct_raw_s_f_mesh_ons_"
            "deeponto_stage1_case4.json"
        ),
    },
}

stage1_config = stage1_configs.get(stage1_run)

if stage1_config is None:
    raise ValueError(
        f"Unknown stage1_run: {stage1_run}. "
        f"Available options: {list(stage1_configs)}"
    )

stage1_model_name = stage1_config["model_name"]
candidate_file = stage1_config["candidate_file"]

if not os.path.isfile(candidate_file):
    raise FileNotFoundError(
        f"Candidate matching file not found: {candidate_file}"
    )


with open(candidate_file, encoding="utf-8") as candidate_input:
    candidates = json.load(candidate_input)

selected_source_iris = {
    candidate["source"]
    for candidate in candidates
}

selected_target_iris = {
    candidate["target"]
    for candidate in candidates
}


pipe = OntoAlignerPipeline(
    task_class=GenericOMDataset,
    source_ontology_path="assets/food-onto/mesh.owl",
    target_ontology_path="assets/food-onto/ons.owl",
    #reference_matching_path="",
    output_dir="results",
    output_format="json",
)

source_enricher = DeepOntoAxiomEnricher(
    ontology_path="assets/food-onto/mesh.owl",
    max_axioms_per_class=8,
    include_named_subclass_axioms=False,
    jvm_memory="4g",
)

target_enricher = DeepOntoAxiomEnricher(
    ontology_path="assets/food-onto/ons.owl",
    max_axioms_per_class=8,
    include_named_subclass_axioms=False,
    jvm_memory="4g",
)


source_enricher.enrich(
    pipe.dataset["source"],
    selected_iris=selected_source_iris,
)

target_enricher.enrich(
    pipe.dataset["target"],
    selected_iris=selected_target_iris,
)


out = pipe(
    method="llm",
    encoder_model=ConceptLLMEncoder(),
    dataset_class=dataset_class,

    candidate_matching_path=candidate_file,
    llm_prompt_preview_count=1,                       # For all the input to the prompt print in log # llm_prompt_preview_count=14, by default to 0

    llm_path="meta-llama/Llama-3.3-70B-Instruct",
    #llm_path="Qwen/Qwen3-32B",
    device="cuda",
    device_map="balanced",

    #apply_top_k_in_llm=True,
    #top_k=5, 

    llm_output_mode="raw",

    batch_size=1,
    max_length=1000,
    max_new_tokens=80,

    # The manual export block below saves the result, so avoid a duplicate.
    save_matchings=False,
    return_matching=True,

    llm_kwargs={
        "num_beams": 1,
        "do_sample": False,
        "quantization_config": BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        ),
    },
)


print("Done. Output type:", type(out))
print("Number of processed candidate pairs:", len(out))

if out:
    print("First item:")
    print(json.dumps(out[0], indent=2, ensure_ascii=False))

stage2_model_name = "Llama-3.3-70B-Instruct"
ontology_pair = "mesh_ons"
experiment_case = "case6"

run_name = (
    f"{stage1_model_name}_Stage1_"
    f"{stage2_model_name}_"
    f"raw_s_f_{ontology_pair}_stage2_"
    f"{experiment_case}_{prompt_mode}_{direction}"
)

print("Run name:", run_name)
out_dir = f"results/Stage_2/{ontology_pair}"


os.makedirs(out_dir, exist_ok=True)

json_path = os.path.join(out_dir, f"{run_name}.json")
csv_path = os.path.join(out_dir, f"{run_name}.csv")

with open(json_path, "w", encoding="utf-8") as output_file:
    json.dump(out, output_file, indent=2, ensure_ascii=False)

pd.DataFrame(out).to_csv(csv_path, index=False)

print("JSON saved to:", json_path)
print("CSV saved to:", csv_path)
PY

echo "# Job $SLURM_JOB_NAME finished at $(date +%F-%T)"
