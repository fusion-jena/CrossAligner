#!/usr/bin/env bash
#SBATCH --job-name=ontoalign-Qwen3-32B-raw_s_f
#SBATCH --nodes=1
#SBATCH --partition=gpu,gpu-test
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --exclude=gpu013,gpu014,gpu015,gpu005,gpu007
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-user=divyasha.sunil.naik@uni-jena.de
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

import os
import json
import pandas as pd
from transformers import BitsAndBytesConfig
from ontoaligner.pipeline import OntoAlignerPipeline
from ontoaligner.ontology.generic import GenericOMDataset
from ontoaligner.encoder.concept_formatting import (
    format_concept_context,
    format_candidate_concept_context,
)
from ontoaligner.encoder import ConceptCandidateLLMEncoder

import warnings
from ontoaligner.ontology.deeponto_verbalizer import DeepOntoAxiomEnricher

source_candidate_root_iri = os.environ.get(
    "SOURCE_CANDIDATE_ROOT_IRI",
    "http://purl.obolibrary.org/obo/OCCO_35000000",                 # food preparation or serving related occupation holder
)             

target_candidate_root_iris = [
    "http://purl.bioontology.org/ontology/MESH/D011634",             # PUBLIC_HEALTH Mesh
    "http://purl.bioontology.org/ontology/MESH/D007221",              # INDUSTRY Mesh
]
            
pipe = OntoAlignerPipeline(
    task_class=GenericOMDataset,
    source_ontology_path="assets/food-onto/occo.owl",   # assets/food-onto/occo.owl  # mesh.owl
    target_ontology_path="assets/food-onto/mesh.owl",
    #reference_matching_path="",      # fill if you evaluate
    output_dir="results",
    output_format="json",            # ensure JSON output file
    source_candidate_root_iris=source_candidate_root_iri,
    target_candidate_root_iris=target_candidate_root_iris,
)

out = pipe(
    method="llm",
    encoder_model=ConceptCandidateLLMEncoder(),
    llm_path="Qwen/Qwen3-32B",
    #llm_path="meta-llama/Llama-3.3-70B-Instruct",
    device="cuda",
    device_map="balanced",
    llm_output_mode="binary",      # relation or "binary"
    #apply_top_k_in_llm=True,
    #top_k=5, 
    batch_size=1,
    max_length=800,
    max_new_tokens=10,

    save_matchings=False,
    return_matching=True,
    llm_kwargs={
        "num_beams": 1,
        "do_sample": False,          # your setting
        "quantization_config": BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        ),
    },
)

print("Done. Output type:", type(out))
if isinstance(out, list) and out:
    print("First item:", out[0])

# ---- post-run export block ----
# raw-mode export
run_name = "Qwen3-32B_raw_s_f_occo_mesh_deeponto_stage1_case4"                     #"Qwen3-32B_raw_s_f_mesh_ons_binary"
out_dir = "results/Stage_1/Qwen/occo_mesh"

#run_name = "Llama-3.3-70B-Instruct_raw_s_f_occo_mesh_deeponto_stage1_case4"
#out_dir = "results/Llama-3.3-70B-Instruct"

os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, f"{run_name}.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

if pd is not None:
    pd.DataFrame(out).to_csv(os.path.join(out_dir, f"{run_name}.csv"), index=False)

PY

echo "# Job $SLURM_JOB_NAME finished at $(date +%F-%T)"
