# CrossAligner

CrossAligner, a two-stage LLM-based methodology for discovering directed semantic bridges across three food-domain ontologies.

## Pipeline overview

1. **Stage 1 — candidate selection:** an LLM performs binary screening and retains concept pairs that are semantically related.
2. **Stage 2 — semantic-bridge generation:** an LLM reassesses the retained pairs using richer ontology context and produces a directed natural-language relationship.

The included experiments cover:

- MeSH → ONS
- OccO → MeSH
- OccO → ONS

Stage 2 supports source-to-target and target-to-source directions, each with zero-shot and few-shot prompts.

## Repository structure

```text
CrossAligner/
├── assets/food-onto/       # MeSH, OccO, and ONS ontology modules
├── bash/                   # Slurm scripts for Stage 1 and Stage 2
├── data/ground_truth/      # Expert reference mappings
├── ontoaligner/            # Pipeline implementation
├── results/
│   ├── Stage_1/            # Candidate-selection outputs
│   ├── Stage_2/            # Semantic-bridge outputs
│   └── CrossAligner_Evaluation_results/
│       ├── figures/        # Generated evaluation figures
│       └── tables/         # Generated evaluation tables
├── evaluation/
│   └── CrossAligner_Evaluation_3RQs_EXECUTED.ipynb
├── evaluation_core.py      # Evaluation functions and metrics
├── requirements.txt
└── pyproject.toml
```

## Requirements

The experimental configuration was developed for:

- Python 3.10 or 3.11
- an NVIDIA CUDA GPU
- Java 11 for DeepOnto/OWLAPI
- a Hugging Face account with access to the selected gated model

Install the required packages:
    pip install -r requirements.txt

```
Check Java and the CrossAligner imports:

```
java -version
from ontoaligner.pipeline import OntoAlignerPipeline
from ontoaligner.encoder import ConceptCandidateLLMEncoder, ConceptLLMEncoder
```

## Running Stage 1

Submit the script for the required ontology pair:

```bash
sbatch bash/Qwen3-32B-raw_s_f_DeepOnto_mesh_ons_stage1.bash
sbatch bash/Qwen3-32B-raw_s_f_DeepOnto_occo_mesh_stage1.bash
sbatch bash/Qwen3-32B-raw_s_f_DeepOnto_occo_ons_stage1.bash
```

Stage 1 writes JSON and CSV outputs to:

```text
results/Stage_1/<model>/<ontology_pair>/
```

Before starting Stage 2, confirm that the selected candidate JSON exists in that directory.

## Running Stage 2

Each Stage-2 script contains two experiment selectors:

```python
experiment_name = "source_to_target_zero_shot"
stage1_run = "qwen"
```

Available `experiment_name` values are:

- `source_to_target_zero_shot`
- `target_to_source_zero_shot`
- `source_to_target_few_shot`
- `target_to_source_few_shot`

Available `stage1_run` values are `qwen` and `llama`.

Submit the selected ontology pair:

```bash
sbatch bash/Qwen3-32B-raw_s_f_DeepOnto_stages2_mesh_ons.bash
sbatch bash/Qwen3-32B-raw_s_f_DeepOnto_stages2_occo_mesh.bash
sbatch bash/Qwen3-32B-raw_s_f_DeepOnto_stages2_occo_ons.bash
```

Stage-2 outputs are stored below:

```text
results/Stage_2/<ontology_pair>/
```

## Running the evaluation:
The evaluation reads the Stage 1 and Stage 2 outputs directly from:

```
results/Stage_1/
results/Stage_2/
```

Run the notebook evaluation/CrossAligner_Evaluation_3RQs_EXECUTED.ipynb. Generated evaluation tables and figures are written to:

```
results/CrossAligner_Evaluation_results/
├── figures/
└── tables/
```

The generated filename records the Stage-1 model, Stage-2 model, ontology pair, prompt mode, and direction.

## Acknowledgement and license

CrossAligner builds on OntoAligner by Babaei Giglou et al. The original copyright notices are retained in inherited files.

The project is distributed under the Apache License 2.0. 
