# CrossAligner

CrossAligner is a two-stage, large-language-model pipeline for discovering directed semantic bridges between ontologies developed from different perspectives. This repository extends the [OntoAligner](https://github.com/sciknoworg/OntoAligner) toolkit with candidate filtering, DeepOnto axiom verbalisation, configurable Stage-2 prompts, and experiments for three food-domain ontology pairs.

## Pipeline overview

1. **Stage 1 — candidate selection:** an LLM performs binary screening and retains concept pairs that may be semantically related.
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
├── ontoaligner/            # Python package and pipeline implementation
├── results/
│   ├── Stage_1/            # Candidate pairs
│   └── Stage_2/            # Directed semantic-bridge outputs
├── evaluation/             # Evaluation notebook, tables, and figures
├── requirements.txt
└── setup.py
```

## Requirements

The experimental configuration was developed for:

- Linux
- Python 3.10 or 3.11
- an NVIDIA CUDA GPU
- Java 11 for DeepOnto/OWLAPI
- a Hugging Face account with access to the selected gated model

The Slurm scripts request one GPU, eight CPUs, and 128 GB RAM. Adjust the `#SBATCH` settings for another cluster.

Create an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

Check Java and the package import:

```bash
java -version

python - <<'PY'
from ontoaligner.pipeline import OntoAlignerPipeline
from ontoaligner.encoder import ConceptCandidateLLMEncoder, ConceptLLMEncoder

print("CrossAligner imports succeeded.")
PY
```

## Hugging Face token

Do not write a Hugging Face token inside a Bash file. Export it in the shell before submitting a job:

```bash
export HF_TOKEN="your_hugging_face_token"
```

The job scripts copy this value to `HUGGINGFACE_HUB_TOKEN`. They stop with a clear error when `HF_TOKEN` is missing.

Consequently, ontology and result paths are relative to the clone itself. No username-specific path such as `/vast/<user>/...` is required.

Slurm must create its output file before the script starts, so create `logs/` and submit jobs from the repository root:

```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p logs
```

## Running Stage 1

Submit the script for the required ontology pair:

```bash
sbatch bash/Qwen3-32B-raw_s_f_DeepOnto_mesh_ons_stage1.bash
sbatch bash/Qwen3-32B-raw_s_f_DeepOnto_occo_mesh_stage1.bash
sbatch bash/Qwen3-32B-raw_s_f_DeepOnto_occo_ons_stage1.bash
```

Stage-1 JSON and CSV files are written below:

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

The generated filename records the Stage-1 model, Stage-2 model, ontology pair, prompt mode, and direction.

## Acknowledgement and license

CrossAligner builds on OntoAligner by Babaei Giglou et al. The original copyright notices are retained in inherited files.

The project is distributed under the Apache License 2.0. See [LICENSE](LICENSE), [CONTRIBUTING.md](CONTRIBUTING.md), and [CITATION.cff](CITATION.cff).
