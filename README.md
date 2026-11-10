# Relik RAG Experiments

Small experiments comparing semantic and entity-aware retrieval on the
[HotpotQA](https://hotpotqa.github.io/) dataset. The project uses BGE
embeddings with LlamaIndex and Relik entity linking to measure document
Recall@K.

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- Optional: [Ollama](https://ollama.com/) for `test_ollama.py`

Install the project and its dependencies:

```bash
uv sync
```

## Usage

The original exploratory scripts download the required dataset and models on
first run:

```bash
uv run python src/inspect_hotpotqa.py
uv run python src/baseline_retrieval.py
uv run python src/relik_retrieval.py
uv run python src/hybrid_retrieval.py
uv run python src/evaluate_retrieval.py
```

`baseline_retrieval.py` runs semantic retrieval, while
`relik_retrieval.py` explores entity matches. `hybrid_retrieval.py` combines
both signals, and `evaluate_retrieval.py` evaluates multiple examples and
writes JSON results under `results/`.

## Slurm-friendly three-stage workflow

For larger runs, prepare the data and run the two model stages as separate
jobs. Each model stage writes its result after every example, so a stopped job
can be rerun and will skip completed examples:

```bash
# CPU job: choose hotpotqa or 2wiki and the number of examples
uv run python src/prepare_hotpotqa.py --dataset hotpotqa --num-results 1000
# 2WikiMultiHopQA uses a deterministic shuffled sample
uv run python src/prepare_hotpotqa.py --dataset 2wiki --num-results 1000

# GPU job 1: semantic embeddings (use the same dataset and count)
uv run python src/compute_embeddings.py \
  --dataset-name hotpotqa --num-results 1000 --device cuda

# GPU job 2: Relik annotations and hybrid evaluation
uv run python src/compute_relik.py \
  --dataset-name hotpotqa --num-results 1000 --device cuda
```

The stages communicate through
`data/hotpotqa_validation_1000.json`,
`data/hotpotqa_embeddings_1000.json`, and
`data/relik_annotations_1000.json`. Use `--dataset`, `--output`, and
`--annotations` to choose different locations. For 2Wiki, replace
`hotpotqa` with `2wiki`; the default files are named using the dataset and
number of results. The final hybrid output is written under `results/`.

To verify an Ollama installation separately:

```bash
ollama run openbmb/minicpm5
uv run python src/test_ollama.py
```

## Project layout

```text
src/       Experiment scripts and the Python package
data/      Local annotation caches (not committed)
results/   Generated evaluation results (not committed)
```

This repository is intended for experimentation; the scripts currently use
configuration constants near the top of each file.