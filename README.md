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

The scripts download the required dataset and models on first run:

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