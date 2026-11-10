"""Compute and persist semantic scores for prepared examples."""

import argparse
import json
import math
from pathlib import Path

from llama_index.embeddings.huggingface import HuggingFaceEmbedding


DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", choices=["hotpotqa", "2wiki"], default="hotpotqa")
    parser.add_argument("--num-results", type=int, default=1000)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def cosine_scores(
    query_embedding: list[float],
    document_embeddings: list[list[float]],
) -> list[float]:
    query_norm = math.sqrt(sum(value * value for value in query_embedding))
    scores = []
    for document_embedding in document_embeddings:
        document_norm = math.sqrt(
            sum(value * value for value in document_embedding)
        )
        denominator = query_norm * document_norm
        scores.append(
            sum(
                query_value * document_value
                for query_value, document_value in zip(
                    query_embedding,
                    document_embedding,
                )
            )
            / denominator
            if denominator
            else 0.0
        )
    return scores


def main() -> None:
    args = parse_args()
    if args.num_results < 1:
        raise ValueError("--num-results must be positive")
    if args.dataset is None:
        split = "validation" if args.dataset_name == "hotpotqa" else "dev"
        args.dataset = Path(f"data/{args.dataset_name}_{split}_{args.num_results}.json")
    if args.output is None:
        args.output = Path(f"data/{args.dataset_name}_embeddings_{args.num_results}.json")
    with args.dataset.open("r", encoding="utf-8") as file:
        prepared = json.load(file)

    examples = prepared["examples"]
    output_by_id = {}
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as file:
            output_by_id = json.load(file).get("results", {})

    pending = [example for example in examples if example["id"] not in output_by_id]
    print(f"Embedding {len(pending)} examples ({len(output_by_id)} cached).")
    if pending:
        model_kwargs = {}
        if args.device:
            model_kwargs["device"] = args.device
        embed_model = HuggingFaceEmbedding(
            model_name=args.model,
            **model_kwargs,
        )

        for number, example in enumerate(pending, start=1):
            documents = example["documents"]
            texts = [document["text"] for document in documents]
            embeddings = embed_model.get_text_embedding_batch(
                [example["question"], *texts]
            )
            scores = cosine_scores(embeddings[0], embeddings[1:])
            ranking = [
                document["title"]
                for _, document in sorted(
                    zip(scores, documents),
                    key=lambda item: item[0],
                    reverse=True,
                )
            ]
            output_by_id[example["id"]] = {
                "question": example["question"],
                "semantic_scores": {
                    document["title"]: score
                    for document, score in zip(documents, scores)
                },
                "semantic_ranking": ranking,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as file:
                json.dump(
                    {
                        "embedding_model": args.model,
                        "dataset": str(args.dataset),
                        "results": output_by_id,
                    },
                    file,
                    indent=2,
                )
            print(f"[{number}/{len(pending)}] {example['id']}")

    print(f"Saved semantic results to {args.output}")


if __name__ == "__main__":
    main()
