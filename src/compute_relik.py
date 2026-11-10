"""Annotate prepared examples with Relik and produce hybrid results."""

import argparse
import json
from pathlib import Path

from relik import Relik
from relik.inference.data.objects import RelikOutput


DEFAULT_MODEL = "sapienzanlp/relik-entity-linking-large"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", choices=["hotpotqa", "2wiki"], default="hotpotqa")
    parser.add_argument("--num-results", type=int, default=1000)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, nargs="+", default=[2, 5])
    return parser.parse_args()


def extract_entities(model: Relik, text: str) -> set[str]:
    result: RelikOutput = model(text)
    return {span.label for span in result.spans}


def recall_at_k(ranking: list[str], gold_titles: set[str], k: int) -> float:
    return len(set(ranking[:k]) & gold_titles) / len(gold_titles) if gold_titles else 0.0


def main() -> None:
    args = parse_args()
    if args.num_results < 1:
        raise ValueError("--num-results must be positive")
    if args.dataset is None:
        split = "validation" if args.dataset_name == "hotpotqa" else "dev"
        args.dataset = Path(f"data/{args.dataset_name}_{split}_{args.num_results}.json")
    if args.embeddings is None:
        args.embeddings = Path(
            f"data/{args.dataset_name}_embeddings_{args.num_results}.json"
        )
    if args.annotations is None:
        args.annotations = Path(
            f"data/{args.dataset_name}_relik_annotations_{args.num_results}.json"
        )
    if args.output is None:
        args.output = Path(
            f"results/{args.dataset_name}_retrieval_{args.num_results}.json"
        )
    with args.dataset.open("r", encoding="utf-8") as file:
        prepared = json.load(file)
    with args.embeddings.open("r", encoding="utf-8") as file:
        embedding_file = json.load(file)
    embeddings = embedding_file["results"]

    annotations = {}
    if args.annotations.exists():
        with args.annotations.open("r", encoding="utf-8") as file:
            annotations = json.load(file)

    pending = [
        example
        for example in prepared["examples"]
        if example["id"] not in annotations
    ]
    print(f"Annotating {len(pending)} examples ({len(annotations)} cached).")
    if pending:
        model_kwargs = {}
        if args.device:
            model_kwargs["device"] = args.device
        relik = Relik.from_pretrained(args.model, **model_kwargs)
        for number, example in enumerate(pending, start=1):
            annotations[example["id"]] = {
                "question": example["question"],
                "query_entities": sorted(
                    extract_entities(relik, example["question"])
                ),
                "document_entities": {
                    document["title"]: sorted(
                        extract_entities(relik, document["text"])
                    )
                    for document in example["documents"]
                },
            }
            args.annotations.parent.mkdir(parents=True, exist_ok=True)
            with args.annotations.open("w", encoding="utf-8") as file:
                json.dump(annotations, file, indent=2, ensure_ascii=False)
            print(f"[{number}/{len(pending)}] {example['id']}")

    results = []
    recalls = {k: [] for k in args.top_k}
    for example in prepared["examples"]:
        example_id = example["id"]
        semantic = embeddings[example_id]
        annotation = annotations[example_id]
        query_entities = set(annotation["query_entities"])
        hybrid_details = []
        document_entities = annotation["document_entities"]
        for title, semantic_score in semantic["semantic_scores"].items():
            matched = query_entities & set(document_entities.get(title, []))
            hybrid_details.append(
                {
                    "title": title,
                    "semantic_score": semantic_score,
                    "entity_match": int(bool(matched)),
                    "matched_entities": sorted(matched),
                    "hybrid_score": semantic_score + args.alpha * bool(matched),
                }
            )
        hybrid_details.sort(key=lambda item: item["hybrid_score"], reverse=True)
        hybrid_ranking = [item["title"] for item in hybrid_details]
        metrics = {"bge": {}, "hybrid": {}}
        gold_titles = set(example["gold_titles"])
        for k in args.top_k:
            bge = recall_at_k(semantic["semantic_ranking"], gold_titles, k)
            hybrid = recall_at_k(hybrid_ranking, gold_titles, k)
            recalls[k].append((bge, hybrid))
            metrics["bge"][f"recall@{k}"] = bge
            metrics["hybrid"][f"recall@{k}"] = hybrid
        results.append(
            {
                "id": example_id,
                "question": example["question"],
                "gold_titles": sorted(gold_titles),
                "query_entities": sorted(query_entities),
                "metrics": metrics,
                "semantic_ranking": semantic["semantic_ranking"],
                "hybrid_ranking": hybrid_ranking,
                "semantic_scores": semantic["semantic_scores"],
                "hybrid_details": hybrid_details,
            }
        )

    summary = {}
    comparison = {}
    for k, values in recalls.items():
        bge_values = [value[0] for value in values]
        hybrid_values = [value[1] for value in values]
        mean_bge = sum(bge_values) / len(values) if values else 0.0
        mean_hybrid = sum(hybrid_values) / len(values) if values else 0.0
        summary[f"recall@{k}"] = {
            "bge": mean_bge,
            "hybrid": mean_hybrid,
            "delta": mean_hybrid - mean_bge,
        }
        comparison[f"recall@{k}"] = {
            "hybrid_wins": sum(h > b for b, h in values),
            "hybrid_losses": sum(h < b for b, h in values),
            "ties": sum(h == b for b, h in values),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "configuration": {
                    "num_examples": len(results),
                    "embedding_model": embedding_file.get("embedding_model"),
                    "relik_model": args.model,
                    "alpha": args.alpha,
                    "top_k_values": args.top_k,
                },
                "summary": summary,
                "comparison": comparison,
                "results": results,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved annotations to {args.annotations}")
    print(f"Saved hybrid results to {args.output}")


if __name__ == "__main__":
    main()
