"""Download and persist HotpotQA or 2WikiMultiHopQA examples."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


DATASETS = {
    "hotpotqa": {
        "path": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "split": "validation",
    },
    "2wiki": {
        "path": "xanhho/2WikiMultihopQA",
        "config": None,
        "split": "dev",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="hotpotqa",
        help="Dataset to download.",
    )
    parser.add_argument(
        "--num-results",
        "--num-examples",
        dest="num_results",
        type=int,
        default=1000,
        help="Number of examples to prepare.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_results < 1:
        raise ValueError("--num-results must be positive")

    metadata = DATASETS[args.dataset]
    dataset = load_dataset(metadata["path"], metadata["config"], split=metadata["split"])
    if args.dataset == "2wiki":
        dataset = dataset.shuffle(seed=args.seed)
    dataset = dataset.select(range(min(args.num_results, len(dataset))))

    examples = []
    for example in dataset:
        if args.dataset == "hotpotqa":
            example_id = example["id"]
            documents = zip(
                example["context"]["title"],
                example["context"]["sentences"],
            )
        else:
            example_id = example["_id"]
            documents = zip(
                example["context"]["title"],
                example["context"]["content"],
            )
        examples.append(
            {
                "id": example_id,
                "question": example["question"],
                "gold_titles": sorted(
                    set(example["supporting_facts"]["title"])
                ),
                "documents": [
                    {
                        "title": title,
                        "text": " ".join(sentences),
                    }
                    for title, sentences in documents
                ],
            }
        )

    if args.output is None:
        args.output = Path(
            f"data/{args.dataset}_{metadata['split']}_{len(examples)}.json"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset": args.dataset,
                "config": metadata["config"],
                "split": metadata["split"],
                "seed": args.seed if args.dataset == "2wiki" else None,
                "num_examples": len(examples),
                "examples": examples,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
