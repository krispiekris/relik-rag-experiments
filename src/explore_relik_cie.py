#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import torch
from relik import Relik


DEFAULT_MODEL = "relik-ie/relik-cie-small"


def get_value(obj, name, default=None):
    """Read a field from either an object or a dictionary."""
    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(obj, name, default)


def span_to_dict(span):
    """Convert a ReLiK span into a JSON-serializable dictionary."""
    if span is None:
        return None

    return {
        "text": get_value(span, "text", ""),
        "label": get_value(span, "label", ""),
        "start": get_value(span, "start"),
        "end": get_value(span, "end"),
    }


def output_to_annotation(output):
    """Convert a ReLiK output object into JSON-serializable data."""

    spans = []

    for span in get_value(output, "spans", []) or []:
        converted_span = span_to_dict(span)

        if converted_span:
            spans.append(converted_span)

    # Some ReLiK versions use `triplets`, while others may use `triples`.
    raw_triplets = (
        get_value(output, "triplets", None)
        or get_value(output, "triples", None)
        or []
    )

    triplets = []

    for triplet in raw_triplets:
        subject = get_value(triplet, "subject")

        relation = (
            get_value(triplet, "label", None)
            or get_value(triplet, "relation", None)
            or ""
        )

        obj = get_value(triplet, "object")
        confidence = get_value(triplet, "confidence", None)

        triplets.append(
            {
                "subject": span_to_dict(subject),
                "relation": relation,
                "object": span_to_dict(obj),
                "confidence": confidence,
            }
        )

    return {
        "spans": spans,
        "triplets": triplets,
    }


def annotate_text(model, text):
    """Run ReLiK CIE on one text."""

    text = (text or "").strip()

    if not text:
        return {
            "spans": [],
            "triplets": [],
        }

    output = model(text)

    return output_to_annotation(output)


def save_json(data, path):
    """Safely write JSON through a temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temporary_path.replace(path)


def percentage(numerator, denominator):
    if denominator == 0:
        return 0.0

    return round(100 * numerator / denominator, 2)


def evaluate_annotations(dataset, annotations):
    """Calculate basic entity and relation coverage statistics."""

    example_by_id = {
        example["id"]: example
        for example in dataset["examples"]
    }

    question_span_counts = []
    question_triplet_counts = []

    document_span_counts = []
    document_triplet_counts = []

    gold_triplet_counts = []
    distractor_triplet_counts = []

    relation_frequencies = Counter()

    questions_with_spans = 0
    questions_with_triplets = 0

    documents_with_spans = 0
    documents_with_triplets = 0

    gold_documents_with_triplets = 0
    distractor_documents_with_triplets = 0

    total_gold_documents = 0
    total_distractor_documents = 0

    examples_with_question_gold_relation_overlap = 0
    inspection_examples = []

    for example_id, annotation in annotations["examples"].items():
        original = example_by_id[example_id]
        gold_titles = set(original["gold_titles"])

        question_annotation = annotation["question"]
        question_spans = question_annotation["spans"]
        question_triplets = question_annotation["triplets"]

        question_span_counts.append(len(question_spans))
        question_triplet_counts.append(len(question_triplets))

        if question_spans:
            questions_with_spans += 1

        if question_triplets:
            questions_with_triplets += 1

        question_relations = {
            str(triplet.get("relation", "")).strip().lower()
            for triplet in question_triplets
            if triplet.get("relation")
        }

        matching_gold_documents = []

        for document in annotation["documents"]:
            title = document["title"]
            document_spans = document["spans"]
            document_triplets = document["triplets"]

            document_span_counts.append(len(document_spans))
            document_triplet_counts.append(len(document_triplets))

            if document_spans:
                documents_with_spans += 1

            if document_triplets:
                documents_with_triplets += 1

            for triplet in document_triplets:
                relation = str(
                    triplet.get("relation", "")
                ).strip()

                if relation:
                    relation_frequencies[relation] += 1

            is_gold = title in gold_titles

            if is_gold:
                total_gold_documents += 1
                gold_triplet_counts.append(len(document_triplets))

                if document_triplets:
                    gold_documents_with_triplets += 1

            else:
                total_distractor_documents += 1
                distractor_triplet_counts.append(
                    len(document_triplets)
                )

                if document_triplets:
                    distractor_documents_with_triplets += 1

            document_relations = {
                str(triplet.get("relation", "")).strip().lower()
                for triplet in document_triplets
                if triplet.get("relation")
            }

            shared_relations = sorted(
                question_relations & document_relations
            )

            if is_gold and shared_relations:
                matching_gold_documents.append(
                    {
                        "title": title,
                        "shared_relations": shared_relations,
                    }
                )

        if matching_gold_documents:
            examples_with_question_gold_relation_overlap += 1

        if question_triplets or matching_gold_documents:
            inspection_examples.append(
                {
                    "id": example_id,
                    "question": original["question"],
                    "gold_titles": original["gold_titles"],
                    "question_spans": question_spans,
                    "question_triplets": question_triplets,
                    "gold_relation_matches": matching_gold_documents,
                }
            )

    number_of_examples = len(annotations["examples"])
    number_of_documents = len(document_triplet_counts)

    return {
        "examples": number_of_examples,
        "documents": number_of_documents,
        "question_coverage": {
            "with_entities": questions_with_spans,
            "with_entities_pct": percentage(
                questions_with_spans,
                number_of_examples,
            ),
            "with_relations": questions_with_triplets,
            "with_relations_pct": percentage(
                questions_with_triplets,
                number_of_examples,
            ),
            "average_entities": round(
                mean(question_span_counts),
                3,
            ) if question_span_counts else 0.0,
            "average_relations": round(
                mean(question_triplet_counts),
                3,
            ) if question_triplet_counts else 0.0,
        },
        "document_coverage": {
            "with_entities": documents_with_spans,
            "with_entities_pct": percentage(
                documents_with_spans,
                number_of_documents,
            ),
            "with_relations": documents_with_triplets,
            "with_relations_pct": percentage(
                documents_with_triplets,
                number_of_documents,
            ),
            "average_entities": round(
                mean(document_span_counts),
                3,
            ) if document_span_counts else 0.0,
            "average_relations": round(
                mean(document_triplet_counts),
                3,
            ) if document_triplet_counts else 0.0,
        },
        "gold_vs_distractor": {
            "gold_documents": total_gold_documents,
            "gold_with_relations": gold_documents_with_triplets,
            "gold_with_relations_pct": percentage(
                gold_documents_with_triplets,
                total_gold_documents,
            ),
            "gold_average_relations": round(
                mean(gold_triplet_counts),
                3,
            ) if gold_triplet_counts else 0.0,
            "distractor_documents": total_distractor_documents,
            "distractor_with_relations": (
                distractor_documents_with_triplets
            ),
            "distractor_with_relations_pct": percentage(
                distractor_documents_with_triplets,
                total_distractor_documents,
            ),
            "distractor_average_relations": round(
                mean(distractor_triplet_counts),
                3,
            ) if distractor_triplet_counts else 0.0,
        },
        "question_gold_relation_overlap": {
            "examples": examples_with_question_gold_relation_overlap,
            "percentage": percentage(
                examples_with_question_gold_relation_overlap,
                number_of_examples,
            ),
        },
        "most_common_document_relations": [
            {
                "relation": relation,
                "count": count,
            }
            for relation, count
            in relation_frequencies.most_common(20)
        ],
        "inspection_examples": inspection_examples[:25],
    }


def print_summary(summary):
    question = summary["question_coverage"]
    document = summary["document_coverage"]
    comparison = summary["gold_vs_distractor"]
    overlap = summary["question_gold_relation_overlap"]

    print("\n" + "=" * 60)
    print("RELIK CIE COVERAGE SUMMARY")
    print("=" * 60)

    print(f"Examples:  {summary['examples']}")
    print(f"Documents: {summary['documents']}")

    print("\nQuestions")
    print(
        f"  With entities:  "
        f"{question['with_entities_pct']:.2f}%"
    )
    print(
        f"  With relations: "
        f"{question['with_relations_pct']:.2f}%"
    )
    print(
        f"  Average entities/question:  "
        f"{question['average_entities']:.3f}"
    )
    print(
        f"  Average relations/question: "
        f"{question['average_relations']:.3f}"
    )

    print("\nDocuments")
    print(
        f"  With entities:  "
        f"{document['with_entities_pct']:.2f}%"
    )
    print(
        f"  With relations: "
        f"{document['with_relations_pct']:.2f}%"
    )
    print(
        f"  Average entities/document:  "
        f"{document['average_entities']:.3f}"
    )
    print(
        f"  Average relations/document: "
        f"{document['average_relations']:.3f}"
    )

    print("\nGold versus distractor documents")
    print(
        f"  Gold with relations: "
        f"{comparison['gold_with_relations_pct']:.2f}%"
    )
    print(
        f"  Distractors with relations: "
        f"{comparison['distractor_with_relations_pct']:.2f}%"
    )
    print(
        f"  Average gold relations: "
        f"{comparison['gold_average_relations']:.3f}"
    )
    print(
        f"  Average distractor relations: "
        f"{comparison['distractor_average_relations']:.3f}"
    )

    print("\nQuestion-to-gold relation-label overlap")
    print(
        f"  Examples with overlap: "
        f"{overlap['examples']}/{summary['examples']} "
        f"({overlap['percentage']:.2f}%)"
    )

    print("\nMost common document relations")

    for item in summary["most_common_document_relations"][:10]:
        print(f"  {item['relation']}: {item['count']}")

    print("=" * 60)


def configure_cuda():
    """Require CUDA and show the selected GPU."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this environment. "
            "Check that the job has a GPU and that the CUDA-enabled "
            "PyTorch build is installed."
        )

    device = "cuda"

    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)

    print("=" * 60)
    print("CUDA CONFIGURATION")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"Selected device: {device}:{gpu_index}")
    print(f"GPU: {gpu_name}")
    print("=" * 60)

    return device


def main():
    parser = argparse.ArgumentParser(
        description="Explore ReLiK CIE coverage on HotpotQA using CUDA."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/hotpotqa_validation_100.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/relik_cie_annotations_100.json"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "results/relik_cie_summary_100.json"
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally process only the first N examples.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    device = configure_cuda()

    with args.input.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    examples = dataset["examples"]

    if args.limit is not None:
        examples = examples[:args.limit]

    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as file:
            annotations = json.load(file)

        print(
            f"Resuming with "
            f"{len(annotations.get('examples', {}))} "
            f"cached examples."
        )

    else:
        annotations = {
            "dataset": dataset.get("dataset"),
            "config": dataset.get("config"),
            "split": dataset.get("split"),
            "model": args.model,
            "device": device,
            "examples": {},
        }

    print(f"\nLoading model: {args.model}")
    print(f"Requested device: {device}")

    model = Relik.from_pretrained(
        args.model,
        device=device,
    )

    print("Model loaded.")

    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)

    print(f"GPU memory allocated: {allocated_gb:.2f} GB")
    print(f"GPU memory reserved:  {reserved_gb:.2f} GB")

    for example_number, example in enumerate(examples, start=1):
        example_id = example["id"]

        if example_id in annotations["examples"]:
            print(
                f"[{example_number}/{len(examples)}] "
                f"Cached: {example_id}"
            )
            continue

        print(
            f"[{example_number}/{len(examples)}] "
            f"Processing: {example_id}"
        )

        question_annotation = annotate_text(
            model,
            example["question"],
        )

        document_annotations = []

        for document_number, document in enumerate(
            example["documents"],
            start=1,
        ):
            print(
                f"    Document "
                f"{document_number}/{len(example['documents'])}: "
                f"{document['title']}"
            )

            # Include the title because it often contains
            # an important entity.
            model_input = (
                f"{document['title']}. {document['text']}"
            )

            result = annotate_text(
                model,
                model_input,
            )

            document_annotations.append(
                {
                    "title": document["title"],
                    "is_gold": (
                        document["title"]
                        in example["gold_titles"]
                    ),
                    "spans": result["spans"],
                    "triplets": result["triplets"],
                }
            )

        annotations["examples"][example_id] = {
            "question": question_annotation,
            "documents": document_annotations,
        }

        if example_number % args.save_every == 0:
            save_json(
                annotations,
                args.output,
            )

            print(f"    Checkpoint saved to {args.output}")

    save_json(
        annotations,
        args.output,
    )

    summary = evaluate_annotations(
        dataset,
        annotations,
    )

    save_json(
        summary,
        args.summary,
    )

    print_summary(summary)

    peak_memory_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
    )

    print(f"\nPeak GPU memory: {peak_memory_gb:.2f} GB")
    print(f"Annotations saved to: {args.output}")
    print(f"Summary saved to:     {args.summary}")


if __name__ == "__main__":
    main()