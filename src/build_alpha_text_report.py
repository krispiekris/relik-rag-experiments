import json
from pathlib import Path

CASES_PATH = Path("results/alpha_0014_cases.json")
DATASET_PATH = Path("data/hotpotqa_validation_1000.json")
OUTPUT_PATH = Path("results/alpha_0014_text_report.json")
TEXT_PATH = Path("results/alpha_0014_text_report.txt")


with CASES_PATH.open(encoding="utf-8") as f:
    cases_data = json.load(f)

with DATASET_PATH.open(encoding="utf-8") as f:
    dataset = json.load(f)


examples_by_id = {
    example["id"]: example
    for example in dataset["examples"]
}


def normalize_title(title):
    return title.strip().lower()


def annotate_case(case):
    source_example = examples_by_id[case["id"]]

    documents_by_title = {
        normalize_title(document["title"]): document
        for document in source_example["documents"]
    }

    enriched_candidates = []

    for candidate in case["hybrid_top_10_comparison"]:
        document = documents_by_title.get(
            normalize_title(candidate["title"])
        )

        enriched_candidates.append({
            **candidate,
            "text": document["text"] if document else None,
            "text_found": document is not None,
        })

    return {
        "id": case["id"],
        "outcome": case["outcome"],
        "question": case["question"],
        "query_entities": case["query_entities"],
        "gold_titles": case["gold_titles"],
        "semantic_recall_at_5": case["semantic_recall_at_5"],
        "hybrid_recall_at_5": case["hybrid_recall_at_5"],
        "candidates": enriched_candidates,
    }


cases = (
    cases_data["wins"]
    + cases_data["losses"]
)

enriched_cases = [
    annotate_case(case)
    for case in cases
]


output = {
    "configuration": cases_data["configuration"],
    "summary": cases_data["summary"],
    "cases": enriched_cases,
}

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)


with TEXT_PATH.open("w", encoding="utf-8") as f:
    for case in enriched_cases:
        f.write("=" * 100 + "\n")
        f.write(f"{case['outcome'].upper()} | {case['id']}\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"QUESTION\n{case['question']}\n\n")
        f.write(
            "QUERY ENTITIES\n"
            + ", ".join(case["query_entities"])
            + "\n\n"
        )
        f.write(
            "GOLD TITLES\n"
            + ", ".join(case["gold_titles"])
            + "\n\n"
        )
        f.write(
            "RECALL@5\n"
            f"{case['semantic_recall_at_5']}"
            f" -> {case['hybrid_recall_at_5']}\n\n"
        )

        for candidate in case["candidates"]:
            flags = []

            if candidate["is_gold"]:
                flags.append("GOLD")

            if candidate["rank"] <= 5:
                flags.append("HYBRID TOP-5")

            if (
                candidate["semantic_rank"] is not None
                and candidate["semantic_rank"] <= 5
            ):
                flags.append("SEMANTIC TOP-5")

            f.write("-" * 100 + "\n")
            f.write(
                f"HYBRID RANK {candidate['rank']} | "
                f"SEMANTIC RANK {candidate['semantic_rank']} | "
                f"MOVE {candidate['rank_change']:+d}\n"
            )
            f.write(f"TITLE: {candidate['title']}\n")
            f.write(
                "FLAGS: "
                + (", ".join(flags) if flags else "none")
                + "\n"
            )
            f.write(
                f"SEMANTIC SCORE: "
                f"{candidate['semantic_score']:.6f}\n"
            )
            f.write(
                f"ENTITY MATCH: "
                f"{candidate['entity_match']}\n"
            )
            f.write(
                "MATCHED ENTITIES: "
                + (
                    ", ".join(candidate["matched_entities"])
                    if candidate["matched_entities"]
                    else "none"
                )
                + "\n"
            )
            f.write(
                f"HYBRID SCORE: "
                f"{candidate['hybrid_score']:.6f}\n\n"
            )
            f.write("TEXT:\n")
            f.write(
                candidate["text"]
                if candidate["text"]
                else "[TEXT NOT FOUND]"
            )
            f.write("\n\n")


missing = [
    (case["id"], candidate["title"])
    for case in enriched_cases
    for candidate in case["candidates"]
    if not candidate["text_found"]
]

print(f"Cases written: {len(enriched_cases)}")
print(f"Missing candidate texts: {len(missing)}")
print(f"JSON report: {OUTPUT_PATH}")
print(f"Readable report: {TEXT_PATH}")

if missing:
    print("\nMissing titles:")
    for example_id, title in missing:
        print(example_id, "|", title)