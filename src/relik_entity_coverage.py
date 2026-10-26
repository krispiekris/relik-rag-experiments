import json
from pathlib import Path

from datasets import load_dataset


CACHE_PATH = Path("data/relik_annotations_100.json")
NUM_EXAMPLES = 100


def normalize_entity(entity: str) -> str:
    """Normalize entity labels before comparison."""
    return " ".join(entity.casefold().split())


with CACHE_PATH.open("r", encoding="utf-8") as file:
    annotations = json.load(file)


dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split=f"validation[:{NUM_EXAMPLES}]",
)

dataset_by_id = {
    example["id"]: example
    for example in dataset
}


questions_with_any_overlap = 0
questions_with_gold_overlap = 0
questions_with_distractor_overlap = 0

questions_with_gold_only_overlap = 0
questions_with_distractor_only_overlap = 0
questions_with_both_overlap = 0

total_shared_any = 0
total_shared_gold = 0
total_shared_distractor = 0

query_entities_total = 0
query_entities_found_anywhere = 0
query_entities_found_in_gold = 0
query_entities_found_in_distractors = 0

gold_documents_total = 0
gold_documents_with_query_overlap = 0

distractor_documents_total = 0
distractor_documents_with_query_overlap = 0

questions_covering_all_gold_documents = 0
questions_covering_one_gold_document = 0

missing_example_ids = []
missing_document_annotations = 0


for example_id, annotation in annotations.items():
    if example_id not in dataset_by_id:
        missing_example_ids.append(example_id)
        continue

    dataset_example = dataset_by_id[example_id]

    gold_titles = set(
        dataset_example["supporting_facts"]["title"]
    )

    query_entities = {
        normalize_entity(entity)
        for entity in annotation.get("query_entities", [])
    }

    document_entities = annotation.get(
        "document_entities",
        {},
    )

    query_entities_total += len(query_entities)

    all_document_entities = set()
    gold_document_entities = set()
    distractor_document_entities = set()

    gold_documents_covered = 0

    for title in dataset_example["context"]["title"]:
        if title not in document_entities:
            missing_document_annotations += 1
            entities = set()
        else:
            entities = {
                normalize_entity(entity)
                for entity in document_entities[title]
            }

        all_document_entities.update(entities)

        shared_with_query = query_entities & entities

        if title in gold_titles:
            gold_documents_total += 1
            gold_document_entities.update(entities)

            if shared_with_query:
                gold_documents_with_query_overlap += 1
                gold_documents_covered += 1
        else:
            distractor_documents_total += 1
            distractor_document_entities.update(entities)

            if shared_with_query:
                distractor_documents_with_query_overlap += 1

    shared_any = query_entities & all_document_entities
    shared_gold = query_entities & gold_document_entities
    shared_distractor = (
        query_entities & distractor_document_entities
    )

    if shared_any:
        questions_with_any_overlap += 1

    if shared_gold:
        questions_with_gold_overlap += 1

    if shared_distractor:
        questions_with_distractor_overlap += 1

    if shared_gold and not shared_distractor:
        questions_with_gold_only_overlap += 1
    elif shared_distractor and not shared_gold:
        questions_with_distractor_only_overlap += 1
    elif shared_gold and shared_distractor:
        questions_with_both_overlap += 1

    total_shared_any += len(shared_any)
    total_shared_gold += len(shared_gold)
    total_shared_distractor += len(shared_distractor)

    query_entities_found_anywhere += len(shared_any)
    query_entities_found_in_gold += len(shared_gold)
    query_entities_found_in_distractors += len(
        shared_distractor
    )

    if gold_documents_covered == len(gold_titles):
        questions_covering_all_gold_documents += 1
    elif gold_documents_covered == 1:
        questions_covering_one_gold_document += 1


evaluated_questions = (
    len(annotations) - len(missing_example_ids)
)


def percentage(value: int, total: int) -> float:
    if total == 0:
        return 0.0

    return 100 * value / total


print("=" * 70)
print("RELIK QUESTION-DOCUMENT ENTITY OVERLAP — HOTPOTQA")
print("=" * 70)

print(f"\nEvaluated questions: {evaluated_questions}")

print("\nQuestion-level overlap:")

print(
    "Any candidate document: "
    f"{questions_with_any_overlap}/{evaluated_questions} "
    f"({percentage(questions_with_any_overlap, evaluated_questions):.1f}%)"
)

print(
    "Gold documents: "
    f"{questions_with_gold_overlap}/{evaluated_questions} "
    f"({percentage(questions_with_gold_overlap, evaluated_questions):.1f}%)"
)

print(
    "Distractor documents: "
    f"{questions_with_distractor_overlap}/{evaluated_questions} "
    f"({percentage(questions_with_distractor_overlap, evaluated_questions):.1f}%)"
)

print("\nOverlap type:")

print(
    "Gold only: "
    f"{questions_with_gold_only_overlap}"
)

print(
    "Distractor only: "
    f"{questions_with_distractor_only_overlap}"
)

print(
    "Both gold and distractors: "
    f"{questions_with_both_overlap}"
)

print("\nQuery-entity coverage:")

print(
    "Query entities found in any document: "
    f"{query_entities_found_anywhere}/{query_entities_total} "
    f"({percentage(query_entities_found_anywhere, query_entities_total):.1f}%)"
)

print(
    "Query entities found in gold documents: "
    f"{query_entities_found_in_gold}/{query_entities_total} "
    f"({percentage(query_entities_found_in_gold, query_entities_total):.1f}%)"
)

print(
    "Query entities found in distractors: "
    f"{query_entities_found_in_distractors}/{query_entities_total} "
    f"({percentage(query_entities_found_in_distractors, query_entities_total):.1f}%)"
)

print("\nDocument-level overlap:")

print(
    "Gold documents sharing a query entity: "
    f"{gold_documents_with_query_overlap}/{gold_documents_total} "
    f"({percentage(gold_documents_with_query_overlap, gold_documents_total):.1f}%)"
)

print(
    "Distractor documents sharing a query entity: "
    f"{distractor_documents_with_query_overlap}/"
    f"{distractor_documents_total} "
    f"({percentage(distractor_documents_with_query_overlap, distractor_documents_total):.1f}%)"
)

print("\nGold-document coverage per question:")

print(
    "Questions with all gold documents covered: "
    f"{questions_covering_all_gold_documents}"
)

print(
    "Questions with exactly one gold document covered: "
    f"{questions_covering_one_gold_document}"
)

print("\nAverage shared entities per question:")

print(
    "Any document: "
    f"{total_shared_any / evaluated_questions:.2f}"
)

print(
    "Gold documents: "
    f"{total_shared_gold / evaluated_questions:.2f}"
)

print(
    "Distractor documents: "
    f"{total_shared_distractor / evaluated_questions:.2f}"
)

print("\nCache consistency:")

print(
    "Missing example IDs: "
    f"{len(missing_example_ids)}"
)

print(
    "Missing document annotations: "
    f"{missing_document_annotations}"
)