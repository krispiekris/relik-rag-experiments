import json
from pathlib import Path


CACHE_PATH = Path("data/relik_annotations_100.json")


with CACHE_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    annotations = json.load(file)


total_questions = len(annotations)

questions_with_entities = 0
total_query_entity_mentions = 0
unique_query_entities = set()

total_documents = 0
documents_with_entities = 0
total_document_entity_mentions = 0
unique_document_entities = set()


for example in annotations.values():
    query_entities = example.get(
        "query_entities",
        [],
    )

    if query_entities:
        questions_with_entities += 1

    total_query_entity_mentions += len(query_entities)
    unique_query_entities.update(query_entities)

    document_entities = example.get(
        "document_entities",
        {},
    )

    for entities in document_entities.values():
        total_documents += 1

        if entities:
            documents_with_entities += 1

        total_document_entity_mentions += len(entities)
        unique_document_entities.update(entities)


question_coverage = (
    100 * questions_with_entities / total_questions
)

document_coverage = (
    100 * documents_with_entities / total_documents
)


print("=" * 60)
print("RELIK ENTITY COVERAGE — HOTPOTQA")
print("=" * 60)

print(
    f"\nQuestions with at least one entity: "
    f"{questions_with_entities} / {total_questions} "
    f"({question_coverage:.1f}%)"
)

print(
    f"Documents with at least one entity: "
    f"{documents_with_entities} / {total_documents} "
    f"({document_coverage:.1f}%)"
)

print("\nAdditional statistics:")

print(
    f"Query entity mentions: "
    f"{total_query_entity_mentions}"
)

print(
    f"Unique query entities: "
    f"{len(unique_query_entities)}"
)

print(
    f"Document entity mentions: "
    f"{total_document_entity_mentions}"
)

print(
    f"Unique document entities: "
    f"{len(unique_document_entities)}"
)