from collections import defaultdict

from datasets import load_dataset
from relik import Relik
from relik.inference.data.objects import RelikOutput


# ---------------------------------------------------------
# 1. Load one HotpotQA example
# ---------------------------------------------------------

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split="validation[:1]",
)

example = dataset[0]

question = example["question"]
gold_titles = set(example["supporting_facts"]["title"])

titles = example["context"]["title"]
sentence_groups = example["context"]["sentences"]


# ---------------------------------------------------------
# 2. Load Relik
# ---------------------------------------------------------

print("Loading Relik...")

relik = Relik.from_pretrained(
    "sapienzanlp/relik-entity-linking-small"
)

print("Relik loaded.")


# ---------------------------------------------------------
# 3. Extract entities from every context document
# ---------------------------------------------------------

entity_to_documents = defaultdict(set)
document_entities = {}

for title, sentences in zip(titles, sentence_groups):
    text = " ".join(sentences)

    result: RelikOutput = relik(text)

    entities = {
        span.label
        for span in result.spans
    }

    document_entities[title] = entities

    for entity in entities:
        entity_to_documents[entity].add(title)


# ---------------------------------------------------------
# 4. Extract entities from the question
# ---------------------------------------------------------

query_result: RelikOutput = relik(question)

query_entities = {
    span.label
    for span in query_result.spans
}


# ---------------------------------------------------------
# 5. Display query entities
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("QUESTION")
print("=" * 80)
print(question)

print("\nQUERY ENTITIES")
print("=" * 80)

for span in query_result.spans:
    print(
        f"{span.text!r} -> {span.label!r}"
    )


# ---------------------------------------------------------
# 6. Display entities found in each document
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("DOCUMENT ENTITIES")
print("=" * 80)

for title in titles:
    marker = "✓ GOLD" if title in gold_titles else ""

    print(f"\n{title} {marker}")

    entities = document_entities[title]

    if entities:
        for entity in sorted(entities):
            print(f"  - {entity}")
    else:
        print("  - No entities found")


# ---------------------------------------------------------
# 7. Entity-based retrieval
# ---------------------------------------------------------

matched_documents = defaultdict(set)

for entity in query_entities:
    for title in entity_to_documents.get(entity, set()):
        matched_documents[title].add(entity)


# ---------------------------------------------------------
# 8. Display entity matches
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("ENTITY-MATCHED DOCUMENTS")
print("=" * 80)

if not matched_documents:
    print("No documents matched query entities.")

else:
    ranked_matches = sorted(
        matched_documents.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    for rank, (title, matched_entities) in enumerate(
        ranked_matches,
        start=1,
    ):
        marker = "✓ GOLD" if title in gold_titles else ""

        print(
            f"{rank}. {title} {marker}"
        )

        print(
            f"   Matched entities: "
            f"{', '.join(sorted(matched_entities))}"
        )