from datasets import load_dataset
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from relik import Relik
from relik.inference.data.objects import RelikOutput


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TOP_K = 5
ALPHA = 0.10


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
# 2. Convert HotpotQA contexts to LlamaIndex nodes
# ---------------------------------------------------------

nodes = []

for title, sentences in zip(titles, sentence_groups):
    text = " ".join(sentences)

    node = TextNode(
        text=text,
        metadata={
            "title": title,
        },
    )

    nodes.append(node)


# ---------------------------------------------------------
# 3. Semantic retrieval with BGE
# ---------------------------------------------------------

print("Loading BGE...")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5"
)

index = VectorStoreIndex(nodes)

# Retrieve all candidates because we need a semantic
# score for every document before hybrid reranking.
semantic_retriever = index.as_retriever(
    similarity_top_k=len(nodes),
)

semantic_results = semantic_retriever.retrieve(question)

semantic_scores = {
    result.node.metadata["title"]: result.score
    for result in semantic_results
}

print("BGE retrieval complete.")


# ---------------------------------------------------------
# 4. Load Relik
# ---------------------------------------------------------

print("\nLoading Relik...")

relik = Relik.from_pretrained(
    "sapienzanlp/relik-entity-linking-small"
)

print("Relik loaded.")


# ---------------------------------------------------------
# 5. Extract entities from the query
# ---------------------------------------------------------

query_result: RelikOutput = relik(question)

query_entities = {
    span.label
    for span in query_result.spans
}


# ---------------------------------------------------------
# 6. Extract entities from every document
# ---------------------------------------------------------

document_entities = {}

for title, sentences in zip(titles, sentence_groups):
    text = " ".join(sentences)

    result: RelikOutput = relik(text)

    entities = {
        span.label
        for span in result.spans
    }

    document_entities[title] = entities


# ---------------------------------------------------------
# 7. Calculate hybrid scores
#
# HybridScore =
#     SemanticScore + ALPHA * EntityMatch
#
# EntityMatch = 1 if at least one linked query entity
# occurs in the document, otherwise 0.
# ---------------------------------------------------------

hybrid_results = []

for title in titles:
    semantic_score = semantic_scores[title]

    matched_entities = (
        query_entities
        & document_entities[title]
    )

    entity_match = int(
        len(matched_entities) > 0
    )

    hybrid_score = (
        semantic_score
        + ALPHA * entity_match
    )

    hybrid_results.append(
        {
            "title": title,
            "semantic_score": semantic_score,
            "entity_match": entity_match,
            "matched_entities": matched_entities,
            "hybrid_score": hybrid_score,
        }
    )


# ---------------------------------------------------------
# 8. Rank by hybrid score
# ---------------------------------------------------------

hybrid_results.sort(
    key=lambda item: item["hybrid_score"],
    reverse=True,
)


# ---------------------------------------------------------
# 9. Display question and query entities
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("QUESTION")
print("=" * 80)

print(question)

print("\nQUERY ENTITIES")
print("=" * 80)

for entity in sorted(query_entities):
    print(f"- {entity}")


# ---------------------------------------------------------
# 10. Display semantic baseline
# ---------------------------------------------------------

print("\n" + "=" * 80)
print(f"TOP-{TOP_K} BGE BASELINE")
print("=" * 80)

semantic_top_k = semantic_results[:TOP_K]

for rank, result in enumerate(
    semantic_top_k,
    start=1,
):
    title = result.node.metadata["title"]

    marker = (
        "✓ GOLD"
        if title in gold_titles
        else ""
    )

    print(
        f"{rank}. {title:<40} "
        f"semantic={result.score:.4f} "
        f"{marker}"
    )


# ---------------------------------------------------------
# 11. Display hybrid ranking
# ---------------------------------------------------------

print("\n" + "=" * 80)
print(
    f"TOP-{TOP_K} HYBRID RETRIEVAL "
    f"(alpha={ALPHA})"
)
print("=" * 80)

for rank, result in enumerate(
    hybrid_results[:TOP_K],
    start=1,
):
    title = result["title"]

    marker = (
        "✓ GOLD"
        if title in gold_titles
        else ""
    )

    matched = ", ".join(
        sorted(result["matched_entities"])
    )

    if not matched:
        matched = "-"

    print(
        f"{rank}. {title:<40} "
        f"semantic={result['semantic_score']:.4f} "
        f"entity={result['entity_match']} "
        f"hybrid={result['hybrid_score']:.4f} "
        f"{marker}"
    )

    print(
        f"   matched entities: {matched}"
    )


# ---------------------------------------------------------
# 12. Calculate Recall@K
# ---------------------------------------------------------

baseline_titles = {
    result.node.metadata["title"]
    for result in semantic_top_k
}

hybrid_titles = {
    result["title"]
    for result in hybrid_results[:TOP_K]
}

baseline_retrieved_gold = (
    baseline_titles
    & gold_titles
)

hybrid_retrieved_gold = (
    hybrid_titles
    & gold_titles
)

baseline_recall = (
    len(baseline_retrieved_gold)
    / len(gold_titles)
)

hybrid_recall = (
    len(hybrid_retrieved_gold)
    / len(gold_titles)
)


# ---------------------------------------------------------
# 13. Display comparison
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print(
    f"BGE Recall@{TOP_K}:    "
    f"{baseline_recall:.2f} "
    f"({len(baseline_retrieved_gold)}/{len(gold_titles)})"
)

print(
    f"Hybrid Recall@{TOP_K}: "
    f"{hybrid_recall:.2f} "
    f"({len(hybrid_retrieved_gold)}/{len(gold_titles)})"
)