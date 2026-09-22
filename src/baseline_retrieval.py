from datasets import load_dataset
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


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


# ---------------------------------------------------------
# 2. Configure embedding model
# ---------------------------------------------------------

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5"
)


# ---------------------------------------------------------
# 3. Convert the 10 HotpotQA contexts to LlamaIndex nodes
# ---------------------------------------------------------

nodes = []

titles = example["context"]["title"]
sentence_groups = example["context"]["sentences"]

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
# 4. Build the semantic index
# ---------------------------------------------------------

index = VectorStoreIndex(nodes)

retriever = index.as_retriever(
    similarity_top_k=5,
)


# ---------------------------------------------------------
# 5. Retrieve
# ---------------------------------------------------------

results = retriever.retrieve(question)


# ---------------------------------------------------------
# 6. Display result
# ---------------------------------------------------------

print("=" * 80)
print("QUESTION")
print("=" * 80)
print(question)

print("\nGOLD SUPPORTING DOCUMENTS")
print("=" * 80)

for title in gold_titles:
    print(f"- {title}")

print("\nTOP-5 SEMANTIC RETRIEVAL")
print("=" * 80)

retrieved_gold = set()

for rank, result in enumerate(results, start=1):
    title = result.node.metadata["title"]
    score = result.score

    is_gold = title in gold_titles

    if is_gold:
        retrieved_gold.add(title)

    marker = "✓ GOLD" if is_gold else ""

    print(
        f"{rank}. {title:<40} "
        f"score={score:.4f}  {marker}"
    )


# ---------------------------------------------------------
# 7. Calculate document Recall@5
# ---------------------------------------------------------

recall_at_5 = len(retrieved_gold) / len(gold_titles)

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

print(
    f"Gold documents retrieved: "
    f"{len(retrieved_gold)}/{len(gold_titles)}"
)

print(f"Recall@5: {recall_at_5:.2f}")