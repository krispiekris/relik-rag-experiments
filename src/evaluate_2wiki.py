import json
from pathlib import Path

from datasets import load_dataset
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from relik import Relik
from relik.inference.data.objects import RelikOutput


# =========================================================
# Configuration
# =========================================================

NUM_EXAMPLES = 20
DATASET_SEED = 42
TOP_K_VALUES = [2, 5]
ALPHA = 0.10

CACHE_PATH = Path(
    "data/2wiki_relik_annotations_20.json"
)

RESULTS_PATH = Path(
    "results/2wiki_retrieval_20.json"
)

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RELIK_MODEL = "sapienzanlp/relik-entity-linking-large"


# =========================================================
# Helper functions
# =========================================================

def extract_entities(relik_model: Relik, text: str) -> set[str]:
    """Extract linked entitym text using Relik."""

    result: RelikOutput = relik_model(text)

    return {
        span.label
        for span in result.spans
    }


def calculate_recall(
    retrieved_titles: list[str],
    gold_titles: set[str],
    k: int,
) -> float:
    """Calculate Recall@K for one multi-hop question."""

    if not gold_titles:
        return 0.0

    retrieved_at_k = set(retrieved_titles[:k])

    retrieved_gold = retrieved_at_k & gold_titles

    return len(retrieved_gold) / len(gold_titles)


def save_relik_cache(cache: dict) -> None:
    """Persist Relik annotations to disk."""

    CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CACHE_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            cache,
            file,
            indent=2,
            ensure_ascii=False,
        )

# =========================================================
# 1. Load 2WikiMultihopQA
# =========================================================

print("=" * 80)
print("LOADING DATASET")
print("=" * 80)

print(
    f"Loading {NUM_EXAMPLES} "
    f"2WikiMultihopQA dev examples..."
)

dataset = load_dataset(
    "xanhho/2WikiMultihopQA",
    split="dev",
)

# Deterministic sample, avoiding dependence on
# the dataset's original ordering.
dataset = (
    dataset
    .shuffle(seed=DATASET_SEED)
    .select(range(NUM_EXAMPLES))
)

print(
    f"Loaded {len(dataset)} examples."
)

# =========================================================
# 2. Load BGE embedding model
# =========================================================

print("\n" + "=" * 80)
print("LOADING BGE")
print("=" * 80)

Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL,
    device="cuda"
)

print("BGE loaded.")


# =========================================================
# 3. Load existing Relik annotation cache
# =========================================================

relik_cache = {}

if CACHE_PATH.exists():

    print("\n" + "=" * 80)
    print("LOADING RELIK CACHE")
    print("=" * 80)

    with CACHE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        relik_cache = json.load(file)

    print(
        f"Loaded cached annotations for "
        f"{len(relik_cache)} examples."
    )


# =========================================================
# 4. Check whether Relik is needed
# =========================================================

example_ids = {
    example["_id"]
    for example in dataset
}

cached_ids = set(
    relik_cache.keys()
)

missing_ids = (
    example_ids - cached_ids
)

relik = None

if missing_ids:

    print("\n" + "=" * 80)
    print("LOADING RELIK")
    print("=" * 80)

    print(
        f"{len(missing_ids)} examples "
        f"still require Relik annotation."
    )

    relik = Relik.from_pretrained(
        RELIK_MODEL,
        device="cuda"
    )

    print("Relik loaded.")

else:

    print(
        "\nAll Relik annotations are already cached."
    )


# =========================================================
# 5. Metric storage
# =========================================================

bge_recall_values = {
    k: []
    for k in TOP_K_VALUES
}

hybrid_recall_values = {
    k: []
    for k in TOP_K_VALUES
}

all_results = []


# =========================================================
# 6. Evaluate each question
# =========================================================

for example_number, example in enumerate(
    dataset,
    start=1,
):

    example_id = example["_id"]
    question = example["question"]

    gold_titles = set(
        example["supporting_facts"]["title"]
    )

    titles = list(
        example["context"]["title"]
    )

    sentence_groups = list(
        example["context"]["content"]
    )

    print("\n" + "-" * 80)

    print(
        f"[{example_number:02d}/{NUM_EXAMPLES}] "
        f"{question}"
    )

    print(
        "Gold: "
        + ", ".join(sorted(gold_titles))
    )


    # -----------------------------------------------------
    # 6.1 Build document nodes
    # -----------------------------------------------------

    nodes = []
    document_texts = {}

    for title, sentences in zip(
        titles,
        sentence_groups,
    ):

        text = " ".join(sentences)

        document_texts[title] = text

        node = TextNode(
            text=text,
            metadata={
                "title": title,
            },
        )

        nodes.append(node)


    # -----------------------------------------------------
    # 6.2 BGE semantic retrieval
    # -----------------------------------------------------

    index = VectorStoreIndex(
        nodes
    )

    semantic_retriever = (
        index.as_retriever(
            similarity_top_k=len(nodes)
        )
    )

    semantic_results = (
        semantic_retriever.retrieve(
            question
        )
    )

    semantic_scores = {
        result.node.metadata["title"]:
            float(result.score)
        for result in semantic_results
    }

    semantic_ranking = [
        result.node.metadata["title"]
        for result in semantic_results
    ]


    # -----------------------------------------------------
    # 6.3 Obtain Relik annotations
    # -----------------------------------------------------

    if example_id in relik_cache:

        print(
            "Relik annotations: cached"
        )

        cached_example = (
            relik_cache[example_id]
        )

        query_entities = set(
            cached_example[
                "query_entities"
            ]
        )

        document_entities = {
            title: set(entities)
            for title, entities in (
                cached_example[
                    "document_entities"
                ].items()
            )
        }

    else:

        print(
            "Relik annotations: computing"
        )

        if relik is None:
            raise RuntimeError(
                "Relik is required but "
                "was not loaded."
            )

        # Query entities

        query_entities = extract_entities(
            relik,
            question,
        )

        # Document entities

        document_entities = {}

        total_documents = len(titles)

        for document_number, title in enumerate(
            titles,
            start=1,
        ):

            print(
                f"  Document "
                f"{document_number:02d}/"
                f"{total_documents}: "
                f"{title}"
            )

            document_entities[title] = (
                extract_entities(
                    relik,
                    document_texts[title],
                )
            )

        # Store JSON-compatible annotations.

        relik_cache[example_id] = {
            "question": question,
            "query_entities": sorted(
                query_entities
            ),
            "document_entities": {
                title: sorted(entities)
                for title, entities
                in document_entities.items()
            },
        }

        # Save immediately after each question.
        # This protects us if WSL or the process
        # stops during a later question.

        save_relik_cache(
            relik_cache
        )

        print(
            "Relik annotations saved."
        )


    # -----------------------------------------------------
    # 6.4 Calculate hybrid scores
    # -----------------------------------------------------

    hybrid_documents = []

    for title in titles:

        semantic_score = (
            semantic_scores[title]
        )

        entities_in_document = (
            document_entities.get(
                title,
                set(),
            )
        )

        matched_entities = (
            query_entities
            & entities_in_document
        )

        entity_match = int(
            bool(matched_entities)
        )

        hybrid_score = (
            semantic_score
            + ALPHA * entity_match
        )

        hybrid_documents.append(
            {
                "title": title,
                "semantic_score":
                    semantic_score,
                "entity_match":
                    entity_match,
                "matched_entities":
                    sorted(matched_entities),
                "hybrid_score":
                    hybrid_score,
            }
        )


    # -----------------------------------------------------
    # 6.5 Sort hybrid ranking
    # -----------------------------------------------------

    hybrid_documents.sort(
        key=lambda item:
            item["hybrid_score"],
        reverse=True,
    )

    hybrid_ranking = [
        item["title"]
        for item in hybrid_documents
    ]


    # -----------------------------------------------------
    # 6.6 Calculate Recall@K
    # -----------------------------------------------------

    example_metrics = {
        "bge": {},
        "hybrid": {},
    }

    for k in TOP_K_VALUES:

        bge_recall = calculate_recall(
            semantic_ranking,
            gold_titles,
            k,
        )

        hybrid_recall = calculate_recall(
            hybrid_ranking,
            gold_titles,
            k,
        )

        bge_recall_values[k].append(
            bge_recall
        )

        hybrid_recall_values[k].append(
            hybrid_recall
        )

        example_metrics["bge"][
            f"recall@{k}"
        ] = bge_recall

        example_metrics["hybrid"][
            f"recall@{k}"
        ] = hybrid_recall


    # -----------------------------------------------------
    # 6.7 Print current question results
    # -----------------------------------------------------

    metric_text = []

    for k in TOP_K_VALUES:

        metric_text.append(
            f"R@{k}: "
            f"BGE="
            f"{example_metrics['bge'][f'recall@{k}']:.2f}, "
            f"Hybrid="
            f"{example_metrics['hybrid'][f'recall@{k}']:.2f}"
        )

    print(
        " | ".join(metric_text)
    )


    # -----------------------------------------------------
    # 6.8 Store detailed result
    # -----------------------------------------------------

    all_results.append(
        {
            "id": example_id,
            "question": question,
            "gold_titles":
                sorted(gold_titles),
            "query_entities":
                sorted(query_entities),
            "metrics":
                example_metrics,
            "semantic_ranking":
                semantic_ranking,
            "hybrid_ranking":
                hybrid_ranking,
            "semantic_scores":
                semantic_scores,
            "hybrid_details":
                hybrid_documents,
        }
    )


# =========================================================
# 7. Calculate aggregate metrics
# =========================================================

summary = {}

for k in TOP_K_VALUES:

    mean_bge = (
        sum(bge_recall_values[k])
        / len(bge_recall_values[k])
    )

    mean_hybrid = (
        sum(hybrid_recall_values[k])
        / len(hybrid_recall_values[k])
    )

    delta = (
        mean_hybrid - mean_bge
    )

    summary[
        f"recall@{k}"
    ] = {
        "bge": mean_bge,
        "hybrid": mean_hybrid,
        "delta": delta,
    }


# =========================================================
# 8. Print summary
# =========================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    f"Examples: {NUM_EXAMPLES}"
)

print(
    f"Embedding model: "
    f"{EMBEDDING_MODEL}"
)

print(
    f"Relik model: "
    f"{RELIK_MODEL}"
)

print(
    f"Hybrid alpha: "
    f"{ALPHA}"
)

print()

for k in TOP_K_VALUES:

    values = summary[
        f"recall@{k}"
    ]

    print(
        f"Recall@{k}"
    )

    print(
        f"  BGE:    "
        f"{values['bge']:.4f}"
    )

    print(
        f"  Hybrid: "
        f"{values['hybrid']:.4f}"
    )

    print(
        f"  Delta:  "
        f"{values['delta']:+.4f}"
    )

    print()


# =========================================================
# 9. Win / loss / tie analysis
# =========================================================

comparison = {}

print("=" * 80)
print("PER-QUESTION COMPARISON")
print("=" * 80)

for k in TOP_K_VALUES:

    wins = 0
    losses = 0
    ties = 0

    for bge_value, hybrid_value in zip(
        bge_recall_values[k],
        hybrid_recall_values[k],
    ):

        if hybrid_value > bge_value:
            wins += 1

        elif hybrid_value < bge_value:
            losses += 1

        else:
            ties += 1

    comparison[
        f"recall@{k}"
    ] = {
        "hybrid_wins": wins,
        "hybrid_losses": losses,
        "ties": ties,
    }

    print(
        f"Recall@{k}: "
        f"Hybrid wins={wins}, "
        f"losses={losses}, "
        f"ties={ties}"
    )


# =========================================================
# 10. Save detailed experiment results
# =========================================================

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output = {
    "configuration": {
    "dataset":
        "xanhho/2WikiMultihopQA",
    "dataset_split":
        "dev",
    "dataset_seed":
        DATASET_SEED,
    "num_examples":
        NUM_EXAMPLES,
    "embedding_model":
        EMBEDDING_MODEL,
    "relik_model":
        RELIK_MODEL,
    "alpha":
        ALPHA,
    "top_k_values":
        TOP_K_VALUES,
    },
    "summary":
        summary,
    "comparison":
        comparison,
    "results":
        all_results,
}

with RESULTS_PATH.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        output,
        file,
        indent=2,
        ensure_ascii=False,
    )


# =========================================================
# 11. Final paths
# =========================================================

print("\n" + "=" * 80)
print("FILES")
print("=" * 80)

print(
    f"Relik annotation cache: "
    f"{CACHE_PATH}"
)

print(
    f"Experiment results:     "
    f"{RESULTS_PATH}"
)

print("\nEvaluation complete.")