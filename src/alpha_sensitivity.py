import json
from pathlib import Path


# =========================================================
# Configuration
# =========================================================

RESULTS_PATH = Path("results/retrieval_20.json")
OUTPUT_PATH = Path("results/alpha_sensitivity_20.json")

ALPHAS = [
    0.00,
    0.01,
    0.025,
    0.05,
    0.075,
    0.10,
    0.20,
]

TOP_K_VALUES = [2, 5]


# =========================================================
# Helper function
# =========================================================

def recall_at_k(
    ranking: list[str],
    gold_titles: set[str],
    k: int,
) -> float:
    """Calculate Recall@K for a single question."""

    if not gold_titles:
        return 0.0

    retrieved_titles = set(ranking[:k])
    retrieved_gold = retrieved_titles & gold_titles

    return len(retrieved_gold) / len(gold_titles)


# =========================================================
# 1. Load previous experiment results
# =========================================================

if not RESULTS_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {RESULTS_PATH}. "
        "Run evaluate_retrieval.py first."
    )

with RESULTS_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    experiment = json.load(file)

examples = experiment["results"]

print(
    f"Loaded {len(examples)} examples "
    f"from {RESULTS_PATH}"
)


# =========================================================
# 2. Calculate BGE baseline
# =========================================================

baseline = {}

for k in TOP_K_VALUES:

    recalls = []

    for example in examples:

        gold_titles = set(
            example["gold_titles"]
        )

        semantic_ranking = (
            example["semantic_ranking"]
        )

        recall = recall_at_k(
            semantic_ranking,
            gold_titles,
            k,
        )

        recalls.append(recall)

    baseline[k] = (
        sum(recalls) / len(recalls)
    )


print("\nBGE BASELINE")

for k in TOP_K_VALUES:
    print(
        f"Recall@{k}: "
        f"{baseline[k]:.4f}"
    )


# =========================================================
# 3. Sensitivity analysis
# =========================================================

sensitivity_results = []

for alpha in ALPHAS:

    recall_values = {
        k: []
        for k in TOP_K_VALUES
    }

    wins = {
        k: 0
        for k in TOP_K_VALUES
    }

    losses = {
        k: 0
        for k in TOP_K_VALUES
    }

    ties = {
        k: 0
        for k in TOP_K_VALUES
    }


    # -----------------------------------------------------
    # Evaluate each question
    # -----------------------------------------------------

    for example in examples:

        gold_titles = set(
            example["gold_titles"]
        )

        semantic_ranking = (
            example["semantic_ranking"]
        )

        semantic_scores = (
            example["semantic_scores"]
        )

        hybrid_details = (
            example["hybrid_details"]
        )


        # -------------------------------------------------
        # Recover Relik entity-match signal
        # -------------------------------------------------

        entity_matches = {
            item["title"]:
                int(item["entity_match"])
            for item in hybrid_details
        }


        # -------------------------------------------------
        # Calculate scores using the current alpha
        #
        # score =
        # semantic_score
        # + alpha * entity_match
        # -------------------------------------------------

        reranked_documents = []

        for title, semantic_score in (
            semantic_scores.items()
        ):

            entity_match = (
                entity_matches.get(
                    title,
                    0,
                )
            )

            hybrid_score = (
                float(semantic_score)
                + alpha * entity_match
            )

            reranked_documents.append(
                {
                    "title": title,
                    "semantic_score":
                        float(semantic_score),
                    "entity_match":
                        entity_match,
                    "hybrid_score":
                        hybrid_score,
                }
            )


        # -------------------------------------------------
        # Sort by hybrid score
        # -------------------------------------------------

        reranked_documents.sort(
            key=lambda item:
                item["hybrid_score"],
            reverse=True,
        )

        hybrid_ranking = [
            item["title"]
            for item in reranked_documents
        ]


        # -------------------------------------------------
        # Evaluate Recall@K
        # -------------------------------------------------

        for k in TOP_K_VALUES:

            bge_recall = recall_at_k(
                semantic_ranking,
                gold_titles,
                k,
            )

            hybrid_recall = recall_at_k(
                hybrid_ranking,
                gold_titles,
                k,
            )

            recall_values[k].append(
                hybrid_recall
            )


            # ---------------------------------------------
            # Per-question comparison
            # ---------------------------------------------

            if hybrid_recall > bge_recall:
                wins[k] += 1

            elif hybrid_recall < bge_recall:
                losses[k] += 1

            else:
                ties[k] += 1


    # -----------------------------------------------------
    # Aggregate results for current alpha
    # -----------------------------------------------------

    alpha_result = {
        "alpha": alpha,
    }

    for k in TOP_K_VALUES:
        mean_recall = (
            sum(recall_values[k])
            / len(recall_values[k])
        )

        delta = (
            mean_recall
            - baseline[k]
        )

        alpha_result[f"recall@{k}"] = mean_recall
        alpha_result[f"delta@{k}"] = delta
        alpha_result[f"wins@{k}"] = wins[k]
        alpha_result[f"losses@{k}"] = losses[k]
        alpha_result[f"ties@{k}"] = ties[k]

    sensitivity_results.append(alpha_result)


# =========================================================
# 4. Print sensitivity table
# =========================================================

print("\n" + "=" * 90)
print("ALPHA SENSITIVITY ANALYSIS")
print("=" * 90)

print(
    f"{'Alpha':>8}"
    f"{'R@2':>10}"
    f"{'Delta@2':>12}"
    f"{'W/L/T @2':>15}"
    f"{'R@5':>10}"
    f"{'Delta@5':>12}"
    f"{'W/L/T @5':>15}"
)

print("-" * 90)

for result in sensitivity_results:
    comparison_2 = (
        f"{result['wins@2']}/"
        f"{result['losses@2']}/"
        f"{result['ties@2']}"
    )

    comparison_5 = (
        f"{result['wins@5']}/"
        f"{result['losses@5']}/"
        f"{result['ties@5']}"
    )

    print(
        f"{result['alpha']:>8.3f}"
        f"{result['recall@2']:>10.4f}"
        f"{result['delta@2']:>+12.4f}"
        f"{comparison_2:>15}"
        f"{result['recall@5']:>10.4f}"
        f"{result['delta@5']:>+12.4f}"
        f"{comparison_5:>15}"
    )


# =========================================================
# 5. Save results
# =========================================================

output = {
    "configuration": {
        "num_examples": len(examples),
        "alphas": ALPHAS,
        "top_k_values": TOP_K_VALUES,
        "source_results": str(RESULTS_PATH),
    },
    "baseline": {
        f"recall@{k}": baseline[k]
        for k in TOP_K_VALUES
    },
    "sensitivity_results": sensitivity_results,
}

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT_PATH.open(
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
# 6. Sanity check
# =========================================================

alpha_zero = next(
    result
    for result in sensitivity_results
    if result["alpha"] == 0.0
)

sanity_check_passed = True

for k in TOP_K_VALUES:
    if abs(
        alpha_zero[f"recall@{k}"]
        - baseline[k]
    ) > 1e-12:
        sanity_check_passed = False


print("\n" + "=" * 90)
print("SANITY CHECK")
print("=" * 90)

if sanity_check_passed:
    print(
        "PASS: alpha=0 reproduces "
        "the BGE baseline exactly."
    )
else:
    print(
        "FAIL: alpha=0 does not reproduce "
        "the BGE baseline."
    )


print("\nResults saved to:")
print(OUTPUT_PATH)