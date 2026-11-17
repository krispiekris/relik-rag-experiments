import json
from pathlib import Path

ALPHA = 0.014
K = 5
INPUT_PATH = Path("results/retrieval_1000.json")
OUTPUT_PATH = Path("results/alpha_0014_cases.json")


def normalise_title(title):
    return title.strip().lower()


def recall_at_k(ranking, gold_titles, k):
    retrieved = {
        normalise_title(item["title"])
        for item in ranking[:k]
    }
    gold = {
        normalise_title(title)
        for title in gold_titles
    }

    if not gold:
        return 0.0

    return len(retrieved & gold) / len(gold)


def build_semantic_ranking(example):
    details_by_title = {
        item["title"]: item
        for item in example["hybrid_details"]
    }

    ranking = []

    for rank, title in enumerate(example["semantic_ranking"], start=1):
        details = details_by_title.get(title, {})

        ranking.append({
            "rank": rank,
            "title": title,
            "semantic_score": details.get(
                "semantic_score",
                example.get("semantic_scores", {}).get(title)
                if isinstance(example.get("semantic_scores"), dict)
                else None
            ),
            "entity_match": details.get("entity_match", 0),
            "matched_entities": details.get("matched_entities", []),
        })

    return ranking


def build_hybrid_ranking(example, alpha):
    ranking = []

    for item in example["hybrid_details"]:
        semantic_score = item["semantic_score"]
        entity_match = item["entity_match"]

        ranking.append({
            "title": item["title"],
            "semantic_score": semantic_score,
            "entity_match": entity_match,
            "matched_entities": item.get("matched_entities", []),
            "hybrid_score": semantic_score + alpha * entity_match,
        })

    ranking.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True
    )

    for rank, item in enumerate(ranking, start=1):
        item["rank"] = rank

    return ranking


def ranking_comparison(semantic_ranking, hybrid_ranking, gold_titles):
    semantic_positions = {
        item["title"]: item["rank"]
        for item in semantic_ranking
    }

    gold_normalised = {
        normalise_title(title)
        for title in gold_titles
    }

    comparison = []

    for item in hybrid_ranking[:10]:
        semantic_rank = semantic_positions.get(item["title"])

        comparison.append({
            **item,
            "semantic_rank": semantic_rank,
            "rank_change": (
                semantic_rank - item["rank"]
                if semantic_rank is not None
                else None
            ),
            "is_gold": (
                normalise_title(item["title"])
                in gold_normalised
            ),
        })

    return comparison


with INPUT_PATH.open(encoding="utf-8") as f:
    data = json.load(f)

selected_cases = []

for example in data["results"]:
    semantic_ranking = build_semantic_ranking(example)
    hybrid_ranking = build_hybrid_ranking(example, ALPHA)

    semantic_recall = recall_at_k(
        semantic_ranking,
        example["gold_titles"],
        K,
    )

    hybrid_recall = recall_at_k(
        hybrid_ranking,
        example["gold_titles"],
        K,
    )

    if hybrid_recall > semantic_recall:
        outcome = "win"
    elif hybrid_recall < semantic_recall:
        outcome = "loss"
    else:
        continue

    selected_cases.append({
        "id": example["id"],
        "outcome": outcome,
        "question": example["question"],
        "query_entities": example.get("query_entities", []),
        "gold_titles": example["gold_titles"],
        "semantic_recall_at_5": semantic_recall,
        "hybrid_recall_at_5": hybrid_recall,
        "semantic_top_10": semantic_ranking[:10],
        "hybrid_top_10_comparison": ranking_comparison(
            semantic_ranking,
            hybrid_ranking,
            example["gold_titles"],
        ),
    })


wins = [
    case for case in selected_cases
    if case["outcome"] == "win"
]

losses = [
    case for case in selected_cases
    if case["outcome"] == "loss"
]

output = {
    "configuration": {
        "alpha": ALPHA,
        "k": K,
        "ranking_formula": (
            "semantic_score + alpha * entity_match"
        ),
    },
    "summary": {
        "wins": len(wins),
        "losses": len(losses),
        "total_changed": len(selected_cases),
    },
    "wins": wins,
    "losses": losses,
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(json.dumps(output["summary"], indent=2))
print(f"\nSaved report to: {OUTPUT_PATH}")

for case in selected_cases:
    print("\n" + "=" * 80)
    print(case["outcome"].upper(), case["id"])
    print(case["question"])
    print("Query entities:", case["query_entities"])
    print("Gold titles:", case["gold_titles"])
    print(
        f"Recall@5: {case['semantic_recall_at_5']}"
        f" -> {case['hybrid_recall_at_5']}"
    )

    for item in case["hybrid_top_10_comparison"]:
        marker = "GOLD" if item["is_gold"] else ""
        print(
            f"{item['rank']:>2}. "
            f"{item['title'][:42]:<42} "
            f"semantic-rank={str(item['semantic_rank']):>3} "
            f"move={str(item['rank_change']):>3} "
            f"sem={item['semantic_score']:.4f} "
            f"entity={item['entity_match']} "
            f"hybrid={item['hybrid_score']:.4f} "
            f"{marker}"
        )