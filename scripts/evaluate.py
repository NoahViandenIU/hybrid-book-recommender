from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.recommender import HybridRecommender, load_books, load_ratings

DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

K = 5
ALPHA_GRID = tuple(round(value / 10, 1) for value in range(11))


def user_holdouts(ratings):
    holdouts = {}
    for user_id, user_ratings in ratings.items():
        positives = sorted(
            book_id for book_id, rating in user_ratings.items() if rating >= 4
        )
        if len(positives) < 2:
            continue
        holdouts[user_id] = {
            "validation": positives[-2],
            "test": positives[-1],
        }
    return holdouts


def metric_row(split, model, user_id, holdout, recommendations, latency_ms):
    recommendation_ids = [book_id for book_id, _ in recommendations]
    rank = (
        recommendation_ids.index(holdout) + 1
        if holdout in recommendation_ids
        else None
    )
    hit = int(rank is not None)
    precision = hit / K
    recall = float(hit)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    reciprocal_rank = 1 / rank if rank else 0.0
    return {
        "split": split,
        "model": model,
        "user_id": user_id,
        "holdout": holdout,
        "rank": rank or "",
        "hit": hit,
        "precision_at_5": precision,
        "recall_at_5": recall,
        "f1_at_5": f1,
        "reciprocal_rank_at_5": reciprocal_rank,
        "latency_ms": latency_ms,
        "recommended": recommendation_ids,
    }


def evaluate_alpha(books, ratings, holdouts, alpha, split):
    rows = []
    for user_id, user_holdout in holdouts.items():
        train = deepcopy(ratings)
        if split == "validation":
            del train[user_id][user_holdout["validation"]]
            del train[user_id][user_holdout["test"]]
            holdout = user_holdout["validation"]
        else:
            del train[user_id][user_holdout["test"]]
            holdout = user_holdout["test"]

        engine = HybridRecommender(books, train, alpha=alpha)
        start = time.perf_counter()
        recommendations = engine.recommend(user_id, k=K)
        latency_ms = (time.perf_counter() - start) * 1000
        rows.append(
            metric_row(
                split,
                f"alpha={alpha:.1f}",
                user_id,
                holdout,
                recommendations,
                latency_ms,
            )
        )
    return rows


def evaluate_popularity(books, ratings, holdouts):
    rows = []
    for user_id, user_holdout in holdouts.items():
        train = deepcopy(ratings)
        holdout = user_holdout["test"]
        del train[user_id][holdout]
        engine = HybridRecommender(books, train)
        seen = set(train[user_id])
        start = time.perf_counter()
        ranked_ids = sorted(
            (book_id for book_id in books if book_id not in seen),
            key=lambda book_id: (-engine.item_popularity[book_id], book_id),
        )[:K]
        latency_ms = (time.perf_counter() - start) * 1000
        recommendations = [
            (book_id, engine.item_popularity[book_id]) for book_id in ranked_ids
        ]
        rows.append(
            metric_row(
                "test",
                "popularity-only",
                user_id,
                holdout,
                recommendations,
                latency_ms,
            )
        )
    return rows


def summarize(rows):
    latencies = [row["latency_ms"] for row in rows]
    return {
        "users_evaluated": len(rows),
        "precision_at_5": statistics.mean(row["precision_at_5"] for row in rows),
        "recall_at_5": statistics.mean(row["recall_at_5"] for row in rows),
        "f1_at_5": statistics.mean(row["f1_at_5"] for row in rows),
        "hit_rate_at_5": statistics.mean(row["hit"] for row in rows),
        "mrr_at_5": statistics.mean(row["reciprocal_rank_at_5"] for row in rows),
        "mean_latency_ms": statistics.mean(latencies),
        "median_latency_ms": statistics.median(latencies),
    }


def select_alpha(validation_results):
    return max(
        ALPHA_GRID,
        key=lambda alpha: (
            validation_results[f"{alpha:.1f}"]["hit_rate_at_5"],
            validation_results[f"{alpha:.1f}"]["mrr_at_5"],
            -abs(alpha - 0.5),
        ),
    )


def run_evaluation():
    books = load_books(DATA_DIR / "books.csv")
    ratings = load_ratings(DATA_DIR / "ratings.csv")
    holdouts = user_holdouts(ratings)

    validation_rows = {}
    validation_results = {}
    for alpha in ALPHA_GRID:
        rows = evaluate_alpha(books, ratings, holdouts, alpha, "validation")
        validation_rows[alpha] = rows
        validation_results[f"{alpha:.1f}"] = summarize(rows)

    selected_alpha = select_alpha(validation_results)
    test_models = {
        "collaborative-only": evaluate_alpha(
            books, ratings, holdouts, 0.0, "test"
        ),
        f"hybrid-alpha-{selected_alpha:.1f}": evaluate_alpha(
            books, ratings, holdouts, selected_alpha, "test"
        ),
        "content-only": evaluate_alpha(books, ratings, holdouts, 1.0, "test"),
        "popularity-only": evaluate_popularity(books, ratings, holdouts),
    }
    test_results = {
        model: summarize(rows) for model, rows in test_models.items()
    }

    all_rows = []
    for alpha in ALPHA_GRID:
        all_rows.extend(validation_rows[alpha])
    for rows in test_models.values():
        all_rows.extend(rows)

    summary = {
        "protocol": {
            "k": K,
            "users_evaluated": len(holdouts),
            "validation_holdout": (
                "second alphabetically last positive item per user"
            ),
            "test_holdout": "alphabetically last positive item per user",
            "alpha_grid": list(ALPHA_GRID),
            "selection_rule": (
                "highest validation hit_rate_at_5, then validation mrr_at_5"
            ),
        },
        "selected_alpha": selected_alpha,
        "validation_results": validation_results,
        "test_results": test_results,
    }

    with open(
        REPORT_DIR / "evaluation_rows.csv", "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    with open(
        REPORT_DIR / "evaluation_summary.json", "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    return summary


if __name__ == "__main__":
    print(json.dumps(run_evaluation(), indent=2))
