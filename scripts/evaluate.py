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

books = load_books(DATA_DIR / "books.csv")
ratings = load_ratings(DATA_DIR / "ratings.csv")

ks = [5]
rows = []
latencies = []

for user_id, user_ratings in ratings.items():
    positives = [bid for bid, rating in user_ratings.items() if rating >= 4]
    if not positives:
        continue
    # Deterministic leave-one-out: hold out the alphabetically last positive item.
    holdout = sorted(positives)[-1]
    train = deepcopy(ratings)
    del train[user_id][holdout]
    engine = HybridRecommender(books, train, alpha=0.6)
    start = time.perf_counter()
    recs = engine.recommend(user_id, k=max(ks))
    elapsed_ms = (time.perf_counter() - start) * 1000
    latencies.append(elapsed_ms)
    rec_ids = [bid for bid, _ in recs]
    for k in ks:
        top_k = rec_ids[:k]
        hit = 1 if holdout in top_k else 0
        precision = hit / k
        recall = hit / 1
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
        rows.append({
            "user_id": user_id,
            "holdout": holdout,
            "k": k,
            "hit": hit,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "latency_ms": elapsed_ms,
            "recommended": top_k,
        })

summary = {
    "users_evaluated": len({r["user_id"] for r in rows}),
    "precision_at_5": statistics.mean(r["precision"] for r in rows),
    "recall_at_5": statistics.mean(r["recall"] for r in rows),
    "f1_at_5": statistics.mean(r["f1"] for r in rows),
    "hit_rate_at_5": statistics.mean(r["hit"] for r in rows),
    "mean_latency_ms": statistics.mean(latencies),
    "median_latency_ms": statistics.median(latencies),
    "alpha": 0.6,
}

with open(REPORT_DIR / "evaluation_rows.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

with open(REPORT_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
