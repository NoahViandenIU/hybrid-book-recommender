"""Hybrid recommendation engine for a small online bookstore.

The implementation is deliberately lightweight and reproducible. It combines
content-based item similarity with user-based collaborative filtering and a
popularity fallback. The code avoids storing personal data and operates on
pseudonymous user IDs only.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

TOKEN_RE = re.compile(r"[a-zA-Z]{3,}")


@dataclass(frozen=True)
class Book:
    book_id: str
    title: str
    author: str
    genre: str
    tags: str
    description: str


RatingMatrix = Dict[str, Dict[str, float]]


def load_books(path: str | Path) -> Dict[str, Book]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["book_id"]: Book(**row) for row in csv.DictReader(f)}


def load_ratings(path: str | Path) -> RatingMatrix:
    ratings: RatingMatrix = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ratings[row["user_id"]][row["book_id"]] = float(row["rating"])
    return dict(ratings)


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def build_tfidf(books: Dict[str, Book]) -> Dict[str, Dict[str, float]]:
    docs = {}
    df = Counter()
    for bid, book in books.items():
        text = " ".join([book.genre, book.tags, book.description])
        terms = _tokens(text)
        docs[bid] = Counter(terms)
        for term in set(terms):
            df[term] += 1
    n = len(books)
    vectors: Dict[str, Dict[str, float]] = {}
    for bid, counts in docs.items():
        total = sum(counts.values()) or 1
        vector = {}
        for term, count in counts.items():
            tf = count / total
            idf = math.log((1 + n) / (1 + df[term])) + 1
            vector[term] = tf * idf
        vectors[bid] = vector
    return vectors


def cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class HybridRecommender:
    def __init__(self, books: Dict[str, Book], ratings: RatingMatrix, alpha: float = 0.6):
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.books = books
        self.ratings = ratings
        self.alpha = alpha
        self.tfidf = build_tfidf(books)
        self.item_popularity = self._compute_popularity()

    def _compute_popularity(self) -> Dict[str, float]:
        totals = defaultdict(float)
        counts = defaultdict(int)
        for user_ratings in self.ratings.values():
            for bid, rating in user_ratings.items():
                totals[bid] += rating
                counts[bid] += 1
        if not counts:
            return {bid: 0.0 for bid in self.books}
        max_count = max(counts.values())
        popularity = {}
        for bid in self.books:
            avg = totals[bid] / counts[bid] if counts[bid] else 0.0
            support = counts[bid] / max_count if max_count else 0.0
            popularity[bid] = (avg / 5.0) * 0.8 + support * 0.2
        return popularity

    def _content_score(self, user_id: str, target_book_id: str) -> float:
        user_ratings = self.ratings.get(user_id, {})
        liked = [(bid, r) for bid, r in user_ratings.items() if r >= 4]
        if not liked:
            return self.item_popularity.get(target_book_id, 0.0)
        weighted_sum = 0.0
        weight_total = 0.0
        for bid, rating in liked:
            sim = cosine_sparse(self.tfidf[target_book_id], self.tfidf[bid])
            weight = rating / 5.0
            weighted_sum += sim * weight
            weight_total += weight
        return weighted_sum / weight_total if weight_total else 0.0

    def _user_similarity(self, u1: str, u2: str) -> float:
        r1 = self.ratings.get(u1, {})
        r2 = self.ratings.get(u2, {})
        common = set(r1) & set(r2)
        if len(common) < 2:
            return 0.0
        mean1 = sum(r1[b] for b in common) / len(common)
        mean2 = sum(r2[b] for b in common) / len(common)
        num = sum((r1[b] - mean1) * (r2[b] - mean2) for b in common)
        den1 = math.sqrt(sum((r1[b] - mean1) ** 2 for b in common))
        den2 = math.sqrt(sum((r2[b] - mean2) ** 2 for b in common))
        if den1 == 0 or den2 == 0:
            return 0.0
        return max(0.0, num / (den1 * den2))

    def _collaborative_score(self, user_id: str, target_book_id: str) -> float:
        similarities: List[Tuple[float, float]] = []
        for other_user, other_ratings in self.ratings.items():
            if other_user == user_id or target_book_id not in other_ratings:
                continue
            sim = self._user_similarity(user_id, other_user)
            if sim > 0:
                similarities.append((sim, other_ratings[target_book_id] / 5.0))
        if not similarities:
            return self.item_popularity.get(target_book_id, 0.0)
        numerator = sum(sim * rating for sim, rating in similarities)
        denominator = sum(sim for sim, _ in similarities)
        return numerator / denominator if denominator else 0.0

    def score(self, user_id: str, book_id: str) -> float:
        content = self._content_score(user_id, book_id)
        collaborative = self._collaborative_score(user_id, book_id)
        return self.alpha * content + (1 - self.alpha) * collaborative

    def recommend(self, user_id: str, k: int = 5) -> List[Tuple[str, float]]:
        seen = set(self.ratings.get(user_id, {}))
        candidates = [bid for bid in self.books if bid not in seen]
        ranked = [(bid, self.score(user_id, bid)) for bid in candidates]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:k]


def create_engine(data_dir: str | Path = "data", alpha: float = 0.6) -> HybridRecommender:
    data_dir = Path(data_dir)
    books = load_books(data_dir / "books.csv")
    ratings = load_ratings(data_dir / "ratings.csv")
    return HybridRecommender(books, ratings, alpha=alpha)


if __name__ == "__main__":
    engine = create_engine(Path(__file__).resolve().parents[1] / "data")
    for bid, score in engine.recommend("U003", k=5):
        book = engine.books[bid]
        print(f"{bid}: {book.title} ({score:.3f})")
