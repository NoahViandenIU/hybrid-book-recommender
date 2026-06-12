from pathlib import Path
import unittest

from src.recommender import (
    HybridRecommender,
    build_tfidf,
    cosine_sparse,
    create_engine,
    load_books,
    load_ratings,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class RecommenderTests(unittest.TestCase):
    def test_data_loads_expected_fixture_size(self):
        self.assertEqual(len(load_books(DATA_DIR / "books.csv")), 20)
        self.assertEqual(len(load_ratings(DATA_DIR / "ratings.csv")), 15)

    def test_tfidf_vectors_cover_all_books(self):
        books = load_books(DATA_DIR / "books.csv")
        vectors = build_tfidf(books)
        self.assertEqual(set(vectors), set(books))
        self.assertTrue(all(vectors.values()))

    def test_cosine_similarity_identity_and_empty_vector(self):
        vector = {"python": 0.8, "testing": 0.6}
        self.assertAlmostEqual(cosine_sparse(vector, vector), 1.0)
        self.assertEqual(cosine_sparse(vector, {}), 0.0)

    def test_invalid_alpha_is_rejected(self):
        books = load_books(DATA_DIR / "books.csv")
        ratings = load_ratings(DATA_DIR / "ratings.csv")
        with self.assertRaises(ValueError):
            HybridRecommender(books, ratings, alpha=1.1)

    def test_recommendations_do_not_include_seen_books(self):
        engine = create_engine(DATA_DIR, alpha=0.7)
        seen = set(engine.ratings["U003"])
        recommendations = engine.recommend("U003", k=5)
        self.assertEqual(len(recommendations), 5)
        self.assertFalse(any(book_id in seen for book_id, _ in recommendations))

    def test_expected_top_recommendation_is_stable(self):
        engine = create_engine(DATA_DIR, alpha=0.7)
        recommendations = engine.recommend("U003", k=5)
        self.assertEqual(recommendations[0][0], "B019")

    def test_scores_are_normalized(self):
        engine = create_engine(DATA_DIR, alpha=0.7)
        for _, score in engine.recommend("U001", k=5):
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_cold_user_gets_deterministic_popularity_fallback(self):
        engine = create_engine(DATA_DIR, alpha=0.7)
        first = engine.recommend("UNKNOWN", k=3)
        second = engine.recommend("UNKNOWN", k=3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_non_positive_k_is_rejected(self):
        engine = create_engine(DATA_DIR)
        with self.assertRaises(ValueError):
            engine.recommend("U001", k=0)


if __name__ == "__main__":
    unittest.main()
