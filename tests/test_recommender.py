from pathlib import Path
import unittest

from src.recommender import create_engine, load_books, load_ratings

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

class RecommenderTests(unittest.TestCase):
    def test_data_loads(self):
        self.assertEqual(len(load_books(DATA_DIR / "books.csv")), 20)
        self.assertGreaterEqual(len(load_ratings(DATA_DIR / "ratings.csv")), 10)

    def test_recommendations_do_not_include_seen_books(self):
        engine = create_engine(DATA_DIR)
        seen = set(engine.ratings["U003"])
        recommendations = engine.recommend("U003", k=5)
        self.assertEqual(len(recommendations), 5)
        self.assertFalse(any(bid in seen for bid, _ in recommendations))

    def test_scores_are_normalized(self):
        engine = create_engine(DATA_DIR)
        for bid, score in engine.recommend("U001", k=5):
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_cold_user_gets_popularity_fallback(self):
        engine = create_engine(DATA_DIR)
        recommendations = engine.recommend("UNKNOWN", k=3)
        self.assertEqual(len(recommendations), 3)
        self.assertTrue(all(score >= 0.0 for _, score in recommendations))

if __name__ == "__main__":
    unittest.main()
