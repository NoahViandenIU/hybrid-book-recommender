from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import run_evaluation


class EvaluationTests(unittest.TestCase):
    def test_evaluation_selects_alpha_from_validation_only(self):
        summary = run_evaluation()
        self.assertEqual(summary["selected_alpha"], 0.7)
        self.assertEqual(summary["protocol"]["users_evaluated"], 15)

    def test_test_split_contains_all_required_baselines(self):
        summary = run_evaluation()
        results = summary["test_results"]
        self.assertEqual(
            set(results),
            {
                "collaborative-only",
                "hybrid-alpha-0.7",
                "content-only",
                "popularity-only",
            },
        )
        self.assertAlmostEqual(
            results["hybrid-alpha-0.7"]["hit_rate_at_5"], 14 / 15
        )


if __name__ == "__main__":
    unittest.main()
