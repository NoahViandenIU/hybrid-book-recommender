import unittest

from app import create_app


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = create_app().test_client()

    def test_index_and_books_endpoints(self):
        index_response = self.client.get("/")
        self.assertEqual(index_response.status_code, 200)
        index_response.close()
        response = self.client.get("/api/books")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.get_json()), 20)

    def test_recommendation_endpoint_returns_five_items(self):
        response = self.client.get("/api/recommendations/U003?k=5")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.get_json()["recommendations"]), 5)

    def test_non_integer_k_returns_400(self):
        response = self.client.get("/api/recommendations/U003?k=abc")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_out_of_range_k_returns_400(self):
        for value in ("0", "-1", "21"):
            with self.subTest(value=value):
                response = self.client.get(
                    f"/api/recommendations/U003?k={value}"
                )
                self.assertEqual(response.status_code, 400)

    def test_blank_user_id_returns_400(self):
        response = self.client.get("/api/recommendations/%20?k=5")
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
