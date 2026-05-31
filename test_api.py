import unittest

from fastapi.testclient import TestClient

from api import app


class BestsellerApiSmokeTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])

    def test_model_info_endpoint(self):
        response = self.client.get("/v1/model-info")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["model_name"], "gradient_boosting")
        self.assertIn("feature_columns", data)

    def test_predict_endpoint(self):
        payload = {
            "title": "Fourth Wing",
            "author": "Rebecca Yarros",
            "publisher": "Red Tower Books",
            "publish_year": 2023,
            "page_count": 528,
            "ol_edition_count": 1,
            "ol_subjects": ["Fantasy", "Romance", "Dragons"],
            "ol_ebook_access": "no_ebook",
            "ol_languages": ["eng"],
            "ol_first_publish_year": 2023,
        }

        response = self.client.post("/v1/predict", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data["bestseller_probability"], 0)
        self.assertLessEqual(data["bestseller_probability"], 1)
        self.assertIn(data["prediction"], [0, 1])
        self.assertIn(data["label"], ["likely_bestseller", "unlikely_bestseller"])
        self.assertEqual(data["model_name"], "gradient_boosting")


if __name__ == "__main__":
    unittest.main()
