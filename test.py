import unittest
import json
import time
from app import app


class TestCropPrediction(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def log_test_results(self, test_name, results):
        """Helper function to log quantitative results to a JSON file."""
        with open("test_results.json", "a") as file:
            file.write(json.dumps({test_name: results}, indent=4) + "\n")

    def test_valid_input(self):
        """Test API with valid input."""
        data = {
            "PH": 6.5,
            "N": 20,
            "P": 10,
            "K": 15,
            "ORG": 3,
            "HUM": 15,
            "REGION_MARMARA": 1
        }
        start_time = time.time()
        response = self.app.post('/predict', data=json.dumps(data), content_type='application/json')
        elapsed_time = time.time() - start_time
        self.assertEqual(response.status_code, 200)
        response_data = response.get_json()
        self.assertIn('predicted_crops', response_data)

        # Log results
        self.log_test_results("test_valid_input", {
            "response_time": elapsed_time,
            "predicted_crops": response_data.get('predicted_crops', [])
        })

    def test_missing_required_parameter(self):
        """Test API with missing required parameters."""
        data = {
            "PH": 6.5,
            "N": 20,
            "P": 10,
            "K": 15,
            "ORG": 3
        }
        response = self.app.post('/predict', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 500)
        response_data = response.get_json()
        self.assertIn('error', response_data)

   
        """Test model accuracy with predefined test cases."""
        test_data = [
            {"input": {"PH": 6.5, "N": 20, "P": 10, "K": 15, "ORG": 3, "HUM": 15, "REGION_MARMARA": 1},
             "expected": ["wheat", "barley"]},  # Replace with actual expected crops
            {"input": {"PH": 7.2, "N": 40, "P": 20, "K": 25, "ORG": 4, "HUM": 30, "REGION_AEGEA": 1},
             "expected": ["corn"]}  # Replace with actual expected crops
        ]
        correct_predictions = 0
        total_predictions = len(test_data)

        for item in test_data:
            response = self.app.post('/predict', data=json.dumps(item["input"]), content_type='application/json')
            self.assertEqual(response.status_code, 200)
            response_data = response.get_json()

            # Debugging: Print the structure of 'response_data'
            print("Response Data:", response_data)

            # Extract crop names from the dictionaries in 'predicted_crops'
            predicted_crops = [
                crop.get('name') for crop in response_data.get('predicted_crops', []) if 'name' in crop
            ]

            if set(predicted_crops) == set(item["expected"]):
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions

        # Log accuracy
        self.log_test_results("test_accuracy", {"accuracy": accuracy})

    def test_response_time(self):
        """Measure average API response time."""
        data = {
            "PH": 6.5,
            "N": 20,
            "P": 10,
            "K": 15,
            "ORG": 3,
            "HUM": 15,
            "REGION_MARMARA": 1
        }
        response_times = []

        for _ in range(10):  # Perform multiple requests to measure average response time
            start_time = time.time()
            response = self.app.post('/predict', data=json.dumps(data), content_type='application/json')
            elapsed_time = time.time() - start_time
            self.assertEqual(response.status_code, 200)
            response_times.append(elapsed_time)

        avg_response_time = sum(response_times) / len(response_times)
        # Log response time
        self.log_test_results("test_response_time", {"average_time": avg_response_time})

    def test_stress_test(self):
        """Perform stress test by sending multiple concurrent requests."""
        data = {
            "PH": 6.5,
            "N": 20,
            "P": 10,
            "K": 15,
            "ORG": 3,
            "HUM": 15,
            "REGION_MARMARA": 1
        }
        responses = [self.app.post('/predict', data=json.dumps(data), content_type='application/json') for _ in range(100)]
        success_count = sum(1 for response in responses if response.status_code == 200)

        # Log stress test results
        self.log_test_results("test_stress_test", {"success_count": success_count, "total_requests": 100})


if __name__ == '__main__':
    unittest.main()
