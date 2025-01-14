import unittest
import json
from app import app





class TestCropPrediction(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_valid_input(self):
        data = {
            "PH": 6.5,
            "N": 20,
            "P": 10,
            "K": 15,
            "ORG": 3,
            "HUM": 15,
            "REGION_MARMARA": 1
        }
        response = self.app.post('/predict', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response_data = response.get_json()
        self.assertIn('predicted_crops', response_data)

    def test_missing_required_parameter(self):
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

    def test_invalid_nutrient_values(self):
        # Sending invalid data for nutrient values
        invalid_data = {
            'PH': 7,
            'N': -1,  # Invalid value
            'P': 10,
            'K': 5,
            'ORG': 2.5,
            'HUM': 50,
            'REGION_MARMARA': 1
        }
        response = self.app.post('/predict', data=json.dumps(invalid_data), content_type='application/json')
        self.assertEqual(response.status_code, 400)  # Expect 400 for invalid input

    def test_minimum_edge_case(self):
        data = {
            "PH": 0,
            "N": 0,
            "P": 0,
            "K": 0,
            "ORG": 0,
            "HUM": 0,
            "REGION_AEGEA": 1
        }
        response = self.app.post('/predict', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_maximum_edge_case(self):
        data = {
            "PH": 14,
            "N": 200,
            "P": 200,
            "K": 200,
            "ORG": 20,
            "HUM": 100,
            "REGION_MEDITERRANEAN": 1
        }
        response = self.app.post('/predict', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_valid_crop_info_retrieval(self):
        data = {
            "crop_name": "wheat"
        }
        response = self.app.post('/get_crop_info', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response_data = response.get_json()
        self.assertIn('info', response_data)

    def test_invalid_region_value(self):
        # Sending invalid region value
        invalid_data = {
            'PH': 7,
            'N': 3,
            'P': 10,
            'K': 5,
            'ORG': 2.5,
            'HUM': 50,
            'REGION_MARMARA': 0  # Invalid region
        }
        response = self.app.post('/predict', data=json.dumps(invalid_data), content_type='application/json')
        self.assertEqual(response.status_code, 400)  # Expect 400 for invalid region

    def test_stress_test(self):
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
        for response in responses:
            self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
