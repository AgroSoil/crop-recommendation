from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import os
import numpy as np
import json


app = Flask(__name__)
CORS(app)

# Model ve LabelEncoder'ı yükledik
model_path = os.path.join(os.path.dirname(__file__), 'random_forest_model.pkl')
label_encoder_path = os.path.join(os.path.dirname(__file__), 'label_encoder.pkl')

model = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)

# Eğitim sırasında elde edilen doğruluk skorunu burada belirttik
average_accuracy = 0.720716566728697  

# Feature columns in the same order as training data
FEATURE_COLUMNS = [
    "PH", "N(mg/kg)", "P(mg/kg)", "K(mg/kg)", "ORG(%)", "HUM(%)",
    "REGION_MARMARA", "REGION_AEGEA", "REGION_MEDITERRANEAN",
    "REGION_CENTRAL_ANATOLIA", "REGION_EASTERN_ANATOLIA",
    "REGION_BLACK_SEA", "REGION_SOUTHERN_ANATOLIA"
]

# Bitki yetiştirme bilgilerini JSON dosyasından okuma kısmı
def load_crop_info():
    crop_info_path = os.path.join(os.path.dirname(__file__), 'crops_info.json')
    try:
        with open(crop_info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Dosya bulunamadı: {crop_info_path}")
        return {}
    except json.JSONDecodeError:
        print("JSON dosyası parse edilemedi")
        return {}

def prepare_features(data):
    """
    Prepare features in the correct order for model prediction
    """
    features = [
        float(data['PH']),
        float(data['N']),
        float(data['P']),
        float(data['K']),
        float(data['ORG']),
        float(data['HUM']),
        int(data.get('REGION_MARMARA', 0)),
        int(data.get('REGION_AEGEA', 0)),
        int(data.get('REGION_MEDITERRANEAN', 0)),
        int(data.get('REGION_CENTRAL_ANATOLIA', 0)),
        int(data.get('REGION_EASTERN_ANATOLIA', 0)),
        int(data.get('REGION_BLACK_SEA', 0)),
        int(data.get('REGION_SOUTHERN_ANATOLIA', 0))
    ]
    return np.array(features).reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)  # Debug print
        
        # Prepare features in the correct order
        try:
            features = prepare_features(data)
            print("Prepared features:", features)  # Debug print
        except Exception as e:
            print("Error in prepare_features:", str(e))
            raise

        # Verify feature shape
        print("Feature shape:", features.shape)  # Debug print

        # Make prediction
        try:
            probabilities = model.predict_proba(features)[0]
            print("Prediction probabilities:", probabilities)  # Debug print
        except Exception as e:
            print("Error in predict_proba:", str(e))
            raise

        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_crops = label_encoder.inverse_transform(top_3_indices)
        top_3_confidences = probabilities[top_3_indices]

        # Debug prints
        print("Top 3 indices:", top_3_indices)
        print("Top 3 crops:", top_3_crops)
        print("Top 3 confidences:", top_3_confidences)

        # Load crop info
        try:
            crops_info = load_crop_info()
        except Exception as e:
            print("Error loading crop info:", str(e))
            crops_info = {}

        results = []
        for i in range(3):
            crop = top_3_crops[i]
            crop_info = crops_info.get(crop, {})
            result = {
                'crop': crop,
                'confidence': round(top_3_confidences[i] * 100, 2),
                'info': crop_info
            }
            results.append(result)

        return jsonify({
            'predicted_crops': results,
            'model_accuracy': average_accuracy
        })
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/get_crop_info', methods=['POST'])
def get_crop_info():
    data = request.get_json()
    crop_name = data.get('crop_name')

    crops_info = load_crop_info()

    crop_info = crops_info.get(crop_name, {})
    print("Alınan bitki bilgisi:", crop_info)

    return jsonify({
        'info': crop_info
    })

@app.route('/')
def index():
    return render_template('index.html')

# Add a health check endpoint
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'label_encoder_loaded': label_encoder is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5008)