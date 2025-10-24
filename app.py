from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import os
import numpy as np
import json


app = Flask(__name__)

# Belirli bir origin listesi ile CORS'u yapılandırıyoruz
allowed_origins = [
    "http://localhost:5008",  # Örneğin bir frontend uygulaması
]

CORS(app, resources={r"/*": {"origins": allowed_origins}})

# Model, LabelEncoder ve Scaler'ı yükledik
model_path = os.path.join(os.path.dirname(__file__), 'random_forest_model.pkl')
label_encoder_path = os.path.join(os.path.dirname(__file__), 'label_encoder.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

model = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)
scaler = joblib.load(scaler_path)

# Eğitim sırasında elde edilen doğruluk skorunu burada belirttik
average_accuracy = 0.720716566728697  

# Sütunlar eğitim verisetindeki sütunlarla aynı isimde
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
    # input doğrulama
    if float(data['PH']) < 0 or float(data['PH']) > 14:
        raise ValueError("PH value must be between 0 and 14.")
    if float(data['N']) < 0 or float(data['P']) < 0 or float(data['K']) < 0:
        raise ValueError("Nutrient values cannot be negative.")
    if float(data['ORG']) < 0 or float(data['HUM']) < 0:
        raise ValueError("Organic matter and humidity cannot be negative.")
    
    # bölge doğrulama
    regions = [
        "REGION_MARMARA", "REGION_AEGEA", "REGION_MEDITERRANEAN",
        "REGION_CENTRAL_ANATOLIA", "REGION_EASTERN_ANATOLIA",
        "REGION_BLACK_SEA", "REGION_SOUTHERN_ANATOLIA"
    ]
    if not any(data.get(region, 0) == 1 for region in regions):
        raise ValueError("At least one valid region must be specified.")
    
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
        
        
        features = prepare_features(data)
        
        # scaling uygulandı
        scaled_features = scaler.transform(features)  # scaler girdilere uygulanıyor
        
        # tahmin yaptığımız kısım
        probabilities = model.predict_proba(scaled_features)[0]
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_crops = label_encoder.inverse_transform(top_3_indices)
        top_3_confidences = probabilities[top_3_indices]
        
        # bitki bilgilerini yüklediğimiz kısım
        crops_info = load_crop_info()
        results = [
            {
                'crop': top_3_crops[i],
                'confidence': round(top_3_confidences[i] * 100, 2),
                'info': crops_info.get(top_3_crops[i], {})
            }
            for i in range(3)
        ]

        return jsonify({
            'predicted_crops': results,
            'model_accuracy': average_accuracy
        })
    
    except ValueError as ve:
        return jsonify({
            'error': 'Invalid input',
            'message': str(ve)
        }), 400
    except Exception as e:
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

# health check endpoint
@app.route('/health')
def health_check():
    try:
        # Kontrol durumlarını belirle
        model_loaded = model is not None
        label_encoder_loaded = label_encoder is not None

        # Eğer herhangi bir bileşen eksikse hata döndür
        if not model_loaded or not label_encoder_loaded:
            return jsonify({
                'status': 'unhealthy',
                'model_loaded': model_loaded,
                'label_encoder_loaded': label_encoder_loaded,
                'error': 'One or more critical components are not loaded'
            }), 500

        # Sağlıklı durum için dönen yanıt
        return jsonify({
            'status': 'healthy',
            'model_loaded': model_loaded,
            'label_encoder_loaded': label_encoder_loaded
        }), 200

    except Exception as e:
        # Beklenmeyen bir hata durumunda yanıt
        return jsonify({
            'status': 'unhealthy',
            'error': 'An unexpected error occurred',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5008)
