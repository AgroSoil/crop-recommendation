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

# Bitki yetiştirme bilgilerini JSON dosyasından okuma kısmı
def load_crop_info():
    crop_info_path ='/app/crops_info.json' 
    if not os.path.exists(crop_info_path):
        print("Dosya bulunamadı:", crop_info_path)
    with open(crop_info_path) as f:
        return json.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([data['PH'], data['N'], data['P'], data['K'], data['ORG'], data['HUM']]).reshape(1, -1)

    # Tahmin yapıyor ve en yüksek olasılığa sahip ilk 3 bitkiyi seçiyor model
    probabilities = model.predict_proba(features)[0]
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3_crops = label_encoder.inverse_transform(top_3_indices)
    top_3_confidences = probabilities[top_3_indices]

    # Bitki yetiştirme bilgilerini JSON dosyasından alıyoruz
    crops_info = load_crop_info()


    

    # Tahmin sonuçları ve yetiştirme bilgilerini açılan popup'ta gösteriyoruz
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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5008)
