# Crop Recommendation System Based on Soil Analysis 🌱

## Project Overview
Soil analysis plays a crucial role in determining crop productivity. This project utilizes soil properties such as **pH**, **nutrient levels (N, P, K)**, organic matter percentage, and humidity to predict the optimal crop using a **Random Forest Classifier**. Regional suitability for crops across Turkey's seven geographical regions is included to narrow down recommendations effectively.

---

## Features
- **Input Parameters**: Soil pH, Nitrogen, Phosphorus, Potassium, Organic Matter, and Humidity.
- **Regional Suitability**: Incorporates regional data (Marmara, Aegean, Mediterranean, Central Anatolia, Eastern Anatolia, Black Sea, and Southeastern Anatolia).
- **Machine Learning Model**: Optimized Random Forest Classifier with hyperparameter tuning.
- **Prediction Capability**: Provides crop predictions based on soil input and region-specific suitability.
- **User-Friendly Dataset**: Clear labeling and well-organized data structure for ease of use.

---

## Dataset
The dataset includes:

### 1. Soil Analysis Columns:
- **PH**: Soil pH value  
- **N(mg/kg)**: Nitrogen content  
- **P(mg/kg)**: Phosphorus content  
- **K(mg/kg)**: Potassium content  
- **ORG(%)**: Organic matter percentage  
- **HUM(%)**: Humidity percentage  

### 2. Target Variable:
- **CROP**: Crop name (e.g., bugday, arpa, misir, etc.)

### 3. Region Columns:
- **REGION_MARMARA**, **REGION_AEGEA**, **REGION_MEDITERRANEAN**, etc.  
  Binary flags (0 or 1) indicating whether the crop is suitable for the corresponding region.

---

## Technologies Used
- **Python**
- **Scikit-learn** (Machine Learning)
- **Pandas** and **NumPy** (Data Preprocessing)
- **Matplotlib** (Visualization)
- **Git** and **GitHub** (Version Control)

---

## Model Details
- **Algorithm**: Random Forest Classifier
- **Hyperparameters**:
  - `n_estimators`: 322  
  - `max_depth`: 14  
  - `max_features`: 'log2'  
  - `min_samples_split`: 5  
  - `min_samples_leaf`: 2  
  - `bootstrap`: False  
- **Evaluation Metrics**:
  - Accuracy on Test Data  
  - Cross-Validation Scores  

---

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Git
- Libraries: `pandas`, `scikit-learn`, `numpy`, `matplotlib`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-organization/crop-recommendation-system.git
   cd crop-recommendation-system
2. Install dependencies:
   ```bash
    pip install -r requirements.txt
3. Run the project scripts:
   ```bash
    python model_training.py


## Usage

### Prepare Input Data:
- Provide soil parameters (pH, N, P, K, ORG, HUM) and regional information.

### Run the Model:
- Use the trained model to predict suitable crops.

### Example Prediction:
```python
new_sample = [[6.2, 33, 28, 290, 2.9, 71]]  # Soil properties
predicted_label = model.predict(new_sample_df)
print("Recommended Crop: ", label_encoder.inverse_transform(predicted_label)[0])
```

## CONTACT:
For any questions or issues, please contact ozgulbaytekin or sedefelms.
