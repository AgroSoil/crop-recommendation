from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib

# dataset yükleniyor
df = pd.read_csv("/home/ozgul/Desktop/crop_dataset.csv")

# features: PH, N, P, K, ORG, HUM, and regional sütunları
feature_columns = ["PH", "N(mg/kg)", "P(mg/kg)", "K(mg/kg)", "ORG(%)", "HUM(%)", 
                   "REGION_MARMARA", "REGION_AEGEA", "REGION_MEDITERRANEAN", 
                   "REGION_CENTRAL_ANATOLIA", "REGION_EASTERN_ANATOLIA", 
                   "REGION_BLACK_SEA", "REGION_SOUTHERN_ANATOLIA"]

# features:x, target: y
x = df[feature_columns]
y = df["CROP"].values.ravel()

# hedef değişken encode ediliyor
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# SMOTE-> veriseti balance
smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x, y_encoded)

# veriyi eğitim ve test olarak ayırdık
x_train, x_test, y_train, y_test = train_test_split(
    x_resampled, y_resampled, test_size=0.22, random_state=42, stratify=y_resampled)

# model için en iyi parametreler belirlendi
best_params = {
    'bootstrap': False,
    'max_depth': 14,
    'max_features': 'log2',
    'min_samples_leaf': 2,
    'min_samples_split': 5,
    'n_estimators': 322
}

# modeli oluşturup eğitme kısmı
model = RandomForestClassifier(**best_params, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)  # fit and transform
x_test_scaled = scaler.transform(x_test)  # transform the test data

model.fit(x_train_scaled, y_train)

# model değerlendirme
test_score = model.score(x_test_scaled, y_test)
print("Test set accuracy score: ", test_score)

# cross-validation scores
scores = cross_val_score(model, x_resampled, y_resampled, cv=5)
print("Cross-validation scores:", scores)
print("Average accuracy score:", scores.mean())

# model ve scaler kaydediyoruz
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# örnek tahmin
new_sample = [[6.2, 33, 28, 290, 2.9, 71, 1, 0, 0, 0, 0, 0, 0]]
new_sample_df = pd.DataFrame(new_sample, columns=feature_columns)
new_sample_scaled = scaler.transform(new_sample_df)  # yeni değere scaling uygulanıyor

predicted_label = model.predict(new_sample_scaled)


#tahin edilen label orijinal ismine çevrilir
predicted_crop = label_encoder.inverse_transform(predicted_label)
print("Predicted crop: ", predicted_crop[0])
