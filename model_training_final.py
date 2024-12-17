
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
# Veriyi yükleme
df = pd.read_csv("/home/ozgul/Desktop/crop_dataset.csv")
x = df[["PH", "N(mg/kg)", "P(mg/kg)", "K(mg/kg)", "ORG(%)", "HUM(%)"]]
y = df[["CROP"]].values.ravel()
# LabelEncoder ile hedef değişkenleri kodla
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)








# Eğitim ve test verisine bölüyoruz (test_size ile %20 test verisi, %80 eğitim verisi ayırıyoruz)
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.22, random_state=42, stratify=y_encoded)

# En iyi parametrelerle model oluşturuyoruz
best_params = {
    'bootstrap': False,
    'max_depth': 14,
    'max_features': 'log2',
    'min_samples_leaf': 2,
    'min_samples_split': 5,
    'n_estimators': 322
}
# Modeli en iyi parametrelerle oluşturuyoruz
model = RandomForestClassifier(**best_params, random_state=42)

# Modeli eğitiyoruz
model.fit(x_train, y_train)









# Test verisi doğruluk skoru
test_score = model.score(x_test, y_test)
print("Test verisi doğruluk skoru: ", test_score)

# Çapraz doğrulama skoru
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x, y_encoded, cv=5)
print("Çapraz doğrulama skorları:", scores)
print("Ortalama doğruluk skoru:", scores.mean())









# Yeni bir örnekle tahmin yapıyoruz
new_sample = [[6.2, 33, 28, 290, 2.9, 71]]  # PH, N, P, K, ORG, HUM değerleri
new_sample_df = pd.DataFrame(new_sample, columns=["PH", "N(mg/kg)", "P(mg/kg)", "K(mg/kg)", "ORG(%)", "HUM(%)"])
predicted_label = model.predict(new_sample_df)
# Tahmin edilen bitkiyi dönüştürüyoruz
predicted_crop = label_encoder.inverse_transform(predicted_label)
print("Tahmin edilen bitki: ", predicted_crop[0])