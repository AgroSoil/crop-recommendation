from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib

# Load the dataset
df = pd.read_csv("/home/ozgul/Desktop/crop_dataset.csv")

# Features: PH, N, P, K, ORG, HUM, and regional columns
feature_columns = ["PH", "N(mg/kg)", "P(mg/kg)", "K(mg/kg)", "ORG(%)", "HUM(%)", 
                   "REGION_MARMARA", "REGION_AEGEA", "REGION_MEDITERRANEAN", 
                   "REGION_CENTRAL_ANATOLIA", "REGION_EASTERN_ANATOLIA", 
                   "REGION_BLACK_SEA", "REGION_SOUTHERN_ANATOLIA"]

# Extract features (X) and target (y)
x = df[feature_columns]
y = df["CROP"].values.ravel()

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x, y_encoded)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x_resampled, y_resampled, test_size=0.22, random_state=42, stratify=y_resampled)

# Define the best parameters for the model
best_params = {
    'bootstrap': False,
    'max_depth': 14,
    'max_features': 'log2',
    'min_samples_leaf': 2,
    'min_samples_split': 5,
    'n_estimators': 322
}

# Create and train the model
model = RandomForestClassifier(**best_params, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)  # Fit and transform the training data
x_test_scaled = scaler.transform(x_test)  # Transform the test data

model.fit(x_train_scaled, y_train)

# Evaluate the model
test_score = model.score(x_test_scaled, y_test)
print("Test set accuracy score: ", test_score)

# Cross-validation scores
scores = cross_val_score(model, x_resampled, y_resampled, cv=5)
print("Cross-validation scores:", scores)
print("Average accuracy score:", scores.mean())

# Save the model and scaler
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Predict on a new sample
new_sample = [[6.2, 33, 28, 290, 2.9, 71, 1, 0, 0, 0, 0, 0, 0]]
new_sample_df = pd.DataFrame(new_sample, columns=feature_columns)
new_sample_scaled = scaler.transform(new_sample_df)  # Apply scaling to the new sample

predicted_label = model.predict(new_sample_scaled)

# Convert the predicted label to the original crop name
predicted_crop = label_encoder.inverse_transform(predicted_label)
print("Predicted crop: ", predicted_crop[0])
