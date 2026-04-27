import joblib
import pandas as pd

# Load everything
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# Build input
sample_dict = {feature: 0 for feature in features}

sample_dict['age'] = 70
sample_dict['hypertension'] = 1
sample_dict['heart_disease'] = 1
sample_dict['avg_glucose_level'] = 250
sample_dict['bmi'] = 38

sample = pd.DataFrame([sample_dict])

# Scale
sample_scaled = scaler.transform(sample)

# Predict
probability = rf_model.predict_proba(sample_scaled)[0][1]

# Apply custom threshold (0.3)
prediction = (probability > 0.3).astype(int)

print("Prediction:", prediction)

if prediction == 1:
    print("This patient has high Risk of a Stroke")
else:
    print("This patient is Low Risk")

print("Stroke Probability:", probability)