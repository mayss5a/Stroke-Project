import joblib
import pandas as pd

# Load everything
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# Build input
sample_dict = {feature: 0 for feature in features}

sample_dict['age'] = 67
sample_dict['hypertension'] = 1
sample_dict['heart_disease'] = 0
sample_dict['avg_glucose_level'] = 228.69
sample_dict['bmi'] = 36.6

sample = pd.DataFrame([sample_dict])

# Scale
sample_scaled = scaler.transform(sample)

# Predict
prediction = rf_model.predict(sample_scaled)
probability = rf_model.predict_proba(sample_scaled)[0][1]

print("Prediction:", prediction)

if prediction[0] == 1:
    print(" High Risk of Stroke")
else:
    print(" Low Risk")

print("Stroke Probability:", probability)