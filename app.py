import pickle
import pandas as pd
import streamlit as st

# load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Try loading scaler (if exists)
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None

def preprocess_and_predict(features):

    # Convert to DataFrame
    input_data = pd.DataFrame([features])

    # Get required columns
    required_columns = model.feature_names_in_

    # Add missing columns
    for col in required_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Arrange columns
    input_data = input_data[required_columns]

    # Apply scaling only if scaler exists
    
        # Prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]
    return prediction[0], probability[0]

# --------------- UI ---------------

st.title("Heart Disease Prediction App")

age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Major Vessels", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])
# Input dictionary
features = {
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}

# Prediction
if st.button("Predict"):
    prediction, probability = preprocess_and_predict(features)

    if prediction == 1:
        st.error(f"HIGH chance of heart disease (Probability: {probability:.2f})")
    else:
        st.success(f"LOW chance of heart disease (Probability: {probability:.2f})")