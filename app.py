import streamlit as st
import joblib
import numpy as np

# Load model, scaler, and label encoder
model = joblib.load("student_performance_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("🎓 Student Performance Predictor")
st.write("Enter student details to predict performance")

# User inputs
studytime = st.slider("Study Time (1-4)", 1, 4)
failures = st.slider("Failures (0-4)", 0, 4)
absences = st.slider("Absences", 0, 50)

if st.button("Predict"):
    input_data = np.array([[studytime, failures, absences]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    result = label_encoder.inverse_transform(prediction)[0]

    st.success(f"Predicted Performance: {result}")