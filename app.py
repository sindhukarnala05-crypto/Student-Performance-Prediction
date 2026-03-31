import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("student_performance_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎓 Student Performance Predictor")
st.write("Enter student details to predict performance")

# User inputs
studytime = st.slider("Study Time (1-4)", 1, 4)
failures = st.slider("Failures (0-4)", 0, 4)
absences = st.slider("Absences", 0, 50)

if st.button("Predict"):
    features = [0]*30
    features[0] = studytime
    features[1] = failures
    features[2] = absences
    input_data = np.array([features])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    labels = ["Low", "Medium", "High"]
    result = labels[prediction[0]]
    st.success(f"Predicted Performance: {result}")