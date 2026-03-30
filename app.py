import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("../student_performance_model.pkl")
scaler = joblib.load("../scaler.pkl")

st.title("🎓 Student Performance Predictor")
st.write("Enter student details to predict performance")

# User inputs
studytime = st.slider("Study Time (1-4)", 1, 4)
failures = st.slider("Failures (0-4)", 0, 4)
absences = st.slider("Absences", 0, 50)

# Predict button
if st.button("Predict"):

    # create 30 feature list
    features = [0]*30

    # put user inputs in first positions
    features[0] = studytime
    features[1] = failures
    features[2] = absences

    input_data = np.array([features])

    # scale
    input_scaled = scaler.transform(input_data)

    # prediction
    prediction = model.predict(input_scaled)

    if prediction[0] == 0:
        result = "Low"
    elif prediction[0] == 1:
        result = "Medium"
    else:
        result = "High"

    st.success(f"🎯 Predicted Performance: {result}")