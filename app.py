import streamlit as st
import joblib
import pandas as pd

# Load model files
model = joblib.load("student_performance_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("🎓 Student Performance Predictor")
st.write("Enter student details to predict performance")

# User inputs
studytime = st.slider("Study Time (1-4)", 1, 4, 2)
failures = st.slider("Failures (0-4)", 0, 4, 1)
absences = st.slider("Absences (0-50)", 0, 50, 10)

if st.button("Predict"):
    # Create dataframe with exact feature names
    input_df = pd.DataFrame([{
        "studytime": studytime,
        "failures": failures,
        "absences": absences
    }])

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)
    result = label_encoder.inverse_transform(prediction)[0]

    st.success(f"🎯 Predicted Performance: {result}")