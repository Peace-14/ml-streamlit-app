import streamlit as st
import joblib
import numpy as np

# Load your model
model = joblib.load("fine.pkl")

# Streamlit UI
st.title("ML Model Predictor")

# Example: Suppose model expects two inputs
input1 = st.number_input("Enter feature 1")
input2 = st.number_input("Enter feature 2")

if st.button("Predict"):
    data = np.array([[input1, input2]])  # Adjust shape as needed
    prediction = model.predict(data)
    st.success(f"Prediction: {prediction[0]}")
