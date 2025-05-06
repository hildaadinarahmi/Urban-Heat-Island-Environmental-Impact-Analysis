import streamlit as st
import pandas as pd
import joblib
import os

# Application title
st.title("City Temperature Band Classifier")

# Check if the model file is available
model_path = "random_forest_classifier.pkl"
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please ensure this file is in the same folder.")
    st.stop()

# Load the model
model = joblib.load(model_path)

# User input
st.sidebar.header("Environmental Data Input")
def user_input_features():
    temperature = st.sidebar.slider("Temperature (°C)", 20.0, 45.0, 30.0)
    humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 50.0)
    wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 15.0, 5.0)
    pollution = st.sidebar.slider("Pollution (PM2.5)", 0.0, 300.0, 50.0)
    data = {
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "pollution": pollution
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Prediction output
st.subheader("Prediction Results")
categories = ["Low", "Medium", "High"]  # Adjust classification labels
st.write(f"Impact Category: **{categories[prediction[0]]}**")

st.subheader("Probabilities")
proba_df = pd.DataFrame(prediction_proba, columns=categories)
st.write(proba_df)
