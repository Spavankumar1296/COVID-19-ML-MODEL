import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Interactive Machine Learning Model Deployment")

# Load the trained model from a fixed file path
MODEL_PATH = "models/covid_decision_tree_.pkl"

try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file not found! Please ensure the file exists in the 'models' directory.")

# Extract feature names if available
try:
    feature_names = model.feature_names_in_
except AttributeError:
    feature_names = [f"Feature {i+1}" for i in range(model.n_features_in_)]

# Get user input for each feature
st.subheader("Enter Feature Values")
user_inputs = {}

for feature in feature_names:
    user_inputs[feature] = st.text_input(f"Enter {feature}")

# Convert input to DataFrame and predict
if st.button("Predict"):
    try:
        # Convert inputs to numerical values
        input_data = pd.DataFrame([user_inputs], dtype=float)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        st.success(f"Prediction: {prediction}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
