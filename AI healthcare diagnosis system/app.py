import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model and label encoder
with open("disease_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Automatically load feature names from the trained model
symptoms = list(model.feature_names_in_)  # Ensures the same 132 symptoms are used

# Streamlit UI
st.set_page_config(layout="wide")

st.sidebar.title("🩺 Select Your Symptoms")
st.sidebar.write("Choose symptoms from the list below:")

# User selects symptoms in sidebar
selected_symptoms = st.sidebar.multiselect("Symptoms:", symptoms)

# Main content area
st.title("AI Healthcare Diagnosis System")

# Display a sample medical image
st.image("sample.png", caption="AI-Based Diagnosis", use_column_width=True)

if st.sidebar.button("Predict Disease"):
    # Prepare input data with all 132 symptoms
    user_symptoms = np.zeros(len(symptoms))
    
    # Ensure selected symptoms are correctly marked as 1
    for symptom in selected_symptoms:
        if symptom in symptoms:
            user_symptoms[symptoms.index(symptom)] = 1

    # Predict disease
    predicted_index = model.predict([user_symptoms])[0]
    predicted_disease = label_encoder.inverse_transform([predicted_index])[0]

    st.subheader("🎯 Predicted Disease:")
    st.success(f"**{predicted_disease}**")
