import streamlit as st
import instaloader
import pandas as pd
import joblib
import os

# Load the trained machine learning model
MODEL_PATH = "fake_instagram_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error("⚠️ Model file not found! Train and save 'fake_instagram_model.pkl' first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Function to scrape Instagram user data
def get_instagram_data(username):
    loader = instaloader.Instaloader()
    try:
        profile = instaloader.Profile.from_username(loader.context, username)
        
        data = {
            "followers": profile.followers,
            "following": profile.followees,
            "posts": profile.mediacount,
            "bio_length": len(profile.biography) if profile.biography else 0,
        }
        
        # Calculate additional features
        data["followers_ratio"] = data["followers"] / (data["following"] + 1)  # Avoid division by zero
        data["posts_per_follower"] = data["posts"] / (data["followers"] + 1)
        
        return data
    except Exception as e:
        st.error(f"❌ Failed to fetch data: {str(e)}")
        return None

# Function to predict if the account is fake
def predict_fake_account(data):
    features = pd.DataFrame([data])  # Convert to DataFrame for ML model
    prediction = model.predict(features)[0]  # Get prediction (0=Real, 1=Fake)
    return "Fake Account ❌" if prediction == 1 else "Real Account ✅"

# Streamlit UI
st.title("🔍 Instagram Fake Account Detector")

# User Input: Instagram Username
username = st.text_input("Enter Instagram Username:", "")

if st.button("Check Account"):
    if username:
        st.info("Fetching Instagram data...")
        user_data = get_instagram_data(username)
        
        if user_data:
            st.write("📊 **Extracted Data:**")
            st.json(user_data)  # Show extracted data
            
            st.info("Analyzing with Machine Learning Model...")
            result = predict_fake_account(user_data)
            
            st.success(f"📝 **Prediction:** {result}")
    else:
        st.warning("⚠️ Please enter a valid Instagram username.")
