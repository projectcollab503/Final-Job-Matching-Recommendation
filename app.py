import streamlit as st
import pickle
import numpy as np

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Udaan – Job Recommender", layout="centered")
st.title("🧠 Udaan – Job Recommender")
st.write("Enter your skills, interests, or qualifications and get top job recommendations!")

# User input
user_input = st.text_area("💬 Describe your background:", "")

if st.button("🔍 Get Recommendations"):
    if user_input.strip() == "":
        st.warning("Please enter something to get job recommendations.")
    else:
        # Transform input
        transformed = vectorizer.transform([user_input])
        
        # Predict
        prediction = model.predict(transformed)[0]

        # Output recommendation
        if prediction == 1:
            st.success("🎯 A suitable job **is recommended** based on your profile!")
        else:
            st.error("❌ No job match is currently recommended based on your input.")
