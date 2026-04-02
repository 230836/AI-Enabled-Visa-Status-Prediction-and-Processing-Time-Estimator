"""
AI Enabled Visa Processing Time Predictor
Streamlit Web Application
"""

import streamlit as st
import joblib
import pandas as pd
from datetime import datetime


model = joblib.load("models/model.pkl")



def preprocess_input(data):
    df = pd.DataFrame([data])

    # Convert date → month
    df['application_date'] = pd.to_datetime(df['application_date'])
    df['month'] = df['application_date'].dt.month

    # Drop original column
    df.drop(columns=['application_date'], inplace=True)

    # Encoding (keep same as training)
    df['country'] = df['country'].map({
        "India": 0,
        "USA": 1,
        "UK": 2
    })

    df['visa_type'] = df['visa_type'].map({
        "Student": 0,
        "Work": 1,
        "Tourist": 2
    })

    return df


def predict_time(data):
    processed = preprocess_input(data)
    prediction = model.predict(processed)[0]
    return round(prediction, 2)


def confidence_range(pred, error=2):
    lower = max(0, pred - error)
    upper = pred + error
    return round(lower, 2), round(upper, 2)

# -------------------------------
# UI Design
# -------------------------------

st.set_page_config(page_title="Visa Predictor", page_icon="🌍", layout="centered")

st.markdown(
    "<h1 style='text-align:center; color:#4CAF50;'>🌍 Visa Processing Time Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

st.write("### Enter Application Details")

# Input Fields
country = st.selectbox("🌎 Country", ["India", "USA", "UK"])
visa_type = st.selectbox("📄 Visa Type", ["Student", "Work", "Tourist"])
application_date = st.date_input("📅 Application Date")

# Predict Button
if st.button("🚀 Predict Processing Time"):

    with st.spinner("Processing..."):

        input_data = {
            "country": country,
            "visa_type": visa_type,
            "application_date": str(application_date)
        }

        try:
            prediction = predict_time(input_data)
            lower, upper = confidence_range(prediction)

            st.success(f"⏱ Estimated Time: **{prediction} days**")

            st.info(f"📊 Expected Range: **{lower} – {upper} days**")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.caption("AI-based prediction system for visa processing time")
