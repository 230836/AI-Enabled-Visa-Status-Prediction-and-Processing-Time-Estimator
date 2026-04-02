"""
AI Enabled Visa Processing Time Predictor
Streamlit Web Application
"""

import streamlit as st
import joblib
import pandas as pd

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("models/best_model.pkl")

# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Convert date → month
    df['application_date'] = pd.to_datetime(df['application_date'])
    df['Application_Month'] = df['application_date'].dt.month
    df['Application_Year'] = df['application_date'].dt.year

    # Drop original date
    df.drop(columns=['application_date'], inplace=True)

    # Encoding (same as training)
    df['Applicant_Country'] = df['Applicant_Country'].map({
        "India": 0,
        "USA": 1,
        "UK": 2
    })

    df['Visa_Type'] = df['Visa_Type'].map({
        "Student": 0,
        "Work": 1,
        "Tourist": 2
    })

    return df

# -------------------------------
# Prediction Function
# -------------------------------
def predict_time(data):
    processed = preprocess_input(data)
    prediction = model.predict(processed)[0]
    return round(prediction, 2)

# -------------------------------
# Confidence Range
# -------------------------------
def confidence_range(pred):
    error = 5  # You can adjust
    return max(0, pred - error), pred + error

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Visa Predictor", page_icon="🌍")

st.markdown(
    "<h1 style='text-align:center; color:#4CAF50;'>🌍 Visa Processing Time Predictor</h1>",
    unsafe_allow_html=True
)

st.write("### Enter Application Details")

# Inputs
country = st.selectbox("🌎 Applicant Country", ["India", "USA", "UK"])
visa_type = st.selectbox("📄 Visa Type", ["Student", "Work", "Tourist"])
application_date = st.date_input("📅 Application Date")

# Button
if st.button("🚀 Predict Processing Time"):

    input_data = {
        "Applicant_Country": country,
        "Visa_Type": visa_type,
        "application_date": str(application_date)
    }

    try:
        prediction = predict_time(input_data)
        low, high = confidence_range(prediction)

        st.success(f"⏱ Estimated Processing Time: **{prediction} days**")
        st.info(f"📊 Expected Range: **{round(low,2)} – {round(high,2)} days**")

    except Exception as e:
        st.error(f"Error: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built using Machine Learning & Streamlit")
