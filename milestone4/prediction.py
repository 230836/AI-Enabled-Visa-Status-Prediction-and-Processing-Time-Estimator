"""
Prediction Module
Handles preprocessing + model prediction
"""

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

    # Convert date → month & year
    df['application_date'] = pd.to_datetime(df['application_date'])
    df['Application_Month'] = df['application_date'].dt.month
    df['Application_Year'] = df['application_date'].dt.year

    # Drop original date column
    df.drop(columns=['application_date'], inplace=True)

    # Encode categorical variables (same as training)
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
def predict_processing_time(data):
    """
    Takes dictionary input and returns predicted processing time
    """
    try:
        processed = preprocess_input(data)
        prediction = model.predict(processed)[0]
        return round(prediction, 2)

    except Exception as e:
        raise ValueError(f"Prediction failed: {e}")
