# AI Enabled Visa Status Prediction and Processing Time Estimator

## 📌 Project Overview

Visa applicants often face uncertainty regarding application outcomes and processing times.  
This project aims to build a data-driven system that:

- Predicts visa approval status
- Estimates visa processing time
- Analyzes processing trends across locations

The project uses historical H1B visa application data to perform preprocessing, exploratory data analysis (EDA), and modeling preparation.

---

## 🎯 Objectives

- Estimate visa processing time in days and months
- Analyze approval vs rejection trends
- Study processing patterns by location (Processing Office)
- Prepare dataset for predictive modeling

---

## 📊 Dataset Information

Dataset Used: H1B Visa Applications Dataset  
Total Records Used: 20,000 (sampled and cleaned)

### Key Features:

- `Application_Date`
- `Decision_Date`
- `Processing_Days`
- `Processing_Months`
- `Visa_Status`
- `Processing_Office`
- `Visa_Type` (H1B)

---

## 🛠 Data Preprocessing Steps

1. Loaded raw dataset (85k+ records)
2. Converted date columns to datetime format
3. Removed rows with missing or invalid dates
4. Calculated Processing_Days
5. Calculated Processing_Months
6. Removed negative processing durations
7. Selected first 20,000 clean records
8. Renamed columns for project clarity

---

## 📈 Key Derived Features

- **Processing_Days** = Decision_Date − Application_Date
- **Processing_Months** = Rounded(Processing_Days / 30)
- **Processing_Office** derived from employer_state

---

---

## 🔍 Tools & Technologies Used

- Python
- Pandas
- NumPy
- Google Colab
- GitHub

---

## 🚀 Future Enhancements

- Build regression model to predict processing time
- Build classification model to predict visa approval
- Develop web-based estimator using Flask or Streamlit
- Deploy model to cloud platform

---

## 📌 Conclusion

This project demonstrates how historical visa data can be cleaned, structured, and transformed into meaningful insights for processing time estimation and approval trend analysis.

The dataset is now ready for advanced machine learning modeling and deployment.
