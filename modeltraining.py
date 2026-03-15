# ==========================================================
# 🚀 MILESTONE 3: PREDICTIVE MODELING
# Visa Processing Time Prediction
# ==========================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

print("Milestone 3 Started 🚀")

# ==========================================================
# 1️⃣ LOAD DATASET
# ==========================================================

df = pd.read_csv("Final_Dataset.csv")

print("Dataset Shape:", df.shape)

# Convert date columns
df["Application_Date"] = pd.to_datetime(df["Application_Date"])
df["Decision_Date"] = pd.to_datetime(df["Decision_Date"])

# ==========================================================
# 2️⃣ FEATURE ENGINEERING
# ==========================================================

df["Application_Year"] = df["Application_Date"].dt.year
df["Application_Month"] = df["Application_Date"].dt.month
df["Application_Quarter"] = df["Application_Date"].dt.quarter

# ==========================================================
# 3️⃣ SELECT FEATURES
# ==========================================================

features = [
    "Application_Year",
    "Application_Month",
    "Application_Quarter",
    "Processing_Office",
    "Visa_Status"
]

target = "Processing_Days"

X = df[features]
y = df[target]

# ==========================================================
# 4️⃣ ENCODE CATEGORICAL FEATURES
# ==========================================================

label_encoders = {}

for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# ==========================================================
# 5️⃣ TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ==========================================================
# 6️⃣ MODEL 1: LINEAR REGRESSION
# ==========================================================

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

# ==========================================================
# 7️⃣ MODEL 2: RANDOM FOREST
# ==========================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

# ==========================================================
# 8️⃣ MODEL 3: GRADIENT BOOSTING
# ==========================================================

gb_model = GradientBoostingRegressor()

gb_model.fit(X_train, y_train)

gb_pred = gb_model.predict(X_test)

# ==========================================================
# 9️⃣ MODEL EVALUATION FUNCTION
# ==========================================================

def evaluate_model(name, y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name} Results")
    print("---------------------")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)

# ==========================================================
# 🔟 EVALUATE ALL MODELS
# ==========================================================

evaluate_model("Linear Regression", y_test, lr_pred)
evaluate_model("Random Forest", y_test, rf_pred)
evaluate_model("Gradient Boosting", y_test, gb_pred)

print("\nMilestone 3 Completed ✅")
