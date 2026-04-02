 ==========================================================
# MILESTONE 3 - MACHINE LEARNING MODELING PIPELINE
# Visa Processing Time Estimation
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

sns.set(style="whitegrid")

print("Milestone 3 Pipeline Started")


# ==========================================================
# 1️⃣ LOAD DATASET
# ==========================================================

df = pd.read_csv("Final_Dataset.csv")

print("Dataset Shape:", df.shape)

df["Application_Date"] = pd.to_datetime(df["Application_Date"])
df["Decision_Date"] = pd.to_datetime(df["Decision_Date"])


# ==========================================================
# 2️⃣ FEATURE ENGINEERING
# ==========================================================

df["Application_Year"] = df["Application_Date"].dt.year
df["Application_Month"] = df["Application_Date"].dt.month
df["Application_Quarter"] = df["Application_Date"].dt.quarter
df["Application_Weekday"] = df["Application_Date"].dt.dayofweek

print("Feature Engineering Completed")


# ==========================================================
# 3️⃣ FEATURE SELECTION
# ==========================================================

features = [
    "Application_Year",
    "Application_Month",
    "Application_Quarter",
    "Application_Weekday",
    "Processing_Office",
    "Visa_Status"
]

target = "Processing_Days"

X = df[features]
y = df[target]


# ==========================================================
# 4️⃣ PREPROCESSING PIPELINE
# ==========================================================

categorical_cols = ["Processing_Office", "Visa_Status"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)


# ==========================================================
# 5️⃣ TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Samples:", X_train.shape)
print("Testing Samples:", X_test.shape)


# ==========================================================
# 6️⃣ MODEL DEFINITIONS
# ==========================================================

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=250,
        max_depth=15,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}


results = []


# ==========================================================
# 7️⃣ MODEL TRAINING + EVALUATION
# ==========================================================

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    cv_score = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error"
    ).mean()

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CV_MAE": abs(cv_score)
    })

    print("\nModel:", name)
    print("MAE :", round(mae,2))
    print("RMSE:", round(rmse,2))
    print("R2  :", round(r2,3))


# ==========================================================
# 8️⃣ MODEL COMPARISON
# ==========================================================

results_df = pd.DataFrame(results)

print("\nModel Comparison Table")
print(results_df.sort_values("MAE"))


# ==========================================================
# 9️⃣ VISUALIZE MODEL PERFORMANCE
# ==========================================================

plt.figure(figsize=(8,5))

sns.barplot(
    data=results_df,
    x="Model",
    y="MAE"
)

plt.title("Model Comparison (Lower MAE is Better)")
plt.xticks(rotation=30)

plt.show()


# ==========================================================
# 🔟 FEATURE IMPORTANCE (RANDOM FOREST)
# ==========================================================

rf_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", RandomForestRegressor(n_estimators=250))
    ]
)

rf_pipeline.fit(X_train, y_train)

model = rf_pipeline.named_steps["model"]

feature_names = rf_pipeline.named_steps["preprocessing"] \
                .get_feature_names_out()

importance = pd.Series(
    model.feature_importances_,
    index=feature_names
).sort_values(ascending=False)[:15]

plt.figure(figsize=(9,6))

sns.barplot(
    x=importance.values,
    y=importance.index
)

plt.title("Top Important Features (Random Forest)")

plt.show()
