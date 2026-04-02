 ==========================================================
# 🚀 MILESTONE 2 - EXPLORATORY DATA ANALYSIS
# Advanced & Original Version
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks")
plt.rcParams["figure.figsize"] = (9,5)

print("Milestone 2 Execution Started 🚀")

# ==========================================================
# 1️⃣ LOAD DATA
# ==========================================================

df = pd.read_csv("Final_Dataset.csv")

df["Application_Date"] = pd.to_datetime(df["Application_Date"])
df["Decision_Date"] = pd.to_datetime(df["Decision_Date"])

print("Dataset Shape:", df.shape)

# ==========================================================
# 2️⃣ CORE STATISTICS
# ==========================================================

print("\nProcessing Days Summary")
print(df["Processing_Days"].describe())

median_val = df["Processing_Days"].median()

# ==========================================================
# 3️⃣ DISTRIBUTION ANALYSIS
# ==========================================================

plt.figure()
sns.histplot(df["Processing_Days"], bins=35, kde=True)
plt.axvline(median_val, linestyle="--", color="red")
plt.title("Distribution of Processing Time (Days)")
plt.xlabel("Processing Days")
plt.ylabel("Frequency")
plt.show()

# ==========================================================
# 4️⃣ STATUS-BASED COMPARISON
# ==========================================================

plt.figure()
sns.violinplot(x="Visa_Status", y="Processing_Days", data=df)
plt.title("Processing Time Spread by Visa Status")
plt.show()

# ==========================================================
# 5️⃣ TIME FEATURE EXTRACTION
# ==========================================================

df["Year"] = df["Application_Date"].dt.year
df["Month"] = df["Application_Date"].dt.month
df["Quarter"] = df["Application_Date"].dt.quarter
df["Week_Number"] = df["Application_Date"].dt.isocalendar().week

print("Time Features Extracted.")

# ==========================================================
# 6️⃣ RESAMPLED WEEKLY TREND (TIME-SERIES STYLE)
# ==========================================================

weekly_trend = df.set_index("Application_Date") \
                 .resample("W")["Processing_Days"] \
                 .mean()

weekly_trend.plot()
plt.title("Weekly Average Processing Trend")
plt.ylabel("Processing Days")
plt.show()

# ==========================================================
# 7️⃣ MONTHLY & QUARTERLY TRENDS
# ==========================================================

monthly_avg = df.groupby("Month")["Processing_Days"].mean()
quarter_avg = df.groupby("Quarter")["Processing_Days"].mean()

plt.figure()
monthly_avg.plot(marker="o")
plt.title("Average Processing Time by Month")
plt.show()

plt.figure()
quarter_avg.plot(kind="bar")
plt.title("Average Processing Time by Quarter")
plt.show()

# ==========================================================
# 8️⃣ STATE-LEVEL PERFORMANCE
# ==========================================================

top_states = df.groupby("Processing_Office")["Processing_Days"] \
               .mean() \
               .sort_values(ascending=False) \
               .head(10)

plt.figure()
top_states.plot(kind="bar")
plt.title("Top 10 States by Avg Processing Time")
plt.ylabel("Processing Days")
plt.show()

# ==========================================================
# 9️⃣ OUTLIER REDUCTION ANALYSIS
# ==========================================================

threshold = df["Processing_Days"].quantile(0.98)
df_trimmed = df[df["Processing_Days"] <= threshold]

plt.figure()
sns.boxplot(x="Visa_Status", y="Processing_Days", data=df_trimmed)
plt.title("Processing Time by Status (Trimmed Data)")
plt.show()

# ==========================================================
# 🔟 ADVANCED FEATURE ENGINEERING
# ==========================================================

# Speed Index
df["Speed_Index"] = df["Processing_Days"] / df["Processing_Days"].mean()

# Processing Category
df["Processing_Level"] = pd.qcut(
    df["Processing_Days"],
    q=3,
    labels=["Low", "Medium", "High"]
)

# State-Level Efficiency Score
state_efficiency = df.groupby("Processing_Office")["Processing_Days"] \
                     .mean()

df["State_Efficiency"] = df["Processing_Office"].map(state_efficiency)

print("Advanced Features Added.")

# ==========================================================
# 1️⃣1️⃣ CORRELATION MATRIX
# ==========================================================

numeric_cols = df.select_dtypes(include=["int64","float64"]).columns

plt.figure(figsize=(8,6))
sns.heatmap(df[numeric_cols].corr(), cmap="viridis", annot=False)
plt.title("Correlation Matrix")
plt.show()

