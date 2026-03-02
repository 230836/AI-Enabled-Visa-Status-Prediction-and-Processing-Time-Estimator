import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Final_Dataset.csv")

print("Dataset Shape:", df.shape)

print("\nProcessing Days Summary:\n")
print(df["Processing_Days"].describe())

plt.figure(figsize=(8,5))
sns.histplot(df["Processing_Days"], bins=30, kde=True)
plt.title("Distribution of Processing Days")
plt.show()

plt.figure(figsize=(6,4))
df["Visa_Status"].value_counts().plot(kind="bar")
plt.title("Visa Status Distribution")
plt.show()


df["Application_Date"] = pd.to_datetime(df["Application_Date"])
df["Application_Year"] = df["Application_Date"].dt.year
df["Application_Month"] = df["Application_Date"].dt.month

df["Processing_Category"] = pd.cut(
    df["Processing_Days"],
    bins=[0, 60, 120, 1000],
    labels=["Fast", "Medium", "Slow"]
)

approval_map = df.groupby("Processing_Office")["Visa_Status"] \
                  .apply(lambda x: (x == "Approved").mean())

df["State_Approval_Rate"] = df["Processing_Office"].map(approval_map)

plt.figure(figsize=(8,5))
df.groupby("Application_Year")["Processing_Days"].mean().plot(marker="o")
plt.title("Average Processing Days by Year")
plt.show()

corr_matrix = df[["Processing_Days","Application_Month"]].corr()

plt.figure(figsize=(5,4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

