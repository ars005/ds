# Practical-2
# To implement data cleaning
# 2.1) Removing leading or lagging spaces from a data entry
# 2.2) Removing nonprintable characters from a data entry
# 2.3) Data cleaning: handling missing values, type conversion,   data transformations, removing duplicates.
# 2.4) To detect outliers in the given data.

# Import libraries
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Step 1: Create sample dataset
data = {
    "ID": [1, 2, 3, 4, 5, 5],
    "Name": ["  Alice  ", " Bob", "Charlie  ", "  David", "Eva", "Eva"],
    "Age": [25, np.nan, 30, 45, 60, 60],
    "Salary": [30000, 40000, np.nan, 80000, 120000, 120000],
    "Comment": ["Hello\t", "World\n", "Good\rDay", "Clean Data\x0c", "Nice", "Nice"],
    "Department": ["HR", "IT", "Finance", "IT", np.nan, "IT"],
}
df = pd.DataFrame(data)
print("Original Dataset:\n", df)

# -----------------------------
# 2.1 Removing Leading/Trailing Spaces
# -----------------------------
df["Name"] = df["Name"].str.strip()


# -----------------------------
# 2.2 Removing Non-printable Characters
# -----------------------------
def remove_nonprintable(text):
    return re.sub(r"[^\x20-\x7E]", "", str(text))


df["Comment"] = df["Comment"].apply(remove_nonprintable)

# -----------------------------
# 2.3 Handling Missing Values, Type Conversion, Transformations, Duplicates
# -----------------------------

# Handling Missing Values
df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Salary"].fillna(df["Salary"].median(), inplace=True)
df["Department"].fillna("Unknown", inplace=True)

# Type Conversion
df["ID"] = df["ID"].astype("int64")
df["Salary"] = df["Salary"].astype("float64")

# Data Transformation (log transform Salary)
df["Log_Salary"] = np.log1p(df["Salary"])

# Removing Duplicates
df = df.drop_duplicates()

print("\nCleaned Dataset:\n", df)

# -----------------------------
# 2.4 Outlier Detection
# -----------------------------

# IQR Method for Age
Q1 = df["Age"].quantile(0.25)
Q3 = df["Age"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers_iqr = df[(df["Age"] < lower) | (df["Age"] > upper)]
print("\nOutliers by IQR (Age):\n", outliers_iqr)

# Z-score Method for Salary
z_scores = np.abs(stats.zscore(df["Salary"]))
outliers_z = df[z_scores > 3]
print("\nOutliers by Z-Score (Salary):\n", outliers_z)

# Visualization (Boxplots)
sns.boxplot(x=df["Age"])
plt.title("Outlier Detection (Age)")
plt.show()

sns.boxplot(x=df["Salary"])
plt.title("Outlier Detection (Salary)")
plt.show()
