# Practical-1
# Descriptive Statistics Methods This section focuses on methods for summarizing and describing the main features of a dataset.
# 1.1) Implement central tendency (mean, median, mode), quartile, and interquartile range calculations.
# 1.2) Implement descriptive statistics for univariate, bivariate, and multivariate analysis.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    "Age": [23, 25, 29, 35, 40, 45, 50, 60],
    "Salary": [25000, 30000, 40000, 50000, 60000, 75000, 90000, 120000],
    "Experience": [1, 2, 4, 6, 8, 12, 15, 20],
}
df = pd.DataFrame(data)
print("Dataset:\n", df)

# -----------------------
# 1.1 CENTRAL TENDENCY
# -----------------------
print("\n--- Central Tendency ---")
print("Mean:\n", df.mean())
print("Median:\n", df.median())
print("Mode:\n", df.mode().iloc[0])

# Quartiles & IQR
print("\n--- Quartiles & IQR ---")
print("Q1:\n", df.quantile(0.25))
print("Q2 (Median):\n", df.quantile(0.50))
print("Q3:\n", df.quantile(0.75))
print("IQR:\n", df.quantile(0.75) - df.quantile(0.25))

# -----------------------
# 1.2 UNIVARIATE ANALYSIS
# -----------------------
print("\n--- Univariate Analysis (Age) ---")
print(df["Age"].describe())

plt.hist(df["Age"], bins=5, edgecolor="black")
plt.title("Histogram of Age")
plt.show()

sns.boxplot(x=df["Age"])
plt.title("Boxplot of Age")
plt.show()

# -----------------------
# BIVARIATE ANALYSIS
# -----------------------
print("\n--- Bivariate Analysis (Age vs Salary) ---")
print("Correlation (Age vs Salary):", df["Age"].corr(df["Salary"]))

sns.scatterplot(x="Age", y="Salary", data=df)
plt.title("Scatterplot: Age vs Salary")
plt.show()

# -----------------------
# MULTIVARIATE ANALYSIS
# -----------------------
print("\n--- Multivariate Analysis ---")
print("Correlation Matrix:\n", df.corr())

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Heatmap of Correlation Matrix")
plt.show()

sns.pairplot(df)
plt.show()
