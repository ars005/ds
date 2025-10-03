# Practical 7: To explore the given data and identify the patterns in it

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# You can replace this with any CSV dataset, e.g. "data.csv"
# For example, we'll use the famous Iris dataset
df = sns.load_dataset("iris")

# Display first few rows
print("First 5 Rows of the Dataset:")
print(df.head())

# Basic info about the dataset
print("\nDataset Information:")
print(df.info())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Check data types and unique values
print("\nUnique Values per Column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# Distribution of numeric features
df.hist(figsize=(10, 6), color="skyblue", edgecolor="black")
plt.suptitle("Feature Distributions", fontsize=14)
plt.show()

# Pairplot to visualize relationships between variables
sns.pairplot(df, hue="species")
plt.suptitle("Pairwise Relationships", y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Boxplots to detect outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title("Boxplot to Detect Outliers")
plt.show()

# Grouped statistics
print("\nAverage feature values grouped by species:")
print(df.groupby("species").mean())

print(
    "\n✅ Data exploration complete. Patterns identified through correlations, group means, and visual analysis."
)
