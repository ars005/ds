#AIM : To implement various descriptive statistics methods central tendency, quartile and interquartile
import pandas as pd 
import numpy as np 

# ✅ Ensure all columns have exactly 10 entries
Data = {
    'Student_id': [101, 102, 103, 104, 105, 106, 207, 108, 109, 110],
    'Age': [18, 19, 18, 20, 19, 21, 18, 20, 18, 22],
    'score': [85, 59, 27, 89, 58, 87, 58, 90, 82, 89],
    'Study_hours': [5, 7, 4, 8, 6, 3, 7, 5, 6, 9]
}

df = pd.DataFrame(Data)

print("Original DataFrame:")
print(df)
print("\n")

print("Descriptive Statistics using .describe():")
print(df.describe())
print("\n")

print("Individual Statistical Measures:")

# ✅ Central tendency
print(f"Mean of score: {df['score'].mean():.2f}")
print(f"Median of score: {df['score'].median():.2f}")
print(f"Mode of Age: {df['Age'].mode().tolist()}")

# ✅ Quartiles
print(f"25th percentile (Q1) of score: {df['score'].quantile(0.25):.2f}")
print(f"50th percentile (Q2 / median) of score: {df['score'].quantile(0.50):.2f}")
print(f"75th percentile (Q3) of score: {df['score'].quantile(0.75):.2f}")

# ✅ Interquartile Range
iqr_score = df['score'].quantile(0.75) - df['score'].quantile(0.25)
print(f"Interquartile Range (IQR) of score: {iqr_score:.2f}")