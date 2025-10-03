# Practical-3
# Regression Analysis
# 3.1) To perform regression analysis using single linear regression.
# 3.2) To perform regression analysis using multiple linear regression.
# 3.3) To perform logistic regression analysis

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


np.random.seed(42)

N_samples = 500
Age = np.random.randint(20, 60, N_samples)
Income = np.random.normal(50000, 15000, N_samples)
Experience = np.random.randint(1, 20, N_samples)
City = np.random.choice(["A", "B", "C", "D"], N_samples)

# Simulate a Continuous Target Variable (for SLR/MLR)
# Target = f(Age, Income, Noise)
Continuous_Target = (
    1500 * Age + 0.5 * Income - 1000 * Experience + np.random.normal(0, 5000, N_samples)
)

# Simulate a Binary Target Variable (for Logistic Regression)
# Probability of 'Purchase' increases with Income and decreases with Age
# We use the logistic function to ensure the probability is between 0 and 1
prob_purchase = 1 / (
    1 + np.exp(-(0.00005 * Income - 0.1 * Age + np.random.normal(0, 1, N_samples)))
)
Binary_Target = (prob_purchase > 0.5).astype(int)  # Convert probability to 0 or 1

data = pd.DataFrame(
    {
        "Age": Age,
        "Income": Income,
        "Experience": Experience,
        "City": City,
        "Continuous_Target": Continuous_Target,
        "Binary_Target": Binary_Target,
    }
)

print("--- Dataset Head (First 5 Rows) ---")
print(data.head())
print("\n" + "=" * 50 + "\n")

# --- 3.1) SINGLE LINEAR REGRESSION (SLR) ---
# Goal: Predict Continuous_Target using only Income

print("--- 3.1) Single Linear Regression (SLR) ---")

# 1. Data Prep: X must be 2D (a DataFrame)
X_slr = data[["Income"]]
y_slr = data["Continuous_Target"]

# 2. Split Data
X_train_slr, X_test_slr, y_train_slr, y_test_slr = train_test_split(
    X_slr, y_slr, test_size=0.3, random_state=42
)

# 3. Model Training
slr_model = LinearRegression()
slr_model.fit(X_train_slr, y_train_slr)

# 4. Prediction
y_pred_slr = slr_model.predict(X_test_slr)

# 5. Evaluation
r2_slr = r2_score(y_test_slr, y_pred_slr)
mse_slr = mean_squared_error(y_test_slr, y_pred_slr)

# 6. Interpretation
print(f"SLR Intercept (b0): {slr_model.intercept_:.2f}")
print(f"SLR Coefficient (b1 for Income): {slr_model.coef_[0]:.4f}")
print(f"SLR R^2 Score (Goodness of Fit): {r2_slr:.4f}")
print(f"SLR Mean Squared Error (MSE): {mse_slr:.2f}")
print(
    "Conclusion: The model explains {:.2f}% of the variance in the target variable.".format(
        r2_slr * 100
    )
)
print("\n" + "=" * 50 + "\n")


# --- 3.2) MULTIPLE LINEAR REGRESSION (MLR) ---
# Goal: Predict Continuous_Target using Age, Income, and Experience

print("--- 3.2) Multiple Linear Regression (MLR) ---")

# 1. Data Prep: Use multiple features
X_mlr = data[
    ["Age", "Income", "Experience"]
]  # NOTE: No categorical features used for simplicity
y_mlr = data["Continuous_Target"]

# 2. Split Data
X_train_mlr, X_test_mlr, y_train_mlr, y_test_mlr = train_test_split(
    X_mlr, y_mlr, test_size=0.3, random_state=42
)

# 3. Model Training
mlr_model = LinearRegression()
mlr_model.fit(X_train_mlr, y_train_mlr)

# 4. Prediction
y_pred_mlr = mlr_model.predict(X_test_mlr)

# 5. Evaluation
r2_mlr = r2_score(y_test_mlr, y_pred_mlr)
mse_mlr = mean_squared_error(y_test_mlr, y_pred_mlr)

# 6. Interpretation
print(f"MLR Intercept (b0): {mlr_model.intercept_:.2f}")
print("MLR Coefficients:")
for feature, coef in zip(X_mlr.columns, mlr_model.coef_):
    print(f"  {feature}: {coef:.4f}")

print(f"\nMLR R^2 Score (Goodness of Fit): {r2_mlr:.4f}")
print(f"MLR Mean Squared Error (MSE): {mse_mlr:.2f}")
print("Conclusion: Adding more features significantly improved the R^2 score.")
print("\n" + "=" * 50 + "\n")


# --- 3.3) LOGISTIC REGRESSION (Classification) ---
# Goal: Predict Binary_Target (Purchase/No Purchase) using Age and Income

print("--- 3.3) Logistic Regression Analysis ---")

# 1. Data Prep
X_log = data[["Age", "Income"]]
y_log = data["Binary_Target"]

# 2. Split Data
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_log, y_log, test_size=0.3, random_state=42
)

# 2.5 Feature Scaling (Crucial for Logistic Regression)
scaler = StandardScaler()
X_train_log_scaled = scaler.fit_transform(X_train_log)
X_test_log_scaled = scaler.transform(X_test_log)

# 3. Model Training
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train_log_scaled, y_train_log)

# 4. Prediction
y_pred_log = logistic_model.predict(X_test_log_scaled)
# Get probabilities (useful for ROC AUC)
y_pred_proba_log = logistic_model.predict_proba(X_test_log_scaled)[
    :, 1
]  # Probability of class 1

# 5. Evaluation (Classification Metrics)
accuracy = accuracy_score(y_test_log, y_pred_log)
conf_matrix = confusion_matrix(y_test_log, y_pred_log)
roc_auc = roc_auc_score(y_test_log, y_pred_proba_log)

# 6. Interpretation
print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("\nConfusion Matrix (True vs Predicted):")
print("          Predicted 0   Predicted 1")
print(f"True 0:    {conf_matrix[0, 0]:<10}    {conf_matrix[0, 1]}")
print(f"True 1:    {conf_matrix[1, 0]:<10}    {conf_matrix[1, 1]}")

print("\nModel Coefficients (Log-Odds):")
for feature, coef in zip(X_log.columns, logistic_model.coef_[0]):
    print(f"  {feature}: {coef:.4f}")

print(
    "\nConclusion: The model predicts Purchase with an accuracy of {:.2f}%. A high ROC AUC score suggests good discriminatory power.".format(
        accuracy * 100
    )
)
