import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn. linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Create a dataset (Heart Disease like) as DataFrame
np.random.seed(42)
n_samples = 500

data = pd.DataFrame({
"Age": np.random.randint( 29,77, n_samples),
"Sex": np. random.randint( 0,2, n_samples), # 0 = female, 1 = male
"Cholesterol": np.random.randint(150, 300, n_samples),
"BloodPressure": np.random.randint( 90, 180, n_samples),
"MaxHeartRate": np.random.randint( 90,200, n_samples)
})

# Target variable (rule-based: high Cholesterol, high BP, or low MaxHR + higher risk)
data["HeartDisease"] =((data["Cholesterol"] > 240) |
                       (data["BloodPressure"] > 140) |
                       (data["MaxHeartRate"] < 120)).astype(int)


print("Sample of Heart Disease Dataset: \n")
print(data.head())

# Step 2: Split features & target
X = data. drop("HeartDisease",axis=1)
y = data["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)

# Step 3: Train Logistic Regression
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Step 4: Predictions & Evaluation
y_pred = model.predict(X_test)

print("\n Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))