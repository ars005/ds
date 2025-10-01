#AIM: Use an appropriate dataset and create a supervised learning model, Analyse the model with ROC-AUC.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc
)

# ==========================
# 1. Breast Cancer Dataset
# ==========================
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=23, stratify=y
)

tree1 = DecisionTreeClassifier(random_state=23)
tree1.fit(X_train, y_train)

y_pred1 = tree1.predict(X_test)
y_proba1 = tree1.predict_proba(X_test)[:, 1]

print("=== Breast Cancer Dataset ===")
print("Accuracy :", accuracy_score(y_test, y_pred1))
print("Precision:", precision_score(y_test, y_pred1))
print("Recall   :", recall_score(y_test, y_pred1))
print("F1-score :", f1_score(y_test, y_pred1))

cm1 = confusion_matrix(y_test, y_pred1)
sns.heatmap(cm1, annot=True, fmt="g",
            xticklabels=["malignant", "benign"],
            yticklabels=["malignant", "benign"])
plt.title("Confusion Matrix - Breast Cancer Dataset")
plt.show()

fpr1, tpr1, _ = roc_curve(y_test, y_proba1)
roc_auc1 = auc(fpr1, tpr1)


# ==========================
# 2. Your Unbalanced CSV Dataset
# ==========================
# Load from CSV if saved, otherwise create DataFrame directly
data = pd.DataFrame({
    "education": ["bach","mast","diploma","mast","diploma","bach","mast","mast","diploma","mast","bach","mast","mast","mast","bach"],
    "job": [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
})

# Encode categorical features
X2 = pd.get_dummies(data.drop("job", axis=1))
y2 = data["job"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.25, random_state=23, stratify=y2
)

tree2 = DecisionTreeClassifier(random_state=23)
tree2.fit(X2_train, y2_train)

y2_pred = tree2.predict(X2_test)
y2_proba = tree2.predict_proba(X2_test)[:, 1]

print("\n=== Unbalanced CSV Dataset ===")
print("Accuracy :", accuracy_score(y2_test, y2_pred))
print("Precision:", precision_score(y2_test, y2_pred))
print("Recall   :", recall_score(y2_test, y2_pred))
print("F1-score :", f1_score(y2_test, y2_pred))

cm2 = confusion_matrix(y2_test, y2_pred)
sns.heatmap(cm2, annot=True, fmt="g")
plt.title("Confusion Matrix - Unbalanced CSV Dataset")
plt.show()

fpr2, tpr2, _ = roc_curve(y2_test, y2_proba)
roc_auc2 = auc(fpr2, tpr2)


# ==========================
# 3. Compare ROC Curves
# ==========================
plt.figure(figsize=(7, 5))
plt.plot(fpr1, tpr1, label=f"Breast Cancer (AUC={roc_auc1:.2f})")
plt.plot(fpr2, tpr2, label=f"Unbalanced CSV (AUC={roc_auc2:.2f})")

plt.plot([0, 1], [0, 1], "r--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend()
plt.grid(True)
plt.show()