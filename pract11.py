# Practical-11
# AIM:bagging and boosting

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import pandas as pd

data = load_breast_cancer()
x = data.data
y = data.target
df = pd.DataFrame(y)
print(df.head())

# splitting data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# initialize model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.3, random_state=42)

# train model
rf.fit(x_train, y_train)
gb.fit(x_train, y_train)

# predict model
y_pred_rf = rf.predict(x_test)
y_pred_gb = gb.predict(x_test)

# Evaluate and print result
print("Random forest (bagging) classification report: ")
print(classification_report(y_test, y_pred_rf))

print("\nGradient boosting (boosting) classification report: ")
print(classification_report(y_test, y_pred_gb))

# AIM:cross validation

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

data = load_breast_cancer()
X = data.data
y = data.target

kf = StratifiedKFold(n_splits=10)

rf = RandomForestClassifier(n_estimators=100, n_jobs=42)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

rf_scores = cross_val_score(rf, X, y, cv=kf, scoring="f1")
gb_scores = cross_val_score(gb, X, y, cv=kf, scoring="f1")

print(
    f"Random forest (Bagging) 10-fold cv f1-score:"
    f"Mean={rf_scores.mean():.4f}:,Std={rf_scores.std():.4f}"
)
print(
    f"Gradient boosting (Boosting) 10-fold cv f1-score:"
    f"Mean={gb_scores.mean():.4f}:,Std={gb_scores.std():.4f}"
)
