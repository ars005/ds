# Write a program to implement Cross validation methods

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
 

data = load_breast_cancer()
X= data.data
Y= data.target

#define models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

#stratified K-Fold ensures balanced splits
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

#cross-validation scores
rf_scores = cross_val_score(rf,X, Y, cv=kf, scoring='f1')
gb_scores = cross_val_score(gb,X, Y, cv=kf, scoring='f1')

print(f"Random Forest (Bagging) 10-fold CV F1-score: "
      f"Mean={rf_scores.mean():.4f}, std={rf_scores.std():.4f}")

print(f"Gradient Boosting (Boosting) 10-fold CV F1-score: "
      f"Mean={gb_scores.mean():.4f}, std={gb_scores.std():.4f}")

#data = pd.read_excel(r"C:\Users\91892\Downloads\heart.xlsx")
#print(load_breast_cancer())
#df = pd.DataFrame(data.data, data)
#print (df)