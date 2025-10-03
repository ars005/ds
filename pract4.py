# Classification
# 4.1) To implement classification using decision tree induction
# 4.2) To implement classification using Naïve Bayes algorithm
# 4.3) To implement classification using decision tree induction   with various attribute selection methods(Information Gain,   Gini index and Gain ratio)

# Import libraries
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# --------------------------
# Reading the dataset
# --------------------------
df = pd.read_csv("dataset.csv")

# Mapping categorical values
df["Nationality"] = df["Nationality"].map({"UK ": 0, "USA ": 1, "N": 2})
df["Go"] = df["Go"].map({"YES": 1, "NO": 0})

# Features and target
features = ["Age", "Experience", "Rank", "Nationality"]
X = df[features]
y = df["Go"]

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# 4.1) Decision Tree Classifier (default)
# --------------------------
dtree = DecisionTreeClassifier(random_state=42)  # default criterion='gini'
dtree.fit(X_train, y_train)
y_pred_dt = dtree.predict(X_test)

print("Decision Tree (Default Gini) Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

plt.figure(figsize=(12, 8))
plot_tree(dtree, feature_names=features, class_names=["NO", "YES"], filled=True)
plt.show()

# --------------------------
# 4.2) Naïve Bayes Classifier
# --------------------------
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

print("Naïve Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))

# --------------------------
# 4.3) Decision Tree with Various Criteria
# --------------------------

# Using Gini Index
dt_gini = DecisionTreeClassifier(criterion="gini", random_state=42)
dt_gini.fit(X_train, y_train)
y_pred_gini = dt_gini.predict(X_test)
print("Decision Tree (Gini Index) Accuracy:", accuracy_score(y_test, y_pred_gini))

# Using Entropy (Information Gain)
dt_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt_entropy.fit(X_train, y_train)
y_pred_entropy = dt_entropy.predict(X_test)
print(
    "Decision Tree (Information Gain) Accuracy:", accuracy_score(y_test, y_pred_entropy)
)
