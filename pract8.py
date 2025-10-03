# Practical -8
# AIM:8.1 To evaluate our binary classification model using confusion matrix along with precision and recall.
##CODE:
# Import the necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the breast cancer dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Train the Model
tree = DecisionTreeClassifier(random_state=23)
tree.fit(X_train, y_train)

# Prediction
y_pred = tree.predict(X_test)

# compute the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# plot the confusion matrix:
sns.heatmap(
    cm,
    annot=True,
    fmt="g",
    xticklabels=["malignant" "benign"],
    yticklabels=["malignant" "benign"],
)
plt.ylabel("Prediction", fontsize=13)
plt.xlabel("Actual", fontsize=13)
plt.title("Confusion Matrix", fontsize=17)
plt.show()

# Finding precision and recall
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy :", accuracy)
precision = precision_score(y_test, y_pred)
print("Precision :", precision)
recall = recall_score(y_test, y_pred)
print("Recall :", recall)
F2_score = f1_score(y_test, y_pred)
print("Fl-score :", F2_score)

# Practical - 8.2
# AIM: To evaluate multi-class classification model using confusion matrix along with precision and recall.

# Import the necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# Load the dataset (Iris dataset is multi-class)
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=23
)

# Train the Model
model = DecisionTreeClassifier(random_state=23)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Compute the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the Confusion Matrix
sns.heatmap(
    cm,
    annot=True,
    fmt="g",
    xticklabels=["Setosa", "Versicolor", "Virginica"],
    yticklabels=["Setosa", "Versicolor", "Virginica"],
)
plt.ylabel("Actual", fontsize=13)
plt.xlabel("Predicted", fontsize=13)
plt.title("Confusion Matrix (Multi-Class)", fontsize=17)
plt.show()

# Finding accuracy, precision, recall and F1-score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy :", accuracy)

# For multi-class, use 'macro' or 'weighted' average
precision = precision_score(y_test, y_pred, average="macro")
print("Precision :", precision)

recall = recall_score(y_test, y_pred, average="macro")
print("Recall :", recall)

F1_score = f1_score(y_test, y_pred, average="macro")
print("F1-score :", F1_score)

# Optional: Detailed report
print("\nClassification Report:\n")
print(
    classification_report(
        y_test, y_pred, target_names=["Setosa", "Versicolor", "Virginica"]
    )
)
