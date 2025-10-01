#AIM: 4C
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Sample Dataset
data = {
'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast','Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild','Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal','High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
'Windy': ['False', 'True', 'False', 'False', 'False', 'True', 'True', 'False','False', 'False', 'True', 'True', 'False', 'True'],
'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes' , 'No', 'Yes', 'No', 'Yes','Yes', 'Yes', 'Yes', 'Yes', 'No' ]
}

df = pd.DataFrame(data)
print(data)

# Encode categorical variables
X = pd.get_dummies(df.drop('PlayTennis', axis=1))
y = df ['PlayTennis' ]
feature_names = X.columns

# Function to plot a clean decision tree (only attributes at nodes)
def plot_clean_tree(clf, title): #3 usage
 plt.figure(figsize=(10, 6))
 plot_tree(clf,
 feature_names=feature_names,
 class_names=None,
 filled=False,
 rounded=True,
 impurity=False,
 proportion=False,
 label='none',
 fontsize=12)

 plt.title(title)
 plt. show()

#Gini Index

clf_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
clf_gini.fit(X, y)
plot_clean_tree(clf_gini,"Decision Tree (Gini Index)")

# Information Gain (Entropy)
clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf_entropy.fit(X, y)
plot_clean_tree(clf_entropy,"Decision Tree (Information Gain - Entropy)")

# Gain Ratio

def entropy(col):
 elements, counts = np. unique(col, return_counts=True)
 return -np.sum([(counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts))
 for i in range(len(elements))])

def gain_ratio(df, feature, target="PlayTennis"):
 total_entropy = entropy(df[target])
 vals, counts = np.unique(df[feature], return_counts=True)
 weighted_entropy = np.sum([(counts[i]/np.sum(counts)) *
 entropy(df[df[feature] == vals[i]][target]) for i in range(len(vals))])
 info_gain = total_entropy - weighted_entropy
 split_info = -np.sum([(counts[i]/np.sum(counts)) *
 np.log2(counts[i]/np.sum(counts)) for i in range(len(vals))])
 return info_gain / split_info if split_info != 0 else 0

# Print gain ratios
print("\nGain Ratio values:")
for col in ['Outlook', 'Temperature', 'Humidity', 'Windy']:
  print(f"{col}: {gain_ratio(df, col):.4f}")

plot_clean_tree(clf_entropy,"Decision Tree (Approximated Gain Ratio using Entropy Tree)")