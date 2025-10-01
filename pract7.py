#AIM: To implement classification algorithm using decision tree
#CODE:
#Decision tree_classification implementaton
import pandas as pd
import sys
from sklearn import tree
from sklearn. tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\91892\Downloads\dataset.csv")

d= {'UK ' : 0, 'USA ' : 1, 'N':2}
df['Nationality']= df['Nationality'].map(d)
d= {'YES' :1, 'NO':0}
df['Go'] = df['Go'].map(d)

features = ['Age','Experience', 'Rank', 'Nationality']

X = df[features]
y = df[ 'Go' ]

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X,y)

tree.plot_tree(dtree, feature_names=features )
plt.savefig("result.png")
plt.show()
# plt.savefig(sys.stdout)
# sys.stdout. flush()
print(dtree.predict([[40,10,7,1]]))

print("[1] means 'GO'")
print("[0] means 'NO'")