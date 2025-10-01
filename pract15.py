# AIM:- boosting and bagging
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report

#load dataset
data = load_breast_cancer()
 
X = data.data
Y = data.target
df=pd.DataFrame(Y)
df['target'] = data.target
print(df.iloc[:, :6].head())  # first 5 features + target column

#print(df.head())
#data1 = pd.DataFrame(data.data, columns= data.feature_names) 
#print(data1)

#split into train and test sets
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2, random_state=42)

#Initialize models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

#train models
rf.fit(X_train, Y_train)
gb.fit(X_train, Y_train)

#predict on test set
Y_pred_rf = rf.predict(X_test)
Y_pred_gb = gb.predict(X_test)

#Evaluate and print results
print("Random Forest (Bagging) Classification Report:")
print(classification_report(Y_test, Y_pred_rf))


print("Gradient Boosting (Boosting) Classification Report:")
print(classification_report(Y_test, Y_pred_gb))
