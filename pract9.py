#AIM: 8.1 To evaluate binary classification model using confusion matrix along with precision and recall.
#import the necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

#load the breast cancer dataset
X, Y= load_breast_cancer(return_X_y=True)
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.025)

#train the model
tree = DecisionTreeClassifier(random_state=23)
tree.fit(X_train,Y_train)
#preduction
Y_pred = tree.predict(X_test)

#compute the confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
#plot the confusion matrix
sns.heatmap(cm, 
            annot=True,
            fmt='g',
            xticklabels=['malignant','benign'],
            yticklabels=['malilgnant','benign'])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()

#finding precision and recall
#accuracy = accuracy_score(Y_test,Y_pred)
#print("Accuracy:",accuracy)
precision = precision_score(Y_test, Y_pred)
print("Precision:", precision)
recall = recall_score(Y_test, Y_pred)
print("Recall:", recall)
#F1_score = f1_score(Y_test,Y_pred)
#print("f1_score:", F1_score)
