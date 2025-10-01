#AIM: 3.1 To perform regression analysis using single linear regression.
import pandas as pd
import matplotlib.pyplot as mtp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#defining data
data_set = pd.read_csv(r"C:\Users\91892\Downloads\Salary_Data.csv")
x = data_set.iloc[:, :-1].values #independent variable(experience)
y = data_set.iloc[:, 1].values #depedent variable (salary)

#splitting the dataset into training and test set
(x_train, x_test, y_train, y_test)= train_test_split(x,y,test_size= 1/3, random_state=0)

#fitting the simple linear regression model to the training dataset
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#prediction of test and training set result
#y_pred = regressor.predict(x_test) #using test data to predict y(salary)
#x_pred = regressor.predict(x_train) #training (x) is used to predict training (y)
y_pred_train = regressor.predict(x_train)

#visualizing training set result
mtp.scatter(x_train, y_train, color="green", label="Actual Salary (training)")  #actual training points
mtp.plot(x_train, y_pred_train, color="red", label="Regression Line")          #regression  line
mtp.title("Salary vs experince (training dataset)")
mtp.xlabel("years os experience")
mtp.ylabel("Salary (In rupees)")
mtp.show()

#predict test data
y_pred_test = regressor.predict(x_test)

#Find accuracy (R2 score)
accuracy = r2_score(y_test, y_pred_test)
print(f"Test Data Accuracy (R2 Score): {accuracy:.2f}")

#visualizing test set result
mtp.scatter(x_test, y_test, color="blue", label="Actual Salary")  #actual test points
mtp.scatter(x_test, y_pred_test, color="red", label="Predicted Salary")
mtp.plot(x_train, regressor.predict(x_train), color="green", label="Regression Line")      #same regression line
mtp.title("Salary vs experince (training dataset)")
mtp.xlabel("years os experience")
mtp.ylabel("Salary (In rupees)")
mtp.legend()
mtp.show()



