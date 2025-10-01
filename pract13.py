#AIM: 4.2 To implement classification using Naïve Bayes algorithm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv(r"C:\Users\91892\Downloads\loan.csv")

# View the first few rows
print(data.head())

# Drop rows with missing values (you can also choose to impute)
data.dropna(inplace=True)

# Encode categorical variables directly in original DataFrame to avoid SettingWithCopyWarning
le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed']:
 data.loc[:, col] = le.fit_transform(data[col])

# Encode target variable
data['Loan_Status'] = le. fit_transform(data[ 'Loan_Status' ])

# Select features and target after
X = data[['Gender', 'Married', 'Education', 'Self_Employed', 'ApplicantIncome', 'LoanAmount' ]]
y = data['Loan_Status' ]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)

# Initialize and train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

#Predict on test set
y_pred = model.predict(X_test)

#Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Create a sample DataFrame with the correct feature names and order
sample = pd.DataFrame ({
'Gender': [1],  # Male encoded as 1
'Married': [1],   # Yes encoded as 1
'Education': [1], # Graduate encoded as 1
'Self_Employed': [0],   # No encoded as 0
'ApplicantIncome': [5000],
'LoanAmount': [128]
})


# Predict the class for the unknown sample or evidence
predicted_class = model.predict(sample)

# Map prediction back to label
loan_status_map = {0: 'N', 1: 'Y'}
print(f"Predicted Loan Status: {loan_status_map[predicted_class[0]]}")