import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Load training and test data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Create copies of the data to preserve original datasets
train = train_data.copy()
test = test_data.copy()

# Display the first five rows of the training data
print("First 5 rows of the training data:")
print(train.head())

# Check for missing values in the training data
print("\nMissing values in the training data:")
print(train.isnull().sum())

# Fill missing values in the 'Age' column with the median age
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)

# Fill missing values in the 'Embarked' column with the mode (most frequent value)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Embarked'].fillna(test['Embarked'].mode()[0], inplace=True)

# Fill missing values in the 'Fare' column in the test data with the median fare
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# Convert 'Sex' column to numerical values: male = 0, female = 1
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

# Perform one-hot encoding on the 'Embarked' column and drop the first to avoid dummy variable trap
train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Embarked'], drop_first=True)

# Drop unnecessary columns
drop_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
train.drop(drop_cols, axis=1, inplace=True)
test_passenger_id = test['PassengerId']  # Save 'PassengerId' for the submission file
test.drop(drop_cols, axis=1, inplace=True)

# Separate features and target variable from training data
X = train.drop('Survived', axis=1)  # Features
y = train['Survived']               # Target variable

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Import the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# Instantiate the model with 100 decision trees
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the validation set
y_pred = model.predict(X_val)

# Evaluate the model's accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_val, y_pred)
print("\nAccuracy of the model: {:.2f}%".format(accuracy * 100))

# Predict on the test data
test_predictions = model.predict(test)

# Create the submission DataFrame
submission = pd.DataFrame({
    'PassengerId': test_passenger_id,
    'Survived': test_predictions
})

# Save the submission file
submission.to_csv('submission.csv', index=False)
print("\nPrediction results have been saved to 'submission.csv'.")
