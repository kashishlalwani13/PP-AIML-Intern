# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Reading the dataset
dataset = pd.read_csv(r'C:\Users\kaash\OneDrive\Desktop\KLIMP\Internship\Project planners\task 3\data.csv')

# Preprocessing the dataset
dataset.dropna(inplace=True)
X = dataset.drop(['ID', 'LOCATION', 'DISASTER'], axis=1)
y = dataset['DISASTER']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating an instance of Logistic Regression model
model = LogisticRegression()

# Training the model
model.fit(X_train, y_train)

# Predicting the labels of the test set
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of the model:', accuracy)

# Predicting the label of a new disaster
new_disaster = pd.DataFrame({'MAGNITUDE': 7.0, 'DEPTH': 50.0, 'INTENSITY': 10.0, 'AREA': 10000.0}, index=[0])
predicted_label = model.predict(new_disaster)

print("The predicted disaster is:", predicted_label[0])
