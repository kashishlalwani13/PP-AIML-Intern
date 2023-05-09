# Importing necessary libraries
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Reading the dataset
dataset = pd.read_csv(r'C:\Users\kaash\OneDrive\Desktop\KLIMP\Internship\Project planners\task 2\data.csv')

# Analyzing the dataset
print(dataset.head())
print(dataset.describe())

# Visualizing the correlation between the features and the target variable
sns.pairplot(dataset, x_vars=['age', 'runs_scored', 'wickets_taken', 'batting_average', 'bowling_average'], y_vars=['selling_price'], height=7, aspect=0.7, kind='reg')
sns.plt.show()

# Splitting the dataset into training and testing sets
X = dataset.drop('selling_price', axis=1)
y = dataset['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating an instance of Linear Regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Predicting the prices of the test set
y_pred = model.predict(X_test)

# Evaluating the model
r_squared = r2_score(y_test, y_pred)
print('R-squared score:', r_squared)

# Predicting the price of a new player
new_player = pd.DataFrame({'age': 28, 'runs_scored': 1000, 'wickets_taken': 30, 'batting_average': 35, 'bowling_average': 25}, index=[0])
predicted_price = model.predict(new_player)

print("The predicted selling price of the player is:", predicted_price)
