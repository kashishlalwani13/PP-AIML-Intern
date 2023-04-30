#projectplanners-aiml intern 
#using ARIMA Model to predict future of software engineers 
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Load historical data on the number of software engineers
data = pd.read_csv('softwareengineers.csv', index_col='Year', parse_dates=['Year'])

# Train an ARIMA model on the historical data
model = ARIMA(data, order=(1, 1, 1))
model_fit = model.fit()

# Make predictions for the next 5 years
future_years = pd.date_range(start='2023', periods=5, freq='Y')
future_predictions = model_fit.predict(start=len(data), end=len(data)+4, typ='levels', dynamic=False)

# Print the predicted number of software engineers for each year
for i in range(len(future_years)):
    year = future_years[i].year
    prediction = int(round(future_predictions[i]))
    print(f"In {year}, the predicted number of software engineers is {prediction}")
