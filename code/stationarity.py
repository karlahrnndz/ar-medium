from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

data = {'original_series': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
df = pd.DataFrame(data)

# Differencing
df['stationary_series'] = df['original_series'].diff(1)

# Seasonal differencing (e.g., for monthly data)
s = 12  # Set the seasonal period to 12
df['stationary_series'] = df['original_series'].diff(s).dropna()

# Log transformation
df['stationary_series'] = np.log(df['original_series'])

# Removing linear trend using linear regression
X = np.arange(len(df)).reshape(-1, 1)
y = df['original_series'].values
model = LinearRegression().fit(X, y)
trend_estimation = model.predict(X)
df['stationary_series'] = y - trend_estimation


# Apply Holt-Winters to remove trend and seasonality
seasonality = 4
model = ExponentialSmoothing(df['original_series'], trend='add', seasonal='add', seasonal_periods=seasonality).fit()
df['stationary_series'] = df['original_series'] - model.fittedvalues
