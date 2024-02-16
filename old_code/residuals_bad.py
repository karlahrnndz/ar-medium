import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Seasonal Multipliers
seasonal_multipliers = np.array([110, 130, 120, 150, 190, 230, 210, 280, 200, 170, 150, 120])

# Generate 100 timesteps with multiplicative trend and seasonality
x_range = np.arange(1, 101)
x_range = np.array([x**2 for x in x_range])

# Generate time series with multiplicative trend, seasonality, and additive noise
np.random.seed(42)
noise = np.random.normal(0, 100, 100)
time_series = []
for i, val in enumerate(x_range):
    time_series.append(val * seasonal_multipliers[i % 12] + noise[i])

# Convert to DataFrame
df_mm = pd.DataFrame(data={'y': time_series}, index=pd.date_range(start='2010-01-01', freq='MS', periods=100))

# Fit AR(3) model to the time series
ar_model = ARIMA(df_mm['y'], order=(3, 0, 0))
ar_results = ar_model.fit()

# Plot original time series, fitted values, and residuals
plt.figure(figsize=(12, 6))
plt.plot(df_mm, label='Original Time Series', linestyle='--', color='orange')
plt.plot(ar_results.fittedvalues, label='AR(3) Fitted Values', linestyle='--', color='red')
plt.title('Original Time Series and AR(3) Fitted Values')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.savefig('ts_vs_fitted_bad.jpeg', format='jpeg')
plt.show()

# Plot AR(3) model residuals
residuals_ar_3 = ar_results.resid

plt.figure(figsize=(12, 4))
plt.plot(residuals_ar_3, label='AR(3) Residuals', linestyle='--', color='green')
plt.axhline(y=0, color='red', linestyle='--', label='Zero Line')
plt.axhline(y=np.mean(residuals_ar_3), color='blue', linestyle='--', label='Mean Residual')
plt.legend()
plt.title('AR(3) Model Residuals')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.savefig('residuals_bad.jpeg', format='jpeg')
plt.show()
