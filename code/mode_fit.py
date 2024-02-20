import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import os


# Function to generate AR(3) time this_series
def generate_ar3_series(n_samples):
    np.random.seed(42)
    ar_params = [0.3, -0.5, 0.2]  # AR(3) coefficients
    noise = np.random.normal(0, 1, n_samples)

    series = np.zeros(n_samples)
    for i in range(3, n_samples):
        series[i] = ar_params[0] * series[i - 1] + ar_params[1] * series[i - 2] + ar_params[2] * series[i - 3] + noise[
            i]

    return series


# Generate AR(3) time this_series
n_samples = 100
ar3_series = generate_ar3_series(n_samples)

# Fit AR(3) model
ar3_model = ARIMA(ar3_series, order=(3, 0, 0))
ar3_results = ar3_model.fit()

# Plot AR(3) time this_series and fitted values
plt.figure(figsize=(12, 6))
plt.plot(ar3_series, label='AR(3) Time Series')
plt.plot(ar3_results.fittedvalues, label='AR(3) Fitted Values', linestyle='--', color='red')
plt.legend()
plt.title('AR(3) Time Series and Fitted Values')
plt.xlabel('t')

plt.savefig(os.path.join('../figures', 'ts_vs_fitted_good.svg'), format='svg')
# plt.savefig('ts_vs_fitted_good.jpeg', format='jpeg')
plt.show()

# Plot AR(3) model residuals
residuals_ar3 = ar3_results.resid


# Plot AR(3) model residuals
plt.figure(figsize=(12, 4))
plt.plot(residuals_ar3, label='AR(3) Residuals', linestyle='--', color='green')
# plt.axhline(y=0, color='red', linestyle='--', label='Zero Line')
plt.axhline(y=np.mean(residuals_ar3), color='blue', linestyle='--', label='Mean Residual')
plt.legend()
plt.title('AR(3) Model Residuals')
plt.xlabel('t')
plt.ylabel('Residuals')
plt.savefig(os.path.join('../figures', 'residuals_good.svg'), format='svg')
# plt.savefig('residuals_good.jpeg', format='jpeg')
plt.show()


# ---
# Residuals bad
# ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Seasonal Multipliers
seasonal_multipliers = np.array([110, 130, 120, 150, 190, 230, 210, 280, 200, 170, 150, 120])

# Generate 100 timesteps with multiplicative trend and seasonality
x_range = np.arange(1, 101)
x_range = np.array([x**2 for x in x_range])

# Generate time this_series with multiplicative trend, seasonality, and additive noise
np.random.seed(42)
noise = np.random.normal(0, 100, 100)
time_series = []
for i, val in enumerate(x_range):
    time_series.append(val * seasonal_multipliers[i % 12] + noise[i])

# Convert to DataFrame
df_mm = pd.DataFrame(data={'y': time_series}, index=pd.date_range(start='2010-01-01', freq='MS', periods=100))

# Fit AR(3) model to the time this_series
ar_model = ARIMA(df_mm['y'], order=(3, 0, 0))
ar_results = ar_model.fit()

# Plot original time this_series, fitted values, and residuals
plt.figure(figsize=(12, 6))
plt.plot(df_mm, label='Original Time Series', linestyle='--', color='orange')
plt.plot(ar_results.fittedvalues, label='AR(3) Fitted Values', linestyle='--', color='red')
plt.title('Original Time Series and AR(3) Fitted Values')
plt.xlabel('t')

plt.legend()
plt.savefig(os.path.join('../figures', 'ts_vs_fitted_bad.svg'), format='svg')
# plt.savefig('ts_vs_fitted_bad.jpeg', format='jpeg')
plt.show()

# Plot AR(3) model residuals
residuals_ar_3 = ar_results.resid

plt.figure(figsize=(12, 4))
plt.plot(residuals_ar_3, label='AR(3) Residuals', linestyle='--', color='green')
# plt.axhline(y=0, color='red', linestyle='--', label='Zero Line')
plt.axhline(y=np.mean(residuals_ar_3), color='blue', linestyle='--', label='Mean Residual')
plt.legend()
plt.title('AR(3) Model Residuals')
plt.xlabel('t')
plt.ylabel('Residuals')
plt.savefig(os.path.join('../figures', 'residuals_bad.svg'), format='svg')
# plt.savefig('residuals_bad.jpeg', format='jpeg')
plt.show()


# ---
# Naive
# ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Seasonal Multipliers
seasonal_multipliers = [110, 130, 120, 150, 190, 230, 210, 280, 200, 170, 150, 120]

# Generate 100 timesteps with multiplicative trend and seasonality
x_range = np.arange(1, 101)
x_range = np.array([x**1.25 for x in x_range])

# Generate time this_series with multiplicative trend, seasonality, and additive noise
np.random.seed(42)
noise = np.random.normal(0, 10, 100)
time_series = []
for i, val in enumerate(x_range):
    time_series.append(val * seasonal_multipliers[i % 12] + noise[i])

# Convert to DataFrame
df_mm = pd.DataFrame(data={'y': time_series}, index=pd.date_range(start='2010-01-01', freq='MS', periods=100))

# Fit AR(3) model to the time this_series
ar_model = ARIMA(df_mm['y'], order=(3, 0, 0))
ar_results = ar_model.fit()

# Fit Naive Model (Persistence Model)
naive_forecast = df_mm['y'].shift(1)

# Plot original time this_series, AR(3) fitted values, Naive Model fitted values, and residuals
plt.figure(figsize=(12, 8))
plt.plot(df_mm, label='Original Time Series', linestyle='-', color='orange')
plt.plot(ar_results.fittedvalues, label='AR(3) Fitted Values', linestyle='--', color='red')
plt.plot(naive_forecast, label='Naive Model Fitted Values', linestyle='--', color='blue')
plt.title('Original Time Series and Model Fitted Values')
plt.xlabel('t')

plt.legend()
plt.savefig(os.path.join('../figures', 'ts_vs_fitted_naive.svg'), format='svg')
# plt.savefig('ts_vs_fitted_naive.jpeg', format='jpeg')
plt.show()

# Plot AR(3) model residuals and Naive Model residuals
residuals_ar_3 = ar_results.resid
residuals_naive = df_mm['y'] - naive_forecast

plt.figure(figsize=(12, 6))
plt.plot(residuals_ar_3, label='AR(3) Residuals', linestyle='-', color='green')
plt.plot(residuals_naive, label='Naive Model Residuals', linestyle='-', color='purple')
# plt.axhline(y=0, color='red', linestyle='--', label='Zero Line')
plt.axhline(y=np.mean(residuals_ar_3), color='blue', linestyle='--', label='Mean Residual (AR(3))')
plt.axhline(y=np.mean(residuals_naive), color='orange', linestyle='--', label='Mean Residual (Naive)')
plt.legend()
plt.title('Model Residuals Comparison')
plt.xlabel('t')
plt.ylabel('Residuals')
plt.savefig(os.path.join('../figures', 'residuals_naive.svg'), format='svg')
# plt.savefig('residuals_naive.jpeg', format='jpeg')
plt.show()

