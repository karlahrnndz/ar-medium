import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


# Function to generate AR(3) time series
def generate_ar3_series(n_samples):
    np.random.seed(42)
    ar_params = [0.3, -0.5, 0.2]  # AR(3) coefficients
    noise = np.random.normal(0, 1, n_samples)

    series = np.zeros(n_samples)
    for i in range(3, n_samples):
        series[i] = ar_params[0] * series[i - 1] + ar_params[1] * series[i - 2] + ar_params[2] * series[i - 3] + noise[
            i]

    return series


# Generate AR(3) time series
n_samples = 100
ar3_series = generate_ar3_series(n_samples)

# Fit AR(3) model
ar3_model = ARIMA(ar3_series, order=(3, 0, 0))
ar3_results = ar3_model.fit()

# Plot AR(3) time series and fitted values
plt.figure(figsize=(12, 6))
plt.plot(ar3_series, label='AR(3) Time Series')
plt.plot(ar3_results.fittedvalues, label='AR(3) Fitted Values', linestyle='--', color='red')
plt.legend()
plt.title('AR(3) Time Series and Fitted Values')
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig('ts_vs_fitted_good.jpeg', format='jpeg')
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
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.savefig('residuals_good.jpeg', format='jpeg')
plt.show()