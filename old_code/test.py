import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

# Assuming 'stationary_series' is your time series data
lags = 5  # Set the maximum lag order you want to test

stationary_series = np.random.randint(1, 3, 50)

# Fit AutoReg model
model = sm.tsa.AutoReg(stationary_series, lags=lags, trend='c')  # You may need to adjust the parameters
model_fit = model.fit()

# Get residuals
residuals = model_fit.resid

# Perform Ljung-Box test
res = acorr_ljungbox(residuals, lags=lags)

# Check if p-values are below a significance level (e.g., 0.05) to reject the null hypothesis
significant_lags = np.where(res['lb_pvalue'] < 1)

# Print the significant lags (you can consider setting p = max(significant_lags)
print(f"Significant Lags: {significant_lags}")
