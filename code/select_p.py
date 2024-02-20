from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import os


# -------------------------------------------------------------- #
#                             ACF/PACF                           #
# -------------------------------------------------------------- #


# Function to generate AR(p) time this_series
def generate_ar_series(p, num_samples):
    ar_params = [1, 2, 3]
    ar_series = [np.random.normal() for _ in range(p)]  # Initialize with p random values

    for _ in range(p, num_samples):
        ar_value = np.sum(ar_params * np.array(ar_series[-p:])) + np.random.normal()
        ar_series.append(ar_value)

    return ar_series


# Parameters
p = 3  # Order of the AR model
num_samples = 100
lags = 20  # Number of lags for ACF and PACF plots

# Generate AR(p) time this_series
ar_series = generate_ar_series(p, num_samples)

# Plot ACF and PACF
fig, ax = plt.subplots(1, 2, figsize=(14, 4))

# Autocorrelation
sm.graphics.tsa.plot_acf(ar_series, lags=lags, ax=ax[0])
ax[0].set_title(f'ACF Plot for AR({p}) Process')

# Partial Autocorrelation
sm.graphics.tsa.plot_pacf(ar_series, lags=lags, ax=ax[1])
ax[1].set_title(f'PACF Plot for AR({p}) Process')

# Save the plot as a JPEG file
plt.savefig(os.path.join('..', 'figures', 'raw_svg', 'acf_pacf.svg'), format='svg')

plt.show()

# -------------------------------------------------------------- #
#                              AIC/BIC                           #
# -------------------------------------------------------------- #

aic_values = []
bic_values = []
p_range = range(1, 4)

for p in p_range:
    model = AutoReg(ar_series, lags=p, trend='c').fit()
    aic_values.append(model.aic)
    bic_values.append(model.bic)

best_p_aic = p_range[np.argmin(aic_values)]  # AIC-suggested p
best_p_bic = p_range[np.argmin(bic_values)]  # BIC-suggested p

print(best_p_aic)
print(best_p_bic)


# -------------------------------------------------------------- #
#                         Cross-Validation                       #
# -------------------------------------------------------------- #

# Assuming you have a stationary time this_series called "ar_series"

# Set up the TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Initialize variables to store results
best_model = None
best_mape = np.inf

# Specify the range of lag orders to consider
p_range = range(1, 4)  # Adjust the range based on your needs

best_lag_order = None
ar_series = pd.Series(ar_series)

# Split the data into train and validation sets using TimeSeriesSplit
for train_index, val_index in tscv.split(ar_series):
    train_data = [ar_series[i] for i in train_index]
    val_data = [ar_series[i] for i in val_index]

    for p in p_range:

        # Fit AutoReg model on the training data
        model = AutoReg(train_data, lags=p, trend='c')
        model_fit = model.fit()

        # Forecast on the validation set
        forecast_values = model_fit.predict(start=len(train_data), end=len(train_data) + len(val_data) - 1)

        # Calculate MAPE on the validation set
        mape = mean_absolute_percentage_error(val_data, forecast_values)

        # Check if the current model has a lower MAPE than the best model so far
        if mape < best_mape:
            best_mape = mape
            best_model = model_fit
            best_lag_order = p  # Update the best lag order

# Print the lag order of the best model
print("Best lag order:", best_lag_order)


# -------------------------------------------------------------- #
#                         Ljung-Box test                         #
# -------------------------------------------------------------- #

# Assuming 'stationary_series' is your time this_series data
lags = 10  # Set the maximum lag order you want to test

# Fit AutoReg model
model = sm.tsa.AutoReg(ar_series, lags=lags, trend='c')  # You may need to adjust the parameters
model_fit = model.fit()

# Get residuals
residuals = model_fit.resid

# Perform Ljung-Box test
res = acorr_ljungbox(residuals, lags=lags)

# Check if p-values are below a significance level (e.g., 0.05) to reject the null hypothesis
significant_lags = np.where(res['lb_pvalue'] < 0.05)[0] + 1

# Print the significant lags (you can consider setting p = max(significant_lags)
print(f"Significant Lags: {significant_lags}")
