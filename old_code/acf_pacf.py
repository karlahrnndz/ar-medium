import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Function to generate AR(p) time series
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

# Generate AR(p) time series
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
plt.savefig('acf_pacf.jpeg', format='jpeg')

plt.show()
