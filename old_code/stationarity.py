# import numpy as np
# import matplotlib.pyplot as plt
#
# np.random.seed(42)
#
# # Number of time points
# n = 100
#
# # Simulate a non-stationary time series
# epsilon = np.random.normal(0, 1, n)
# X = np.zeros(n)
#
# for t in range(1, n):
#     X[t] = 2 * X[t-1] + epsilon[t]
#
# # Plot the non-stationary time series
# plt.plot(X)
# plt.title("Non-Stationary AR(1) Process with $\phi = 2$")
# plt.xlabel("Time")
# plt.ylabel("$X_t$")
# plt.show()
#
#

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

np.random.seed(42)

# Number of time points
n = 100

# Number of simulations
num_simulations = 100

# True autoregressive parameter
true_phi = 2
true_c = 1

# Storage for estimated parameters
estimated_phis = np.zeros((num_simulations,))

# Simulate and fit AR(1) models
for i in range(num_simulations):
    # Simulate a non-stationary time series
    epsilon = np.random.normal(0, 1, n)
    X = np.zeros(n)

    for t in range(1, n):
        X[t] = true_c + true_phi * X[t-1] + epsilon[t]

    # Fit AR(1) model to the non-stationary time series
    model = AutoReg(X, lags=1)
    result = model.fit()

    # Store the estimated autoregressive parameter
    estimated_phis[i] = result.params[1]

# Plot the distribution of estimated parameters
plt.hist(estimated_phis, edgecolor='black')
# plt.axvline(x=true_phi, color='red', linestyle='dashed', linewidth=2, label='True $\phi$')
# plt.title("Distribution of Estimated $\phi$ in Non-Stationary AR(1) Process")
# plt.xlabel("$\phi$")
# plt.ylabel("Frequency")
# plt.legend()
plt.show()