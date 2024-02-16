import matplotlib.pyplot as plt
import numpy as np
import os

# Set the random seed for reproducibility
np.random.seed(42)

def generate_ar1_with_c(c, phi, sigma, num_samples):
    # Generate AR(1) time series with a constant term
    ar1_series = np.zeros(num_samples)
    ar1_series[0] = c + np.random.normal(0, sigma)  # Corrected initialization

    for t in range(1, num_samples):
        ar1_series[t] = c + phi * ar1_series[t - 1] + np.random.normal(0, sigma)

    return ar1_series

# Number of samples
num_samples = 100

# Common parameters
phi = 1  # Set phi to 1
sigma = 1

# Generate AR(1) process with c = 1 (one realization)
c_minus_1_series = generate_ar1_with_c(1, phi, sigma, num_samples)

# Generate and plot five realizations of AR(1) process with c = 1
c_1_series = [generate_ar1_with_c(1, phi, sigma, num_samples) for _ in range(5)]

# Plot both realizations in one figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=False)

# Plot one realization
axs[0].plot(c_minus_1_series, label=f'AR(1) with c={1}, σ={sigma}, φ={phi}')
axs[0].set_ylabel('Value')
axs[0].legend()
axs[0].set_title(f'One Realization of AR(1) Process with c={1}')

# Plot five realizations
for i, series in enumerate(c_1_series, 1):
    axs[1].plot(series, label=f'Realization {i}')

axs[1].set_xlabel('Time')
axs[1].set_ylabel('Value')
axs[1].legend()
axs[1].set_title(f'Five Realizations of AR(1) Process with c={1}')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
# plt.show()


# Save the plot as an SVG file
plt.savefig(os.path.join('../plots', 'ar1_processes_phi_eq1.jpeg'), format='jpeg')

# Show the plot
plt.show()
