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
phi = 0.9  # Set phi to 1
sigma = 0.2

# Generate and plot AR(1) processes with different constant terms in separate figures
c_values = [-5, 1, 5]

fig, axs = plt.subplots(len(c_values), 1, figsize=(10, 6), sharex=True, sharey=True)

for i, c in enumerate(c_values, 1):
    ar1_series = generate_ar1_with_c(c, phi, sigma, num_samples)
    axs[i - 1].plot(ar1_series, label=f'AR(1) with c={c}, σ={sigma}, φ={phi}')
    axs[i - 1].set_ylabel('Value')
    axs[i - 1].legend()

axs[-1].set_xlabel('Time')
plt.suptitle(f'AR(1) Processes with Varying c Terms (σ={sigma}, φ={phi})')

# Save the plot as an SVG file
plt.savefig(os.path.join('../plots', 'ar1_processes.jpeg'), format='jpeg')

# Show the plot
plt.show()
