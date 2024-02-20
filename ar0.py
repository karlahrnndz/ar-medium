import matplotlib.pyplot as plt
import numpy as np
import os

# Set the random seed for reproducibility
np.random.seed(42)


def generate_ar0(c, std_dev, num_samples):
    # Generate white noise with the specified standard deviation
    white_noise = np.random.normal(0, std_dev, num_samples)

    # Generate AR(0) time series
    ar0_series = c + white_noise

    return ar0_series


# Set parameters
num_samples = 100
std_dev = 0.2

# Generate and plot AR(0) processes with different constant terms
c_values = [1, -1, 5]
fig, axs = plt.subplots(len(c_values), 1, figsize=(10, 6), sharex=True, sharey=True)

for i, c in enumerate(c_values, 1):
    ar0_series = generate_ar0(c, std_dev, num_samples)
    axs[i - 1].plot(ar0_series, label=f'AR(0) with c={c}, σ={std_dev}')
    axs[i - 1].legend()

axs[-1].set_xlabel('t')

# Save the plot as an SVG file
plt.savefig(os.path.join('figures', 'ar0_processes.svg'), format='svg')

# Show plot
plt.show()
