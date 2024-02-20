import matplotlib.pyplot as plt
import numpy as np
import os


# -------------------------------------------------------------- #
#                            AR(1) - c                           #
# -------------------------------------------------------------- #

def generate_ar1_with_c(c, phi, sigma, num_samples):

    # Generate AR(1) time this_series with a constant term
    ar1_series = np.zeros(num_samples)
    ar1_series[0] = c + np.random.normal(0, sigma)  # Corrected initialization

    for t in range(1, num_samples):
        ar1_series[t] = c + phi * ar1_series[t - 1] + np.random.normal(0, sigma)

    return ar1_series


# Set parameters
num_samples = 100
phi = 0.9
sigma = 5

# Generate and plot AR(1) processes with different constant terms in separate figures
c_values = [-5, 1, 5]
fig, axs = plt.subplots(len(c_values), 1, figsize=(10, 6), sharex=True, sharey=True)

for i, c in enumerate(c_values):
    this_ar1_series = generate_ar1_with_c(c, phi, sigma, num_samples)
    axs[i - 1].plot(this_ar1_series, label=f'AR(1) with c={c}, σ={sigma}, ϕ={phi}')
    axs[i - 1].legend()

axs[-1].set_xlabel('t')

# Save the plot as an SVG file
plt.savefig(os.path.join('..', 'figures', 'raw_svg' 'ar1_c.svg'), format='svg')

# Show the plot
plt.show()

# -------------------------------------------------------------- #
#                     AR(1) - c, when phi ≥ 1                    #
# -------------------------------------------------------------- #


# Ser parameters
num_samples = 100
phi = 1
sigma = 10

# Generate and plot AR(1) processes with different constant terms in separate figures
c_values = [-5, 5]
fig, axs = plt.subplots(len(c_values), 1, figsize=(10, 6), sharex=True, sharey=True)

for i, c in enumerate(c_values):
    this_ar1_series = generate_ar1_with_c(c, phi, sigma, num_samples)
    axs[i - 1].plot(this_ar1_series, label=f'AR(1) with c={c}, σ={sigma}, ϕ={phi}')
    axs[i - 1].legend()

axs[-1].set_xlabel('t')

# Save the plot as an SVG file
plt.savefig(os.path.join('..', 'figures', 'raw_svg' 'ar1_c_phieq1.svg'), format='svg')

# Show the plot
plt.show()


# -------------------------------------------------------------- #
#                          AR(1) - phi = 1                       #
# -------------------------------------------------------------- #

# Number of samples
num_samples = 100

# Common parameters
phi = 1
sigma = 1

# Generate AR(1) process with ϕ = 1 (one realization)
phi_eq1_series = generate_ar1_with_c(1, phi, sigma, num_samples)

# Generate and plot five realizations of AR(1) process with ϕ = 1
phi_eq1_multiseries = [generate_ar1_with_c(1, phi, sigma, num_samples) for _ in range(20)]

# Plot both realizations in one figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=False)

# Plot one realization
axs[0].plot(phi_eq1_series, label=f'AR(1) with c={1}, σ={sigma}, ϕ={phi}')
axs[0].legend()
axs[0].set_title(f'One Realization of AR(1) Process with ϕ={phi}')

# Plot five realizations
for i, this_series in enumerate(phi_eq1_multiseries):
    axs[1].plot(this_series)

axs[1].set_xlabel('t')
axs[1].set_title(f'20 Realizations of AR(1) Process with ϕ={phi}')

# Adjust layout for better spacing
plt.tight_layout()

# Save the plot as an SVG file
plt.savefig(os.path.join('..', 'figures', 'raw_svg' 'ar1_phieq1.svg'), format='svg')

# Show the plot
plt.show()


# -------------------------------------------------------------- #
#                         AR(1) - phi = -1                       #
# -------------------------------------------------------------- #

# Number of samples
num_samples = 100

# Common parameters
phi = -1  # Set phi to 1
sigma = 1
c = 1

# Generate AR(1) process with c = 1 (one realization)
phi_eqm1_series = generate_ar1_with_c(c, phi, sigma, num_samples)

# Generate and plot five realizations of AR(1) process with c = 1
phi_eqm1_multiseries = [generate_ar1_with_c(c, phi, sigma, num_samples) for _ in range(20)]

# Plot both realizations in one figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=False)

# Plot one realization
axs[0].plot(phi_eqm1_series, label=f'AR(1) with c={c}, σ={sigma}, ϕ={phi}')
axs[0].legend()
axs[0].set_title(f'One Realization of AR(1) Process with ϕ={phi}')

# Plot five realizations
for i, this_series in enumerate(phi_eqm1_multiseries):
    axs[1].plot(this_series)

axs[1].set_xlabel('t')
axs[1].set_title(f'20 Realizations of AR(1) Process with ϕ={phi}')

# Adjust layout for better spacing
plt.tight_layout()

# Save the plot as an SVG file
plt.savefig(os.path.join('..', 'figures', 'raw_svg' 'ar1_phieqm1.svg'), format='svg')

# Show the plot
plt.show()


# -------------------------------------------------------------- #
#                         AR(1) - phi < - 1                      #
# -------------------------------------------------------------- #

# Number of samples
num_samples = 100

# Common parameters
phi = -1.1  # Set phi to 1
sigma = 1
c = 1

# Generate AR(1) process with c = 1 (one realization)
phi_ltm1_series = generate_ar1_with_c(c, phi, sigma, num_samples)

# Generate and plot five realizations of AR(1) process with c = 1
phi_ltm1_multiseries = [generate_ar1_with_c(c, phi, sigma, num_samples) for _ in range(20)]

# Plot both realizations in one figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=False)

# Plot one realization
axs[0].plot(phi_ltm1_series, label=f'AR(1) with c={c}, σ={sigma}, ϕ={phi}')
axs[0].legend()
axs[0].set_title(f'One Realization of AR(1) Process with ϕ={phi}')

# Plot five realizations
for i, this_series in enumerate(phi_ltm1_multiseries):
    axs[1].plot(this_series)

axs[1].set_xlabel('t')
axs[1].set_title(f'20 Realizations of AR(1) Process with ϕ={phi}')

# Adjust layout for better spacing
plt.tight_layout()

# Save the plot as an SVG file
plt.savefig(os.path.join('..', 'figures', 'raw_svg' 'ar1_philtm1.svg'), format='svg')

# Show the plot
plt.show()


# -------------------------------------------------------------- #
#                         AR(1) - phi >  1                       #
# -------------------------------------------------------------- #

# Number of samples
num_samples = 100

# Common parameters
phi = 1.1  # Set phi to 1
sigma = 10
c = 0

# Generate AR(1) process with c = 1 (one realization)
phi_gt1_series = generate_ar1_with_c(c, phi, sigma, num_samples)

# Generate and plot five realizations of AR(1) process with c = 1
phi_gt1_multiseries = [generate_ar1_with_c(c, phi, sigma, num_samples) for _ in range(20)]

# Plot both realizations in one figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=False)

# Plot one realization
axs[0].plot(phi_gt1_series, label=f'AR(1) with c={c}, σ={sigma}, ϕ={phi}')
axs[0].legend()
axs[0].set_title(f'One Realization of AR(1) Process with ϕ={phi}')

# Plot five realizations
for i, this_series in enumerate(phi_gt1_multiseries):
    axs[1].plot(this_series)

axs[1].set_xlabel('t')
axs[1].set_title(f'20 Realizations of AR(1) Process with ϕ={phi}')

# Adjust layout for better spacing
plt.tight_layout()

# Save the plot as an SVG file
plt.savefig(os.path.join('..', 'figures', 'raw_svg' 'ar1_phigt1.svg'), format='svg')

# Show the plot
plt.show()


# -------------------------------------------------------------- #
#                        AR(1) - |phi| <  1                      #
# -------------------------------------------------------------- #
# -------------------------------------------------------------- #

# Common parameters
phi = 0.5  # Set phi to 1
sigma = 1
c = 1

# Generate AR(1) process with c = 1 (one realization)
phi_abslt1_series = generate_ar1_with_c(c, phi, sigma, num_samples)

# Generate and plot five realizations of AR(1) process with c = 1
phi_abslt1_multiseries = [generate_ar1_with_c(c, phi, sigma, num_samples) for _ in range(20)]

# Plot both realizations in one figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=False)

# Plot one realization
axs[0].plot(phi_abslt1_series, label=f'AR(1) with c={c}, σ={sigma}, ϕ={phi}')
axs[0].legend()
axs[0].set_title(f'One Realization of AR(1) Process with ϕ={phi}')

# Plot five realizations
for i, this_series in enumerate(phi_abslt1_multiseries):
    axs[1].plot(this_series)

axs[1].set_xlabel('t')
axs[1].set_title(f'20 Realizations of AR(1) Process with ϕ={phi}')

# Adjust layout for better spacing
plt.tight_layout()

# Save the plot as an SVG file
plt.savefig(os.path.join('..', 'figures', 'raw_svg' 'ar1_abslt1.svg'), format='svg')

# Show the plot
plt.show()
