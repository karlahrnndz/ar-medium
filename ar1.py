import matplotlib.pyplot as plt
import numpy as np
import os

# Set the random seed for reproducibility
np.random.seed(42)

#
# # -------------------------------------------------------------- #
# #                            AR(1) - c                           #
# # -------------------------------------------------------------- #
#
# def generate_ar1_with_c(c, phi, sigma, num_samples):
#
#     # Generate AR(1) time series with a constant term
#     ar1_series = np.zeros(num_samples)
#     ar1_series[0] = c + np.random.normal(0, sigma)  # Corrected initialization
#
#     for t in range(1, num_samples):
#         ar1_series[t] = c + phi * ar1_series[t - 1] + np.random.normal(0, sigma)
#
#     return ar1_series
#
#
# # Ser parameters
# num_samples = 100
# phi = 0.9
# sigma = 5
#
# # Generate and plot AR(1) processes with different constant terms in separate figures
# c_values = [-5, 1, 5]
# fig, axs = plt.subplots(len(c_values), 1, figsize=(10, 6), sharex=True, sharey=True)
#
# for i, c in enumerate(c_values, 1):
#     ar1_series = generate_ar1_with_c(c, phi, sigma, num_samples)
#     axs[i - 1].plot(ar1_series, label=f'AR(1) with c={c}, σ={sigma}, ϕ={phi}')
#     axs[i - 1].legend()
#
# axs[-1].set_xlabel('t')
# # Save the plot as an SVG file
# plt.savefig(os.path.join('figures', 'ar1_c.svg'), format='svg')
#
# # Show the plot
# plt.show()
#
# # -------------------------------------------------------------- #
# #                     AR(1) - c, when phi > 1                    #
# # -------------------------------------------------------------- #
#
#
# # Ser parameters
# num_samples = 100
# phi = 1
# sigma = 10
#
# # Generate and plot AR(1) processes with different constant terms in separate figures
# c_values = [-5, 5]
# fig, axs = plt.subplots(len(c_values), 1, figsize=(10, 6), sharex=True, sharey=True)
#
# for i, c in enumerate(c_values, 1):
#     ar1_series = generate_ar1_with_c(c, phi, sigma, num_samples)
#     axs[i - 1].plot(ar1_series, label=f'AR(1) with c={c}, σ={sigma}, ϕ={phi}')
#     axs[i - 1].legend()
#
# axs[-1].set_xlabel('t')
# # Save the plot as an SVG file
# plt.savefig(os.path.join('figures', 'ar1_c_phieq1.svg'), format='svg')
#
# # Show the plot
# plt.show()
#
#
# # -------------------------------------------------------------- #
# #                          AR(1) - phi = 1                       #
# # -------------------------------------------------------------- #
#
# # Set the random seed for reproducibility
# np.random.seed(42)
#
# # Number of samples
# num_samples = 100
#
# # Common parameters
# phi = 1  # Set phi to 1
# sigma = 1
#
# # Generate AR(1) process with c = 1 (one realization)
# c_minus_1_series = generate_ar1_with_c(1, phi, sigma, num_samples)
#
# # Generate and plot five realizations of AR(1) process with c = 1
# c_1_series = [generate_ar1_with_c(1, phi, sigma, num_samples) for _ in range(20)]
#
# # Plot both realizations in one figure with two subplots
# fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=False)
#
# # Plot one realization
# axs[0].plot(c_minus_1_series, label=f'AR(1) with c={1}, σ={sigma}, ϕ={phi}')
# axs[0].legend()
# axs[0].set_title(f'One Realization of AR(1) Process with ϕ={1}')
#
# # Plot five realizations
# for i, series in enumerate(c_1_series, 1):
#     axs[1].plot(series, label=f'Realization {i}')
#
# axs[1].set_xlabel('t')
# axs[1].set_title(f'Five Realizations of AR(1) Process with ϕ={1}')
#
# # Adjust layout for better spacing
# plt.tight_layout()
#
# # Save the plot as an SVG file
# plt.savefig(os.path.join('figures', 'ar1_phieq1.svg'), format='svg')
#
# # Show the plot
# plt.show()
#
#
# # -------------------------------------------------------------- #
# #                         AR(1) - phi = -1                       #
# # -------------------------------------------------------------- #
#
# # Number of samples
# num_samples = 100
#
# # Common parameters
# phi = -1  # Set phi to 1
# sigma = 1
# c = 1
#
# # Generate AR(1) process with c = 1 (one realization)
# c_minus_1_series = generate_ar1_with_c(c, phi, sigma, num_samples)
#
# # Generate and plot five realizations of AR(1) process with c = 1
# c_1_series = [generate_ar1_with_c(c, phi, sigma, num_samples) for _ in range(20)]
#
# # Plot both realizations in one figure with two subplots
# fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=False)
#
# # Plot one realization
# axs[0].plot(c_minus_1_series, label=f'AR(1) with c={c}, σ={sigma}, ϕ={phi}')
# axs[0].legend()
# axs[0].set_title('One Realization of AR(1) Process with ϕ=-1')
#
# # Plot five realizations
# for i, series in enumerate(c_1_series, 1):
#     axs[1].plot(series, label=f'Realization {i}')
#
# axs[1].set_xlabel('t')
# axs[1].set_title('Five Realizations of AR(1) Process with ϕ=-1')
#
# # Adjust layout for better spacing
# plt.tight_layout()
#
# # Show the plot
# # plt.show()
#
#
# # Save the plot as an SVG file
# plt.savefig(os.path.join('figures', 'ar1_phieqm1.svg'), format='svg')
#
# # Show the plot
# plt.show()
#
#
# # -------------------------------------------------------------- #
# #                         AR(1) - phi < - 1                      #
# # -------------------------------------------------------------- #
#
# # Number of samples
# num_samples = 100
#
# # Common parameters
# phi = -1.1  # Set phi to 1
# sigma = 1
# c = 1
#
# # Generate AR(1) process with c = 1 (one realization)
# c_minus_1_series = generate_ar1_with_c(c, phi, sigma, num_samples)
#
# # Generate and plot five realizations of AR(1) process with c = 1
# c_1_series = [generate_ar1_with_c(c, phi, sigma, num_samples) for _ in range(20)]
#
# # Plot both realizations in one figure with two subplots
# fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=False)
#
# # Plot one realization
# axs[0].plot(c_minus_1_series, label=f'AR(1) with c={c}, σ={sigma}, ϕ={phi}')
# axs[0].legend()
# axs[0].set_title(f'One Realization of AR(1) Process with ϕ={phi}')
#
# # Plot five realizations
# for i, series in enumerate(c_1_series, 1):
#     axs[1].plot(series, label=f'Realization {i}')
#
# axs[1].set_xlabel('t')
# axs[1].set_title(f'Five Realizations of AR(1) Process with ϕ={phi}')
#
# # Adjust layout for better spacing
# plt.tight_layout()
#
# # Save the plot as an SVG file
# plt.savefig(os.path.join('figures', 'ar1_philtm1.svg'), format='svg')
#
# # Show the plot
# plt.show()
#
#
# # -------------------------------------------------------------- #
# #                         AR(1) - phi >  1                       #
# # -------------------------------------------------------------- #
#
# # Number of samples
# num_samples = 100
#
# # Common parameters
# phi = 1.1  # Set phi to 1
# sigma = 10
# c = 0
#
# # Generate AR(1) process with c = 1 (one realization)
# c_minus_1_series = generate_ar1_with_c(c, phi, sigma, num_samples)
#
# # Generate and plot five realizations of AR(1) process with c = 1
# c_1_series = [generate_ar1_with_c(c, phi, sigma, num_samples) for _ in range(20)]
#
# # Plot both realizations in one figure with two subplots
# fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=False)
#
# # Plot one realization
# axs[0].plot(c_minus_1_series, label=f'AR(1) with c={c}, σ={sigma}, ϕ={phi}')
# axs[0].legend()
# axs[0].set_title(f'One Realization of AR(1) Process with ϕ={phi}')
#
# # Plot five realizations
# for i, series in enumerate(c_1_series, 1):
#     axs[1].plot(series, label=f'Realization {i}')
#
# axs[1].set_xlabel('t')
# axs[1].set_title(f'Five Realizations of AR(1) Process with ϕ={phi}')
#
# # Adjust layout for better spacing
# plt.tight_layout()
#
# # Save the plot as an SVG file
# plt.savefig(os.path.join('figures', 'ar1_phigt1.svg'), format='svg')
#
# # Show the plot
# plt.show()
#
#
# # -------------------------------------------------------------- #
# #                        AR(1) - |phi| <  1                      #
# # -------------------------------------------------------------- #
#
#
# # Common parameters
# phi = 0.5  # Set phi to 1
# sigma = 1
# c = 1
#
# # Generate AR(1) process with c = 1 (one realization)
# c_minus_1_series = generate_ar1_with_c(c, phi, sigma, num_samples)
#
# # Generate and plot five realizations of AR(1) process with c = 1
# c_1_series = [generate_ar1_with_c(c, phi, sigma, num_samples) for _ in range(20)]
#
# # Plot both realizations in one figure with two subplots
# fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=False)
#
# # Plot one realization
# axs[0].plot(c_minus_1_series, label=f'AR(1) with c={c}, σ={sigma}, ϕ={phi}')
# axs[0].legend()
# axs[0].set_title(f'One Realization of AR(1) Process with ϕ={phi}')
#
# # Plot five realizations
# for i, series in enumerate(c_1_series, 1):
#     axs[1].plot(series, label=f'Realization {i}')
#
# axs[1].set_xlabel('t')
# axs[1].set_title(f'Five Realizations of AR(1) Process with ϕ={phi}')
#
# # Adjust layout for better spacing
# plt.tight_layout()
#
# # Show the plot
# # plt.show()
#
#
# # Save the plot as an SVG file
# plt.savefig(os.path.join('figures', 'ar1_absphilt1.svg'), format='svg')
#
# # Show the plot
# plt.show()
#


from statsmodels.tsa.ar_model import AutoReg

def generate_ar1_with_c(c, phi, sigma, num_samples):

    # Generate AR(1) time series with a constant term
    ar1_series = np.zeros(num_samples)
    ar1_series[0] = c + np.random.normal(0, sigma)  # Corrected initialization

    for t in range(1, num_samples):
        ar1_series[t] = c + phi * ar1_series[t - 1] + np.random.normal(0, sigma)

    return ar1_series

phi = 1
sigma = 10
c = 1
num_samples = 100
c_1_series = generate_ar1_with_c(c, phi, sigma, num_samples)

# # Dataframe with original time series to forecast
# original_series = c_1_series# some series of values
#
# # Make time series stationary using one or more transformations
# stationaty_series = c_1_series# Some transformation(s) on original_series

# Fit AR(p) model to the stationary series
p = 1
model_ar = AutoReg(c_1_series, lags=p, trend='c').fit()

forecast_steps = 10
forecast_values = model_ar.predict(start=len(c_1_series), end=len(c_1_series) + forecast_steps - 1)


def interleave_cumsum(lst, s):
    result_list = []
    cumulative_sums = {}

    for i, value in enumerate(lst):
        floor = i % s
        cumulative_sums[floor] = cumulative_sums.get(floor, 0) + value
        result_list.append(cumulative_sums[floor])

    return result_list

forecast_final = df['original_series'].iloc[-s:] + interleave_cumsum(forecast_values)

print('hi')