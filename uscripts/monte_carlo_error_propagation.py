#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 10:08:08 2026

@author: umbertor
"""

import numpy as np
from irfpy.scidat import plotting, io
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import os
from scipy.optimize import minimize
from scipy.optimize import curve_fit

#%%dtaa loading
mp = 1.6726e-27
e = 1.6022e-19
os.chdir("/home/umbertor/Documents/PhD/courses/statistics/space_race/JICA/uscripts")
data_pos = np.load('../data/jica_datarequest_nr20.npz')
time_pos = data_pos['time']
full_data_pos = data_pos['p1']
full_data_pos = .12 * full_data_pos
reduced_data_pos = np.vstack((
    full_data_pos[:-1, :] - full_data_pos[1:, :],
    (full_data_pos[:-1, :] - full_data_pos[1:, :])[-1,:]))
reduced_data_pos[reduced_data_pos < 0] = 0
# transform to counts and account for picoampere
dt = 2e-3
reduced_counts_pos = reduced_data_pos * dt * 1e-12/e

data_neg = np.load('../data/jica_datarequest_nr21.npz')
time_neg = data_neg['time']
full_data_neg = data_neg['n1']
full_data_neg = .12 * full_data_neg
reduced_data_neg = np.vstack((
    full_data_neg[:-1, :] - full_data_neg[1:, :],
    (full_data_neg[:-1, :] - full_data_neg[1:, :])[-1,:]))

reduced_data_neg[reduced_data_neg < 0] = 0
# transform to counts and account for picoampere
reduced_counts_neg = reduced_data_neg * dt * 1e-12/e

U_prel = np.load('../data/jica_datarequest_nr10.npz')['scpot'][0]
v_prel = np.load('../data/jica_datarequest_nr7.npz')['speed'][0] * 1e3

dictionary, footer = io.read_from_txt('energy_calibration_jica.txt', 
                                      '  ', ':', '#')   
en_table = dictionary['Energy_center'].data
en_table_pos = en_table + 1.3160564888762565
en_table_neg = en_table + 2.585570822484897


#%%functions

def mass_to_energy_positive(m, U, v, q=1):
        return 1/2*(m * mp * v**2)/e - U
    
def mass_to_energy_negative(m, U, v, q=1):
        return 1/2*(m * mp * v**2)/e + U

def gaussian(x, A, mu, sigma):
    return (
        A / (sigma * np.sqrt(2 * np.pi)) * np.e ** (- (x - mu) ** 2 / (2 * sigma ** 2))
        )


#%%
np.random.seed(0)
n_mc = 1000


fit_mask_pos = (en_table_pos > 690) & (en_table_pos < 732)
fit_mask_neg = (en_table_pos > 1058) & (en_table_pos < 1148)


en_table_fit_pos = en_table_pos[fit_mask_pos]
en_table_fit_neg = en_table_neg[fit_mask_neg]

avg_ratios_peaks = []
avg_ratios_amplitudes = []
std_ratios_peaks = []
std_ratios_amplitudes = []

for i in range(len(time_pos)):

    ratios = []
    ratios_amplitude = []
    
    for _ in range(n_mc):
        # Step 1: draw Poisson realization
        N_mc_pos = np.random.poisson(reduced_counts_pos[:,i])

        N_fit_mc_pos = N_mc_pos[fit_mask_pos]

        sigma_y = np.sqrt(np.maximum(N_fit_mc_pos, 1))
        # Step 2: fit Gaussian (mu fixed if you want)
        try:
            popt, pcov = curve_fit(
                gaussian,
                en_table_fit_pos,
                N_fit_mc_pos,
                p0=[100, 706, 15],
                bounds=([0, 705, 3],
                        [np.inf, 707, 60]),
                sigma=sigma_y,
                absolute_sigma=True,
                maxfev=2000
                )
        except RuntimeError:
            continue

        A_mc_pos, mu_mc_pos, sigma_mc_pos = popt
        fit_result_pos = gaussian(en_table_fit_pos, A_mc_pos, mu_mc_pos, sigma_mc_pos)
        
        # Step 1: draw Poisson realization
        N_mc_neg = np.random.poisson(reduced_counts_neg[:,i])

        N_fit_mc_neg = N_mc_neg[fit_mask_neg]
        sigma_y = np.sqrt(np.maximum(N_fit_mc_neg, 1))
        # Step 2: fit Gaussian (mu fixed if you want)
        try:
            popt, pcov = curve_fit(
                gaussian,
                en_table_fit_neg,
                N_fit_mc_neg,
                p0=[10, 1117, 15],
                bounds=([0, 1105, 3],
                        [np.inf, 1125, 60]),
                sigma=sigma_y,
                absolute_sigma=True,
                maxfev=2000
                )
        except RuntimeError:
            continue

        A_mc_neg, mu_mc_neg, sigma_mc_neg = popt
        fit_result_neg = gaussian(en_table_fit_neg, A_mc_neg, mu_mc_neg, sigma_mc_neg)
        ratios.append(np.max(N_fit_mc_neg) / np.max(N_fit_mc_pos))
        ratios_amplitude.append(A_mc_neg / A_mc_pos)

        # sp = plotting.SimplePlot(rows = 2, xlabel = 'Energy [eV]', ylabel = 'counts []', yscale='log',
        #                          title = f'ratio = {ratios_amplitude[-1]}, t = {time_pos[i]}',
        #                          figure_ratio=.9, tight_layout=2)
        # sp.ax[0].plot(en_table_pos, N_mc_pos)
        # sp.ax[0].axvline(mass_to_energy_positive(60.03, U_prel, v_prel), color = 'r', linestyle = 'dotted')
        # sp.ax[0].set_ylim([1e2, None])
        # sp.ax[0].set_title('Positive')
        # sp.ax[0].plot(en_table_fit_pos, fit_result_pos, 'tab:red', lw=2, label = 'Gaussian fit')
    
    
        # sp.ax[1].plot(en_table_neg, N_mc_neg)
        # sp.ax[1].axvline(mass_to_energy_positive(94.9401, U_prel, v_prel), color = 'r', linestyle = 'dotted')
        # sp.ax[1].set_ylim([1e2, None])
        # sp.ax[1].set_title('Negative')
        # sp.ax[1].plot(en_table_fit_neg, fit_result_neg, 'tab:red', lw=2, label = 'Gaussian fit')
        # plt.show()

    ratios = np.array(ratios)
    ratios_amplitude = np.array(ratios_amplitude)
    print(f'avg ratio at time {time_pos[i]} is {np.mean(ratios)}, with std {np.std(ratios)}')
    print(f'avg ratio of amplitudes at time {time_pos[i]} is {np.mean(ratios_amplitude)}, with std {np.std(ratios_amplitude)}')
    avg_ratios_peaks.append(np.mean(ratios))
    avg_ratios_amplitudes.append(np.mean(ratios_amplitude))
    std_ratios_peaks.append(np.std(ratios))
    std_ratios_amplitudes.append(np.std(ratios_amplitude))

#excluding the first ratio as we don't have negative data
avg_ratios_peaks = np.array(avg_ratios_peaks[1:])
avg_ratios_amplitudes = np.array(avg_ratios_amplitudes[1:])
std_ratios_peaks = np.array(std_ratios_peaks[1:])
std_ratios_amplitudes = np.array(std_ratios_amplitudes[1:])

#%% hypothesis 1: statistical test with the ratio of the amplitudes
mean = np.mean(avg_ratios_amplitudes)

weighted_mean = np.average(avg_ratios_amplitudes, weights=std_ratios_amplitudes)

sigma = np.std(avg_ratios_amplitudes, ddof = 1)

sem = sigma / len(avg_ratios_amplitudes)

weighted_sem = np.sqrt(1/np.sum(1/std_ratios_amplitudes**2))


mu0 = 0.5


t = (mean - mu0) / sem
df = len(avg_ratios_amplitudes) - 1

p_value = 2 * (1 - stats.t.cdf(abs(t), df))

#%% hypothesis 2: statistical test with the ratio of the peaks
mean = np.mean(avg_ratios_peaks)

weighted_mean = np.average(avg_ratios_peaks, weights=std_ratios_peaks)

sigma = np.std(avg_ratios_peaks, ddof = 1)

sem = sigma / len(avg_ratios_peaks)

weighted_sem = np.sqrt(1/np.sum(1/std_ratios_peaks**2))


mu0 = 0.5


t = (mean - mu0) / sem
df = len(avg_ratios_peaks) - 1

p_value = 2 * (1 - stats.t.cdf(abs(t), df))


