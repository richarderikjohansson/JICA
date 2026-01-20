#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 10:49:18 2026

@author: umbertor
"""


import numpy as np
from irfpy.scidat import plotting, io
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import os
from scipy.special import erfc, erf
from scipy.optimize import minimize



data = np.load('../data/jica_datarequest_nr11.npz')
a = data['pipeline_version']
b = data['dataproductnr']
time = data['time']
full_data = data['n1']
full_data = .12 * full_data

reduced_data = np.vstack((
    full_data[:-1, :] - full_data[1:, :],
    (full_data[:-1, :] - full_data[1:, :])[-1,:]))
energy_offset = 2.585570822484897
dictionary, footer = io.read_from_txt('energy_calibration_jica.txt', 
                                      '  ', ':', '#')   
en_table = dictionary['Energy_center'].data + energy_offset

#%% energy-current
# unit is amu
mp = 1.6726e-27
e = 1.6022e-19
negative_masses = np.array([12.0111, 13.019200000000001, 14.0273, 15.035400000000001, 16.0435, 
                            17.0516, 24.0222, 25.0303, 26.038400000000003, 27.0465, 28.0546, 
                            29.0627, 31.0789, 37.0414, 38.0495, 39.05760000000001, 
                            40.06570000000001, 41.073800000000006, 42.081900000000005, 43.09, 
                            44.0981, 45.1062, 49.0525, 50.0606, 51.06870000000001, 
                            53.084900000000005, 55.1011, 57.1173, 0.00055])



dictionary, footer = io.read_from_txt('energy_calibration_jica.txt', 
                                      '  ', ':', '#')   
en_table = dictionary['Energy_center'].data + energy_offset

U_prel = np.load('../data/jica_datarequest_nr10.npz')['scpot'][0]
v_prel = np.load('../data/jica_datarequest_nr7.npz')['speed'][0] * 1e3
en_table = en_table[350:380]
reduced_data = reduced_data[350:380, 0]
def mass_to_energy_negative(m, U, v, q=1):
        return 1/2*(m * mp * v**2)/e + U

energy_signal = mass_to_energy_negative(94.9401, U = U_prel, v = v_prel) 

def gaussian_model(x, A, mu, sigma):
    return (
        A / (sigma * np.sqrt(2 * np.pi)) * np.e ** (- (x - mu) ** 2 / (2 * sigma ** 2))
        )

# all these errors and sigma are BS, have to crrect for them
energy_width = np.array(len(en_table) * [3.003])
N_total = np.sum(reduced_data)
sigma_current = np.ones(len(reduced_data))
sigma_current[reduced_data > 0] = np.sqrt(reduced_data[reduced_data > 0])

# ---- Define chi-square ----
def chi2_gaussian(params):
    A, mu, sigma = params
    model = gaussian_model(en_table, A, mu, sigma)
    return np.sum(((reduced_data - model) / sigma_current)**2)



# ---- Initial guesses ----
p0_gauss = [10, 1115, 15]

# ---- Fit via chi-square minimization ----
res_gauss = minimize(chi2_gaussian, p0_gauss, method='Nelder-Mead')

A, mu, sigma = res_gauss.x

# ---- Plot results ----
x_fit = np.linspace(min(en_table), max(en_table), 500)
fit_result = gaussian_model(en_table, A, mu, sigma)




sp = plotting.SimplePlot(xlabel = 'Energy [eV]', ylabel = 'current [pA]', yscale='log')
sp.ax.plot(en_table, reduced_data)
sp.ax.plot(en_table, fit_result , 'tab:red', lw=2, label = 'Gaussian fit')
sp.ax.axvline(energy_signal, color = 'red', linestyle = 'dotted')
sp.ax.set_ylim([1, None])


plt.show()
print(f'the integrated signal is {A}')
