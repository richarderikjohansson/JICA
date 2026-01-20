#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 15:31:45 2026

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

data = np.load('../data/jica_datarequest_nr8.npz')
a = data['pipeline_version']
b = data['dataproductnr']
time = data['time']
full_data = data['p1']
full_data = .12 * full_data

reduced_data = np.vstack((
    full_data[:-1, :] - full_data[1:, :],
    (full_data[:-1, :] - full_data[1:, :])[-1,:]))
energy_offset = 1.3160564888762565
dictionary, footer = io.read_from_txt('energy_calibration_jica.txt', 
                                      '  ', ':', '#')   
en_table = dictionary['Energy_center'].data + energy_offset
#%% energy-current
# unit is amu
mp = 1.6726e-27
e = 1.6022e-19

U_prel = np.load('../data/jica_datarequest_nr10.npz')['scpot'][0]
v_prel = np.load('../data/jica_datarequest_nr7.npz')['speed'][0] * 1e3
def mass_to_energy_positive(m, U, v, q=1):
        return 1/2*(m * mp * v**2)/e - U
energy_signal = mass_to_energy_positive(60.03, U = U_prel, v = v_prel) 
en_table = en_table[220:242]
reduced_data = reduced_data[220:242, 0]


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
p0_gauss = [100, 707, 15]

# ---- Fit via chi-square minimization ----
res_gauss = minimize(chi2_gaussian, p0_gauss, method='Nelder-Mead')

A, mu, sigma = res_gauss.x

# ---- Plot results ----
x_fit = np.linspace(min(en_table), max(en_table), 500)
fit_result = gaussian_model(en_table, A, mu, sigma)




sp = plotting.SimplePlot(xlabel = 'Energy [eV]', ylabel = 'current [pA]', yscale='log')
sp.ax.plot(en_table, reduced_data)
sp.ax.axvline(energy_signal, color = 'red', linestyle = 'dotted')
sp.ax.plot(en_table, fit_result , 'tab:red', lw=2, label = 'Gaussian fit')

sp.ax.set_ylim([1, None])

plt.show()
print(f'the integrated signal is {A}')

