#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 08:56:20 2026

@author: umbertor
"""

import numpy as np
from irfpy.scidat import plotting, io
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import os
from scipy.optimize import minimize



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
def mass_to_energy_positive(m, U, v, q=1):
        return 1/2*(m * mp * v**2)/e - U

data_neg = np.load('../data/jica_datarequest_nr21.npz')
time_neg = data_neg['time']
full_data_neg = data_neg['n1']
full_data_neg = .12 * full_data_neg
reduced_data_neg = np.vstack((
    full_data_neg[:-1, :] - full_data_neg[1:, :],
    (full_data_neg[:-1, :] - full_data_neg[1:, :])[-1,:]))
def mass_to_energy_negative(m, U, v, q=1):
        return 1/2*(m * mp * v**2)/e + U

U_prel = np.load('../data/jica_datarequest_nr10.npz')['scpot'][0]
v_prel = np.load('../data/jica_datarequest_nr7.npz')['speed'][0] * 1e3


dictionary, footer = io.read_from_txt('energy_calibration_jica.txt', 
                                      '  ', ':', '#')   
en_table = dictionary['Energy_center'].data
en_table_pos = en_table + 1.3160564888762565
en_table_neg = en_table + 2.585570822484897

def gaussian_model(x, A, mu, sigma):
    return (
        A / (sigma * np.sqrt(2 * np.pi)) * np.e ** (- (x - mu) ** 2 / (2 * sigma ** 2))
        )



# ---- Define chi-square ----
def chi2_gaussian(params):
    A, mu, sigma = params
    model = gaussian_model(en_table_reduced, A, mu, sigma)
    return np.sum(((reduced_data - model) / sigma_current)**2)


for i in range(len(time_pos)):
    en_table_reduced_pos = en_table_pos[228:242]
    en_table_reduced = en_table_reduced_pos
    reduced_data = reduced_data_pos[228:242, i]
    # all these errors and sigma are BS, have to crrect for them
    sigma_current = np.ones(len(reduced_data))
    sigma_current[reduced_data > 0] = np.sqrt(reduced_data[reduced_data > 0])
    # ---- Initial guesses ----
    p0_gauss_pos = [100, 707, 15]
    
    # ---- Fit via chi-square minimization ----
    res_gauss = minimize(chi2_gaussian, p0_gauss_pos,
                         method='Nelder-Mead')

    A_pos, mu_pos, sigma_pos = res_gauss.x
    print(A_pos, np.max(reduced_data))

    x_fit = np.linspace(min(en_table), max(en_table), 500)
    fit_result_pos = gaussian_model(en_table_reduced, A_pos, mu_pos, sigma_pos)
    
    en_table_reduced_neg = en_table_neg[350:380]
    en_table_reduced = en_table_reduced_neg
    reduced_data = reduced_data_neg[350:380, i]
    sigma_current = np.ones(len(reduced_data))
    sigma_current[reduced_data > 0] = np.sqrt(reduced_data[reduced_data > 0])
    p0_gauss_neg = [10, 1115, 15]
    # ---- Fit via chi-square minimization ----

    res_gauss = minimize(chi2_gaussian, p0_gauss_neg,
                         method='Nelder-Mead')
    
    A_neg, mu_neg, sigma_neg = res_gauss.x
    fit_result_neg = gaussian_model(en_table_reduced, A_neg, mu_neg, sigma_neg)
    print(A_neg, np.max(reduced_data))

    sp = plotting.SimplePlot(rows = 2, xlabel = 'Energy [eV]', ylabel = 'current [pA]', yscale='log',
                             title = f'ratio = {round(A_neg/A_pos, 3)}, t = {time_pos[i]}',
                             figure_ratio=.9, tight_layout=2)
    sp.ax[0].plot(en_table_pos, reduced_data_pos[:, i])
    sp.ax[0].axvline(mass_to_energy_positive(60.03, U_prel, v_prel), color = 'r', linestyle = 'dotted')
    sp.ax[0].set_ylim([.1, None])
    sp.ax[0].set_title('Positive')
    sp.ax[0].plot(en_table_reduced_pos, fit_result_pos, 'tab:red', lw=2, label = 'Gaussian fit')


    sp.ax[1].plot(en_table_neg, reduced_data_neg[:, i])
    sp.ax[1].axvline(mass_to_energy_positive(94.9401, U_prel, v_prel), color = 'r', linestyle = 'dotted')
    sp.ax[1].set_ylim([.1, None])
    sp.ax[1].set_title('Negative')
    sp.ax[1].plot(en_table_reduced_neg, fit_result_neg, 'tab:red', lw=2, label = 'Gaussian fit')
    plt.show()
    filename = f'plots/pos_neg_spectra_{-time_pos[i]}.png'
    plotting.savefig(sp.fig, filename)  
    plt.close(sp.fig)