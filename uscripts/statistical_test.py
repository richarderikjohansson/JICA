#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 15:22:01 2026

@author: umbertor
"""

import numpy as np
from irfpy.scidat import plotting, io
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import os
from scipy.optimize import minimize


dt = 2e-3
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
reduced_data_pos = reduced_data_pos * dt * 1e-12/e

def mass_to_energy_positive(m, U, v, q=1):
        return 1/2*(m * mp * v**2)/e - U

data_neg = np.load('../data/jica_datarequest_nr21.npz')
time_neg = data_neg['time']
full_data_neg = data_neg['n1']
full_data_neg = .12 * full_data_neg
reduced_data_neg = np.vstack((
    full_data_neg[:-1, :] - full_data_neg[1:, :],
    (full_data_neg[:-1, :] - full_data_neg[1:, :])[-1,:]))

reduced_data_neg = reduced_data_neg * dt * 1e-12/e
def mass_to_energy_negative(m, U, v, q=1):
        return 1/2*(m * mp * v**2)/e + U

U_prel = np.load('../data/jica_datarequest_nr10.npz')['scpot'][0]
v_prel = np.load('../data/jica_datarequest_nr7.npz')['speed'][0] * 1e3


dictionary, footer = io.read_from_txt('energy_calibration_jica.txt', 
                                      '  ', ':', '#')   
en_table = dictionary['Energy_center'].data
en_table_pos = en_table + 1.3160564888762565
en_table_neg = en_table + 2.585570822484897


max_pos = []
max_neg = []
ratios = []
for i in range(len(time_pos)):
    en_table_reduced_pos = en_table_pos[228:242]
    en_table_reduced = en_table_reduced_pos
    reduced_data = reduced_data_pos[228:242, i]
    # all these errors and sigma are BS, have to crrect for them
    max_pos.append(np.max(reduced_data))
    
    en_table_reduced_neg = en_table_neg[350:380]
    en_table_reduced = en_table_reduced_neg
    reduced_data = reduced_data_neg[350:380, i]
    max_neg.append(np.max(reduced_data))
    
    ratios.append(max_neg[-1] / max_pos[-1])


#excluding the first ratio as we don't have negative data
max_pos = np.array(max_pos[1:])
max_neg = np.array(max_neg[1:])
ratios = np.array(ratios[1:])

sigmas_ratio = np.sqrt(max_neg / max_pos **2 + max_neg**2/max_pos**3)


#%% student t test


mean = np.mean(ratios)

weighted_mean = np.average(ratios, weights=sigmas_ratio)

sigma = np.std(ratios, ddof = 1)

sem = sigma / len(ratios)

weighted_sem = np.sqrt(1/np.sum(1/sigmas_ratio**2))


mu0 = 0.5


t = (mean - mu0) / sem
df = len(ratios) - 1

p_value = 2 * (1 - stats.t.cdf(abs(t), df))
