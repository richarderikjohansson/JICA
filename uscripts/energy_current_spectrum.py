#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 19:30:46 2026

@author: umbertor
The function plots the energy-current spectrum at a given time before CA (given by target_time).
Ideally this should be changed to have mass on the x-axis instead of energy, assuming some 
spacecraft velocity and potential. All the expected compounds should have 1 as charge state.
"""
import numpy as np
from irfpy.scidat import plotting, io
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
data = np.load('../data/jica_datarequest_nr0.npz')
a = data['pipeline_version']
b = data['dataproductnr']
time = data['time']
full_data = data['p1p2m1m2']

dictionary, footer = io.read_from_txt('energy_calibration_jica.txt', 
                                      '  ', ':', '#')   

en_table = dictionary['Energy_center'].data


reduced_data = np.vstack((
    full_data[:-1, :] - full_data[1:, :],
    (full_data[:-1, :] - full_data[1:, :])[-1,:]))
noise = reduced_data[10:, :45].flatten()

reduced_data_clean = reduced_data - noise.mean()

reduced_data_clean[reduced_data_clean < 3*noise.std()] = 0


target_time = -160
reduced_en_table = np.linspace(en_table[0], en_table[-1], len(full_data[:,0]))
index = np.argmin(np.abs(target_time - time))
sp = plotting.SimplePlot(xlabel = 'Energy', ylabel = 'Raw integer current', yscale='log')
sp.ax.plot(reduced_en_table, reduced_data_clean[:, index])
plt.show()
