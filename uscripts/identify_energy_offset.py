#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 10:54:35 2026

@author: umbertor
"""

import numpy as np
from irfpy.scidat import plotting, io
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import os
from matplotlib.patches import Rectangle
os.chdir("/home/umbertor/Documents/PhD/courses/statistics/space_race/JICA/uscripts")
data = np.load('../data/jica_datarequest_nr9.npz')
a = data['pipeline_version']
b = data['dataproductnr']
time = data['time']
full_data = data['n2']
full_data = .1 * full_data

reduced_data = np.vstack((
    full_data[:-1, :] - full_data[1:, :],
    (full_data[:-1, :] - full_data[1:, :])[-1,:]))

#%% energy-current, negative sensors
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
en_table = dictionary['Energy_center'].data

U_prel = np.load('../data/jica_datarequest_nr10.npz')['scpot'][0]
v_prel = np.load('../data/jica_datarequest_nr7.npz')['speed'][0] * 1e3
def mass_to_energy_negative(m, U, v, q=1):
        return 1/2*(m * mp * v**2)/e + U
negative_energies = []
for i in range(len(negative_masses)):
    negative_energies.append(mass_to_energy_negative(negative_masses[i], U_prel, v_prel))


sp = plotting.SimplePlot(xlabel = 'Energy [eV]', ylabel = '- current [pA]', yscale='log')
for i in negative_energies:
    sp.ax.axvline(i, color ='k', linestyle = 'dotted')
sp.ax.plot(en_table, reduced_data[:, 0])
sp.ax.axvline(mass_to_energy_negative(94.9401, U_prel, v_prel), color = 'red', linestyle = 'dotted')
sp.ax.set_ylim([1, None])

plt.show()
#%%identify offset, here we are using negative biomarker to find the offset

energy_offset = en_table[293:410]
reduced_data_offset = reduced_data[293:410, 0]
sp = plotting.SimplePlot(xlabel = 'Energy [eV]', ylabel = '- current [pA]', yscale='log',
                         title = f'U = {U_prel} V, v = {v_prel/1e3} km s-1')
sp.ax.plot(energy_offset, reduced_data_offset)
sp.ax.axvline(mass_to_energy_negative(94.9401, U_prel, v_prel), color = 'red', linestyle = 'dotted')
sp.ax.set_ylim([1, None])
plt.show()
print(f'Energy offset is {mass_to_energy_negative(94.9401, U_prel, v_prel) - energy_offset[np.argmax(reduced_data_offset)]}')

#%% other sensor
data = np.load('../data/jica_datarequest_nr11.npz')
a = data['pipeline_version']
b = data['dataproductnr']
time = data['time']
full_data = data['n1']
full_data = .12 * full_data

reduced_data = np.vstack((
    full_data[:-1, :] - full_data[1:, :],
    (full_data[:-1, :] - full_data[1:, :])[-1,:]))

#%% energy-current
# unit is amu

sp = plotting.SimplePlot(xlabel = 'Energy [eV]', ylabel = '- current [pA]', yscale='log')
for i in negative_energies:
    sp.ax.axvline(i, color ='k', linestyle = 'dotted')
sp.ax.plot(en_table, reduced_data[:, 0])
sp.ax.axvline(mass_to_energy_negative(94.9401, U_prel, v_prel), color = 'red', linestyle = 'dotted')
sp.ax.set_ylim([1, None])

plt.show()

#%%identify offset, here we are using negative biomarker to find the offset

energy_offset = en_table[293:410]
reduced_data_offset = reduced_data[293:410, 0]
sp = plotting.SimplePlot(xlabel = 'Energy [eV]', ylabel = '- current [pA]', yscale='log',
                         title = f'U = {U_prel} V, v = {v_prel/1e3} km s-1')
sp.ax.plot(energy_offset, reduced_data_offset)
sp.ax.axvline(mass_to_energy_negative(94.9401, U_prel, v_prel), color = 'red', linestyle = 'dotted')
sp.ax.set_ylim([1, None])
plt.show()
print(f'Energy offset is {mass_to_energy_negative(94.9401, U_prel, v_prel) - energy_offset[np.argmax(reduced_data_offset)]}')


#%% energy-current, positive sensors
# unit is amu
data = np.load('../data/jica_datarequest_nr6.npz')
a = data['pipeline_version']
b = data['dataproductnr']
time = data['time']
full_data = data['p2']
full_data = .1 * full_data

reduced_data = np.vstack((
    full_data[:-1, :] - full_data[1:, :],
    (full_data[:-1, :] - full_data[1:, :])[-1,:]))

#%% energy-current
# unit is amu
mp = 1.6726e-27
e = 1.6022e-19
positive_masses = np.array([1.0081, 2.0162, 3.0243, 4.0026, 5.0107, 12.0111, 13.019200000000001, 
                             14.0273, 15.035400000000001, 16.0435, 17.0516, 24.0222, 25.0303, 
                             26.038400000000003, 27.0465, 28.0546, 29.0627, 31.0789, 37.0414, 
                             38.0495, 39.05760000000001, 40.06570000000001, 41.073800000000006, 
                             42.081900000000005, 43.09, 44.0981, 45.1062, 49.0525, 50.0606,
                             51.06870000000001, 53.084900000000005, 55.1011, 57.1173, 55.8452, 
                             81.8836, 83.8998, 71.8887, 56.8533, 57.861399999999996, 24.3052, 
                             50.343599999999995, 52.3598, 40.3487, 25.313299999999998, 26.3214, 
                             22.9897, 49.028099999999995, 51.0443, 39.0332, 25.0059, 28.0852, 
                             29.0933, 30.1014, 60.03640000000001])


def mass_to_energy_positive(m, U, v, q=1):
        return 1/2*(m * mp * v**2)/e - U
positive_energies = []
for i in range(len(positive_masses)):
    positive_energies.append(mass_to_energy_positive(positive_masses[i], U_prel, v_prel))


sp = plotting.SimplePlot(xlabel = 'Energy [eV]', ylabel = 'current [pA]', yscale='log')
for i in positive_energies:
    sp.ax.axvline(i, color ='k', linestyle = 'dotted')
sp.ax.plot(en_table, reduced_data[:, 0])
sp.ax.axvline(mass_to_energy_negative(60.03, U_prel, v_prel), color = 'red', linestyle = 'dotted')
sp.ax.set_ylim([1, None])

plt.show()

#%%identify offset, here we are using C3H5+ to find the offset

energy_offset = en_table[145:175]
reduced_data_offset = reduced_data[145:175, 0]
sp = plotting.SimplePlot(xlabel = 'Energy [eV]', ylabel = '- current [pA]', yscale='log',
                         title = f'U = {U_prel} V, v = {v_prel/1e3} km s-1')
sp.ax.plot(energy_offset, reduced_data_offset)
sp.ax.axvline(mass_to_energy_positive(41.073800000000006, U_prel, v_prel), color = 'red', linestyle = 'dotted')
sp.ax.set_ylim([1, None])
plt.show()
print(f'Energy offset is {mass_to_energy_positive(41.073800000000006, U_prel, v_prel) - energy_offset[np.argmax(reduced_data_offset)]}')

#%% other sensor
data = np.load('../data/jica_datarequest_nr8.npz')
a = data['pipeline_version']
b = data['dataproductnr']
time = data['time']
full_data = data['p1']
full_data = .12 * full_data

reduced_data = np.vstack((
    full_data[:-1, :] - full_data[1:, :],
    (full_data[:-1, :] - full_data[1:, :])[-1,:]))

#%% energy-current
positive_energies = []
for i in range(len(positive_masses)):
    positive_energies.append(mass_to_energy_positive(positive_masses[i], U_prel, v_prel))


sp = plotting.SimplePlot(xlabel = 'Energy [eV]', ylabel = 'current [pA]', yscale='log')
for i in positive_energies:
    sp.ax.axvline(i, color ='k', linestyle = 'dotted')
sp.ax.plot(en_table, reduced_data[:, 0])
sp.ax.axvline(mass_to_energy_negative(60.03, U_prel, v_prel), color = 'red', linestyle = 'dotted')
sp.ax.set_ylim([1, None])

plt.show()

#%%identify offset, here we are using C3H5+ to find the offset

energy_offset = en_table[145:175]
reduced_data_offset = reduced_data[145:175, 0]
sp = plotting.SimplePlot(xlabel = 'Energy [eV]', ylabel = '- current [pA]', yscale='log',
                         title = f'U = {U_prel} V, v = {v_prel/1e3} km s-1')
sp.ax.plot(energy_offset, reduced_data_offset)
sp.ax.axvline(mass_to_energy_positive(41.073800000000006, U_prel, v_prel), color = 'red', linestyle = 'dotted')
sp.ax.set_ylim([1, None])
plt.show()
print(f'Energy offset is {mass_to_energy_positive(41.073800000000006, U_prel, v_prel) - energy_offset[np.argmax(reduced_data_offset)]}')