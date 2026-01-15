#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 08:56:04 2026

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
data = np.load('../data/jica_datarequest_nr4.npz')
a = data['pipeline_version']
b = data['dataproductnr']
time = data['time']
full_data = data['n1']
full_data = .12 * full_data

sp = plotting.SimplePlot(xlabel='time', ylabel='energy bin')
cmap = sp.cmap('inferno', replace_lowest_color_by_white = False)

im = sp.ax.pcolormesh(time, np.arange(256), full_data, cmap = cmap, 
                      norm=mpl.colors.LogNorm(vmax = full_data.min(), vmin= full_data.max()))
cmap = sp.add_colorbar(im, '- current [pA]', pad=0.05, aspect=15)
# noise indexes
i0, i1 = 5, len(time) - 1      # time indices
j0, j1 = 130, len(full_data[:,0])-1        # energy-bin indices

rect = Rectangle(
    (time[i0], j0),                 # bottom-left corner
    time[i1] - time[i0],             # width
    j1 - j0,                          # height
    linewidth=2,
    edgecolor='cyan',
    facecolor='none'                
)

sp.ax.add_patch(rect)
plt.show()
#%%
reduced_data = np.vstack((
    full_data[:-1, :] - full_data[1:, :],
    (full_data[:-1, :] - full_data[1:, :])[-1,:]))
noise = reduced_data[j0:, i0:].flatten()
plt.hist(noise, bins=20, density=True)
plt.xlabel('counts')
plt.ylabel('probability')
x = np.arange(-50, 50, 1)
plt.plot(x, stats.norm(noise.mean(), noise.std()).pdf(x))
plt.show()
reduced_data_clean = reduced_data - noise.mean()

reduced_data_clean[reduced_data_clean < 2*noise.std()] = 0


sp = plotting.SimplePlot(rows=2, figure_ratio=.8)
cmap = sp.cmap('inferno', replace_lowest_color_by_white = False)

im = sp.ax[0].pcolormesh(time, np.arange(256), reduced_data_clean, cmap = cmap, 
                      norm=mpl.colors.LogNorm(vmax = full_data.max(), vmin= full_data.min()))
cmap = sp.add_colorbar(im, '- current [pA]', pad=0.05, aspect=15)
sp.ax[0].set_xlabel('time')
sp.ax[0].set_ylabel('energy bin')


pos = np.load('../data/jica_datarequest_nr1_and_nr2_interp.npz')

alt = pos["alt"]
tim = pos["time"]
                                              
sp.ax[1].plot(tim, alt, color = 'k')
sp.ax[1].set_xlabel('time')
sp.ax[1].set_ylabel('altitude [km]')
plt.show()

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
en_table = dictionary['Energy_center'].data
reduced_en_table = np.linspace(en_table[0], en_table[-1], len(full_data[:,0]))

U_prel = -1
v_prel = np.load('../data/jica_datarequest_nr7.npz')['speed'][0] * 1e3
def energy_to_mass_negative(E, U, v, q=1):
        return 2*q*e*(E - U)/(mp * v**2)
mass = energy_to_mass_negative(reduced_en_table, U = U_prel, v = v_prel)

target_times = [-144, -142, -140]
for target_time in target_times:
    index = np.argmin(np.abs(target_time - time))
    sp = plotting.SimplePlot(xlabel = 'Energy [eV]', ylabel = '- current [pA]', yscale='log')
    sp.ax.plot(reduced_en_table, reduced_data_clean[:, index])
    plt.show()

    # plotting mass instead of energy, assuming velocity and potential

    sp = plotting.SimplePlot(xlabel = 'Mass [amu]', ylabel = '- current [pA]', yscale='log',
                         title = f'U = {U_prel} V, v = {v_prel/1e3} km s-1, t = {target_time}')
    for i in negative_masses:
        sp.ax.axvline(i, color ='k', linestyle = 'dotted')
    sp.ax.plot(mass, reduced_data_clean[:, index])
    sp.ax.axvline(94.9401, color = 'red', linestyle = 'dotted')
    plt.show()

    