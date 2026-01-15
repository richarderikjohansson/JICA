#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 19:26:11 2026

@author: umbertor
Function to plot energy-time spectrum as received in data request zero. The function also estimates
the background from an area with very low counts (upper left corner on the spectrum), fits a 
gaussian to it and sets to zero all the counts which are less than a given number of standard 
deviations from the gaussian mean. It also plots the counts in bin 0 (should be the sum of all
energies) as a function of time (altitude)
"""

import numpy as np
from irfpy.scidat import plotting, io
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import os
os.chdir("/home/umbertor/Documents/PhD/courses/statistics/space_race/JICA/uscripts")

data = np.load('../data/jica_datarequest_nr0.npz')
a = data['pipeline_version']
b = data['dataproductnr']
time = data['time']
full_data = data['p1p2m1m2']

dictionary, footer = io.read_from_txt('energy_calibration_jica.txt', 
                                      '  ', ':', '#')   

en_table = dictionary['Energy_center'].data
sp = plotting.SimplePlot(xlabel = 'time', ylabel = 'energy bin')
cmap = sp.cmap('inferno', replace_lowest_color_by_white = False)

im = sp.ax.pcolormesh(time, np.arange(64), full_data, cmap = cmap, 
                      norm=mpl.colors.LogNorm(vmax = 5e8, vmin= 1e3))
cmap = sp.add_colorbar(im, 'raw integer current', pad=0.05, aspect=15)


plt.show()

noise = full_data[10:, :45].flatten()
plt.hist(noise, bins=25, density=True)
plt.xlabel('counts')
plt.ylabel('probability')

x = np.arange(1500, 2300, 10)
plt.plot(x, stats.norm(noise.mean(), noise.std()).pdf(x))
plt.show()

full_data_clean = full_data - noise.mean()
full_data_clean[full_data_clean < 2*noise.std()] = 0

sp = plotting.SimplePlot(xlabel = 'time', ylabel = 'energy bin')
cmap = sp.cmap('inferno', replace_lowest_color_by_white = False)

im = sp.ax.pcolormesh(time, np.arange(64), full_data_clean, cmap = cmap, 
                      norm=mpl.colors.LogNorm(vmax = full_data.max(), vmin= full_data.min()))
cmap = sp.add_colorbar(im, 'raw integer current', pad=0.05, aspect=15)

plt.show()

plt.semilogx(full_data_clean[0], -time)
plt.xlabel('raw integer current')
plt.ylabel('-time')
plt.show()