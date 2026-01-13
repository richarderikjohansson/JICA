#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 19:28:12 2026

@author: umbertor
Function to plot energy-time spectrum as received in data request zero. Here we first remove from
each energy bin the counts of the subsequent energy bin, so that now in each energy bin there are 
only the counts in that specific bin.
After this processing, the function also estimates
the background from an area with very low counts (upper left corner on the spectrum), fits a 
gaussian to it and sets to zero all the counts which are less than a given number of standard 
deviations from the gaussian mean. 
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


reduced_data = np.vstack((
    full_data[:-1, :] - full_data[1:, :],
    (full_data[:-1, :] - full_data[1:, :])[-1,:]))
noise = reduced_data[10:, :45].flatten()
plt.hist(noise, bins=25, density=True)
plt.xlabel('counts')
plt.ylabel('probability')
x = np.arange(-500, 500, 10)
plt.plot(x, stats.norm(noise.mean(), noise.std()).pdf(x))
plt.show()
reduced_data_clean = reduced_data - noise.mean()

reduced_data_clean[reduced_data_clean < 3*noise.std()] = 0


sp = plotting.SimplePlot(xlabel = 'time', ylabel = 'energy bin')
cmap = sp.cmap('inferno', replace_lowest_color_by_white = False)

im = sp.ax.pcolormesh(time, np.arange(64), reduced_data_clean, cmap = cmap, 
                      norm=mpl.colors.LogNorm(vmax = full_data.max(), vmin= full_data.min()))
cmap = sp.add_colorbar(im, 'raw integer current', pad=0.05, aspect=15)

plt.show()
