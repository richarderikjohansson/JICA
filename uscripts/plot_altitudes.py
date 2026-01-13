#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 19:34:09 2026

@author: umbertor
"""

from irfpy.scidat import plotting
import numpy as np

pos1 = np.load('../data/jica_datarequest_nr1.npz')

alt1 = pos1["alt"]
time1 = pos1["time"]
                                              
pos2 = np.load('../data/jica_datarequest_nr2.npz')

alt2 = pos2["alt"]
time2 = pos2["time"]



sp = plotting.SimplePlot()
sp.ax.plot(time1, alt1, color = 'k')
sp.ax.plot(time2, alt2, color = 'k')