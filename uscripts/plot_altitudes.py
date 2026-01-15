#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 19:34:09 2026

@author: umbertor
"""

from irfpy.scidat import plotting
import numpy as np

pos = np.load('../data/jica_datarequest_nr2_and_nr1.npz')

alt = pos["alt"]
tim = pos["time"]
sp = plotting.SimplePlot(rows = 2)
                       
sp.ax[0].plot(tim, alt, color = 'k')
sp.ax[0].set_xlabel('time')
sp.ax[0].set_ylabel('altitude [km]')


v = np.gradient(alt, tim)

sp.ax[1].plot(tim, v, color = 'k')
sp.ax[1].set_xlabel('time')
sp.ax[1].set_ylabel('vertical velocity [kms-1]')