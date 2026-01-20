#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 13:30:10 2026

@author: umbertor
"""

import numpy as np
from scipy import stats


ratios = np.array([0.099, 0.122, 0.125, 0.132, 0.128, 0.131, 0.112])

mean = np.mean(ratios)

sigma = np.std(ratios, ddof = 1)

sem = sigma / len(ratios)



R = np.array([0.11, 0.13, 0.12, 0.10, 0.14, 0.12])
mu0 = 0.5


t = (mean - mu0) / sem
df = len(ratios) - 1

p_value = 2 * (1 - stats.t.cdf(abs(t), df))
