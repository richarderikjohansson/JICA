from jica.io import find_file_from_name
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
import numpy as np

fp = find_file_from_name(name="energy_calibration_jica.txt")
caldata = np.loadtxt(fp, dtype=float, skiprows=4)

fig = plt.figure()
gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.plot(caldata[:, 1], caldata[:, 0])
plt.show()
