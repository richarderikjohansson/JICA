from jica.io import find_file_from_id, find_file_from_name, get_datadir
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

data = np.load(find_file_from_id(id=79))

fig = plt.figure()
gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.plot(data["time"], data["alt"])
fig.savefig("figs/id79.pdf")
