from jica.io import find_file_from_id, find_file_from_name
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
from numpy import interp
from scipy.stats import norm


@dataclass
class DataN1:
    time: np.ndarray
    mat: np.ndarray
    noise: np.float64
    filtmat: np.ndarray


@dataclass
class DataALT:
    time: np.ndarray
    alt: np.ndarray


id68 = np.load(find_file_from_id(id=68))
id79 = np.load(find_file_from_name(name="jica_datarequest_nr1_and_nr2_interp.npz"))
datan1 = DataN1(time=id68["time"], mat=id68["n1"], noise=None, filtmat=None)
dataalt = DataALT(time=id79["time"], alt=id79["alt"])

fig = plt.figure(figsize=(8, 6))
gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.hist(datan1.mat[-2, :], density=True, bins=5, label="Energy bin -2")

mu, std = norm.fit(datan1.mat[-1, :])
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
ax.plot(x, p, linewidth=2, color="red", label="Fit")
ax.set_title(rf"Energy bin data for n1 with Gaussian fit ($\sigma$={std}, $\mu$={mu})")
ax.set_xlabel("Raw int count")
ax.set_ylabel("Density")
ax.legend()
fig.savefig("figs/id68_n1_noise.png")
