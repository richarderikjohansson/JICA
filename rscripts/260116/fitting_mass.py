from jica.io import read_data_from_id, read_data_from_name
from jica.utils import calculate_mass
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.optimize import curve_fit


def gaussian_offset(t, A, mu, sigma):
    return A * np.exp(-((t - mu) ** 2) / (2 * sigma**2))


ion_pos = read_data_from_id(id=65)
en_table_pos = read_data_from_name("pos_en_table_adjusted.npz")
en_table_neg = read_data_from_name("neg_en_table_adjusted.npz")
ions = read_data_from_name("ions.npz")
ion_neg = read_data_from_id(id=73)
u = read_data_from_id(id=81)
v = read_data_from_id(id=80)


vel = v.speed[0] * 1e3
scpot = u.scpot
ion_neg = 0.1 * ion_neg.n2
ion_pos = 0.1 * ion_pos.p2
reduced_ion_neg = np.vstack(
    (
        ion_neg[:-1, :] - ion_neg[1:, :],
        (ion_neg[:-1, :] - ion_neg[1:, :])[-1, :],
    )
)
reduced_ion_pos = np.vstack(
    (
        ion_pos[:-1, :] - ion_pos[1:, :],
        (ion_pos[:-1, :] - ion_pos[1:, :])[-1, :],
    )
)

pos_masses = calculate_mass(en_table_pos.en_adjusted, vel, scpot, 1)
neg_masses = calculate_mass(en_table_neg.en_adjusted, vel, scpot, -1)

pt = float(ions.pos_target[1])
nt = float(ions.neg_target[1])
pmask = (pt - 2 <= pos_masses) & (pt + 2 >= pos_masses)
nmask = (nt - 2 <= neg_masses) & (nt + 2 >= neg_masses)

fig = plt.figure(figsize=(10, 6))
gs = GridSpec(1, 2)

left = fig.add_subplot(gs[0, 0])
right = fig.add_subplot(gs[0, 1])

for ax in fig.axes:
    ax.set_yscale("log")
left.plot(pos_masses[pmask], reduced_ion_pos[pmask])
right.plot(neg_masses[nmask], reduced_ion_neg[nmask])
fig.savefig("figs/species_zoomed.pdf")


popt, pcov = curve_fit(
    gaussian_offset,
    pos_masses[pmask].flatten(),
    reduced_ion_pos[pmask].flatten(),
    p0=[100, 60, 1],
)
nopt, ncov = curve_fit(
    gaussian_offset,
    neg_masses[nmask].flatten(),
    reduced_ion_neg[nmask].flatten(),
    p0=[10, 95, 1],
)


print(pos_masses[pmask][-1])
pmass = np.linspace(pos_masses[pmask][0], pos_masses[pmask][-1], 200)
nmass = np.linspace(neg_masses[nmask][0], neg_masses[nmask][-1], 200)
posfit = gaussian_offset(pmass, *popt)
negfit = gaussian_offset(nmass, *nopt)
fig = plt.figure(figsize=(10, 6))
gs = GridSpec(1, 2)

left = fig.add_subplot(gs[0, 0])
right = fig.add_subplot(gs[0, 1])

left.plot(pos_masses[pmask], reduced_ion_pos[pmask])
left.plot(pmass, posfit, color="red")
right.plot(nmass, negfit, color="red")
right.plot(neg_masses[nmask], reduced_ion_neg[nmask])
for ax in fig.axes:
    ymin, ymax = ax.get_ylim()
    ax.set_yscale("log")
    ax.set_ylim((1, ymax))
fig.savefig("figs/species_zoomed_fit.png")
