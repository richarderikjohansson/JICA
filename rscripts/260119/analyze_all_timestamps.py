from jica.io import read_data_from_name, read_data_from_id
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit


def mass_to_energy(ion_mass):
    u = read_data_from_id(id=81)
    v = read_data_from_id(id=80)
    e = 1.602176634e-19  # electron charge
    mp = 1.67262192595e-27  # proton mass
    sc_vel = v.speed * 1e3
    sc_pot = u.scpot

    energy = (ion_mass * mp * sc_vel * sc_vel) / (2 * e) + sc_pot
    return energy


def calc_ion_energies(ions):
    ptm = float(ions.pos_target[1])
    ntm = float(ions.neg_target[1])
    pos_en = mass_to_energy(ptm)
    neg_en = mass_to_energy(ntm)
    return pos_en, neg_en


def gaussian_offset(t, A, mu, sigma):
    return A * np.exp(-((t - mu) ** 2) / (2 * sigma**2))


table_pos = read_data_from_name("pos_en_table_adjusted.npz")
table_neg = read_data_from_name("neg_en_table_adjusted.npz")
ions = read_data_from_name("ions.npz")
pos_en, neg_en = calc_ion_energies(ions)

pos_spec = read_data_from_name("jica_datarequest_nr16.npz")
neg_spec = read_data_from_name("jica_datarequest_nr17.npz")

p1_all = 0.12 * pos_spec.p1
n1_all = 0.12 * neg_spec.n1

p1_time = pos_spec.time
n1_time = neg_spec.time

p1_reduced = []
n1_reduced = []

for i in range(0, 8):
    p1 = p1_all[:, i]
    n1 = n1_all[:, i]
    p1 = np.reshape(p1, (512, 1))
    n1 = np.reshape(n1, (512, 1))
    p1red = np.vstack(
        (
            p1[:-1, :] - p1[1:, :],
            (p1[:-1, :] - p1[1:, :])[-1, :],
        )
    )
    n1red = np.vstack(
        (
            n1[:-1, :] - n1[1:, :],
            (n1[:-1, :] - n1[1:, :])[-1, :],
        )
    )

    p1_reduced.append(p1red)
    n1_reduced.append(n1red)

p1_reduced = np.array(p1_reduced)
n1_reduced = np.array(n1_reduced)

pos_energy = table_pos.en_adjusted
neg_energy = table_neg.en_adjusted
p1mask = (695 <= pos_energy) & (720 >= pos_energy)
n1mask = (1105 <= neg_energy) & (1130 >= neg_energy)

fig = plt.figure(figsize=(12, 6))
gs = GridSpec(1, 2)
left = fig.add_subplot(gs[0, 0])
right = fig.add_subplot(gs[0, 1])
for p1, t in zip(p1_reduced, p1_time):
    left.plot(pos_energy[p1mask], p1[p1mask], label=f"t: {t}")

for n1, t in zip(n1_reduced, n1_time):
    right.plot(neg_energy[n1mask], n1[n1mask], label=f"t: {t}")


for ax in fig.axes:
    ax.set_yscale("log")
    ax.set_xlabel("Energy [E/q]")
    ax.set_ylabel("ion current [pA]")
    ax.legend()
fig.savefig("figs/all_time_stamps.png")

n1_fits = []
n1_vars = []
p1_fits = []
p1_vars = []

for n1 in n1_reduced:
    xdata = neg_energy[n1mask]
    ydata = n1[n1mask]

    enfit = np.linspace(xdata[0], xdata[-1], 200)
    popt, pcov = curve_fit(
        gaussian_offset,
        xdata.flatten(),
        ydata.flatten(),
        p0=[10, 1115, 5],
    )
    fit = gaussian_offset(enfit, *popt)
    n1_fits.append(fit)
    n1_vars.append(popt)


for p1 in p1_reduced:
    xdata = pos_energy[p1mask]
    ydata = p1[p1mask]

    enfit = np.linspace(xdata[0], xdata[-1], 200)
    popt, pcov = curve_fit(
        gaussian_offset,
        xdata.flatten(),
        ydata.flatten(),
        p0=[100, 707, 5],
    )
    fit = gaussian_offset(enfit, *popt)
    p1_fits.append(fit)
    p1_vars.append(popt)


for n, p in zip(n1_fits, p1_fits):
    n_amp = max(n)
    p_amp = max(p)
    print(n_amp / p_amp)


for p1, n1, fp, fn, t in zip(p1_reduced, n1_reduced, p1_fits, n1_fits, p1_time):
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 1)
    enpos = np.linspace(pos_energy[p1mask][0], pos_energy[p1mask][-1], 200)
    enneg = np.linspace(neg_energy[n1mask][0], neg_energy[n1mask][-1], 200)
    top = fig.add_subplot(gs[0, 0])
    bottom = fig.add_subplot(gs[1, 0])
    top.set_yscale("log")
    bottom.set_yscale("log")
    nmax = max(fn)
    pmax = max(fp)
    plt.suptitle(f"t: {t}, ratio:{nmax / pmax} ")
    #
    # positive ions
    top.set_title("Positive ions")
    top.plot(pos_energy, p1, color="black", label="Measurement")
    top.plot(enpos, fp, color="red", label="fit")
    top.set_ylim((1, 1e5))
    # negative ions
    bottom.set_title("Negative ions")
    bottom.plot(neg_energy, n1, color="black", label="Measurement")
    bottom.plot(enneg, fn, color="red", label="fit")
    bottom.set_ylim((1, 1e5))
    fig.savefig(f"figs/meas_and_fit_{t}.png")
    plt.close()
