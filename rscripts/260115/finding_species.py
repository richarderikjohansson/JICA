from jica.io import find_file_from_id, find_file_from_name
from jica.const import POS, NEG
from jica.utils import red_j_spec_err, calc_mass, calc_current
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from types import SimpleNamespace


def read_data_from_id(id: int):
    fp = find_file_from_id(id=id)
    npdata = np.load(fp)
    dct = {k: npdata[k] for k in npdata.keys()}
    dc = SimpleNamespace(**dct)
    return dc


def read_data_from_name(name: str):
    fp = find_file_from_name(name)
    npdata = np.load(fp)
    dct = {k: npdata[k] for k in npdata.keys()}
    dc = SimpleNamespace(**dct)
    return dc


def calculate_mass(E, v, U, q_sign):
    """

    :param E: Energy in Ev/q
    :param v: Velocity in km/s
    :param U: Voltage
    :param q_sign: ion sign
    :return: mass in amu
    """
    e = 1.602e-19
    mp = 1.6726e-27

    E = E * e
    if q_sign > 0:
        kinetic_energy_J = E + U * e
    else:
        kinetic_energy_J = E - U * e

    m_over_q_mp = 2 * kinetic_energy_J / (v**2)
    mass_kg = m_over_q_mp
    mass_amu = mass_kg / mp

    return mass_amu


ion_pos = read_data_from_id(id=65)
en_table_pos = read_data_from_name("pos_en_table_adjusted.npz")
en_table_neg = read_data_from_name("neg_en_table_adjusted.npz")
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

fig = plt.figure(figsize=(12, 6))
plt.suptitle(f"v={vel / 1e3} km/s, U={scpot}")

gs = GridSpec(1, 1)
pos = fig.add_subplot(gs[0, 0])
pos.set_ylabel("current [pA]")
for ax in fig.axes:
    ax.set_yscale("log")
    ax.set_ylim((2, 1e4))
    ax.set_xlabel("Mass [amu]")


posmin, posmax = pos.get_ylim()
for i in POS:
    pos.vlines(
        i, ymin=posmin, ymax=posmax, colors="grey", alpha=0.2, linestyles="dotted"
    )
pos.plot(pos_masses, reduced_ion_pos[:, 0], color="black")
pos.vlines(
    60.035, ymin=posmin, ymax=posmax, label=r"$PC_2H_5^+$", color="red", alpha=0.5
)
pos.legend()
fig.savefig("biomarker_pos.pdf")


fig = plt.figure(figsize=(12, 6))
plt.suptitle(f"v={vel / 1e3} km/s, U={scpot}")

gs = GridSpec(1, 1)
neg = fig.add_subplot(gs[0, 0])
neg.set_ylabel("current [pA]")
for ax in fig.axes:
    ax.set_yscale("log")
    ax.set_ylim((2, 1e4))
    ax.set_xlabel("Mass [amu]")


negmin, negmax = neg.get_ylim()
for i in NEG:
    neg.vlines(
        i, ymin=negmin, ymax=negmax, colors="grey", alpha=0.2, linestyles="dotted"
    )
neg.plot(neg_masses, reduced_ion_neg[:, 0], color="black")
neg.vlines(94.93, ymin=negmin, ymax=negmax, label=r"$CH_3Br^-$", color="red", alpha=0.5)
neg.legend()
fig.savefig("biomarker_neg.pdf")
