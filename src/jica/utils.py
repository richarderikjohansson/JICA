import numpy as np
import matplotlib.pyplot as plt
from .io import find_file_from_id, find_file_from_name
from .const import e, mp
from scipy import stats
from types import SimpleNamespace


def red_j_spec_err(spec_arr, e_range, t_range, savename):  # , corner_j, corner_e
    """
    red_j_spec_err
    This function cleans the energy-current spectrum from noise, which is estimated by a given empty corner of the matrix.
    :param spec_arr: spectrum data
    :param e_range: energy range that is assumed empty
    :param t_range: time range that is assumed empty
    :param savename: noise fit is saved under 'figs/noise_fit_{savename}.png'

    Returns:
    --------
    :param spec_cleaned: an array with one energy dimension less than the input, cleaned.
    """
    # reduce higher energy bins
    spec_arr = spec_arr.copy()
    e_l = e_range[0]
    e_u = e_range[1]
    j_l = t_range[0]
    j_u = t_range[1]
    spec_c = spec_arr[:-1, :] - spec_arr[1:, :]
    # plot and savefig of counts
    counts = spec_c[e_l:e_u, j_l:j_u].flatten()
    plt.hist(counts, bins=15, density=True)
    plt.xlabel("counts")
    plt.ylabel("probability")
    x = np.arange(-500, 500, 10)
    count_dist = stats.norm(counts.mean(), counts.std()).pdf(x)
    plt.plot(x, count_dist)
    plt.savefig(f"figs/noise_fit_{savename}.png")
    # reduce 3sigma
    spec_cleaned = spec_c - counts.mean()
    spec_cleaned[spec_cleaned < 3 * counts.std()] = 0

    return spec_cleaned


def calc_current(spec_arr, instrument):
    factor = 0.12
    if instrument == "p2":
        factor = 0.1
    if instrument == "n1":
        factor = -0.12
    if instrument == "n2":
        factor = -0.1
    return factor * spec_arr


def calc_mass(energy, sc_pot, vel):
    """Lenas mass function

    :param energy:
    :param sc_pot:
    :param vel:
    :return:
    """
    e = 1.602176634e-19  # electron charge
    m_p = 1.67262192595e-27  # proton mass
    return e / m_p * energy * (2 - sc_pot) / vel / vel


def calculate_mass(E: float, v: float, U: float, q_sign: int) -> float:
    """Richards mass function

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


def current2counts(ion_current: np.ndarray, dt: int):
    """Function to calculate counts from current in pA

    :param ion_current: array with ion currents
    :param dt: time resolution
    :return: counts
    """
    current_in_ampere = ion_current / 1e12
    return (current_in_ampere * dt) / e
