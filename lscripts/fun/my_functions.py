import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from molmass import Formula


def red_j_spec_err(spec_arr, e_range, t_range, savename=None): # , corner_j, corner_e
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
    print(spec_arr.ndim)
    if spec_arr.ndim== 1:
        spec_c = spec_arr[:-1] - spec_arr[1:]
    else:
        spec_c = spec_arr[:-1, :] - spec_arr[1:, :]
    # plot and savefig of counts
    if spec_arr.ndim== 1:
        counts = spec_c[e_l:e_u]
    else:
        counts = spec_c[e_l:e_u, j_l:j_u].flatten()
    plt.hist(counts, bins=15, density=True)
    plt.xlabel('counts')
    plt.ylabel('probability')
    x = np.arange(-500, 500, 10)
    count_dist = stats.norm(counts.mean(), counts.std()).pdf(x)
    plt.plot(x, count_dist)
    
    if savename:
        plt.savefig(f'figs/noise_fit_{savename}.png')
    # reduce 3sigma
    spec_cleaned = spec_c - counts.mean()
    spec_cleaned[spec_cleaned < 3 * counts.std()] = 0

    return spec_cleaned


def calc_current(spec_arr, instrument):
    factor = 0.12
    if instrument=='p2':
        factor = 0.1
    if instrument=='n1':
        factor = -0.12
    if instrument=='n2':
        factor = -0.1
    return factor * spec_arr


def calc_mass(energy, sc_pot, vel):
    e = 1.6022e-19 # electron charge
    m_p = 1.6726e-27 # proton mass
    return (energy + sc_pot) * 2 * e / (m_p * vel**2)


def calc_energy(mass, sc_pot, vel, instrument='n'):
    """
    Docstring for calc_energy_n
    
    :param mass: mass value or array
    :param sc_pot: spacecraft potential value or array
    :param vel: velocity value or array
    :param instrument: 'n' if negative ion instrument, 'p' if positive ion instrument
    """
    e = 1.6022e-19 # electron charge
    m_p = 1.6726e-27 # proton mass
    if instrument=='n':
        return mass * m_p * vel * vel / 2 / e + sc_pot
    else:
        return mass * m_p * vel * vel / 2 / e - sc_pot
    

def energy_calibration(instrument='n'):
    """
    This function gives an output energy table, 
    it performs both calibration for the instrument and subtracts the remaining offset. 
    """
    sc_pot = np.load('../data/jica_datarequest_nr10.npz')['scpot'][0]
    vel = np.load('../data/jica_datarequest_nr7.npz')['speed'][0] * 1e3
    en_cal = np.loadtxt('../data/energy_calibration_jica.txt')[:,1]

    if instrument=='n':
        spec = np.load("../data/jica_datarequest_nr11.npz")['n1'].flatten()
        spec_c = red_j_spec_err(spec, [400, 512], [0, 1])
        MOL = 'CH3Br-'
        MOI = Formula(MOL).mass
        offset_arr = en_cal[293:410]
        offset_spec = spec_c[293:410]
    else: 
        spec = np.load("../data/jica_datarequest_nr8.npz")['p1'].flatten()
        spec_c = red_j_spec_err(spec, [400, 512], [0, 1])
        MOL = 'PC2H5+'
        MOI = Formula(MOL).mass
        offset_arr = en_cal[225:245]
        offset_spec = spec_c[225:245]        

    offset = calc_energy(MOI, sc_pot, vel, instrument) - offset_arr[np.argmax(offset_spec)]
    
    return en_cal + offset

sc_pot = np.load("../data/jica_datarequest_nr10.npz")['scpot']
velocity = np.load("../data/jica_datarequest_nr7.npz")['speed'] * 1e3

def gauss_fixed_mean_n(x, A, std):
    mean = calc_energy(Formula('CH3Br-').mass, sc_pot, velocity, instrument='n')
    return A * np.exp(- (x - mean) ** 2 / 2 / std / std)

def gauss_fixed_mean_p(x, A, std):
    mean = calc_energy(Formula('PC2H5+').mass, sc_pot, velocity, instrument='p')
    return A * np.exp(- (x - mean) ** 2 / 2 / std / std)

def gauss(x, mean, A, std):
    return A * np.exp(- (x - mean) ** 2 / 2 / std / std)