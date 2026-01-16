from jica.io import find_file_from_name, get_datadir
import numpy as np


def read_calibration():
    fp = find_file_from_name("energy_calibration_jica.txt")
    data = np.loadtxt(fp, skiprows=4)
    dct = {int(row[0]): row[1] for row in data}

    return dct


def adjust_calibration(data, ds):
    ddir = get_datadir()
    energy = np.array([val for val in data.values()])
    bin_num = np.array([k for k in data.keys()])
    energy_adjusted = energy + ds["offset"]
    np.savez_compressed(ddir / ds["name"], bin_num=bin_num, en_adjusted=energy_adjusted)


pos = {"offset": 1.3160564888762565, "name": "pos_en_table_adjusted.npz"}
neg = {"offset": 2.585570822484897, "name": "neg_en_table_adjusted.npz"}

cal = read_calibration()
adjust_calibration(cal, pos)
adjust_calibration(cal, neg)
