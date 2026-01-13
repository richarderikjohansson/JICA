from jica.io import find_file_from_id, find_file_from_name, get_datadir
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


f1 = find_file_from_name(name="jica_datarequest_nr1.npz")
f2 = find_file_from_name(name="jica_datarequest_nr2.npz")

d1 = np.load(f1)
d2 = np.load(f2)
id1 = d1["dataproductnr"]
id2 = d2["dataproductnr"]


assert id1 == id2

time = np.concatenate([d2["time"], d1["time"]])
alt = np.concatenate([d2["alt"], d1["alt"]])
np.savez_compressed(
    get_datadir() / "jica_datarequest_nr2_and_nr1",
    dataproductid=id1,
    time=time,
    alt=alt,
)
