from jica.io import find_file_from_id
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


fp = find_file_from_id(id=50)
dataset = np.load(fp)
print(dataset)

time = dataset["time"]
specsum = dataset["p1p2m1m2"]
