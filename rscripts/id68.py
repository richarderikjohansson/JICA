from jica.io import find_file_from_id
import numpy as np
import plotly.graph_objects as go
from numpy import interp

n1 = np.load(find_file_from_id(id=68))
alt = np.load(find_file_from_id(id=79))

print(alt["time"])
newt = np.arange(-150, -104, step=2)
print(newt)
