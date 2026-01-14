from jica.io import find_file_from_id
import numpy as np
import plotly.graph_objects as go

n1 = np.load(find_file_from_id(id=68))
print(n1)
