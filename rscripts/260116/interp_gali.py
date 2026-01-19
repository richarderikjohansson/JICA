from jica.io import read_data_from_name
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go

data = read_data_from_name(name="jica_datarequest_nr2_and_nr1.npz")
gali = np.loadtxt("gali.txt")
namespace = SimpleNamespace()

for i in range(0, 12):
    match i:
        case 0:
            namespace.t = gali[:, i]
        case 1:
            namespace.z = gali[:, i]
        case 2:
            namespace.p = gali[:, i]
        case 3:
            namespace.temp = gali[:, i]
        case 4:
            namespace.rho = gali[:, i]
        case 5:
            namespace.molmass = gali[:, i]
        case 8:
            namespace.v = gali[:, i]
        case 9:
            namespace.fpa = gali[:, i]
        case 10:
            namespace.lat = gali[:, i]
        case 11:
            namespace.lon = gali[:, i]

time = []
for t in namespace.t:
    time.append(t + 3)

fig = go.Figure(
    layout=dict(
        template="plotly_white",
        width=1000,
        height=800,
        xaxis_title="Time [s]",
        yaxis_title="Altitude [km] above 1 bar level",
    ),
)
fig.add_trace(
    go.Scatter(
        x=time,
        y=namespace.z,
        name="Galileo probe data",
    )
)
fig.add_trace(
    go.Scatter(x=data.time, y=data.alt, name="JICA Probe data", mode="markers")
)
fig.write_html("figs/JICA_GALILEO.html")
print(namespace.lon)
