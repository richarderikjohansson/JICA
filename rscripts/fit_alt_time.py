from jica.io import find_file_from_id
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from dataclasses import dataclass


@dataclass
class Data:
    time: np.ndarray
    alt: np.ndarray
    fit: np.ndarray
    xfit: np.ndarray


def f2(x, a, b, c):
    return x * x * a + x * b + c


def f(x, a, b):
    return a * x + b


data = np.load(find_file_from_id(id=79))
alt = data["alt"]
time = data["time"]
mask = time < -127

ds = Data(time=time[mask], alt=alt[mask], xfit=None, fit=None)
popt, pcov = curve_fit(f, ds.time, ds.alt)
var_m = pcov[0, 0]
ds.xfit = np.arange(-200, -127, step=1)
ds.fit = f(ds.xfit, *popt)

fig = go.Figure()
fig.update_layout(
    template="plotly_white",
    width=1000,
    height=800,
    xaxis_title="Time [s]",
    yaxis_title="Altitude [km] above 1 bar level",
)

fig.add_trace(
    go.Scatter(
        x=time,
        y=alt,
        name="Probe data",
        mode="markers",
    )
)

fig.add_trace(
    go.Scatter(
        x=ds.xfit,
        y=ds.fit,
        name="Fit",
    )
)

fig.write_html("figs/fit_alt_time.html")
fig.write_image("figs/fit_alt_time.png")
