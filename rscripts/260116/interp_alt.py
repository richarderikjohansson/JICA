from jica.io import read_data_from_name
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import numpy as np

data = read_data_from_name(name="jica_datarequest_nr2_and_nr1.npz")

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
        x=data.time,
        y=data.alt,
        name="Probe data",
    )
)

fig.write_html("figs/alt_time.html")


def func(t, A, B, C, K):
    pos = (A / B) * np.exp(B * t) + C * t + K
    return pos


def func2(t, A, B, C):
    pos = A * t * t + B * t + C
    return pos


def func3(t, A, B, C, D):
    pos = A * (t * t * t) + B * (t * t) + C * t + D
    return pos


popt, pcov = curve_fit(func2, data.time, data.alt, p0=[1, -1, 3])
t = np.arange(-200, -100, 1)
fit_poly2 = func2(t, *popt)
popt, pcov = curve_fit(func3, data.time, data.alt, p0=[1, 1, -1, 3])
fit_poly3 = func3(t, *popt)


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
        x=t,
        y=fit_poly2,
        name="Fit poly 2",
        mode="lines",
    )
)
fig.add_trace(
    go.Scatter(
        x=t,
        y=fit_poly3,
        name="Fit poly 3",
        mode="lines",
    )
)
fig.add_trace(
    go.Scatter(
        x=data.time,
        y=data.alt,
        name="Probe",
        mode="markers",
    )
)
fig.write_html("figs/fit_alt_time.html")
