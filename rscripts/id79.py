from jica.io import find_file_from_id, find_file_from_name
from dataclasses import dataclass
import plotly.graph_objects as go
import numpy as np


@dataclass
class AltitudeData:
    time: np.ndarray
    altitude: np.ndarray
    prodnr: int


def read_from_id(id) -> AltitudeData:
    fp = find_file_from_id(id=id)
    temp = np.load(fp)
    data = AltitudeData
    data.time = temp["time"]
    data.altitude = temp["alt"]
    data.prodnr = temp["dataproductid"]
    return data


data = read_from_id(id=79)

# figure
fig = go.Figure()
fig.update_layout(
    xaxis_title="time [s]",
    yaxis_title="Altitude (above 1 bar level)",
    template="plotly_white",
    width=1000,
    height=800,
)

fig.add_trace(
    go.Scatter(
        x=data.time,
        y=data.altitude,
        name="Altitude",
    )
)
fig.write_html("figs/id79.html")
