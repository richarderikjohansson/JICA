from jica.io import find_file_from_id, get_datadir
import numpy as np
import plotly.graph_objects as go
from numpy import interp

n1 = np.load(find_file_from_id(id=68))
alt = np.load(find_file_from_id(id=79))
newt = np.arange(-150, -104, step=2)

newalt = interp(newt, alt["time"], alt["alt"])

fig = go.Figure()
fig.update_layout(
    template="plotly_white",
    width=1000,
    height=800,
    xaxis_title="Time [s]",
    yaxis_title="Altitude [km] (above the 1 bar level)",
    legend=dict(
        x=0.98,  # near right edge
        y=0.98,  # near top edge
        xanchor="right",
        yanchor="top",
    ),
)
fig.add_trace(
    go.Scatter(
        x=alt["time"],
        y=alt["alt"],
        mode="lines+markers",
        marker=dict(
            symbol="x",
            size=10,
            line=dict(width=1),  # thickness of the cross strokes
        ),
        name="Original concatinated alt data (id79)",
    )
)
fig.add_trace(
    go.Scatter(
        x=newt,
        y=newalt,
        mode="markers",
        name="Interpolated to a coarser time grid",
    )
)

fig.write_html("figs/interp_alt.html")
fig.write_image("figs/interp_alt.png")

fn = "jica_datarequest_nr1_and_nr2_interp.npz"
np.savez_compressed(
    get_datadir() / fn,
    time=newt,
    alt=newalt,
    dataproductid=79,
)
