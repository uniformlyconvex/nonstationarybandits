"""
Plotting utilities
"""
import re
import typing as t

import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS

from nsb.trackers.base import NSBTracker
from nsb.distributions import NSDist

def _rgb_to_rgba(rgb: str, opacity: float) -> str:
    """Dumb hacky way to add opacity to the color"""
    # Regex needs to match rgb(a, b, c)
    match = re.match(r"rgb\((\d+), (\d+), (\d+)\)", rgb)
    if match is None:
        raise ValueError(f"Could not parse color {rgb}")
    r, g, b = match.groups()
    
    # Now we have the three values, add the opacity
    return f"rgba({r}, {g}, {b}, {opacity})"

def plot_nsdists(dists: t.Iterable[NSDist]) -> None:
    """
    Make plotting NSDists easy.
    """
    figure = go.Figure()
    for i, dist in enumerate(dists):
        color = DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]
        std = 2.0
        traces = dist.traces(std=std)
        prefix = dist.name or f"[Arm {i}]"
        
        # Add the upper and lower std traces, with filling in between

        for trace in [traces.upper_std, traces.lower_std]:
            trace.line['dash'] = 'dash'
            trace.fillcolor = _rgb_to_rgba(color, 0.1)
            # trace.legendgrouptitle = dict(text=f"Mean ± {std} * StdDev")
            # trace.legendgroup = f"std_{i}"

        for trace in traces:
            trace.name = f"{prefix} {trace.name}"
            trace.line['color'] = color
            if trace is not traces.mean:
                trace.visible = 'legendonly'

        figure.add_trace(traces.upper_std)
        traces.lower_std.fill='tonexty'
        figure.add_trace(traces.lower_std)

        figure.add_trace(traces.mean)
        figure.add_trace(traces.samples) 

    figure.update_layout(
        title_text="Arm distributions over time",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Legend",
    )

    figure.show()


def plot_regret(tracker: NSBTracker) -> None:
    """
    Plot the regret.
    """
    figure = go.Figure()

    for i, (agent, traces) in enumerate(tracker.regret_traces().items()):
        color = DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]
        for trace in traces:
            trace.line['color'] = color
            trace.line['dash'] = 'dash' if "pseudo" in trace.name.lower() else 'solid'
            figure.add_trace(trace)

    figure.update_layout(
        title_text="Regret over time",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Legend",
    )

    figure.show()


def plot_arms(tracker: NSBTracker) -> None:
    figure = go.Figure()

    for i, (agent, traces) in enumerate(tracker.arm_traces().items()):
        color = DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]
        for trace in traces:
            trace.line['color'] = color
            figure.add_trace(trace)

    figure.update_layout(
        title_text="Arm chosen over time",
        xaxis_title="Time",
        yaxis_title="Arm",
        legend_title="Legend",
    )

    figure.show()