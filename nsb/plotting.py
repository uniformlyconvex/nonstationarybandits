"""
Plotting utilities
"""
import re
import typing as t

import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots

from nsb.agents.thompson import TSAgent
from nsb.distributions.nonstationary import NSDist
from nsb.experiments.base import Experiment
from nsb.trackers.base import MABTracker


def _rgb_to_rgba(rgb: str, opacity: float) -> str:
    """Dumb hacky way to add opacity to the color"""
    # Regex needs to match rgb(a, b, c)
    match = re.match(r"rgb\((\d+), (\d+), (\d+)\)", rgb)
    if match is None:
        raise ValueError(f"Could not parse color {rgb}")
    r, g, b = match.groups()
    
    # Now we have the three values, add the opacity
    return f"rgba({r}, {g}, {b}, {opacity})"

def _to_multiline_string(text: str) -> str:
    return "<br>".join(text.split("\n"))

def _estimate_title_size(title: str, font_size: int) -> int:
    return 20 + 1.2 * font_size * title.count("\n")

def get_distributions_figure(dists: t.Iterable[NSDist]) -> go.Scatter:
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
            if trace is not traces.mean and 'std' not in trace.name.lower():
                trace.visible = 'legendonly'

        figure.add_trace(traces.upper_std)
        traces.lower_std.fill='tonexty'
        figure.add_trace(traces.lower_std)

        figure.add_trace(traces.mean)
        figure.add_trace(traces.samples)
        figure.add_trace(traces.mean_to_time_t)

    figure.update_layout(
        title_text="Arm distributions over time",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Legend",
    )

    return figure

def get_regret_figure(trackers: t.Iterable[MABTracker]) -> go.Figure:
    figure = go.Figure()

    for i, tracker in enumerate(trackers):
        color = DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]
        traces = tracker.traces(std=2.0)
        agent_name = str(tracker.agent)

        for trace in traces:
            trace.line['color'] = color
            trace.line['dash'] = 'dash' if "pseudo" in trace.name.lower() else 'solid'
            if 'std' in trace.name.lower():
                trace.fillcolor = _rgb_to_rgba(color, 0.1)
                trace.visible = 'legendonly'
            trace.name = f"[{agent_name}] {trace.name}"

        figure.add_trace(traces.mean_random_regret)
        figure.add_trace(traces.upper_std_random_regret)
        traces.lower_std_random_regret.fill='tonexty'
        figure.add_trace(traces.lower_std_random_regret)

        figure.add_trace(traces.mean_pseudo_regret)
        figure.add_trace(traces.upper_std_pseudo_regret)
        traces.lower_std_pseudo_regret.fill='tonexty'
        figure.add_trace(traces.lower_std_pseudo_regret)

    figure.update_layout(
        title_text="Regret over time",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Legend",
    )   

    return figure


def get_ts_beliefs_figure(exp: t.Type[Experiment]) -> go.Figure | None:
    for agent in exp.agents:
        if isinstance(agent, TSAgent):
            break
    else:
        return None
    
    agent: TSAgent
    beliefs = agent._posteriors
    duration = len(beliefs)

    def get_violins(idx: int) -> list[go.Violin]:
        curr_posteriors = beliefs[idx]
        samples = [post.sample((1000,)) for post in curr_posteriors]
        violins = [
            go.Violin(
                x=i,
                y=samps,
                name=f'Arm {i}'
            )
            for i, samps in enumerate(samples)
        ]
        return violins
    
    buttons=[dict(
        label="Play",
        method="animate",
        args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True)]
    )]
    layout=go.Layout(
        title="Beliefs over time",
        updatemenus=[dict(type="buttons", showactive=False, buttons=buttons)]
    )
    fig = go.Figure(
        data=get_violins(0),
        layout=layout
    )
    fig.frames = [go.Frame(data=get_violins(i)) for i in range(duration)]

    sliders = [{
        "currentvalue": {"prefix": "Frame: "},
        "steps": [{
            "args": [
                [f.name],
                {
                    "frame": {"duration": 50, "redraw": True},
                    "mode": "immediate"
                }
            ],
            "label": str(f.name),
            "method": "animate"
        }
        for f in fig.frames
    ]}]

    fig.show()

def plot_experiment(exp: t.Type[Experiment]) -> None:
    if isinstance(exp, type):
        exp = exp()

    results = exp.get_results()
    trackers = [
        MABTracker.from_results(
            agent=agent,
            environment=exp.environment,
            results=results[agent]
        )
        for agent in exp.agents
    ]

    regret_figure = get_regret_figure(trackers)
    distributions_figure = get_distributions_figure(exp.environment.arms)

    fig: go.Figure = make_subplots(rows=1, cols=2, subplot_titles=("Distributions", "Regret"))


    for trace in distributions_figure.data:
        fig.add_trace(trace, row=1, col=1)

    for trace in regret_figure.data:
        fig.add_trace(trace, row=1, col=2)

    FONT_SIZE = 12

    if exp.__doc__:
        fig.update_layout(
            title={
                'text': _to_multiline_string(exp.__class__.__name__ + "\n" + exp.__doc__),
                'font': {'size': FONT_SIZE},
            },
            margin=dict(t=_estimate_title_size(exp.__doc__, font_size=FONT_SIZE)),
        )

    # Make the legend tiny
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=5)
    ))

    fig.show()

    beliefs_fig = get_ts_beliefs_figure(exp)
    beliefs_fig.show()
