"""
Plotting utilities
"""
import time
import typing as t

import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS

from nsb.agents.base import MABAgent
from nsb.agents.thompson import TSAgent
from nsb.distributions.nonstationary import NSDist
from nsb.distributions.stationary import NormalInverseGamma
from nsb.environment import MABResult
from nsb.experiments.base import Experiment
from nsb.plotting.utils import *
from nsb.trackers.base import MABTracker


STD_DEVS = 2.0

def get_distributions_figure(
        dists: t.Iterable[NSDist],
        frames: bool=False
    ) -> go.Scatter:
    """
    Make plotting NSDists easy.
    """
    fig = go.Figure()
    for i, dist in enumerate(dists):
        color = DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]
        traces = dist.traces(std=STD_DEVS)
        prefix = dist.name or f"[Arm {i}]"

        for trace in traces:
            trace.name = f"{prefix} {trace.name}"
            trace.line['color'] = color
            if trace is not traces.mean and 'std' not in trace.name.lower():
                trace.visible = 'legendonly'
            if trace in [traces.upper_std, traces.lower_std]:
                trace.line['dash'] = 'dash'
                trace.fillcolor = rgb_to_rgba(color, 0.1)
            if trace is traces.lower_std:
                trace.fill='tonexty'

        for trace in [traces.upper_std, traces.lower_std, traces.mean, traces.samples, traces.mean_to_time_t]:
            fig.add_trace(trace)

    fig.update_layout(
        title_text="Arm distributions over time",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Legend",
    )

    if frames:
        # Add vertical dashed line
        fig.frames = get_moving_line_frames(fig)

    return fig

def get_regret_figure(trackers: t.Iterable[MABTracker]) -> go.Figure:
    figure = go.Figure()

    for i, tracker in enumerate(trackers):
        color = DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]
        traces = tracker.traces(std=2.0)
        agent_name = repr(tracker.agent)

        for trace in traces:
            trace.line['color'] = color
            trace.line['dash'] = 'dash' if "pseudo" in trace.name.lower() else 'solid'
            if 'std' in trace.name.lower():
                trace.fillcolor = rgb_to_rgba(color, 0.1)
                trace.visible = 'legendonly'
            trace.name = f"[{agent_name}] {trace.name}"
            if trace in [traces.lower_std_random_regret, traces.lower_std_pseudo_regret]:
                trace.fill='tonexty'

        for trace in [
            traces.mean_random_regret, traces.upper_std_random_regret, traces.lower_std_random_regret,
            traces.mean_pseudo_regret, traces.upper_std_pseudo_regret, traces.lower_std_pseudo_regret
        ]:
            figure.add_trace(trace)

    figure.update_layout(
        title_text="Regret over time",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Legend",
    )   

    return figure


def get_ts_beliefs_figure(exp: t.Type[Experiment]) -> list[go.Figure]:
    if isinstance(exp, type):
        exp = exp()
    
    def _handle_agent(agent: TSAgent) -> go.Figure:
        beliefs = agent._posteriors

        def get_violins(idx: int) -> list[go.Violin]:
            curr_posteriors = beliefs[idx]
            samples = [
                post.sample((1000,)) if not isinstance(post, NormalInverseGamma)
                else post.sample((1000,))[0]
                for post in curr_posteriors
            ]
            violins = [
                go.Violin(
                    y=samps,
                    name=f'Arm {i}'
                )
                for i, samps in enumerate(samples)
            ]
            return violins

        agent_name = repr(agent)

        fig = go.Figure(
            data=get_violins(0),  # Initial plot setup
            layout=go.Layout(title=f"Beliefs over time for {agent_name}")
        )

        all_violins = [get_violins(i) for i in range(len(beliefs))]
        ymin, ymax = get_figure_bounds(fig, 'y')
        # Frames for animation
        fig.frames = [
            go.Frame(
                data=all_violins[i],
                name=f"Timestep {i}",
                layout=go.Layout(
                    yaxis=dict(range=[ymin, ymax])
                )
            )
            for i in range(len(beliefs))
        ]

        # fig = add_slider(fig)
        # fig = add_play_pause_buttons(fig)

        return fig

    figs: list[go.Figure] = []
    all_results: dict[MABAgent, list[list[MABResult]]] = exp.get_results()
    for agent in all_results:
        if isinstance(agent, TSAgent):
            figs.append(_handle_agent(agent))
    
    return figs

def plot_experiment(exp: t.Type[Experiment]) -> None:
    if isinstance(exp, type):
        exp = exp()

    print("Fetching results...")
    tic = time.time()
    results = exp.get_results()
    trackers = [
        MABTracker.from_results(
            agent=agent,
            environment=exp.environment,
            results=results[agent]
        )
        for agent in exp.agents
    ]
    toc = time.time()
    print(f"Results fetched in {toc-tic:.2f} seconds")

    regret_fig = get_regret_figure(trackers)
    distributions_fig = get_distributions_figure(exp.environment.arms, frames=True)
    beliefs_figs = get_ts_beliefs_figure(exp)

    regret_metafig = slap_together([distributions_fig, regret_fig], (2, 1))
    regret_metafig = make_legend_smol(regret_metafig)
    regret_metafig = add_explanation(regret_metafig, exp.__doc__)
    regret_metafig.show()

    for beliefs_fig in beliefs_figs:
        beliefs_metafig = slap_together([beliefs_fig, distributions_fig], (2, 1), animate=True)
        beliefs_metafig = add_play_pause_buttons(beliefs_metafig)
        beliefs_metafig = add_slider(beliefs_metafig)
        beliefs_metafig = make_legend_smol(beliefs_metafig)
        beliefs_metafig = add_explanation(beliefs_metafig, exp.__doc__)
        beliefs_metafig.show()