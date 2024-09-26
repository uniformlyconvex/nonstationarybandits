import typing as t

from plotly import graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots

import nsb.plotting.utils as putils
from nsb.agents.base import MABAgent
from nsb.agents.thompson import TSAgent
from nsb.distributions.conjugacy import NormalInverseGamma
from nsb.distributions.nonstationary import NSDist
from nsb.environment import MABResult
from nsb.experiments.base import Experiment
from nsb.trackers.base import MABTracker


STD_DEVS=2.0

def double_animate(dists: t.Iterable[NSDist], exp: t.Type[Experiment]) -> go.Figure:
    metafig = make_subplots(2,1)

    
    if isinstance(exp, type):
        exp = exp()

    all_results: dict[MABAgent, list[list[MABResult]]] = exp.get_results()
    for agent in all_results:
        if isinstance(agent, TSAgent):
            break
    else:
        return None
    
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
    
    metafig.add_trace(get_violins(0), row=1, col=1)

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
                trace.fillcolor = putils.rgb_to_rgba(color, 0.1)
            if trace is traces.lower_std:
                trace.fill='tonexty'

        for trace in [
            traces.upper_std, traces.lower_std,
            traces.mean, traces.samples, traces.mean_to_time_t
        ]:
            metafig.add_trace(trace, row=2, col=1)

        dist_xmax = max([max(trace.x) for trace in traces])
        dist_ymin = min([min(trace.y) for trace in traces])
        dist_ymax = max([max(trace.y) for trace in traces])

        

        frames = [

        ]

        