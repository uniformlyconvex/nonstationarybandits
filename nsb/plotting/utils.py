import re
import typing as t

from plotly import graph_objects as go
from plotly.subplots import make_subplots

def rgb_to_rgba(rgb: str, opacity: float) -> str:
    """Dumb hacky way to add opacity to the color"""
    # Regex needs to match rgb(a, b, c)
    match = re.match(r"rgb\((\d+), (\d+), (\d+)\)", rgb)
    if match is None:
        raise ValueError(f"Could not parse color {rgb}")
    r, g, b = match.groups()
    
    # Now we have the three values, add the opacity
    return f"rgba({r}, {g}, {b}, {opacity})"


def to_multiline_string(text: str) -> str:
    """Replace newlines with <br> to display multiline text in Plotly"""
    return "<br>".join(text.split("\n"))

def estimate_title_size(title: str, font_size: int) -> float:
    """Estimate the size of a title based on the number of newlines"""
    return 20 + 1.2 * font_size * title.count("\n")

def add_slider(fig: go.Figure) -> go.Figure:
    """Add a slider to a figure with frames"""
    sliders = [dict(
        steps=[dict(
            method='animate',
            args=[[f.name], dict(
                mode='immediate',
                frame=dict(duration=5, redraw=True),
                transition=dict(duration=0))
            ],
            label=f.name if (i % 100 == 0) else str(i),
            visible=True
        ) for i, f in enumerate(fig.frames)
        ],
        active=0,
        currentvalue={"visible": True},  # Show the current slider value
        pad={"t": 50},  # Padding for the slider
        len=1.0
    )]
    fig.update_layout(sliders=sliders, overwrite=True)
    return fig

def add_explanation(fig: go.Figure, explanation: str) -> go.Figure:
    """Add an explanation to a figure"""
    top_padding = (fig.layout.margin.t or 0.0) + estimate_title_size(explanation, font_size=12)
    fig.update_layout(
        title={
            'text': to_multiline_string(explanation),
            'font': {'size': 12},
        },
        margin=dict(t=top_padding),
        overwrite=True
    )
    return fig

def get_figure_bounds(fig: go.Figure, axis: str) -> tuple[float, float]:
    """Get the bounds of a figure"""
    def _get_attr(attr_name: str, func: t.Callable) -> tuple[float, float]:
        return func([func(getattr(trace, attr_name)) for trace in fig.data])
    
    min_, max_ = _get_attr(axis, min), _get_attr(axis, max)
    return min_, max_


def get_moving_line_frames(fig: go.Figure) -> list[go.Frame]:
    """Get frames for a moving line plot"""
    _, x_max = get_figure_bounds(fig, "x")
    y_min, y_max = get_figure_bounds(fig, "y")
    frames = [
        go.Frame(
           data=[
                go.Scatter(
                    x=[timestep, timestep],
                    y=[y_min, y_max],
                    mode='lines',
                    line=dict(color='black', dash='dot', width=2),
                    name="Time"
                )
            ],
            name=f"Timestep {timestep}",
            # layout=go.Layout(
            #     shapes=[
            #         dict(
            #             type="line",
            #             x0=timestep, x1=timestep,
            #             y0=y_min, y1=y_max,
            #             line=dict(color='black', dash='dot', width=2),
            #         )
            #     ]
            # )
        )
        for timestep in range(x_max+1)
    ]
    return frames

def add_play_pause_buttons(fig: go.Figure) -> go.Figure:
    """Add play and pause buttons to a figure"""
    play_button = dict(
        label="Play",
        method="animate",
        args=[None, dict(frame=dict(duration=5, redraw=True), fromcurrent=True)]
    )

    pause_button = dict(
        label="Pause",
        method="animate",
        args=[[None], dict(
            mode="immediate",
            frame=None,  # Stop frame updates
            transition=dict(duration=0)  # No transition when paused
        )]
    )

    # Layout with update menus for buttons
    layout_changes = dict(
        updatemenus=[dict(type="buttons", showactive=True, buttons=[play_button, pause_button])]
    )

    fig.update_layout(layout_changes, overwrite=True)
    return fig

def slap_together(figs: list[go.Figure], shape: tuple[int, int], animate: bool=False) -> go.Figure:
    """Combine multiple figures into one"""
    if shape[0] * shape[1] < len(figs):
        raise ValueError(f"Shape {shape} is too small for {len(figs)} figures")
    metafig = make_subplots(
        rows=shape[0], cols=shape[1],
        subplot_titles=[f.layout.title.text for f in figs]
    )
    for i, fig in enumerate(figs):
        # Fill the rows first
        for trace in fig.data:
            metafig.add_trace(
                trace,
                row=i // shape[1] + 1,
                col=i % shape[1] + 1
            )
    if animate:
        metafig.frames = figs[0].frames

        # Combine all frames.
        # dummy_fig = go.Figure()
        # for fig in figs:
        #     dummy_fig.update_layout(
        #         fig.layout.to_plotly_json(),
        #         overwrite=False
        #     )
        
        pass
        
        # max_frames = max(len(f.frames) for f in figs)
        # frame_data: list[list[go.BaseTraceType]] = [[] for _ in range(max_frames)]

        # for axis, fig in enumerate(figs):
        #     for i, frame in enumerate(fig.frames):
        #         data = fig.frames[i].data if i < len(fig.frames) else fig.frames[-1].data
        #         for datum in data:
        #             datum.xaxis = f"x{axis+1}" if axis > 0 else "x"
        #             datum.yaxis = f"y{axis+1}" if axis > 0 else "y"
        #             datum.name = f"{datum.name} (Axis {axis+1})"
        #         frame_data[i].extend(frame.data)
        
        # metafig.update(frames=[go.Frame(data=data) for data in frame_data])

    return metafig

def make_legend_smol(fig: go.Figure) -> go.Figure:
    """Make the legend smaller"""
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.02,
            xanchor="left",
            font=dict(size=10),
        ),
        overwrite=True
    )
    return fig