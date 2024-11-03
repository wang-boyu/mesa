"""Matplotlib based solara components for visualization MESA spaces and plots."""

import warnings
from collections.abc import Callable

import matplotlib.pyplot as plt
import solara
from matplotlib.figure import Figure

import mesa
from mesa.experimental.cell_space import (
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
)
from mesa.space import (
    HexMultiGrid,
    HexSingleGrid,
    MultiGrid,
    NetworkGrid,
    SingleGrid,
)
from mesa.visualization.matplotlib_renderer import (
    MeasureRendererMatplotlib,
    SpaceRenderMatplotlib,
)
from mesa.visualization.utils import update_counter

# For typing
OrthogonalGrid = SingleGrid | MultiGrid | OrthogonalMooreGrid | OrthogonalVonNeumannGrid
HexGrid = HexSingleGrid | HexMultiGrid | mesa.experimental.cell_space.HexGrid
Network = NetworkGrid | mesa.experimental.cell_space.Network


def make_space_matplotlib(*args, **kwargs):  # noqa: D103
    warnings.warn(
        "make_space_matplotlib has been renamed to make_space_component",
        DeprecationWarning,
        stacklevel=2,
    )
    return make_space_component(*args, **kwargs)


def make_space_component(
    agent_portrayal: Callable | None = None,
    propertylayer_portrayal: dict | None = None,
    post_process: Callable | None = None,
    **space_drawing_kwargs,
):
    """Create a Matplotlib-based space visualization component.

    Args:
        agent_portrayal: Function to portray agents.
        propertylayer_portrayal: Dictionary of PropertyLayer portrayal specifications
        post_process : a callable that will be called with the Axes instance. Allows for fine tuning plots (e.g., control ticks)
        space_drawing_kwargs : additional keyword arguments to be passed on to the underlying space drawer function. See
                               the functions for drawing the various spaces for further details.

    ``agent_portrayal`` is called with an agent and should return a dict. Valid fields in this dict are "color",
    "size", "marker", and "zorder". Other field are ignored and will result in a user warning.


    Returns:
        function: A function that creates a SpaceMatplotlib component
    """
    if agent_portrayal is None:

        def agent_portrayal(a):
            return {}

    def MakeSpaceMatplotlib(model):
        return SpaceMatplotlib(
            model,
            agent_portrayal,
            propertylayer_portrayal,
            post_process=post_process,
            **space_drawing_kwargs,
        )

    return MakeSpaceMatplotlib


@solara.component
def SpaceMatplotlib(
    model,
    agent_portrayal,
    propertylayer_portrayal,
    dependencies: list[any] | None = None,
    post_process: Callable | None = None,
    **space_drawing_kwargs,
):
    """Create a Matplotlib-based space visualization component."""
    update_counter.get()

    fig = Figure()
    ax = fig.add_subplot()

    renderer = SpaceRenderMatplotlib(
        agent_portrayal=agent_portrayal,
        propertylayer_portrayal=propertylayer_portrayal,
        post_process=post_process,
        **space_drawing_kwargs,
    )
    renderer.draw(model, ax)

    solara.FigureMatplotlib(
        fig, format="png", bbox_inches="tight", dependencies=dependencies
    )


def make_plot_measure(*args, **kwargs):  # noqa: D103
    warnings.warn(
        "make_plot_measure has been renamed to make_plot_component",
        DeprecationWarning,
        stacklevel=2,
    )
    return make_plot_component(*args, **kwargs)


def make_plot_component(
    measure: str | dict[str, str] | list[str] | tuple[str],
    post_process: Callable | None = None,
    save_format="png",
):
    """Create a plotting function for a specified measure.

    Args:
        measure (str | dict[str, str] | list[str] | tuple[str]): Measure(s) to plot.
        post_process: a user-specified callable to do post-processing called with the Axes instance.
        save_format: save format of figure in solara backend

    Returns:
        function: A function that creates a PlotMatplotlib component.
    """

    def MakePlotMatplotlib(model):
        return PlotMatplotlib(
            model, measure, post_process=post_process, save_format=save_format
        )

    return MakePlotMatplotlib


@solara.component
def PlotMatplotlib(
    model,
    measure,
    dependencies: list[any] | None = None,
    post_process: Callable | None = None,
    save_format="png",
):
    """Create a Matplotlib-based plot for a measure or measures.

    Args:
        model (mesa.Model): The model instance.
        measure (str | dict[str, str] | list[str] | tuple[str]): Measure(s) to plot.
        dependencies (list[any] | None): Optional dependencies for the plot.
        post_process: a user-specified callable to do post-processing called with the Axes instance.
        save_format: format used for saving the figure.

    Returns:
        solara.FigureMatplotlib: A component for rendering the plot.
    """
    update_counter.get()
    fig, ax = plt.subplots()
    renderer = MeasureRendererMatplotlib(
        measure=measure, post_process=post_process, save_format=save_format
    )
    renderer.render(model, ax)
    solara.FigureMatplotlib(
        fig, format=save_format, bbox_inches="tight", dependencies=dependencies
    )
