"""ASCII-based Solara components for visualization of MESA spaces and plots."""

import solara

from mesa.visualization.ascii_renderer import MeasureRenderAscii, SpaceRenderAscii
from mesa.visualization.utils import update_counter


@solara.component
def SpaceAscii(
    model,
    agent_portrayal,
    width: int = 80,
    height: int = 24,
):
    """Create an ASCII-based space visualization component."""
    # Get the current step to trigger updates
    update_counter.get()

    renderer = SpaceRenderAscii(
        agent_portrayal=agent_portrayal,
        width=width,
        height=height,
    )

    # Use Solara's pre-formatted text component to maintain spacing
    solara.Text(renderer.draw(model))


@solara.component
def PlotAscii(
    model,
    measure: str | list[str],
    width: int = 80,
    height: int = 24,
):
    """Create an ASCII-based plot visualization component."""
    # Get the current step to trigger updates
    update_counter.get()

    renderer = MeasureRenderAscii(
        measure=measure,
        width=width,
        height=height,
    )

    solara.Text(renderer.draw(model))
