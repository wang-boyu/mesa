"""Built-in renderer for Mesa spaces using ASCII characters."""

from collections.abc import Callable
from typing import Protocol

import numpy as np

import mesa
from mesa.experimental.cell_space import OrthogonalMooreGrid, OrthogonalVonNeumannGrid
from mesa.space import ContinuousSpace, MultiGrid, SingleGrid

# For typing
OrthogonalGrid = SingleGrid | MultiGrid | OrthogonalMooreGrid | OrthogonalVonNeumannGrid

class AsciiRenderer(Protocol):
    """A protocol for ASCII renderers."""

    def draw(self, model: mesa.Model) -> str:
        """Render an ASCII visualization of the model."""
        ...

class SpaceRenderAscii:
    """Built-in renderer for Mesa spaces using ASCII characters."""

    def __init__(
        self,
        agent_portrayal: Callable | None = None,
        width: int = 80,  # Default terminal width
        height: int = 24,  # Default terminal height
    ):
        """Initialize a SpaceRenderAscii instance.

        Args:
            agent_portrayal: Function to portray agents. Should return a single character.
            width: Maximum width of the ASCII output
            height: Maximum height of the ASCII output
        """
        self.agent_portrayal = agent_portrayal or (lambda a: "●")
        self.width = width
        self.height = height

    def draw(self, model: mesa.Model) -> str:
        """Render an ASCII visualization of the model space."""
        space = getattr(model, "grid", None)
        if space is None:
            space = getattr(model, "space", None)

        match space:
            case (SingleGrid() | MultiGrid() | OrthogonalMooreGrid() | OrthogonalVonNeumannGrid()):
                return self._draw_orthogonal_grid(space)
            case ContinuousSpace():
                return self._draw_continuous_space(space)
            case _:
                return "Unsupported space type"

    def _draw_orthogonal_grid(self, space: OrthogonalGrid) -> str:
        """Draw an orthogonal grid using ASCII characters."""
        # Create empty grid
        grid = [[" " for _ in range(space.width)] for _ in range(space.height)]

        # Fill in agents
        for cell_content, pos in space.coord_iter():
            x, y = pos
            if isinstance(cell_content, list):  # MultiGrid
                if cell_content:  # If there are agents in this cell
                    grid[y][x] = self.agent_portrayal(cell_content[0])
            elif cell_content:  # SingleGrid with an agent
                grid[y][x] = self.agent_portrayal(cell_content)

        # Convert to string with borders
        horizontal_border = "+" + "-" * space.width + "+"
        return (
            horizontal_border + "\n" +
            "\n".join("|" + "".join(row) + "|" for row in reversed(grid)) +
            "\n" + horizontal_border
        )

    def _draw_continuous_space(self, space: ContinuousSpace) -> str:
        """Draw a continuous space using ASCII characters."""
        # Create empty grid scaled to terminal size
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]

        # Scale factors for converting coordinates to grid positions
        scale_x = (self.width - 2) / (space.x_max - space.x_min)
        scale_y = (self.height - 2) / (space.y_max - space.y_min)

        # Plot each agent
        for agent in space.agents:
            x, y = agent.pos
            # Scale coordinates to grid positions
            grid_x = int((x - space.x_min) * scale_x)
            grid_y = int((y - space.y_min) * scale_y)
            # Ensure within bounds
            grid_x = min(max(0, grid_x), self.width - 1)
            grid_y = min(max(0, grid_y), self.height - 1)
            grid[grid_y][grid_x] = self.agent_portrayal(agent)

        # Convert to string with borders
        horizontal_border = "+" + "-" * (self.width - 2) + "+"
        return (
            horizontal_border + "\n" +
            "\n".join("|" + "".join(row) + "|" for row in reversed(grid)) +
            "\n" + horizontal_border
        )

class MeasureRenderAscii:
    """Built-in renderer for Mesa measures using ASCII characters."""

    def __init__(
        self,
        measure: str | list[str],
        width: int = 80,
        height: int = 24,
    ):
        """Initialize a MeasureRenderAscii instance.

        Args:
            measure: Measure(s) to plot
            width: Width of the ASCII plot
            height: Height of the ASCII plot
        """
        self.measure = measure
        self.width = width
        self.height = height

    def draw(self, model: mesa.Model) -> str:
        """Render an ASCII visualization of the model measures."""
        df = model.datacollector.get_model_vars_dataframe()

        if isinstance(self.measure, str):
            data = df[self.measure].values
        else:  # list of measures
            # For multiple measures, just plot the first one for now
            data = df[self.measure[0]].values

        if len(data) == 0:
            return "No data to plot"

        # Create simple ASCII line plot
        normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
        points = [int(n * (self.height - 1)) if not np.isnan(n) else 0 for n in normalized]

        grid = [[" " for _ in range(min(len(data), self.width))] for _ in range(self.height)]
        for x, y in enumerate(points[:self.width]):
            grid[y][x] = "*"

        return "\n".join("".join(row) for row in reversed(grid))
