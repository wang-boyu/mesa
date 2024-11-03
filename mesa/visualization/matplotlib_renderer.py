"""Built-in renderer for Mesa spaces using matplotlib."""

import itertools
import math
import warnings
from collections.abc import Callable
from typing import Protocol

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.patches import RegularPolygon

import mesa
from mesa.experimental.cell_space import (
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
    VoronoiGrid,
)
from mesa.space import (
    ContinuousSpace,
    HexMultiGrid,
    HexSingleGrid,
    MultiGrid,
    NetworkGrid,
    PropertyLayer,
    SingleGrid,
)

# For typing
OrthogonalGrid = SingleGrid | MultiGrid | OrthogonalMooreGrid | OrthogonalVonNeumannGrid
HexGrid = HexSingleGrid | HexMultiGrid | mesa.experimental.cell_space.HexGrid
Network = NetworkGrid | mesa.experimental.cell_space.Network


class MatplotlibRenderer(Protocol):
    """A protocol for renderers."""

    def draw(self, model: mesa.Model, ax: plt.Axes | None) -> plt.Axes:
        """Render a matplotlib visualization of the model."""
        ...


class SpaceRenderMatplotlib:
    """Built-in renderer for Mesa spaces using matplotlib."""

    def __init__(
        self,
        agent_portrayal: Callable | None = None,
        propertylayer_portrayal: dict | None = None,
        post_process: Callable | None = None,
        **space_drawing_kwargs,
    ):
        """Initialize a SpaceRenderMatplotlib instance.

        Args:
            agent_portrayal: Function to portray agents. It is called with an agent and should return a dict. Valid fields in this dict are "color",
                             "size", "marker", and "zorder". Other fields are ignored and will result in a user warning.
            propertylayer_portrayal: Dictionary of PropertyLayer portrayal specifications. The key is the name of the layer, the value is a dict with
                                     fields specifying how the layer is to be portrayed. valid fields in in the inner dict of propertylayer_portrayal
                                     are "alpha", "vmin", "vmax", "color" or "colormap", and "colorbar" so you can do
                                     `{"some_layer":{"colormap":'viridis', 'alpha':.25, "colorbar":False}}`.
            post_process: a callable that will be called with the Axes instance. Allows for fine tuning plots (e.g., control ticks)
            space_drawing_kwargs: additional keyword arguments to be passed on to the underlying space drawer function. See
                               the functions for drawing the various spaces for further details.
        """
        self.agent_portrayal = agent_portrayal
        self.propertylayer_portrayal = propertylayer_portrayal
        self.post_process = post_process
        self.space_drawing_kwargs = space_drawing_kwargs

    def draw(self, model: mesa.Model, ax: plt.Axes | None) -> plt.Axes:
        """Render a matplotlib visualization of the model space."""
        if ax is None:
            _, ax = plt.subplots()
        space = getattr(model, "grid", None)
        if space is None:
            space = getattr(model, "space", None)
        # https://stackoverflow.com/questions/67524641/convert-multiple-isinstance-checks-to-structural-pattern-
        match space:
            case (
                mesa.space._Grid()
                | OrthogonalMooreGrid()
                | OrthogonalVonNeumannGrid()
            ):
                self._draw_orthogonal_grid(space, ax=ax)
            case (
                HexSingleGrid()
                | HexMultiGrid()
                | mesa.experimental.cell_space.HexGrid()
            ):
                self._draw_hex_grid(space, ax=ax)
            case mesa.space.NetworkGrid() | mesa.experimental.cell_space.Network():
                self._draw_network(space, ax=ax)
            case mesa.space.ContinuousSpace():
                self._draw_continuous_space(space, ax=ax)
            case VoronoiGrid():
                self._draw_voroinoi_grid(space, ax=ax)

        if self.propertylayer_portrayal:
            self._draw_property_layers(space, ax=ax)

        if self.post_process is not None:
            self.post_process(ax=ax)

        return ax

    def _draw_orthogonal_grid(self, space: OrthogonalGrid, ax: plt.Axes):
        """Visualize a orthogonal grid.

        Args:
            space: the space to visualize
            ax: a Matplotlib Axes instance. If none is provided a new figure and ax will be created using plt.subplots

        Returns:
            Returns the Axes object with the plot drawn onto it.
        """
        # gather agent data
        s_default = (180 / max(space.width, space.height)) ** 2
        arguments = self._collect_agent_data(
            space, self.agent_portrayal, size=s_default
        )

        # plot the agents
        self._scatter(ax, arguments)

        # further styling
        ax.set_xlim(-0.5, space.width - 0.5)
        ax.set_ylim(-0.5, space.height - 0.5)

        if self.space_drawing_kwargs.get("draw_grid", True):
            # Draw grid lines
            for x in np.arange(-0.5, space.width - 0.5, 1):
                ax.axvline(x, color="gray", linestyle=":")
            for y in np.arange(-0.5, space.height - 0.5, 1):
                ax.axhline(y, color="gray", linestyle=":")

        return ax

    def _draw_property_layers(self, space: OrthogonalGrid, ax: plt.Axes):
        """Draw PropertyLayers on the given axes.

        Args:
            space (mesa.space._Grid): The space containing the PropertyLayers.
            ax (matplotlib.axes.Axes): The axes to draw on.
        """
        try:
            # old style spaces
            property_layers = space.properties
        except AttributeError:
            # new style spaces
            property_layers = space.property_layers

        for layer_name, portrayal in self.propertylayer_portrayal.items():
            layer = property_layers.get(layer_name, None)
            if not isinstance(layer, PropertyLayer):
                continue

            data = layer.data.astype(float) if layer.data.dtype == bool else layer.data
            width, height = data.shape if space is None else (space.width, space.height)

            if space and data.shape != (width, height):
                warnings.warn(
                    f"Layer {layer_name} dimensions ({data.shape}) do not match space dimensions ({width}, {height}).",
                    UserWarning,
                    stacklevel=2,
                )

            # Get portrayal properties, or use defaults
            alpha = portrayal.get("alpha", 1)
            vmin = portrayal.get("vmin", np.min(data))
            vmax = portrayal.get("vmax", np.max(data))
            colorbar = portrayal.get("colorbar", True)

            # Draw the layer
            if "color" in portrayal:
                rgba_color = to_rgba(portrayal["color"])
                normalized_data = (data - vmin) / (vmax - vmin)
                rgba_data = np.full((*data.shape, 4), rgba_color)
                rgba_data[..., 3] *= normalized_data * alpha
                rgba_data = np.clip(rgba_data, 0, 1)
                cmap = LinearSegmentedColormap.from_list(
                    layer_name, [(0, 0, 0, 0), (*rgba_color[:3], alpha)]
                )
                im = ax.imshow(
                    rgba_data.transpose(1, 0, 2),
                    origin="lower",
                )
                if colorbar:
                    norm = Normalize(vmin=vmin, vmax=vmax)
                    sm = ScalarMappable(norm=norm, cmap=cmap)
                    sm.set_array([])
                    ax.figure.colorbar(sm, ax=ax, orientation="vertical")
            elif "colormap" in portrayal:
                cmap = portrayal.get("colormap", "viridis")
                if isinstance(cmap, list):
                    cmap = LinearSegmentedColormap.from_list(layer_name, cmap)
                im = ax.imshow(
                    data.T,
                    cmap=cmap,
                    alpha=alpha,
                    vmin=vmin,
                    vmax=vmax,
                    origin="lower",
                )
                if colorbar:
                    plt.colorbar(im, ax=ax, label=layer_name)
            else:
                raise ValueError(
                    f"PropertyLayer {layer_name} portrayal must include 'color' or 'colormap'."
                )

    def _collect_agent_data(
        self,
        space: OrthogonalGrid | HexGrid | Network | ContinuousSpace | VoronoiGrid,
        color="tab:blue",
        size=25,
        marker="o",
        zorder: int = 1,
    ):
        """Collect the plotting data for all agents in the space.

        Args:
            space: The space containing the Agents.
            color: default color
            size: default size
            marker: default marker
            zorder: default zorder
        """
        arguments = {"s": [], "c": [], "marker": [], "zorder": [], "loc": []}

        for agent in space.agents:
            portray = self.agent_portrayal(agent)
            loc = agent.pos
            if loc is None:
                loc = agent.cell.coordinate

            arguments["loc"].append(loc)
            arguments["s"].append(portray.pop("size", size))
            arguments["c"].append(portray.pop("color", color))
            arguments["marker"].append(portray.pop("marker", marker))
            arguments["zorder"].append(portray.pop("zorder", zorder))

            if len(portray) > 0:
                ignored_fields = list(portray.keys())
                msg = ", ".join(ignored_fields)
                warnings.warn(
                    f"the following fields are not used in agent portrayal and thus ignored: {msg}.",
                    stacklevel=2,
                )

        return {k: np.asarray(v) for k, v in arguments.items()}

    def _scatter(self, ax: plt.Axes, arguments):
        """Helper function for plotting the agents.

        Args:
            ax: a Matplotlib Axes instance
            arguments: the agents specific arguments for platting
        """
        loc = arguments.pop("loc")
        # if there are no agents to plot, return the axes
        if len(loc) == 0:
            return ax

        x = loc[:, 0]
        y = loc[:, 1]
        marker = arguments.pop("marker")
        zorder = arguments.pop("zorder")

        for mark in np.unique(marker):
            mark_mask = marker == mark
            for z_order in np.unique(zorder):
                zorder_mask = z_order == zorder
                logical = mark_mask & zorder_mask
                ax.scatter(
                    x[logical],
                    y[logical],
                    marker=mark,
                    zorder=z_order,
                    **{k: v[logical] for k, v in arguments.items()},
                    **self.space_drawing_kwargs,
                )

    def _draw_hex_grid(self, space: HexGrid, ax: plt.Axes | None = None):
        """Visualize a hex grid.

        Args:
            space: the space to visualize
            ax: a Matplotlib Axes instance. If none is provided a new figure and ax will be created using plt.subplots

        Returns:
            Returns the Axes object with the plot drawn onto it.

        """
        # gather data
        s_default = (180 / max(space.width, space.height)) ** 2
        arguments = self._collect_agent_data(space, size=s_default)

        # for hexgrids we have to go from logical coordinates to visual coordinates
        # this is a bit messy.

        # give all even rows an offset in the x direction
        # give all rows an offset in the y direction

        # numbers here are based on a distance of 1 between centers of hexes
        offset = math.sqrt(0.75)

        loc = arguments["loc"].astype(float)

        logical = np.mod(loc[:, 1], 2) == 0
        loc[:, 0][logical] += 0.5
        loc[:, 1] *= offset
        arguments["loc"] = loc

        # plot the agents
        self._scatter(ax, arguments)

        # further styling and adding of grid
        ax.set_xlim(-1, space.width + 0.5)
        ax.set_ylim(-offset, space.height * offset)

        def setup_hexmesh(
            width,
            height,
        ):
            """Helper function for creating the hexmaesh."""
            # fixme: this should be done once, rather than in each update
            # fixme check coordinate system in hexgrid (see https://www.redblobgames.com/grids/hexagons/#coordinates-offset)

            patches = []
            for x, y in itertools.product(range(width), range(height)):
                if y % 2 == 0:
                    x += 0.5  # noqa: PLW2901
                y *= offset  # noqa: PLW2901
                hex = RegularPolygon(
                    (x, y),
                    numVertices=6,
                    radius=math.sqrt(1 / 3),
                    orientation=np.radians(120),
                )
                patches.append(hex)
            mesh = PatchCollection(
                patches, edgecolor="k", facecolor=(1, 1, 1, 0), linestyle="dotted", lw=1
            )
            return mesh

        if self.space_drawing_kwargs.get("draw_grid", True):
            # add grid
            ax.add_collection(
                setup_hexmesh(
                    space.width,
                    space.height,
                )
            )
        return ax

    def _draw_network(
        self,
        space: Network,
        ax: Axes,
    ):
        """Visualize a network space.

        Args:
            space: the space to visualize
            ax: a Matplotlib Axes instance. If none is provided a new figure and ax will be created using plt.subplots

        Returns:
            Returns the Axes object with the plot drawn onto it.
        """
        layout_alg = self.space_drawing_kwargs.get("layout_alg", nx.spring_layout)
        layout_kwargs = self.space_drawing_kwargs.get("layout_kwargs", {"seed": 0})

        # gather locations for nodes in network
        graph = space.G
        pos = layout_alg(graph, **layout_kwargs)
        x, y = list(zip(*pos.values()))
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)

        width = xmax - xmin
        height = ymax - ymin
        x_padding = width / 20
        y_padding = height / 20

        # gather agent data
        s_default = (180 / max(width, height)) ** 2
        arguments = self._collect_agent_data(space, size=s_default)

        # this assumes that nodes are identified by an integer
        # which is true for default nx graphs but might user changeable
        pos = np.asarray(list(pos.values()))
        arguments["loc"] = pos[arguments["loc"]]

        # plot the agents
        self._scatter(ax, arguments)

        # further styling
        ax.set_axis_off()
        ax.set_xlim(xmin=xmin - x_padding, xmax=xmax + x_padding)
        ax.set_ylim(ymin=ymin - y_padding, ymax=ymax + y_padding)

        if self.space_drawing_kwargs.get("draw_grid", True):
            # fixme we need to draw the empty nodes as well
            edge_collection = nx.draw_networkx_edges(
                graph, pos, ax=ax, alpha=0.5, style="--"
            )
            edge_collection.set_zorder(0)

        return ax

    def _draw_continuous_space(self, space: ContinuousSpace, ax: Axes):
        """Visualize a continuous space.

        Args:
            space: the space to visualize
            ax: a Matplotlib Axes instance. If none is provided a new figure and ax will be created using plt.subplots

        Returns:
            Returns the Axes object with the plot drawn onto it.
        """
        # space related setup
        width = space.x_max - space.x_min
        x_padding = width / 20
        height = space.y_max - space.y_min
        y_padding = height / 20

        # gather agent data
        s_default = (180 / max(width, height)) ** 2
        arguments = self._collect_agent_data(space, size=s_default)

        # plot the agents
        self._scatter(ax, arguments)

        # further visual styling
        border_style = "solid" if not space.torus else (0, (5, 10))
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color("black")
            spine.set_linestyle(border_style)

        ax.set_xlim(space.x_min - x_padding, space.x_max + x_padding)
        ax.set_ylim(space.y_min - y_padding, space.y_max + y_padding)

        return ax

    def _draw_voroinoi_grid(self, space: VoronoiGrid, ax: Axes):
        """Visualize a voronoi grid.

        Args:
            space: the space to visualize
            ax: a Matplotlib Axes instance. If none is provided a new figure and ax will be created using plt.subplots

        Returns:
            Returns the Axes object with the plot drawn onto it.
        """
        x_list = [i[0] for i in space.centroids_coordinates]
        y_list = [i[1] for i in space.centroids_coordinates]
        x_max = max(x_list)
        x_min = min(x_list)
        y_max = max(y_list)
        y_min = min(y_list)

        width = x_max - x_min
        x_padding = width / 20
        height = y_max - y_min
        y_padding = height / 20

        s_default = (180 / max(width, height)) ** 2
        arguments = self._collect_agent_data(space, size=s_default)

        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        self._scatter(ax, arguments)

        for cell in space.all_cells:
            polygon = cell.properties["polygon"]
            ax.fill(
                *zip(*polygon),
                alpha=min(1, cell.properties[space.cell_coloring_property]),
                c="red",
                zorder=0,
            )  # Plot filled polygon
            ax.plot(*zip(*polygon), color="black")  # Plot polygon edges in black

        return ax


class MeasureRendererMatplotlib:
    """Built-in renderer for Mesa measures using matplotlib."""

    def __init__(
        self,
        measure: str | dict[str, str] | list[str] | tuple[str],
        post_process: Callable | None = None,
        save_format="png",
    ):
        """Initialize a MeasureRendererMatplotlib instance.

        Args:
            measure: Measure(s) to plot.
            post_process: a user-specified callable to do post-processing called with the Axes instance.
            save_format: format used for saving the figure.
        """
        self.measure = measure
        self.post_process = post_process
        self.save_format = save_format

    def render(self, model: mesa.Model, ax: plt.Axes | None = None) -> plt.Axes:
        """Render a matplotlib visualization of the model."""
        if ax is None:
            _, ax = plt.subplots()
        df = model.datacollector.get_model_vars_dataframe()
        if isinstance(self.measure, str):
            ax.plot(df.loc[:, self.measure])
            ax.set_ylabel(self.measure)
        elif isinstance(self.measure, dict):
            for m, color in self.measure.items():
                ax.plot(df.loc[:, m], label=m, color=color)
            ax.legend(loc="best")
        elif isinstance(self.measure, list | tuple):
            for m in self.measure:
                ax.plot(df.loc[:, m], label=m)
            ax.legend(loc="best")

        ax.set_xlabel("Step")
        # Set integer x axis
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if self.post_process is not None:
            self.post_process(ax)
        return ax
