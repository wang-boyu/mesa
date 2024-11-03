"""tests for matplotlib components."""

import unittest

import matplotlib.pyplot as plt

from mesa import Agent, Model
from mesa.experimental.cell_space import (
    CellAgent,
    HexGrid,
    Network,
    OrthogonalMooreGrid,
    VoronoiGrid,
)
from mesa.space import (
    ContinuousSpace,
    HexSingleGrid,
    NetworkGrid,
    PropertyLayer,
    SingleGrid,
)
from mesa.visualization.matplotlib_renderer import SpaceRenderMatplotlib


def agent_portrayal(agent):
    """Simple portrayal of an agent.

    Args:
        agent (Agent): The agent to portray

    """
    return {
        "size": 10,
        "color": "tab:blue",
        "marker": "s" if (agent.unique_id % 2) == 0 else "o",
    }


class TestSpaceRenderMatplotlib(unittest.TestCase):
    """Tests for the SpaceRenderMatplotlib class."""

    def setUp(self):
        """Set up the test case."""
        self.space_renderer = SpaceRenderMatplotlib(agent_portrayal=agent_portrayal)
        self.space_renderer_property_layer = SpaceRenderMatplotlib(
            propertylayer_portrayal={"test": {"colormap": "viridis", "colorbar": True}}
        )

    def test_draw_hex_grid(self):
        """Test drawing hexgrids."""
        model = Model(seed=42)
        model.space = HexSingleGrid(10, 10, torus=True)
        for _ in range(10):
            agent = Agent(model)
            model.space.move_to_empty(agent)

        _, ax = plt.subplots()
        self.space_renderer.draw(model, ax)

        model = Model(seed=42)
        model.space = HexGrid((10, 10), torus=True, random=model.random, capacity=1)
        for _ in range(10):
            agent = CellAgent(model)
            agent.cell = model.space.select_random_empty_cell()

        _, ax = plt.subplots()
        self.space_renderer.draw(model, ax)

    def test_draw_voroinoi_grid(self):
        """Test drawing voroinoi grids."""
        model = Model(seed=42)

        coordinates = model.rng.random((100, 2)) * 10

        model.space = VoronoiGrid(coordinates.tolist(), random=model.random, capacity=1)
        for _ in range(10):
            agent = CellAgent(model)
            agent.cell = model.space.select_random_empty_cell()

        _, ax = plt.subplots()
        self.space_renderer.draw(model, ax)

    def test_draw_orthogonal_grid(self):
        """Test drawing orthogonal grids."""
        model = Model(seed=42)
        model.space = SingleGrid(10, 10, torus=True)
        for _ in range(10):
            agent = Agent(model)
            model.space.move_to_empty(agent)

        _, ax = plt.subplots()
        self.space_renderer.draw(model, ax)

        model = Model(seed=42)
        model.space = OrthogonalMooreGrid(
            (10, 10), torus=True, random=model.random, capacity=1
        )
        for _ in range(10):
            agent = CellAgent(model)
            agent.cell = model.space.select_random_empty_cell()

        _, ax = plt.subplots()
        self.space_renderer.draw(model, ax)

    def test_draw_continuous_space(self):
        """Test drawing continuous space."""
        model = Model(seed=42)
        model.space = ContinuousSpace(10, 10, torus=True)
        for _ in range(10):
            x = model.random.random() * 10
            y = model.random.random() * 10
            agent = Agent(model)
            model.space.place_agent(agent, (x, y))

        _, ax = plt.subplots()
        self.space_renderer.draw(model, ax)

    def test_draw_network(self):
        """Test drawing network."""
        import networkx as nx

        n = 10
        m = 20
        seed = 42
        graph = nx.gnm_random_graph(n, m, seed=seed)

        model = Model(seed=42)
        model.space = NetworkGrid(graph)
        for _ in range(10):
            agent = Agent(model)
            pos = agent.random.randint(0, len(graph.nodes) - 1)
            model.space.place_agent(agent, pos)

        _, ax = plt.subplots()
        self.space_renderer.draw(model, ax)

        model = Model(seed=42)
        model.space = Network(graph, random=model.random, capacity=1)
        for _ in range(10):
            agent = CellAgent(model)
            agent.cell = model.space.select_random_empty_cell()

        _, ax = plt.subplots()
        self.space_renderer.draw(model, ax)

    def test_draw_property_layers(self):
        """Test drawing property layers."""
        model = Model(seed=42)
        model.space = SingleGrid(10, 10, torus=True)
        model.space.add_property_layer(
            PropertyLayer("test", model.space.width, model.space.height, 0)
        )

        _, ax = plt.subplots()
        self.space_renderer_property_layer.draw(model, ax)

        model = Model(seed=42)
        model.space = OrthogonalMooreGrid(
            (10, 10), torus=True, random=model.random, capacity=1
        )
        model.space.add_property_layer(
            PropertyLayer("test", model.space.width, model.space.height, 0)
        )

        _, ax = plt.subplots()
        self.space_renderer_property_layer.draw(model, ax)
