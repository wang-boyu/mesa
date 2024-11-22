from collections import Counter
from functools import partial

import numpy as np
from scipy.stats import entropy

import mesa
from mesa import Model
from mesa.examples.basic.schelling.agents import SchellingAgent


def compute_shannon_entropy(model):
    observed = [agent.same_type_neighbours for agent in model.agents]
    c = Counter(observed)
    pk = [p / sum(c.values()) for p in c.values()]
    return entropy(pk, base=2)


def compute_dissimilarity_index(model, radius=1):
    minority_list = np.zeros(model.grid.width * model.grid.height)
    majority_list = np.zeros(model.grid.width * model.grid.height)
    for i, (_, cell_pos) in enumerate(model.grid.coord_iter()):
        minority_neighbors, majority_neighbors = 0, 0
        neighborhood = model.grid.iter_neighbors(
            cell_pos, moore=True, include_center=True, radius=radius
        )
        for neighbor in neighborhood:
            if neighbor.type == 1:
                minority_neighbors += 1
            elif neighbor.type == 0:
                majority_neighbors += 1
        minority_list[i] = minority_neighbors
        majority_list[i] = majority_neighbors
    minority_list /= model.total_minority
    majority_list /= model.total_majority
    return abs(minority_list - majority_list).sum() / 2


class Schelling(Model):
    """Model class for the Schelling segregation model."""

    def __init__(
        self,
        height=20,
        width=20,
        homophily=3,
        radius=1,
        density=0.8,
        minority_pc=0.2,
        seed=None,
    ):
        """Create a new Schelling model.

        Args:
            width, height: Size of the space.
            density: Initial Chance for a cell to populated
            minority_pc: Chances for an agent to be in minority class
            homophily: Minimum number of agents of same class needed to be happy
            radius: Search radius for checking similarity
            seed: Seed for Reproducibility
        """
        super().__init__(seed=seed)
        self.homophily = homophily
        self.radius = radius

        self.grid = mesa.space.SingleGrid(width, height, torus=True)

        self.happy = 0
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "happy": "happy",
                "shannon_entropy": compute_shannon_entropy,
                "dissimilarity_index": compute_dissimilarity_index,
                "dissimilarity_index_2": partial(compute_dissimilarity_index, radius=2),
                "dissimilarity_index_3": partial(compute_dissimilarity_index, radius=3),
            },
        )
        self.total_minority = 0
        self.total_majority = 0

        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        for _, pos in self.grid.coord_iter():
            if self.random.random() < density:
                agent_type = 1 if self.random.random() < minority_pc else 0
                if agent_type == 1:
                    self.total_minority += 1
                else:
                    self.total_majority += 1
                agent = SchellingAgent(self, agent_type)
                self.grid.place_agent(agent, pos)

        self.datacollector.collect(self)

    def step(self):
        """Run one step of the model."""
        self.happy = 0  # Reset counter of happy agents
        self.agents.shuffle_do("step")

        self.datacollector.collect(self)

        self.running = self.happy != len(self.agents)
