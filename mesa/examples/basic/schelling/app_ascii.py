import solara

from mesa.examples.basic.schelling.model import Schelling
from mesa.visualization import (
    Slider,
    SolaraViz,
    make_plot_component,
    make_space_component,
)


def get_happy_agents(model):
    """Display a text count of how many happy agents there are."""
    return solara.Markdown(f"**Happy agents: {model.happy}**")


def agent_portrayal(agent):
    return "O" if agent.type == 0 else "X"

model_params = {
    "density": Slider("Agent density", 0.8, 0.1, 1.0, 0.1),
    "minority_pc": Slider("Fraction minority", 0.2, 0.0, 1.0, 0.05),
    "homophily": Slider("Homophily", 3, 0, 8, 1),
    "width": 20,
    "height": 20,
}

model1 = Schelling(20, 20, 0.8, 0.2, 3)

HappyPlot = make_plot_component("happy", backend="ascii")

page = SolaraViz(
    model1,
    components=[
        make_space_component(agent_portrayal, backend="ascii"),
        HappyPlot,
        get_happy_agents,
    ],
    model_params=model_params,
)
page  # noqa
