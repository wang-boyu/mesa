"""Mesa visualization module for creating interactive model visualizations.

This module provides components to create browser- and Jupyter notebook-based visualizations of
Mesa models, allowing users to watch models run step-by-step and interact with model parameters.

Key features:
    - SolaraViz: Main component for creating visualizations, supporting grid displays and plots
    - ModelController: Handles model execution controls (step, play, pause, reset)
    - UserInputs: Generates UI elements for adjusting model parameters

The module uses Solara for rendering in Jupyter notebooks or as standalone web applications.
It supports various types of visualizations including matplotlib plots, agent grids, and
custom visualization components.

Usage:
    1. Define an agent_portrayal function to specify how agents should be displayed
    2. Set up model_params to define adjustable parameters
    3. Create a SolaraViz instance with your model, parameters, and desired measures
    4. Display the visualization in a Jupyter notebook or run as a Solara app

See the Visualization Tutorial and example models for more details.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import reacton.core
import solara

import mesa.visualization.components.altair as components_altair
from mesa.visualization.UserParam import Slider
from mesa.visualization.utils import force_update, update_counter

if TYPE_CHECKING:
    from mesa.model import Model


@solara.component
def SolaraViz(
    model: Model | solara.Reactive[Model],
    components: list[reacton.core.Component]
    | list[Callable[[Model], reacton.core.Component]]
    | Literal["default"] = "default",
    play_interval: int = 100,
    model_params=None,
    seed: float = 0,
    name: str | None = None,
):
    """Solara visualization component.

    This component provides a visualization interface for a given model using Solara.
    It supports various visualization components and allows for interactive model
    stepping and parameter adjustments.

    Args:
        model (Model | solara.Reactive[Model]): A Model instance or a reactive Model.
            This is the main model to be visualized. If a non-reactive model is provided,
            it will be converted to a reactive model.
        components (list[solara.component] | Literal["default"], optional): List of solara
            components or functions that return a solara component.
            These components are used to render different parts of the model visualization.
            Defaults to "default", which uses the default Altair space visualization.
        play_interval (int, optional): Interval for playing the model steps in milliseconds.
            This controls the speed of the model's automatic stepping. Defaults to 100 ms.
        model_params (dict, optional): Parameters for (re-)instantiating a model.
            Can include user-adjustable parameters and fixed parameters. Defaults to None.
        seed (int, optional): Seed for the random number generator. This ensures reproducibility
            of the model's behavior. Defaults to 0.
        name (str | None, optional): Name of the visualization. Defaults to the models class name.

    Returns:
        solara.component: A Solara component that renders the visualization interface for the model.

    Example:
        >>> model = MyModel()
        >>> page = SolaraViz(model)
        >>> page

    Notes:
        - The `model` argument can be either a direct model instance or a reactive model. If a direct
          model instance is provided, it will be converted to a reactive model using `solara.use_reactive`.
        - The `play_interval` argument controls the speed of the model's automatic stepping. A lower
          value results in faster stepping, while a higher value results in slower stepping.
    """
    if components == "default":
        components = [components_altair.make_space_altair()]

    # Convert model to reactive
    if not isinstance(model, solara.Reactive):
        model = solara.use_reactive(model)  # noqa: SH102, RUF100

    def connect_to_model():
        # Patch the step function to force updates
        original_step = model.value.step

        def step():
            original_step()
            force_update()

        model.value.step = step
        # Add a trigger to model itself
        model.value.force_update = force_update
        force_update()

    solara.use_effect(connect_to_model, [model.value])

    with solara.AppBar():
        solara.AppBarTitle(name if name else model.value.__class__.__name__)

    with solara.Sidebar(), solara.Column():
        with solara.Card("Controls"):
            ModelController(model, play_interval)

        if model_params is not None:
            with solara.Card("Model Parameters"):
                ModelCreator(
                    model,
                    model_params,
                    seed=seed,
                )
        with solara.Card("Information"):
            ShowSteps(model.value)

    ComponentsView(components, model.value)


def _wrap_component(
    component: reacton.core.Component | Callable[[Model], reacton.core.Component],
) -> reacton.core.Component:
    """Wrap a component in an auto-updated Solara component if needed."""
    if isinstance(component, reacton.core.Component):
        return component

    @solara.component
    def WrappedComponent(model):
        update_counter.get()
        return component(model)

    return WrappedComponent


@solara.component
def ComponentsView(
    components: list[reacton.core.Component]
    | list[Callable[[Model], reacton.core.Component]],
    model: Model,
):
    """Display a list of components.

    Args:
        components: List of components to display
        model: Model instance to pass to each component
    """
    wrapped_components = [_wrap_component(component) for component in components]
    items = [component(model) for component in wrapped_components]
    grid_layout_initial = make_initial_grid_layout(num_components=len(items))
    grid_layout, set_grid_layout = solara.use_state(grid_layout_initial)
    solara.GridDraggable(
        items=items,
        grid_layout=grid_layout,
        resizable=True,
        draggable=True,
        on_grid_layout=set_grid_layout,
    )


JupyterViz = SolaraViz


@solara.component
def ModelController(model: solara.Reactive[Model], play_interval=100):
    """Create controls for model execution (step, play, pause, reset).

    Args:
        model (solara.Reactive[Model]): Reactive model instance
        play_interval (int, optional): Interval for playing the model steps in milliseconds.
    """
    playing = solara.use_reactive(False)
    running = solara.use_reactive(True)

    async def step():
        while playing.value and running.value:
            await asyncio.sleep(play_interval / 1000)
            do_step()

    solara.lab.use_task(
        step, dependencies=[playing.value, running.value], prefer_threaded=False
    )

    def do_step():
        """Advance the model by one step."""
        model.value.step()
        running.value = model.value.running

    def do_play_pause():
        """Toggle play/pause."""
        playing.value = not playing.value

    with solara.Row(justify="space-between"):
        solara.Button(
            label="▶" if not playing.value else "❚❚",
            color="primary",
            on_click=do_play_pause,
            disabled=not running.value,
        )
        solara.Button(
            label="Step",
            color="primary",
            on_click=do_step,
            disabled=playing.value or not running.value,
        )


def split_model_params(model_params):
    """Split model parameters into user-adjustable and fixed parameters.

    Args:
        model_params: Dictionary of all model parameters

    Returns:
        tuple: (user_adjustable_params, fixed_params)
    """
    model_params_input = {}
    model_params_fixed = {}
    for k, v in model_params.items():
        if check_param_is_fixed(v):
            model_params_fixed[k] = v
        else:
            model_params_input[k] = v
    return model_params_input, model_params_fixed


def check_param_is_fixed(param):
    """Check if a parameter is fixed (not user-adjustable).

    Args:
        param: Parameter to check

    Returns:
        bool: True if parameter is fixed, False otherwise
    """
    if isinstance(param, Slider):
        return False
    if not isinstance(param, dict):
        return True
    if "type" not in param:
        return True


@solara.component
def ModelCreator(model, model_params, seed=1):
    """Solara component for creating and managing a model instance with user-defined parameters.

    This component allows users to create a model instance with specified parameters and seed.
    It provides an interface for adjusting model parameters and reseeding the model's random
    number generator.

    Args:
        model (solara.Reactive[Model]): A reactive model instance. This is the main model to be created and managed.
        model_params (dict): Dictionary of model parameters. This includes both user-adjustable parameters and fixed parameters.
        seed (int, optional): Initial seed for the random number generator. Defaults to 1.

    Returns:
        solara.component: A Solara component that renders the model creation and management interface.

    Example:
        >>> model = solara.reactive(MyModel())
        >>> model_params = {
        >>>     "param1": {"type": "slider", "value": 10, "min": 0, "max": 100},
        >>>     "param2": {"type": "slider", "value": 5, "min": 1, "max": 10},
        >>> }
        >>> creator = ModelCreator(model, model_params)
        >>> creator

    Notes:
        - The `model_params` argument should be a dictionary where keys are parameter names and values either fixed values
          or are dictionaries containing parameter details such as type, value, min, and max.
        - The `seed` argument ensures reproducibility by setting the initial seed for the model's random number generator.
        - The component provides an interface for adjusting user-defined parameters and reseeding the model.

    """
    user_params, fixed_params = split_model_params(model_params)

    reactive_seed = solara.use_reactive(seed)

    model_parameters, set_model_parameters = solara.use_state(
        {
            **fixed_params,
            **{k: v.get("value") for k, v in user_params.items()},
        }
    )

    def do_reseed():
        """Update the random seed for the model."""
        reactive_seed.value = model.value.random.random()

    def on_change(name, value):
        set_model_parameters({**model_parameters, name: value})

    def create_model():
        model.value = model.value.__class__(**model_parameters)
        model.value._seed = reactive_seed.value

    solara.use_effect(create_model, [model_parameters, reactive_seed.value])

    with solara.Row(justify="space-between"):
        solara.InputText(
            label="Seed",
            value=reactive_seed,
            continuous_update=True,
        )

        solara.Button(label="Reseed", color="primary", on_click=do_reseed)

    UserInputs(user_params, on_change=on_change)


@solara.component
def UserInputs(user_params, on_change=None):
    """Initialize user inputs for configurable model parameters.

    Currently supports :class:`solara.SliderInt`, :class:`solara.SliderFloat`,
    :class:`solara.Select`, and :class:`solara.Checkbox`.

    Args:
        user_params: Dictionary with options for the input, including label, min and max values, and other fields specific to the input type.
        on_change: Function to be called with (name, value) when the value of an input changes.
    """
    for name, options in user_params.items():

        def change_handler(value, name=name):
            on_change(name, value)

        if isinstance(options, Slider):
            slider_class = (
                solara.SliderFloat if options.is_float_slider else solara.SliderInt
            )
            slider_class(
                options.label,
                value=options.value,
                on_value=change_handler,
                min=options.min,
                max=options.max,
                step=options.step,
            )
            continue

        # label for the input is "label" from options or name
        label = options.get("label", name)
        input_type = options.get("type")
        if input_type == "SliderInt":
            solara.SliderInt(
                label,
                value=options.get("value"),
                on_value=change_handler,
                min=options.get("min"),
                max=options.get("max"),
                step=options.get("step"),
            )
        elif input_type == "SliderFloat":
            solara.SliderFloat(
                label,
                value=options.get("value"),
                on_value=change_handler,
                min=options.get("min"),
                max=options.get("max"),
                step=options.get("step"),
            )
        elif input_type == "Select":
            solara.Select(
                label,
                value=options.get("value"),
                on_value=change_handler,
                values=options.get("values"),
            )
        elif input_type == "Checkbox":
            solara.Checkbox(
                label=label,
                on_value=change_handler,
                value=options.get("value"),
            )
        else:
            raise ValueError(f"{input_type} is not a supported input type")


def make_initial_grid_layout(num_components):
    """Create an initial grid layout for visualization components.

    Args:
        num_components: Number of components to display

    Returns:
        list: Initial grid layout configuration
    """
    return [
        {
            "i": i,
            "w": 6,
            "h": 10,
            "moved": False,
            "x": 6 * (i % 2),
            "y": 16 * (i - i % 2),
        }
        for i in range(num_components)
    ]


@solara.component
def ShowSteps(model):
    """Display the current step of the model."""
    update_counter.get()
    return solara.Text(f"Step: {model.steps}")
