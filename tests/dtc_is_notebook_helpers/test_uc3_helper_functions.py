"""Tests for the uc3_helper_functions.py module."""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import ipywidgets as widgets
import pandas as pd
import plotly.graph_objects as go
import pytest
import yaml

import dtc_is_notebook_helpers.uc3_helper_functions as uc3
from dtc_is_notebook_helpers.uc3_helper_functions import (
    LoadedButton,
    _dataset_name_to_pretty_dataset_name,
    _dataset_names_to_pretty_dataset_names,
    _get_analysis_type,
    _load_dataset_yaml,
    _parse_datasets,
    _pretty_dataset_name_to_dataset_name,
    _pretty_dataset_names_to_dataset_names,
    _var_name_to_pretty_var_name,
    _var_name_units,
    _var_names_to_pretty_var_names,
    analysis_type_selector,
    authenticate_with_token,
    build_covariate_analysis_input_selector,
    customwrap,
    dataset_selector,
    display_datasets_and_variables,
    get_analysis_type,
    get_client,
    get_dataset,
    get_ice_shelf,
    get_ice_shelves,
    get_input_datasets,
    get_time_range,
    get_variables,
    ice_shelf_selector,
    plot_covariate_analysis,
    plot_timeseries_data,
    time_range_selector,
    update_client,
    variable_selector,
    widget_credentials_make,
)


@pytest.fixture
def variable_mappings():
    with open(Path(__file__).resolve().parents[1] / "test_inputs" / "variable_mappings_test.yaml") as f:
        return yaml.safe_load(f)


def test_authenticate_with_token_constructs_configuration_and_client(monkeypatch: pytest.MonkeyPatch) -> None:
    token = "fake-token"  # noqa: S105
    fake_config = object()
    configuration_ctor = MagicMock(return_value=fake_config)

    class FakeApiClient:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(uc3, "Configuration", configuration_ctor)
    monkeypatch.setattr(uc3, "ApiClient", FakeApiClient)

    result = authenticate_with_token(token)

    configuration_ctor.assert_called_once_with(host="https://query.dtc-ice-sheets.org", access_token=token)
    assert isinstance(result, FakeApiClient)
    assert result.config is fake_config


def test_loaded_button_initialization():
    label = "Test Button"
    button = LoadedButton(label=label)
    assert button.value is None


def test_update_client():
    button = LoadedButton(label="Test Button")
    assert update_client(button) is None


def test_load_dataset_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    test_yaml_path = Path(__file__).resolve().parents[1] / "test_inputs" / "variable_mappings_test.yaml"

    class MockPath:
        @property
        def parent(self):
            return self

        def __truediv__(self, _):
            return test_yaml_path

    monkeypatch.setattr(uc3, "Path", lambda _p: MockPath())
    result = _load_dataset_yaml()

    assert isinstance(result, dict)
    assert result["TEST_DATASET_1"]["label"] == "Test Dataset 1"


def test_parse_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    test_yaml_path = Path(__file__).resolve().parents[1] / "test_inputs" / "variable_mappings_test.yaml"

    class MockPath:
        @property
        def parent(self):
            return self

        def __truediv__(self, _):
            return test_yaml_path

    monkeypatch.setattr(uc3, "Path", lambda _p: MockPath())
    dataset_config = _load_dataset_yaml()
    result = _parse_datasets(dataset_config)

    assert isinstance(result, list)
    assert "Test Dataset 1" in result
    assert "Test Dataset 2" in result


def test_dataset_names_to_pretty_dataset_names(monkeypatch: pytest.MonkeyPatch, variable_mappings: dict) -> None:
    monkeypatch.setattr(uc3, "_load_dataset_yaml", lambda: variable_mappings)

    result = _dataset_names_to_pretty_dataset_names()

    assert result == {
        "TEST_DATASET_1": "Test Dataset 1",
        "TEST_DATASET_2": "Test Dataset 2",
    }


def test_pretty_dataset_names_to_dataset_names(monkeypatch: pytest.MonkeyPatch, variable_mappings: dict) -> None:
    monkeypatch.setattr(uc3, "_load_dataset_yaml", lambda: variable_mappings)

    result = _pretty_dataset_names_to_dataset_names()

    assert result == {
        "Test Dataset 1": "TEST_DATASET_1",
        "Test Dataset 2": "TEST_DATASET_2",
    }


def test_pretty_dataset_name_to_dataset_name(monkeypatch: pytest.MonkeyPatch, variable_mappings: dict) -> None:
    monkeypatch.setattr(uc3, "_load_dataset_yaml", lambda: variable_mappings)

    result = _pretty_dataset_name_to_dataset_name("Test Dataset 1")

    assert result == "TEST_DATASET_1"


def test_pretty_dataset_name_to_dataset_name_unknown_returns_none(
    monkeypatch: pytest.MonkeyPatch, variable_mappings: dict
) -> None:
    monkeypatch.setattr(uc3, "_load_dataset_yaml", lambda: variable_mappings)

    with pytest.raises(ValueError, match="Dataset label 'Unknown Dataset' not found in configuration."):
        _pretty_dataset_name_to_dataset_name("Unknown Dataset")


def test_dataset_name_to_pretty_dataset_name(monkeypatch: pytest.MonkeyPatch, variable_mappings: dict) -> None:
    monkeypatch.setattr(uc3, "_load_dataset_yaml", lambda: variable_mappings)

    result = _dataset_name_to_pretty_dataset_name("TEST_DATASET_2")

    assert result == "Test Dataset 2"


def test_var_names_to_pretty_var_names(monkeypatch: pytest.MonkeyPatch, variable_mappings: dict) -> None:
    monkeypatch.setattr(uc3, "_load_dataset_yaml", lambda: variable_mappings)

    result = _var_names_to_pretty_var_names()

    assert result == {
        "test_variable_1": "Test Variable 1",
        "test_variable_2": "Test Variable 2",
    }


def test_var_name_to_pretty_var_name(monkeypatch: pytest.MonkeyPatch, variable_mappings: dict) -> None:
    monkeypatch.setattr(uc3, "_load_dataset_yaml", lambda: variable_mappings)

    result = _var_name_to_pretty_var_name("test_variable_2")

    assert result == "Test Variable 2"


def test_var_name_units(monkeypatch: pytest.MonkeyPatch, variable_mappings: dict) -> None:
    monkeypatch.setattr(uc3, "_load_dataset_yaml", lambda: variable_mappings)

    result = _var_name_units("test_variable_1", "TEST_DATASET_1")

    assert result == "m"


def _build_test_client() -> uc3.ApiClient:
    return uc3.ApiClient(uc3.Configuration(host="https://query.dtc-ice-sheets.org", access_token="fake-token"))  # noqa: S106


def test_get_ice_shelves(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _build_test_client()

    class FakeStateAndFateApi:
        def __init__(self, client):
            self.client = client

        async def list_ice_shelves(self):
            return ["thwaites", "pine_island"]

    monkeypatch.setattr(uc3, "StateAndFateApi", FakeStateAndFateApi)

    result = asyncio.run(get_ice_shelves(fake_client))

    assert result == ["thwaites", "pine_island"]


def test_get_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _build_test_client()

    class FakeGenericApi:
        def __init__(self, client):
            self.client = client

        async def dataset_overviews(self):
            return ["dataset_a", "dataset_b"]

    monkeypatch.setattr(uc3, "GenericApi", FakeGenericApi)

    result = asyncio.run(get_dataset(fake_client))

    assert result == ["dataset_a", "dataset_b"]


def test_ice_shelf_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get_ice_shelves(client):
        return ["thwaites", "pine_island"]

    monkeypatch.setattr(uc3, "get_ice_shelves", fake_get_ice_shelves)

    result = asyncio.run(ice_shelf_selector(_build_test_client()))

    assert isinstance(result, widgets.VBox)
    assert result.children[0].value == "Select Ice Shelf:"
    assert tuple(result.children[1].options) == ("thwaites", "pine_island")
    assert result.children[1].value == "thwaites"


def test_display_datasets_and_variables(monkeypatch: pytest.MonkeyPatch, variable_mappings: dict) -> None:
    dataset_select = widgets.SelectMultiple(options=["Test Dataset 1", "Test Dataset 2"], value=("Test Dataset 1",))
    dataset_widget = widgets.VBox([widgets.Label(value="Select Datasets:"), dataset_select])

    variable_select = widgets.SelectMultiple(options=[])
    update_button = widgets.Button(description="Update variables")
    variable_widget = widgets.VBox([widgets.Label(value="Select Variables:"), variable_select, update_button])

    async def fake_dataset_selector(client):
        return dataset_widget

    async def fake_variable_selector(client):
        return variable_widget

    monkeypatch.setattr(uc3, "dataset_selector", fake_dataset_selector)
    monkeypatch.setattr(uc3, "variable_selector", fake_variable_selector)
    monkeypatch.setattr(uc3, "_load_dataset_yaml", lambda: variable_mappings)

    result_dataset_widget, result_variable_widget = asyncio.run(display_datasets_and_variables(_build_test_client()))

    assert result_dataset_widget is dataset_widget
    assert result_variable_widget is variable_widget

    update_button.click()

    assert tuple(variable_select.options) == ("test_variable_1",)


def test_dataset_selector(monkeypatch: pytest.MonkeyPatch, variable_mappings: dict) -> None:
    monkeypatch.setattr(uc3, "_load_dataset_yaml", lambda: variable_mappings)
    monkeypatch.setattr("typeguard._functions.check_type_internal", lambda value, annotation, memo: None)

    result = asyncio.run(dataset_selector(_build_test_client()))

    assert isinstance(result, widgets.VBox)
    assert result.children[0].value == "Select Datasets:"
    assert tuple(result.children[1].options) == ("Test Dataset 1", "Test Dataset 2")
    assert result.children[1].rows == 10
    assert result.children[1].disabled is False


def test_variable_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("typeguard._functions.check_type_internal", lambda value, annotation, memo: None)

    result = asyncio.run(variable_selector(_build_test_client()))

    assert isinstance(result, widgets.VBox)
    assert result.children[0].value == "Select Variables:"
    assert tuple(result.children[1].options) == ()
    assert result.children[1].rows == 10
    assert result.children[2].description == "Update variables"


def test_analysis_type_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("typeguard._functions.check_type_internal", lambda value, annotation, memo: None)

    result = analysis_type_selector()

    assert isinstance(result, widgets.HBox)
    assert result.children[0].value == "Select Analysis Type: "
    assert tuple(result.children[1].options) == (
        "Correlation",
        "Cross-Correlation",
        "Granger Causality",
    )


def test_time_range_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("typeguard._functions.check_type_internal", lambda value, annotation, memo: None)

    result = time_range_selector()

    assert isinstance(result, widgets.Box)
    assert result.children[0].value == "Pick a start datetime:"
    assert result.children[1].value == "2010-01-01 00:00:00"
    assert result.children[1].placeholder == "YYYY-MM-DD HH:MM:SS"
    assert result.children[2].value == "Pick an end datetime:"
    assert result.children[3].placeholder == "YYYY-MM-DD HH:MM:SS"

    end_time = datetime.strptime(result.children[3].value, "%Y-%m-%d %H:%M:%S")
    assert end_time.date() <= datetime.now().date()


def _build_input_selector(
    datasets_value: tuple[str, ...] = ("Test Dataset 1",),
    variables_value: tuple[str, ...] = ("test_variable_1",),
    analysis_value: str = "Correlation",
) -> widgets.VBox:
    ice_shelf_widget = widgets.VBox(
        [widgets.Label(value="Select Ice Shelf:"), widgets.Dropdown(options=["thwaites", "pine"], value="thwaites")]
    )
    dataset_widget = widgets.VBox(
        [
            widgets.Label(value="Select Datasets:"),
            widgets.SelectMultiple(options=["Test Dataset 1", "Test Dataset 2"], value=datasets_value),
        ]
    )
    variable_widget = widgets.VBox(
        [
            widgets.Label(value="Select Variables:"),
            widgets.SelectMultiple(options=["test_variable_1", "test_variable_2"], value=variables_value),
            widgets.Button(description="Update variables"),
        ]
    )
    analysis_widget = widgets.HBox(
        [
            widgets.Label(value="Select Analysis Type: "),
            widgets.ToggleButtons(
                options=["Correlation", "Cross-Correlation", "Granger Causality"],
                value=analysis_value,
            ),
        ]
    )
    time_widget = widgets.Box(
        [
            widgets.Label(value="Pick a start datetime:"),
            widgets.Text(value="2010-01-01 00:00:00"),
            widgets.Label(value="Pick an end datetime:"),
            widgets.Text(value="2011-01-01 00:00:00"),
        ]
    )
    return widgets.VBox(
        [
            widgets.HBox([ice_shelf_widget, dataset_widget, variable_widget]),
            widgets.HBox([analysis_widget]),
            widgets.HBox([time_widget]),
        ]
    )


def test_widget_credentials_make(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("typeguard._functions.check_type_internal", lambda value, annotation, memo: None)
    monkeypatch.setattr(uc3, "display", lambda *args, **kwargs: None, raising=False)

    result = widget_credentials_make()

    assert isinstance(result, widgets.Box)
    assert isinstance(result.children[0], widgets.Text)
    assert isinstance(result.children[1], LoadedButton)
    assert isinstance(result.children[2], widgets.Output)


def test_get_client() -> None:
    fake_client = _build_test_client()
    credentials_container = widgets.Box([widgets.Text(), LoadedButton(value=fake_client), widgets.Output()])

    result = get_client(credentials_container)

    assert result is fake_client


def test_get_ice_shelf() -> None:
    input_selector = _build_input_selector()

    result = get_ice_shelf(input_selector)

    assert result == "thwaites"


def test_get_analysis_type() -> None:
    input_selector = _build_input_selector(analysis_value="Cross-Correlation")

    result = get_analysis_type(input_selector)

    assert result == "Cross-Correlation"


def test_get_time_range() -> None:
    input_selector = _build_input_selector()

    result = get_time_range(input_selector)

    assert result == ("2010-01-01 00:00:00", "2011-01-01 00:00:00")


def test_get_input_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("typeguard._functions.check_type_internal", lambda value, annotation, memo: None)
    input_selector = _build_input_selector(datasets_value=("Test Dataset 1", "Test Dataset 2"))

    result = get_input_datasets(input_selector)

    assert result == ("Test Dataset 1", "Test Dataset 2")


def test_get_variables(monkeypatch: pytest.MonkeyPatch, variable_mappings: dict) -> None:
    monkeypatch.setattr("typeguard._functions.check_type_internal", lambda value, annotation, memo: None)
    monkeypatch.setattr(uc3, "_load_dataset_yaml", lambda: variable_mappings)
    monkeypatch.setattr(
        uc3,
        "_pretty_dataset_name_to_dataset_name",
        lambda name: {"Test Dataset 1": "TEST_DATASET_1", "Test Dataset 2": "TEST_DATASET_2"}[name],
    )

    class FakeVariableItem:
        def __init__(self, name, dataset, variable, extent):
            self.name = name
            self.dataset = dataset
            self.variable = variable
            self.extent = extent

    monkeypatch.setattr(uc3, "VariableItem", FakeVariableItem)
    input_selector = _build_input_selector(datasets_value=("Test Dataset 1",), variables_value=("test_variable_1",))

    result = get_variables(input_selector)

    assert len(result) == 1
    assert result[0].dataset == "TEST_DATASET_1"
    assert result[0].variable == "test_variable_1"
    assert result[0].extent == "ice_shelf"


def test_build_covariate_analysis_input_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uc3, "display", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr("typeguard._functions.check_type_internal", lambda value, annotation, memo: None)

    ice_shelf_widget = widgets.VBox([widgets.Label(value="Select Ice Shelf:"), widgets.Dropdown(options=["thwaites"])])
    dataset_widget = widgets.VBox(
        [widgets.Label(value="Select Datasets:"), widgets.SelectMultiple(options=["Test Dataset 1"])]
    )
    variable_widget = widgets.VBox(
        [
            widgets.Label(value="Select Variables:"),
            widgets.SelectMultiple(options=["test_variable_1"]),
            widgets.Button(),
        ]
    )
    analysis_widget = widgets.HBox(
        [widgets.Label(value="Select Analysis Type: "), widgets.ToggleButtons(options=["Correlation"])]
    )
    time_widget = widgets.Box(
        [widgets.Label(value="start"), widgets.Text(), widgets.Label(value="end"), widgets.Text()]
    )

    async def fake_ice_shelf_selector(client):
        return ice_shelf_widget

    async def fake_display_datasets_and_variables(client):
        return dataset_widget, variable_widget

    monkeypatch.setattr(uc3, "ice_shelf_selector", fake_ice_shelf_selector)
    monkeypatch.setattr(uc3, "display_datasets_and_variables", fake_display_datasets_and_variables)
    monkeypatch.setattr(uc3, "analysis_type_selector", lambda: analysis_widget)
    monkeypatch.setattr(uc3, "time_range_selector", lambda: time_widget)

    result = asyncio.run(build_covariate_analysis_input_selector(_build_test_client()))

    assert isinstance(result, widgets.VBox)
    assert len(result.children) == 3


@pytest.mark.parametrize(
    ("analysis_value", "expected"),
    [
        ("Correlation", uc3.CovariateAnalysisType.CORR),
        ("Cross-Correlation", uc3.CovariateAnalysisType.CROSS_CORR),
        ("Granger Causality", uc3.CovariateAnalysisType.CAUSAL),
    ],
)
def test__get_analysis_type(analysis_value: str, expected) -> None:
    input_selector = _build_input_selector(analysis_value=analysis_value)

    result = _get_analysis_type(input_selector)

    assert result == expected


def test__get_analysis_type_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("typeguard._functions.check_type_internal", lambda value, annotation, memo: None)

    class Node:
        def __init__(self, children=None, value=None):
            self.children = children if children is not None else []
            self.value = value

    input_selector = Node(
        children=[
            Node(children=[]),
            Node(children=[Node(children=[Node(value="label"), Node(value="Unknown")])]),
            Node(children=[]),
        ]
    )

    with pytest.raises(ValueError, match="Unsupported analysis type"):
        _get_analysis_type(input_selector)


def test_customwrap() -> None:
    result = customwrap("alpha beta gamma", width=6)

    assert result == "alpha<br>beta<br>gamma"


def test_plot_covariate_analysis() -> None:
    plot_data = {
        "variables": {
            "TEST_DATASET_1: Test Variable 1 [m]": {
                "hist_edges": [0.0, 1.0, 2.0],
                "hist_values": [2, 1],
                "values": [0.1, 0.5, 1.2],
            },
            "TEST_DATASET_2: Test Variable 2 [m]": {
                "hist_edges": [0.0, 1.0, 2.0],
                "hist_values": [1, 2],
                "values": [0.2, 0.4, 1.1],
            },
        },
        "covariate_stats": {
            "TEST_DATASET_1: Test Variable 1 [m]/TEST_DATASET_2: Test Variable 2 [m]": {"corr": 0.7},
            "TEST_DATASET_2: Test Variable 2 [m]/TEST_DATASET_1: Test Variable 1 [m]": {"corr": 0.7},
        },
    }

    result = plot_covariate_analysis(plot_data, uc3.CovariateAnalysisType.CORR)

    assert isinstance(result, go.Figure)
    assert len(result.data) > 0
    assert len(result.layout.annotations) > 0


def test_plot_timeseries_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uc3, "_var_name_units", lambda var_name, dataset: "m")
    monkeypatch.setattr(uc3, "_var_name_to_pretty_var_name", lambda var_name: "Pretty Variable")

    timeseries_data = {
        "test_variable_1_0": (
            pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-01-02"]), name="test_variable_1"),
            "TEST_DATASET_1",
        )
    }

    result = plot_timeseries_data(timeseries_data)

    assert isinstance(result, go.Figure)
    assert len(result.data) == 1
    assert result.data[0].name == "Pretty Variable (m)"
