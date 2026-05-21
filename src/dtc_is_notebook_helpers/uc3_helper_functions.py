"""Provide helper functions and widgets for the UC3 notebooks."""

from __future__ import annotations

import ipywidgets as widgets
from dtc_query_client import ApiClient, Configuration, GenericApi, StateAndFateApi
from traitlets import traitlets


def authenticate_with_token(token: str) -> ApiClient:
    """
    Create an authenticated API client.

    Parameters
    ----------
    token : str
        API token used to authenticate requests against the DTC Query API.

    Returns
    -------
    ApiClient
        Configured API client instance.
    """
    config = Configuration(host="https://query.dtc-ice-sheets.org", access_token=token)
    client = ApiClient(config)
    return client


class LoadedButton(widgets.Button):
    """A button that can hold a value as an attribute."""

    def __init__(self: LoadedButton, value: object = None, *args: object, **kwargs: object) -> None:
        """
        Initialize the button with an attached value trait.

        Parameters
        ----------
        value : any, optional
            Initial value stored on the button, by default None.
        *args
            Positional arguments forwarded to ``widgets.Button``.
        **kwargs
            Keyword arguments forwarded to ``widgets.Button``.
        """
        super(LoadedButton, self).__init__(*args, **kwargs)
        # Create the value attribute.
        self.add_traits(value=traitlets.Any(value))


def update_client(b: LoadedButton) -> object:
    """
    Return the client stored on a loaded button.

    Parameters
    ----------
    b : LoadedButton
        Button carrying the client in its ``value`` attribute.

    Returns
    -------
    object
        Stored client value.
    """
    return b.value


async def get_ice_shelves(client: ApiClient) -> list[str]:
    """
    Fetch available ice shelf names.

    Parameters
    ----------
    client : ApiClient
        Authenticated API client.

    Returns
    -------
    list[str]
        Available ice shelf names.
    """
    return await StateAndFateApi(client).list_ice_shelves()


async def get_datasets(client: ApiClient) -> list[object]:
    """
    Fetch available dataset overviews.

    Parameters
    ----------
    client : ApiClient
        Authenticated API client.

    Returns
    -------
    list[object]
        Dataset overview objects returned by the API.
    """
    return await GenericApi(client).dataset_overviews()


async def ice_shelf_selector(client: ApiClient) -> widgets.Dropdown:
    """
    Create an ice shelf selection widget.

    Parameters
    ----------
    client : ApiClient
        Authenticated API client.

    Returns
    -------
    widgets.Dropdown
        Dropdown populated with available ice shelves.
    """
    ice_shelves = await get_ice_shelves(client=client)

    return widgets.Dropdown(
        options=ice_shelves,
        value=ice_shelves[0],
        description="Ice Shelf:",
        disabled=False,
    )


async def dataset_selector(client: ApiClient) -> widgets.SelectMultiple:
    """
    Create a dataset selection widget.

    Parameters
    ----------
    client : ApiClient
        Authenticated API client.

    Returns
    -------
    widgets.SelectMultiple
        Multi-select widget populated with dataset identifiers.
    """
    datasets = await get_datasets(client=client)

    datasets = [dataset.dataset_id for dataset in datasets if dataset.dataset_id != ""]

    return widgets.SelectMultiple(
        options=datasets,
        value=[datasets[0]],
        rows=10,
        description="Datasets",
        disabled=False,
        layout=widgets.Layout(width="80%"),
    )


def analysis_type_selector() -> widgets.ToggleButtons:
    """
    Create an analysis type toggle widget.

    Returns
    -------
    widgets.ToggleButtons
        Toggle buttons for the supported analysis modes.
    """
    return widgets.ToggleButtons(
        options=["Correlation", "Cross-Correlation", "Granger Causality"],
        description="Select Analysis Type:",
        disabled=False,
        button_style="info",  # 'success', 'info', 'warning', 'danger' or ''
    )


def widget_credentials_make() -> widgets.Box:
    """
    Create a credential entry widget group.

    Returns
    -------
    widgets.Box
        Container holding the token input, submit button, and output area.
    """

    def on_button_clicked(b: LoadedButton) -> None:
        """
        Authenticate the entered token and store the client on the button.

        Parameters
        ----------
        b : LoadedButton
            Button receiving the authenticated client in its ``value`` attribute.
        """
        with credentials_output:
            credentials_output.clear_output()
            if credentials_box.value == "":
                print("Please enter an API token before submitting.")
            else:
                print(f"API token submitted: {credentials_box.value}")

                client = authenticate_with_token(credentials_box.value)

        b.value = client

    credentials_box = widgets.Text(
        value="", placeholder="Enter your API token here...", description="API token:", disabled=False
    )

    credentials_button = LoadedButton(
        description="submit API token",
        disabled=False,
        button_style="info",  # 'success', 'info', 'warning', 'danger' or ''
        tooltip="",
        icon="",
    )

    credentials_output = widgets.Output()
    credentials_button.on_click(on_button_clicked)

    credentials_container = [credentials_box, credentials_button, credentials_output]
    return widgets.Box(credentials_container)
