"""Helper functions for the data provenance notebook."""

from ipywidgets import Layout, widgets


def sanitise_for_graphviz(identifier: str) -> str:
    """Sanitise a string (e.g. PROV identifier) so it can be used as a Graphviz ID.

    Parameters
    ----------
    identifier : str
        The input string to sanitise.

    Returns
    -------
    str
        The output string that is safe to use as a Graphviz ID.
    """
    return str(hash(identifier))


def dict_to_ipy_widget(d: dict, special: list[str] | None = None) -> widgets.Widget:
    """Convert a dictionary to an IPython widget for display.

    Parameters
    ----------
    d : dict
        A dictionary of items that can be cast to `str`.
    special : list[str] | None
        Optional list of keys to highlight as special.

    Returns
    -------
    widgets.Widget
        The resulting IPython widget.
    """
    items = []
    if special is None:
        special = []
    for key, value in d.items():
        items.append(
            widgets.Label(
                value=str(key),
                style={"font_weight": "bold", **({"text_color": "blue"} if key in special else {})},
                layout=Layout(justify_content="flex-end", align_items="center"),
            )
        )
        items.append(
            widgets.Label(
                value=str(value), layout=Layout(border="1px solid black", padding="0.5em", overflow="auto hidden")
            )
        )

    # Fix for Google Colab: it was adding `text-oveflow: ellipsis` by default, which breaks the scrolling
    css_fix = widgets.HTML("<style>div.widget-label { text-overflow: clip; }</style>")

    return widgets.GridBox(children=(css_fix, *items), layout=Layout(grid_template_columns=r"min-content 1fr"))
