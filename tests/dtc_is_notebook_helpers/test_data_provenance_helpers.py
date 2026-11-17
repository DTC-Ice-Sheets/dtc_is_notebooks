import pytest

from dtc_is_notebook_helpers.data_provenance_helpers import dict_to_ipy_widget, sanitise_for_graphviz


@pytest.mark.parametrize(
    "identifier,expected",
    [
        ("", ""),
        (" ", " "),
        ("??:?", "??&colon;?"),
        (
            "a long' :|\\ \" id full of - _ weird? characters 🔥\n!",
            "a long' &colon;|\\ \" id full of - _ weird? characters 🔥\n!",
        ),
    ],
)
def test_sanitise_for_graphviz(identifier, expected):
    output = sanitise_for_graphviz(identifier)
    assert output == expected
    assert ":" not in output


def test_dict_to_ipy_widget():
    d = {"key1": "value1", "key2": "value2", "key3": "value3"}
    widget = dict_to_ipy_widget(d, special=["key1"])
    assert widget is not None
    # Basic smoke test to ensure input values are in the output
    assert hasattr(widget, "children")
    assert any("key1" in str(child) for child in widget.children)
    assert any("value1" in str(child) for child in widget.children)
    assert any("key3" in str(child) for child in widget.children)
    assert any("value3" in str(child) for child in widget.children)
    assert any("key2" in str(child) for child in widget.children)
    assert any("value2" in str(child) for child in widget.children)
