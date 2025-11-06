"""Testing the entrypoint."""

import logging
import sys
from unittest.mock import patch

import pytest


@patch.object(sys, "argv", ["dtc_python_template", "--integer_a", "4", "--integer_b", "5"])
def test_main(caplog: pytest.LogCaptureFixture) -> None:
    """Test main for example_entrypoint_b.

    Parameters
    ----------
    caplog : LogCaptureFixture
        The caplog fixture
    """
    from dtc_python_template.example_entrypoint_b import main

    with caplog.at_level(logging.INFO):
        main()
        assert "Worse Calculator Output" in caplog.records[0].message
        assert "9" in caplog.records[0].message
