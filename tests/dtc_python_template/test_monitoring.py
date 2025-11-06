"""Testing the monitoring module."""

import logging
from unittest.mock import patch

from dtc_python_template.monitoring import setup_logging


def test_setup_logging() -> None:
    with patch("dtc_python_template.monitoring.logging.getLogger") as mock_get_logger:
        mock_logger = mock_get_logger.return_value
        setup_logging()

    mock_logger.setLevel.assert_called_once_with(logging.INFO)
    mock_logger.addHandler.assert_called_once()
    added_handler = mock_logger.addHandler.call_args[0][0]
    assert isinstance(added_handler, logging.StreamHandler)
    assert added_handler.level == logging.INFO
    assert isinstance(added_handler.formatter, logging.Formatter)
    assert added_handler.formatter._fmt == "[%(asctime)s] %(levelname)s in %(name)s: %(message)s"
