"""Monitoring and logging tools."""

import logging


def setup_logging() -> None:
    """
    Set up root logger.

    This configuration will be inherited by all module-level loggers created within the package.
    """
    # Create the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a console handler and set the level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and set it for the console handler
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(name)s: %(message)s")
    console_handler.setFormatter(formatter)

    # Add the console handler to the root logger
    logger.addHandler(console_handler)
