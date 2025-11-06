"""Entrypoint for subpackage a.

Note that not all packages need a command line entry point.
"""

import argparse
import logging

from dtc_python_template.example_sub_package_a.example_module_a import BadCalculator
from dtc_python_template.monitoring import setup_logging
from dtc_python_template.schemas import ExampleEntryPointSchema

logger = logging.getLogger(__name__)


def main() -> None:
    """Calculate numbers poorly."""
    setup_logging()
    parser = argparse.ArgumentParser(
        prog="dtc_python_template",
        description="Perform some bad calculations on two integers.",
    )

    # Add arguments to the parser
    for name, field in ExampleEntryPointSchema.model_fields.items():
        parser.add_argument(
            f"--{name}",
            dest=name,
            type=field.annotation if field.annotation is not None else int,
            default=field.default,
            help=field.description,
            required=name in ["integer_a", "integer_b"],
        )

    # Parse the input arguments
    args = parser.parse_args()

    # Create the model from parsed arguments
    model = ExampleEntryPointSchema(**vars(args))

    # Instantiate the calculator and perform calculations
    bad_calculator = BadCalculator(output_array_length=5)
    logger.info(
        "Bad Calculator Output: %s",
        bad_calculator.add_two_numbers(model.integer_a, model.integer_b),
    )


if __name__ == "__main__":
    main()
