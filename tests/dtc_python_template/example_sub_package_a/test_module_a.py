"""Testing the a module."""

import pytest

from dtc_python_template.example_sub_package_a.example_module_a import BadCalculator


@pytest.fixture
def example_bad_calculator() -> BadCalculator:
    """Initialise bad calculator.

    Returns
    -------
    BadCalculator
        An instance of a 4 length BadCalculator

    """
    return BadCalculator(output_array_length=4)


def test_bad_calculator_can_be_instantiated(example_bad_calculator: BadCalculator) -> None:
    assert example_bad_calculator is not None


def test_bad_calculator_output_length_obeyed(example_bad_calculator: BadCalculator) -> None:
    assert len(example_bad_calculator.add_two_numbers(1, 2)) == 4


def test_bad_calculator_output_correct(example_bad_calculator: BadCalculator) -> None:
    assert all(example_bad_calculator.add_two_numbers(2, 6) == 8)
