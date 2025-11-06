"""Example for dtc_python_template."""

import numpy as np
from numpy.typing import NDArray


class BadCalculator:
    """Simple class for Python testing.

    Parameters
    ----------
    output_array_length : int
        Output is returned as an array of this length for no apparent reason.
    """

    def __init__(self, output_array_length: int) -> None:
        self.output_array_length = output_array_length

    def add_two_numbers(self, number_a: float, number_b: float) -> NDArray[np.float64]:
        """Add two numbers and return the result repeated several times.

        Parameters
        ----------
        number_a : float
            The first number
        number_b : float
            The second number

        Returns
        -------
        NDArray[np.float64]
            The result of the bad calculation

        """
        return np.ones(self.output_array_length) * (number_a + number_b)
