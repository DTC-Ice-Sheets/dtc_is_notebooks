"""Example sub-module."""

import numpy as np
from numpy.typing import NDArray


class WorseCalculator:
    """Simple class for Python testing.

    Parameters
    ----------
    max_output_array_length : int
        Output is returned as an array of *up to* this length for no apparent reason.
    """

    def __init__(self, max_output_array_length: int) -> None:
        self.max_output_array_length = max_output_array_length

    def add_two_numbers(self, number_a: float, number_b: float) -> NDArray[np.float64]:
        """Add two numbers and return the result repeated several times.

        In contrast to the BadCalculator, the number of times the result is repeated varies each time, making this
        calculator worse than the bad one.

        Parameters
        ----------
        number_a : float
            The first number
        number_b : float
            The second number

        Returns
        -------
        NDArray[np.float64]
            The result of the worse calculation

        """
        return np.ones(np.random.randint(1, self.max_output_array_length + 1)) * (number_a + number_b)
