
----------------------
Meeting Code Standards
----------------------
The DTC Python template is set up to enforce a set of code standards defined for this project. This section provides an
overview of these standards as well as tips on how to adhere to them :)

PEP8 Code Style Guidelines
---------------------------
The `PEP8 Style Guide For Python Code <https://peps.python.org/pep-0008/>`__ provides a comprehensive overview of
widely accepted conventions in Python programming. It covers topics like proper indentation, use of whitespace, and
variable/class/module naming conventions (please use meaningful names!).

The DTC Python template's pre-commit hooks automatically apply formatters to ensure your code adheres to many of these
conventions by default.

Type hints
-----------
Type hints allow us to explicitly specify the type of a variable, function parameter, or function return value. This
helps make the code more readable, provides useful information to linters and IDEs, and allows us to catch some bugs
early on.

We specify the type of a variable like so:

.. code-block:: python

    x: int = 10
    name: str = "Alice"

and define a function's parameter and return type with:

.. code-block:: python

     def greet(name: str) -> str:
          return "Hello, " + name

Python's ``typing`` module provides additional options such as ``Any``, ``Callable``, and ``Optional`` to annotate more
complex data structures. If a function does not have a return statement, ``-> None:`` should still be specified.

All functions written as part of DTC must include type hints for both their input parameters and return types.
Additionally, the typeguard library is used to validate these types at runtime during test executions, raising an
error if a type mismatch is detected.

Docstrings
-----------
Docstrings are essential for documenting Python functions, classes, and modules. They provide a convenient way to
describe the purpose, inputs, and outputs of your code, making it easier for others (and ourselves!) to understand and
maintain. For consistency and clarity, we will use the NumPy style for docstrings.

The NumPy style includes:

1. A brief description of the function's purpose.
2. Descriptions of each input parameter, including their type and purpose.
3. A description of the return value, including its type and purpose.

For example:

.. code-block:: python

     def greet(name: str) -> str:
          """
          Greet the user with a personalised message.

          Parameters
          ----------
          name : str
               The name of the person to greet.

          Returns
          -------
          str
               A greeting message.
          """
          return f"Hello, {name}!"

A correctly configured VSCode can quickly generate a docstring template for you. To do this, right-click below the
function definition and select ``Generate Docstring`` in the dropdown. Make sure to use this feature after adding type
hints to the function signature so they are automatically included in the template for you!

In DTC, every Python file and every method must have a docstring.

Logging
--------
Logging is essential for tracking the behaviour and performance of your module, especially once it is deployed. It
allows us to record errors, warnings, or debugging details in a consistent and configurable way. Unlike ``print``
statements, logging provides more flexibility and can be easily directed to different outputs and filtered based on
severity levels.

The DTC Python template provides a core logging configuration for the entire package in ``src/dtc_python_template/monitoring.py::setup_logging()``.
This is run before any other package code (see ``src/dtc_python_template/__main__.py``). This ensures that all logs
follow a consistent format and are appropriately handled.

Here is an example of how to define and use a logger in your Python file:

.. code-block:: python

     import logging

     # Define the logger for the current module
     logger = logging.getLogger(__name__)

     def important_function() -> None:
          # Use the logger for different levels of logging
          logger.info("This is an info message")
          logger.debug("This is a debug message")

     def other_important_function() -> None:
          logger.warning("This is a warning message")

The above code creates a new logger object for the entire file, which is then used by the various functions within the
file. By passing ``__name__``, we ensure that the logger is correctly associated with the file's name, making it easier
to identify the source of the logs. Depending on the type of information we wish to convey, we use different log levels
such as INFO, DEBUG, and WARNING to categorize messages based on their severity and importance.

In DTC, we enforce the use of logging instead of ``print`` statements in the code.

Unit Testing & Code Coverage
-----------------------------
Unit testing is a critical part of the development process. It involves writing tests for individual units of code
(usually functions or methods) to ensure they behave as expected. It is essential that unit tests are written as you
develop your code to catch errors early and ensure that your code remains reliable as it evolves.

Code coverage measures the percentage of lines of code executed by your unit tests. We require a minimum of 80% code
coverage in DTC to ensure that a sufficient portion of your codebase is being tested. Code coverage is displayed
whenever you run the ``pytest`` command in a correctly configured poetry environment.

All unit tests should be placed in the ``tests/`` directory. The location and naming of the test files should mirror
that of the files they are testing. For example, if your file is ``src/dtc_python_template/example_subpackage/example_module_b.py``,
the corresponding test file should be ``tests/dtc_python_template/example_sub_package/test_example_module_b.py``.

Unit tests often require ``mocking`` of complex objects or external dependencies, replacing them under test with
controlled, simplified versions that simulate specific behaviors. Please see `unittest.mock library <https://docs.python.org/3/library/unittest.mock.html>`__
for more information.
