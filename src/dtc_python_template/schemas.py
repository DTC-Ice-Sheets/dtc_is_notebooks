"""Here we define the schemas for package entrypoints and outputs.

This helps us ensure the intended usage of the package is well defined and well understood.
"""

# schemas/entry_point_schemas.py
from pydantic import BaseModel, Field


# Example schema for the example entry point
class ExampleEntryPointSchema(BaseModel):
    """Example schema."""

    integer_a: int = Field(..., description="The first integer.")
    integer_b: int = Field(..., description="The second integer.")
