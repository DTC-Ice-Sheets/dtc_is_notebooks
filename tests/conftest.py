"""Shared test fixtures."""

from pathlib import Path

import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def test_inputs_dir() -> Path:
    return Path(__file__).resolve().parent / "test_inputs"


@pytest.fixture
def example_mass_balance_dataset(test_inputs_dir: Path) -> xr.Dataset:
    path = test_inputs_dir / "jakobshavn_mass_balance.zarr"
    return xr.open_dataset(path, engine="zarr")


@pytest.fixture
def example_global_slr_dataset(test_inputs_dir: Path) -> xr.Dataset:
    path = test_inputs_dir / "expected_global_slr_for_jakobshavn_mb_sampled.zarr"
    return xr.open_dataset(path, engine="zarr")


@pytest.fixture
def example_annual_slr_dataset(test_inputs_dir: Path) -> xr.Dataset:
    path = test_inputs_dir / "expected_annual_slr_for_jakobshavn_mb_sampled.zarr"
    return xr.open_dataset(path, engine="zarr")


@pytest.fixture
def basin_csv(tmp_path: Path) -> Path:
    """
    Write a catchment outline: a 10 x 10 square with a 2 x 2 square hole in it.

    What is inside and what is not is then arithmetic rather than a judgement.
    """
    rows = [(0, 0.0, 0.0), (0, 10.0, 0.0), (0, 10.0, 10.0), (0, 0.0, 10.0), (0, 0.0, 0.0),
            (1, 4.0, 4.0), (1, 6.0, 4.0), (1, 6.0, 6.0), (1, 4.0, 6.0), (1, 4.0, 4.0)]
    path = tmp_path / "basin.csv"
    pd.DataFrame(rows, columns=["ring", "x", "y"]).to_csv(path, index=False)
    return path
