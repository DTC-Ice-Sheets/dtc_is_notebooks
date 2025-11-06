"""Shared test fixtures for use_case_2 tests."""
from pathlib import Path

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
