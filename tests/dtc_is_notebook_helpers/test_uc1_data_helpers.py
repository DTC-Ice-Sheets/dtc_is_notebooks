"""Tests for reading the published Use Case 1 datasets.

The stores here are built by the tests and written to a temporary directory,
so nothing reaches the network and the expected values are known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dtc_is_notebook_helpers import uc1_data_helpers as uc1_data


@pytest.fixture
def store_root(tmp_path, monkeypatch):
    """Point the helpers at a temporary directory and return it."""
    monkeypatch.setenv(uc1_data.BASE_URL_VARIABLE, str(tmp_path))
    return tmp_path


def _write(root, collection: str, ds: xr.Dataset) -> None:
    ds.to_zarr(root / uc1_data.STORES[collection], mode="w", consolidated=True, zarr_format=2)


def _discharge_store() -> xr.Dataset:
    """Two days of gauge record, one filled value differing from the measured one."""
    times = pd.to_datetime(["2014-06-01", "2014-06-02"])
    return xr.Dataset(
        data_vars={
            uc1_data.DISCHARGE_VARIABLE: (("point", "time"), np.array([[10.0, 20.0]])),
            f"measured_{uc1_data.DISCHARGE_VARIABLE}": (("point", "time"), np.array([[10.0, np.nan]])),
        },
        coords={"point": [0], "time": times},
    )


def _gridded_store() -> xr.Dataset:
    """Two timesteps on a 2 x 3 grid, every cell a distinct number."""
    times = pd.to_datetime(["2014-06-15", "2014-07-15"])
    values = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
    return xr.Dataset(
        data_vars={"log10_supraglacial_lake_area_fraction": (("time", "y", "x"), values)},
        coords={"time": times, "y": [0.0, 1.0], "x": [0.0, 1.0, 2.0]},
    )


def test_an_unset_base_url_says_so(monkeypatch):
    monkeypatch.delenv(uc1_data.BASE_URL_VARIABLE, raising=False)
    with pytest.raises(RuntimeError, match="is not set"):
        uc1_data.base_url()


def test_a_trailing_separator_is_not_doubled(monkeypatch):
    monkeypatch.setenv(uc1_data.BASE_URL_VARIABLE, "s3://bucket/uc1/")
    assert uc1_data.dataset_url("UC1_WATSON_LAKE_INTENSITY") == (
        "s3://bucket/uc1/uc1_watson_lake_intensity.zarr")


def test_an_unknown_collection_lists_the_real_ones(store_root):  # noqa: ARG001
    with pytest.raises(KeyError, match="UC1_WATSON_RUNOFF_MAR"):
        uc1_data.dataset_url("UC1_WATSON_SOMETHING_ELSE")


def test_every_collection_in_the_catalogue_has_a_store():
    # The eight collections published for Use Case 1. If one is added, it has
    # to be added here too, which is the point of the assertion.
    assert len(uc1_data.STORES) == 8
    assert all(name.startswith("UC1_WATSON_") for name in uc1_data.STORES)
    assert all(store.endswith(".zarr") for store in uc1_data.STORES.values())


def test_a_store_is_opened_by_collection_name(store_root):
    _write(store_root, "UC1_WATSON_DISCHARGE_GAUGE", _discharge_store())
    ds = uc1_data.open_dataset("UC1_WATSON_DISCHARGE_GAUGE")
    assert uc1_data.DISCHARGE_VARIABLE in ds.data_vars


def test_a_store_without_consolidated_metadata_still_opens(store_root):
    ds = _discharge_store()
    ds.to_zarr(store_root / uc1_data.STORES["UC1_WATSON_DISCHARGE_GAUGE"],
               mode="w", consolidated=False, zarr_format=2)
    assert uc1_data.open_dataset("UC1_WATSON_DISCHARGE_GAUGE").sizes["time"] == 2


def test_the_gap_filled_series_is_the_default(store_root):
    _write(store_root, "UC1_WATSON_DISCHARGE_GAUGE", _discharge_store())
    ds = uc1_data.open_dataset("UC1_WATSON_DISCHARGE_GAUGE")

    filled = uc1_data.discharge(ds)
    measured = uc1_data.discharge(ds, measured_only=True)

    assert list(filled.to_numpy()) == [10.0, 20.0]
    assert np.isnan(measured.to_numpy()[1])
    assert filled.index[0] == pd.Timestamp("2014-06-01")
    assert filled.name == "discharge_m3s"


def test_a_gridded_store_comes_back_one_column_per_cell(store_root):
    _write(store_root, "UC1_WATSON_LAKE_INTENSITY", _gridded_store())
    ds = uc1_data.open_dataset("UC1_WATSON_LAKE_INTENSITY")

    table = uc1_data.fields(ds, "log10_supraglacial_lake_area_fraction")

    assert table.shape == (2, 6)          # two timesteps, 2 x 3 cells flattened
    assert list(table.index) == [pd.Timestamp("2014-06-15"), pd.Timestamp("2014-07-15")]
    # The first row is the first timestep's grid, read in order.
    np.testing.assert_allclose(table.iloc[0].to_numpy(), np.arange(6.0))


def test_a_point_store_keeps_one_column_per_node(store_root):
    times = pd.to_datetime(["2014-06-15", "2014-07-15"])
    ds = xr.Dataset(
        data_vars={"surface_runoff_amount": (("point", "time"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={"point": [0, 1], "time": times},
    )
    _write(store_root, "UC1_WATSON_RUNOFF_RACMO55", ds)

    table = uc1_data.fields(uc1_data.open_dataset("UC1_WATSON_RUNOFF_RACMO55"))

    assert table.shape == (2, 2)
    np.testing.assert_allclose(table.iloc[0].to_numpy(), [1.0, 3.0])


def test_asking_for_a_variable_that_is_not_there_lists_what_is(store_root):
    _write(store_root, "UC1_WATSON_LAKE_INTENSITY", _gridded_store())
    ds = uc1_data.open_dataset("UC1_WATSON_LAKE_INTENSITY")

    with pytest.raises(KeyError, match="log10_supraglacial_lake_area_fraction"):
        uc1_data.fields(ds, "surface_runoff_amount")
