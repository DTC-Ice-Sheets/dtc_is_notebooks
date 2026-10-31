"""Tests for running the deployed Use Case 1 modules through the Query API.

Nothing here reaches the network. The client is replaced with stand-ins, and
the job output is a small Zarr store written into a temporary directory, so
the expected values are known.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dtc_is_notebook_helpers import uc1_api_helpers as api


@pytest.fixture
def token(monkeypatch):
    """Put a token in the environment, as a caller is expected to."""
    monkeypatch.setenv(api.TOKEN_VARIABLE, "not-a-real-token")


def _grf_store(path, variable: str = "log10_surface_runoff_amount") -> str:
    """Write a GRF-shaped output: two timesteps on a 2 x 3 grid, with spread."""
    times = pd.to_datetime(["2019-06-15", "2019-07-15"])
    values = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
    ds = xr.Dataset(
        data_vars={
            variable: (("time", "x", "y"), values),
            f"{variable}_uncertainty": (("time", "x", "y"), values * 0.1),
        },
        coords={"time": times, "x": [0.0, 1000.0], "y": [0.0, 1000.0, 2000.0]},
    )
    store = path / "grf.zarr"
    ds.to_zarr(store, mode="w", consolidated=True, zarr_format=2)
    return str(store)


def _gauge_store(path) -> str:
    """Write a gauge-shaped output: four days of the gap-filled and measured series."""
    times = pd.to_datetime(["2019-07-01", "2019-07-02", "2019-07-03", "2019-07-04"])
    gap_filled = np.array([10.0, 20.0, 30.0, 40.0])
    measured = np.array([10.0, np.nan, 30.0, 40.0])
    ds = xr.Dataset(
        data_vars={
            api.DISCHARGE_VARIABLE: (("time",), gap_filled),
            f"measured_{api.DISCHARGE_VARIABLE}": (("time",), measured),
        },
        coords={"time": times},
    )
    store = path / "gauge.zarr"
    ds.to_zarr(store, mode="w", consolidated=True, zarr_format=2)
    return str(store)


def _fts_store(path, names: tuple[str, ...]) -> str:
    """Write an FTS-shaped output: one (curve_id, coef_id) variable per name."""
    values = np.arange(4 * 3, dtype=float).reshape(4, 3)
    ds = xr.Dataset(
        data_vars={name: (("curve_id", "coef_id"), values + n)
                   for n, name in enumerate(names)},
        coords={"curve_id": np.arange(4), "coef_id": np.arange(3)},
    )
    store = path / "fts.zarr"
    ds.to_zarr(store, mode="w", consolidated=True, zarr_format=2)
    return str(store)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def test_a_missing_token_says_where_to_get_one(monkeypatch):
    monkeypatch.delenv(api.TOKEN_VARIABLE, raising=False)
    with pytest.raises(RuntimeError, match="auth/get-token"):
        api._configuration()


def test_a_token_in_the_environment_is_used(token):  # noqa: ARG001
    assert api._configuration().access_token == "not-a-real-token"  # noqa: S105


# --------------------------------------------------------------------------
# Waiting on a job
# --------------------------------------------------------------------------

def test_a_succeeded_job_gives_back_its_output_path(monkeypatch):
    finished = SimpleNamespace(
        status=api.JobStatus.SUCCEEDED,
        outputs={"grf-module": {"output_path": "somewhere.zarr"}})

    async def fake_wait(client, job_id, **kwargs):      # noqa: ANN001, ARG001, ANN003
        return finished

    monkeypatch.setattr(api, "wait_for_job", fake_wait)
    path = asyncio.run(api._await_job(None, "job-1", "grf-module", "GRF"))
    assert path == "somewhere.zarr"


def test_a_job_that_did_not_succeed_is_raised_not_returned(monkeypatch):
    finished = SimpleNamespace(status="FAILED", outputs={})

    async def fake_wait(client, job_id, **kwargs):      # noqa: ANN001, ARG001, ANN003
        return finished

    monkeypatch.setattr(api, "wait_for_job", fake_wait)
    with pytest.raises(api.JobFailedError, match="FAILED"):
        asyncio.run(api._await_job(None, "job-1", "grf-module", "GRF"))


# --------------------------------------------------------------------------
# Reading a finished GRF job back
# --------------------------------------------------------------------------

def test_a_finished_job_is_read_back_as_a_table_and_its_cell_positions(tmp_path, monkeypatch):
    store = _grf_store(tmp_path)

    async def fake_run(*args, **kwargs):                # noqa: ANN002, ANN003, ARG001
        return store

    monkeypatch.setattr(api, "run_grf_job", fake_run)
    result = asyncio.run(api.grf_fields("UC1_WATSON_RUNOFF_INPUT_MAR",
                                        progress=lambda _: None))

    assert result["variable"] == "log10_surface_runoff_amount"
    assert result["fields"].shape == (2, 6)             # two timesteps, 2 x 3 cells
    assert result["uncertainty"].shape == (2, 6)
    assert result["points"].shape == (6, 2)
    assert list(result["fields"].index) == [pd.Timestamp("2019-06-15"),
                                            pd.Timestamp("2019-07-15")]


def test_the_cell_positions_line_up_with_the_columns(tmp_path, monkeypatch):
    # Columns are stacked x-major, so the first three cells share the first x.
    store = _grf_store(tmp_path)

    async def fake_run(*args, **kwargs):                # noqa: ANN002, ANN003, ARG001
        return store

    monkeypatch.setattr(api, "run_grf_job", fake_run)
    points = asyncio.run(api.grf_fields("UC1_WATSON_RUNOFF_INPUT_MAR",
                                        progress=lambda _: None))["points"]

    np.testing.assert_allclose(points[:3, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(points[:3, 1], [0.0, 1000.0, 2000.0])
    np.testing.assert_allclose(points[3:, 0], [1000.0, 1000.0, 1000.0])


def test_a_log10_variable_is_returned_on_its_linear_scale(tmp_path, monkeypatch):
    store = _grf_store(tmp_path)

    async def fake_run(*args, **kwargs):                # noqa: ANN002, ANN003, ARG001
        return store

    monkeypatch.setattr(api, "run_grf_job", fake_run)
    result = asyncio.run(api.grf_fields("UC1_WATSON_RUNOFF_INPUT_MAR",
                                        progress=lambda _: None))

    # The store holds 0..11 as log10, so the first cell comes back as 10**0.
    np.testing.assert_allclose(result["fields"].to_numpy()[0, 0], 1.0)
    # The posterior spread stays in log10 units.
    np.testing.assert_allclose(result["uncertainty"].to_numpy()[0, 0], 0.0)


def test_a_variable_that_is_not_a_log_is_left_alone(tmp_path, monkeypatch):
    store = _grf_store(tmp_path, variable="surface_runoff_amount")

    async def fake_run(*args, **kwargs):                # noqa: ANN002, ANN003, ARG001
        return store

    monkeypatch.setattr(api, "run_grf_job", fake_run)
    result = asyncio.run(api.grf_fields("UC1_WATSON_RUNOFF_INPUT_MAR",
                                        progress=lambda _: None))

    np.testing.assert_allclose(result["fields"].to_numpy()[0, :3], [0.0, 1.0, 2.0])


def test_an_output_with_no_gridded_field_is_refused(tmp_path, monkeypatch):
    times = pd.to_datetime(["2019-06-15"])
    ds = xr.Dataset(data_vars={"something": (("time",), np.array([1.0]))},
                    coords={"time": times})
    store = tmp_path / "empty.zarr"
    ds.to_zarr(store, mode="w", consolidated=True, zarr_format=2)

    async def fake_run(*args, **kwargs):                # noqa: ANN002, ANN003, ARG001
        return str(store)

    monkeypatch.setattr(api, "run_grf_job", fake_run)
    with pytest.raises(ValueError, match="no gridded field"):
        asyncio.run(api.grf_fields("UC1_WATSON_RUNOFF_INPUT_MAR",
                                   progress=lambda _: None))


# --------------------------------------------------------------------------
# Fetching a published dataset
# --------------------------------------------------------------------------

def test_the_spatial_filter_is_built_in_a_form_the_endpoint_accepts():
    """
    The polygon field is a one-of wrapper, and handing it a plain dict leaves it unbound.

    That failure is silent until the request is serialised, so it is worth pinning here
    rather than discovering it against the deployed endpoint.
    """
    wrapped = api.dtc_query_client.models.Epsg4326PolygonJson.from_dict(api.WHOLE_WORLD)

    assert wrapped.to_dict() == api.WHOLE_WORLD


def test_the_gauge_record_comes_back_as_a_daily_series(tmp_path, monkeypatch):
    store = _gauge_store(tmp_path)
    asked = {}

    async def fake_download(dataset_id, *args, **kwargs):   # noqa: ANN001, ANN002, ANN003
        asked["dataset_id"] = dataset_id
        asked["variables"] = kwargs.get("variables")
        return store

    monkeypatch.setattr(api, "download_dataset", fake_download)
    series = asyncio.run(api.discharge_series(progress=lambda _: None))

    assert asked["dataset_id"] == "UC1_WATSON_DISCHARGE_GAUGE"
    assert asked["variables"] == [api.DISCHARGE_VARIABLE]
    assert len(series) == 4
    assert series.index[0] == pd.Timestamp("2019-07-01")
    np.testing.assert_allclose(series.to_numpy(), [10.0, 20.0, 30.0, 40.0])


def test_the_measured_only_record_is_a_different_variable(tmp_path, monkeypatch):
    store = _gauge_store(tmp_path)
    asked = {}

    async def fake_download(dataset_id, *args, **kwargs):   # noqa: ANN001, ANN002, ANN003, ARG001
        asked["variables"] = kwargs.get("variables")
        return store

    monkeypatch.setattr(api, "download_dataset", fake_download)
    series = asyncio.run(api.discharge_series(measured_only=True, progress=lambda _: None))

    assert asked["variables"] == [f"measured_{api.DISCHARGE_VARIABLE}"]
    assert np.isnan(series.to_numpy()[1])            # the gap is kept, not filled


# --------------------------------------------------------------------------
# Reading a finished FTS job back
# --------------------------------------------------------------------------

def test_the_coefficients_are_read_as_curves_by_factor(tmp_path, monkeypatch):
    store = _fts_store(tmp_path, ("log10_surface_runoff_amount",))

    async def fake_run(*args, **kwargs):                # noqa: ANN002, ANN003, ARG001
        return store

    monkeypatch.setattr(api, "run_fts_job", fake_run)
    result = asyncio.run(api.fts_coefficients("somewhere.zarr", progress=lambda _: None))

    assert result["coefficients"].shape == (4, 3)
    assert list(result["coefficients"].columns) == ["factor_1", "factor_2", "factor_3"]


def test_the_fields_coefficients_are_taken_over_the_uncertainty_ones(tmp_path, monkeypatch):
    """
    The module fits every spatial variable, and the GRF output carries the spread too.

    Both sets land in one store with nothing in the names to say which is which, so the
    reader has to choose rather than refuse.
    """
    store = _fts_store(tmp_path, ("log10_surface_runoff_amount",
                                  "log10_surface_runoff_amount_uncertainty"))

    async def fake_run(*args, **kwargs):                # noqa: ANN002, ANN003, ARG001
        return store

    monkeypatch.setattr(api, "run_fts_job", fake_run)
    result = asyncio.run(api.fts_coefficients("somewhere.zarr", progress=lambda _: None))

    assert "log10_surface_runoff_amount" in result["layout"]
    assert "uncertainty" not in result["layout"]
    # The field's values are the lower set, written first.
    np.testing.assert_allclose(result["coefficients"].to_numpy()[0], [0.0, 1.0, 2.0])


def test_an_output_with_nothing_readable_names_what_it_holds(tmp_path, monkeypatch):
    ds = xr.Dataset(data_vars={"only_one": (("coef_id",), np.array([1.0, 2.0]))},
                    coords={"coef_id": np.arange(2)})
    store = tmp_path / "odd.zarr"
    ds.to_zarr(store, mode="w", consolidated=True, zarr_format=2)

    async def fake_run(*args, **kwargs):                # noqa: ANN002, ANN003, ARG001
        return str(store)

    monkeypatch.setattr(api, "run_fts_job", fake_run)
    with pytest.raises(ValueError, match="only_one"):
        asyncio.run(api.fts_coefficients("somewhere.zarr", progress=lambda _: None))
