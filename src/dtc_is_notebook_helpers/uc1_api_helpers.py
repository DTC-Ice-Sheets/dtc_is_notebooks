"""
Use Case 1 - running the deployed modules through the DTC Query API.

The notebook can fit the GRF and FTS modules locally, which needs R and INLA,
or it can submit them as jobs to the deployed workflow. This module is the
second route. Each call returns the path of the output the workflow wrote,
which the next step takes as its input.

A token is required. Generate one at
https://query.dtc-ice-sheets.org/auth/get-token and set it as
``DTC_IS_AUTH_TOKEN`` in the environment.

The calls are coroutines, so a notebook awaits them directly::

    grf_path = await run_grf_job("UC1_WATSON_RUNOFF_INPUT_RACMO55")
    fts_path = await run_fts_job(grf_path)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import dtc_query_client
from dtc_query_client.helpers import wait_for_job
from dtc_query_client.models.job_status import JobStatus

logger = logging.getLogger(__name__)

#: A sink for progress lines, as in `uc1_chain_helpers`.
Progress = Callable[[str], None]

#: Where the deployed Query API lives.
QUERY_API_HOST = "https://query.dtc-ice-sheets.org"

#: Environment variable holding the caller's API token.
TOKEN_VARIABLE = "DTC_IS_AUTH_TOKEN"  # noqa: S105

#: How long to keep waiting for a job that has not reported back, in seconds.
MAX_JOB_MISSING_DELAY = 300

#: The key each module's output path sits under in a finished job.
GRF_OUTPUT_KEY = "grf-module"
FTS_OUTPUT_KEY = "main"


class JobFailedError(RuntimeError):
    """A submitted job finished in a state other than succeeded."""


def _configuration() -> dtc_query_client.configuration.Configuration:
    """
    Build an API configuration from the token in the environment.

    Returns
    -------
    dtc_query_client.configuration.Configuration
        Configured for the deployed Query API.

    Raises
    ------
    RuntimeError
        If no token is set, since every endpoint requires one.
    """
    token = os.environ.get(TOKEN_VARIABLE)
    if not token:
        raise RuntimeError(
            f"{TOKEN_VARIABLE} is not set, so the Query API cannot be reached. "
            f"Generate a token at {QUERY_API_HOST}/auth/get-token and set it in "
            "the environment.",
        )
    return dtc_query_client.configuration.Configuration(
        access_token=token, host=QUERY_API_HOST,
    )


async def _await_job(api_client: "object", job_id: str, output_key: str,
                     what: str) -> str:
    """
    Wait for one submitted job and return the path of what it wrote.

    Parameters
    ----------
    api_client : dtc_query_client.api_client.ApiClient
        The client the job was submitted through.
    job_id : str
        Identifier returned by the submission call.
    output_key : str
        Key the module's output path sits under.
    what : str
        Name used in log messages and errors.

    Returns
    -------
    str
        Path of the output the workflow wrote.

    Raises
    ------
    JobFailedError
        If the job finished in any state other than succeeded.
    """
    result = await wait_for_job(
        api_client, job_id, max_job_missing_delay=MAX_JOB_MISSING_DELAY,
    )
    if result.status != JobStatus.SUCCEEDED:
        raise JobFailedError(f"the {what} job finished as {result.status}, not SUCCEEDED")
    path = result.outputs[output_key]["output_path"]
    logger.info("%s job %s wrote %s", what, job_id, path)
    return path


async def run_grf_job(dataset_name: str, start_date: str = "2006-01-01",
                      end_date: str = "2026-01-01", grid_res_x: int = 100,
                      grid_res_y: int = 100, progress: Progress = print) -> str:
    """
    Fit the GRF module to a published dataset through the Query API.

    Parameters
    ----------
    dataset_name : str
        A Use Case 1 collection, e.g. ``UC1_WATSON_RUNOFF_INPUT_RACMO55``.
    start_date, end_date : str, optional
        Time window, as ISO dates. The default spans the whole record.
    grid_res_x, grid_res_y : int, optional
        Resolution of the regular grid the fitted field is returned on.
    progress : callable, optional
        Called with a line of progress. `print` by default.

    Returns
    -------
    str
        Path of the harmonised field the workflow wrote.
    """
    configuration = _configuration()
    async with dtc_query_client.api_client.ApiClient(configuration) as api_client:
        api = dtc_query_client.api.HydropowerApi(api_client)
        request = dtc_query_client.models.HydropowerGaussianRandomFieldRequest(
            start_date=start_date,
            end_date=end_date,
            dataset_name=dataset_name,
            grid_res_x=grid_res_x,
            grid_res_y=grid_res_y,
        )
        progress(f"GRF: submitting {dataset_name} ...")
        submitted = await api.submit_grf_job(request)
        progress(f"GRF: job {submitted.job_id}, waiting (a cached run returns in seconds)")
        path = await _await_job(api_client, submitted.job_id, GRF_OUTPUT_KEY, "GRF")
    progress("GRF: done")
    return path


async def run_fts_job(data_path: str, curve_time_points: int = 5, num_pca: int = 2,
                      num_factors: int = 3, forecast_horizon: int = 3,
                      progress: Progress = print) -> str:
    """
    Run the FTS module over a harmonised field through the Query API.

    Parameters
    ----------
    data_path : str
        Output of a GRF job, as :func:`run_grf_job` returns it.
    curve_time_points : int, optional
        Timesteps treated as one curve. Five, one per melt-season month.
    num_pca, num_factors : int, optional
        Components and factors retained by the two reduction stages.
    forecast_horizon : int, optional
        Number of curves to forecast ahead.
    progress : callable, optional
        Called with a line of progress. `print` by default.

    Returns
    -------
    str
        Path of the coefficients the workflow wrote.
    """
    configuration = _configuration()
    # A fresh client for the second call, as in the API's own test script.
    async with dtc_query_client.api_client.ApiClient(configuration) as api_client:
        api = dtc_query_client.api.HydropowerApi(api_client)
        request = dtc_query_client.models.HydropowerFunctionalTimeSeriesRequest(
            data_path=data_path,
            curve_time_points=curve_time_points,
            num_pca=num_pca,
            num_factors=num_factors,
            forecast_horizon=forecast_horizon,
        )
        progress("FTS: submitting ...")
        submitted = await api.submit_fts_job(request)
        progress(f"FTS: job {submitted.job_id}, waiting")
        path = await _await_job(api_client, submitted.job_id, FTS_OUTPUT_KEY, "FTS")
    progress("FTS: done")
    return path


async def grf_fields(dataset_name: str, start_date: str = "2006-01-01",
                     end_date: str = "2026-01-01", grid_res_x: int = 100,
                     grid_res_y: int = 100, progress: Progress = print) -> dict:
    """
    Run a GRF job and read its output back in the shape the notebook uses.

    The deployed module returns its field interpolated onto a regular grid, so
    the columns here are grid cells rather than mesh nodes. `points` gives the
    coordinate of each column, in the same order, so a catchment mask can be
    built for them exactly as one is for the mesh.

    Parameters
    ----------
    dataset_name : str
        A Use Case 1 collection, e.g. ``UC1_WATSON_RUNOFF_INPUT_RACMO55``.
    start_date, end_date : str, optional
        Time window, as ISO dates.
    grid_res_x, grid_res_y : int, optional
        Resolution of the grid the field is returned on.
    progress : callable, optional
        Called with a line of progress.

    Returns
    -------
    dict
        ``path`` the job's output location, ``fields`` a frame of timestamps by
        grid cell on the variable's own scale, ``uncertainty`` the matching
        posterior standard deviations or None, ``points`` an (n, 2) array of
        cell coordinates, and ``variable`` the name the module gave the field.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    path = await run_grf_job(dataset_name, start_date, end_date,
                             grid_res_x, grid_res_y, progress=progress)
    progress("GRF: reading the fitted field back")
    try:
        ds = xr.open_zarr(path, consolidated=True)
    except (KeyError, ValueError):
        ds = xr.open_zarr(path, consolidated=False)

    gridded = [str(name) for name, array in ds.data_vars.items()
               if set(array.dims) >= {"time", "x", "y"}]
    variable = next((name for name in gridded if not name.endswith("_uncertainty")), None)
    if variable is None:
        raise ValueError(
            f"no gridded field in the job output; it holds: {', '.join(gridded) or 'nothing'}",
        )

    def table(name: str) -> tuple[pd.DataFrame, np.ndarray]:
        """Flatten one gridded variable, and give the coordinates of its columns."""
        stacked = ds[name].stack(position=("x", "y")).transpose("time", "position")
        frame = pd.DataFrame(np.asarray(stacked.values, dtype=float),
                             index=pd.DatetimeIndex(ds["time"].values))
        coords = np.column_stack([
            np.asarray(stacked["position"]["x"].values, dtype=float),
            np.asarray(stacked["position"]["y"].values, dtype=float),
        ])
        return frame, coords

    fields, points = table(variable)
    uncertainty = None
    if f"{variable}_uncertainty" in ds.data_vars:
        uncertainty, _ = table(f"{variable}_uncertainty")

    # The runoff collections are stored as log10. The rest of the chain works
    # in the linear quantity, so undo it here; the posterior spread stays in
    # log10 units, which is the form `uc1_hydro_helpers.seasonal_sd` expects.
    if variable.startswith("log10_"):
        fields = 10.0 ** fields

    progress(f"GRF: {fields.shape[0]} timesteps over {fields.shape[1]} grid cells")
    return {"path": path, "fields": fields, "uncertainty": uncertainty,
            "points": points, "variable": variable}


async def fts_coefficients(data_path: str, curve_time_points: int = 5, num_pca: int = 2,
                           num_factors: int = 3, forecast_horizon: int = 3,
                           progress: Progress = print) -> dict:
    """
    Run an FTS job and read its coefficients back as one row per curve.

    The module writes its coefficients as a GeoZarr. The layout is read from
    the store rather than assumed: a single two-dimensional variable is taken
    as curves by factors, and several equal-length one-dimensional variables
    are taken as one factor each.

    Parameters
    ----------
    data_path : str
        Output of a GRF job.
    curve_time_points, num_pca, num_factors, forecast_horizon : int, optional
        Passed to the module. See :func:`run_fts_job`.
    progress : callable, optional
        Called with a line of progress.

    Returns
    -------
    dict
        ``path`` the job's output location, ``coefficients`` a frame of curves
        by factor, and ``layout`` a line describing how the store was read.

    Raises
    ------
    ValueError
        If the store holds nothing that can be read as coefficients. The
        message names what it does hold.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    path = await run_fts_job(data_path, curve_time_points, num_pca,
                             num_factors, forecast_horizon, progress=progress)
    progress("FTS: reading the coefficients back")
    try:
        ds = xr.open_zarr(path, consolidated=True)
    except (KeyError, ValueError):
        ds = xr.open_zarr(path, consolidated=False)

    held = {str(name): tuple(array.dims) for name, array in ds.data_vars.items()}
    two_d = [name for name, dims in held.items() if len(dims) == 2]
    one_d = [name for name, dims in held.items() if len(dims) == 1]

    if len(two_d) == 1:
        name = two_d[0]
        values = np.asarray(ds[name].values, dtype=float)
        # Curves are the longer axis; factors are few.
        if values.shape[0] < values.shape[1]:
            values = values.T
        frame = pd.DataFrame(
            values, columns=[f"factor_{i + 1}" for i in range(values.shape[1])])
        layout = f"read {name} with dims {held[name]} as curves by factor"
    elif len(one_d) > 1 and len({ds[name].size for name in one_d}) == 1:
        frame = pd.DataFrame({f"factor_{i + 1}": np.asarray(ds[name].values, dtype=float)
                              for i, name in enumerate(sorted(one_d))})
        layout = f"stacked {len(one_d)} one-dimensional variables: {', '.join(sorted(one_d))}"
    else:
        raise ValueError(
            "the FTS output holds nothing that reads as coefficients. It holds: "
            + "; ".join(f"{name} {dims}" for name, dims in held.items()),
        )

    progress(f"FTS: {frame.shape[1]} coefficients for each of {frame.shape[0]} curves")
    return {"path": path, "coefficients": frame, "layout": layout}
