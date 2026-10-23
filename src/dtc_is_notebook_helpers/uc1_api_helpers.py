"""
Use Case 1 - running the deployed modules through the DTC Query API.

Everything the notebook needs comes from here: the published datasets, through
the data download endpoint, and the fitted fields and curve coefficients, by
submitting the GRF and FTS modules as jobs. Each call returns the path of the
output the workflow wrote, which the next step takes as its input.

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

#: A sink for progress lines, so a caller can route them where it likes.
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
DOWNLOAD_OUTPUT_KEY = "main"

#: The gap-filled gauge series. The measurement-only record, which has holes in
#: it, sits beside it under a ``measured_`` prefix.
DISCHARGE_VARIABLE = "water_volume_transport_in_river_channel"

#: The spatial filter the download endpoint requires. Every Use Case 1
#: collection covers the Watson catchment and nothing else, so there is nothing
#: for a tighter polygon to select; this one is here to satisfy the endpoint.
WHOLE_WORLD = {
    "type": "Polygon",
    "coordinates": [[[-180.0, -90.0], [180.0, -90.0], [180.0, 90.0],
                     [-180.0, 90.0], [-180.0, -90.0]]],
}


class JobFailedError(RuntimeError):
    """A submitted job finished in a state other than succeeded."""


def open_store(path: str) -> "object":
    """
    Open a GeoZarr the workflow wrote.

    Parameters
    ----------
    path : str
        Location of the store, as a job's output gives it.

    Returns
    -------
    xarray.Dataset
        The store as written.
    """
    import xarray as xr

    try:
        return xr.open_zarr(path, consolidated=True)
    except (KeyError, ValueError):
        # Stores written by older converter versions carry no consolidated
        # metadata. Reading them is slower but otherwise identical.
        logger.debug("%s has no consolidated metadata", path)
        return xr.open_zarr(path, consolidated=False)


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


async def download_dataset(dataset_id: str, start_date: str = "2006-01-01",
                           end_date: str = "2026-01-01",
                           variables: "list[str] | None" = None,
                           polygon: "dict | None" = None,
                           progress: Progress = print) -> str:
    """
    Fetch one published dataset through the Query API's download endpoint.

    Parameters
    ----------
    dataset_id : str
        A Use Case 1 collection, e.g. ``UC1_WATSON_DISCHARGE_GAUGE``.
    start_date, end_date : str, optional
        Time window, as ISO dates. The default spans the whole record.
    variables : list of str, optional
        Variables to fetch. All of them by default.
    polygon : dict, optional
        GeoJSON polygon in EPSG:4326. :data:`WHOLE_WORLD` by default.
    progress : callable, optional
        Called with a line of progress.

    Returns
    -------
    str
        Path of the GeoZarr the endpoint wrote.
    """
    configuration = _configuration()
    async with dtc_query_client.api_client.ApiClient(configuration) as api_client:
        api = dtc_query_client.api.GenericApi(api_client)
        request = dtc_query_client.models.DataDownloadRequest(
            dataset_id=dataset_id,
            start_time=start_date,
            end_time=end_date,
            # from_dict rather than the plain dict: the field is a one-of
            # wrapper, and handing it a dict leaves it unbound.
            epsg4326_polygon_json=dtc_query_client.models.Epsg4326PolygonJson.from_dict(
                polygon or WHOLE_WORLD),
            variables=variables,
        )
        progress(f"download: submitting {dataset_id} ...")
        submitted = await api.download_data(request)
        progress(f"download: job {submitted.job_id}, waiting")
        path = await _await_job(api_client, submitted.job_id,
                                DOWNLOAD_OUTPUT_KEY, "download")
    progress("download: done")
    return path


async def discharge_series(start_date: str = "2006-01-01", end_date: str = "2026-01-01",
                           measured_only: bool = False,
                           progress: Progress = print) -> "object":
    """
    Fetch the Watson gauge record as a daily series in cubic metres per second.

    Parameters
    ----------
    start_date, end_date : str, optional
        Time window, as ISO dates.
    measured_only : bool, optional
        True for the pressure-transducer record without its gap filling. The
        default is the gap-filled series, which is what the use case models.
    progress : callable, optional
        Called with a line of progress.

    Returns
    -------
    pandas.Series
        Discharge in m3 s-1, indexed by date, with gaps kept as NaN.
    """
    import numpy as np
    import pandas as pd

    name = f"measured_{DISCHARGE_VARIABLE}" if measured_only else DISCHARGE_VARIABLE
    path = await download_dataset("UC1_WATSON_DISCHARGE_GAUGE", start_date, end_date,
                                  variables=[name], progress=progress)
    progress("download: reading the gauge record back")
    ds = open_store(path)
    values = np.asarray(ds[name].values, dtype=float).ravel()
    series = pd.Series(values, index=pd.DatetimeIndex(ds["time"].values),
                       name="discharge_m3s").sort_index()
    progress(f"gauge: {len(series)} days, {series.index.min():%Y-%m-%d} to "
             f"{series.index.max():%Y-%m-%d}")
    return series


def linear_scale(frame: "object", variable: str) -> "object":
    """
    Undo the log10 the published collections are stored on, where they are.

    Parameters
    ----------
    frame : pandas.DataFrame
        Values as the store holds them.
    variable : str
        The variable's name, which is what says whether it is a log.

    Returns
    -------
    pandas.DataFrame
        The values on their linear scale, or the input unchanged.
    """
    return 10.0 ** frame if variable.startswith("log10_") else frame


async def published_fields(dataset_id: str, start_date: str = "2006-01-01",
                           end_date: str = "2026-01-01",
                           variable: "str | None" = None,
                           progress: Progress = print) -> dict:
    """
    Fetch one published collection and read its field back as a table.

    Parameters
    ----------
    dataset_id : str
        A Use Case 1 collection, e.g. ``UC1_WATSON_LAKE_INTENSITY``.
    start_date, end_date : str, optional
        Time window, as ISO dates.
    variable : str, optional
        Which variable to read. By default the only one that varies in space
        as well as time, which is what every input collection holds.
    progress : callable, optional
        Called with a line of progress.

    Returns
    -------
    dict
        ``path`` where the download was written, ``fields`` a frame of
        timestamps by position on the variable's linear scale, and
        ``variable`` the name it was read from.

    Raises
    ------
    ValueError
        If the store holds no variable that varies in space and time.
    """
    import numpy as np
    import pandas as pd

    path = await download_dataset(dataset_id, start_date, end_date, progress=progress)
    progress("download: reading the field back")
    ds = open_store(path)
    if variable is None:
        spatial = [str(name) for name, array in ds.data_vars.items()
                   if "time" in array.dims and len(array.dims) > 1]
        if not spatial:
            raise ValueError(
                f"{dataset_id} holds no variable that varies in space and time. "
                f"It holds: {', '.join(sorted(str(v) for v in ds.data_vars)) or 'nothing'}",
            )
        variable = spatial[0]

    array = ds[variable]
    positions = [str(dimension) for dimension in array.dims if dimension != "time"]
    flattened = array.stack(position=positions) if len(positions) > 1 else array
    frame = pd.DataFrame(
        np.asarray(flattened.transpose("time", ...).values, dtype=float),
        index=pd.DatetimeIndex(ds["time"].values))
    progress(f"download: {frame.shape[0]} timesteps over {frame.shape[1]} cells")
    return {"path": path, "fields": linear_scale(frame, variable), "variable": variable}


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
        posterior standard deviations or None, ``uncertainty_mean`` the spread
        of the spatial mean itself, one value per timestep, or None,
        ``points`` an (n, 2) array of cell coordinates, and ``variable`` the
        name the module gave the field.
    """
    import numpy as np
    import pandas as pd

    path = await run_grf_job(dataset_name, start_date, end_date,
                             grid_res_x, grid_res_y, progress=progress)
    progress("GRF: reading the fitted field back")
    ds = open_store(path)

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

    # The spread of the spatial mean is taken from the joint posterior by the
    # module itself, and is not the same as collapsing the per-cell spreads,
    # which assumes they are independent. It carries no spatial dimension, so
    # it comes back as a series rather than through `table`.
    uncertainty_mean = None
    if f"{variable}_uncertainty_mean" in ds.data_vars:
        uncertainty_mean = pd.Series(
            np.asarray(ds[f"{variable}_uncertainty_mean"].values, dtype=float).ravel(),
            index=pd.DatetimeIndex(ds["time"].values))

    # The runoff collections are stored as log10. The rest of the chain works
    # in the linear quantity, so undo it here; the posterior spread stays in
    # log10 units, which is the form `uc1_hydro_helpers.seasonal_sd` expects.
    fields = linear_scale(fields, variable)

    progress(f"GRF: {fields.shape[0]} timesteps over {fields.shape[1]} grid cells")
    return {"path": path, "fields": fields, "uncertainty": uncertainty,
            "uncertainty_mean": uncertainty_mean, "points": points,
            "variable": variable}


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

    path = await run_fts_job(data_path, curve_time_points, num_pca,
                             num_factors, forecast_horizon, progress=progress)
    progress("FTS: reading the coefficients back")
    ds = open_store(path)

    held = {str(name): tuple(array.dims) for name, array in ds.data_vars.items()}
    two_d = [name for name, dims in held.items() if len(dims) == 2]
    one_d = [name for name, dims in held.items() if len(dims) == 1]

    # The module analyses every spatial variable it is given, and the GRF
    # output carries the posterior spread alongside the field itself, so the
    # store normally holds a set of coefficients for each. The ones wanted here
    # are the field's: curves fitted to the spread describe a different
    # quantity, and nothing in the name says so.
    if len(two_d) > 1:
        fields = [name for name in two_d if not name.endswith("_uncertainty")]
        if fields:
            two_d = fields

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
