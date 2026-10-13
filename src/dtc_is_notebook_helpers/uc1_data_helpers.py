"""
Use Case 1 - reading the published datasets.

Everything the hydropower notebook reads comes through here, so there is one
place to change when the datasets move. The eight Use Case 1 collections are
GeoZarr stores; this opens one by its collection name and reshapes it into the
pandas objects the rest of the helpers expect.

The stores live wherever ``UC1_DATA_BASE_URL`` points. Set that environment
variable to the DTC-IS store, or to a local directory of unpacked ``.zarr``
stores when working offline. Until the Query API exposes the collections
directly, the base URL is the only thing that needs to change.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

#: Where the stores are read from. A directory of ``.zarr`` stores, local or remote.
BASE_URL_VARIABLE = "UC1_DATA_BASE_URL"

#: Collection name to store name. The collection names are the ones in
#: ``collections.yaml``, so the notebook can name a dataset the way the
#: catalogue does rather than by filename.
STORES: Mapping[str, str] = {
    "UC1_WATSON_RUNOFF_RACMO55": "uc1_watson_runoff_racmo55.zarr",
    "UC1_WATSON_RUNOFF_RACMO": "uc1_watson_runoff_racmo.zarr",
    "UC1_WATSON_RUNOFF_MAR": "uc1_watson_runoff_mar.zarr",
    "UC1_WATSON_RUNOFF_INPUT_RACMO55": "uc1_watson_runoff_input_racmo55.zarr",
    "UC1_WATSON_RUNOFF_INPUT_RACMO": "uc1_watson_runoff_input_racmo.zarr",
    "UC1_WATSON_RUNOFF_INPUT_MAR": "uc1_watson_runoff_input_mar.zarr",
    "UC1_WATSON_DISCHARGE_GAUGE": "uc1_watson_discharge_gauge.zarr",
    "UC1_WATSON_LAKE_INTENSITY": "uc1_watson_lake_intensity.zarr",
}

#: The gap-filled discharge series, which is the one the use case models. The
#: measurement-only series sits beside it under a ``measured_`` prefix.
DISCHARGE_VARIABLE = "water_volume_transport_in_river_channel"


def base_url() -> str:
    """
    Report where the datasets are being read from.

    Returns
    -------
    str
        The configured base location, without a trailing separator.

    Raises
    ------
    RuntimeError
        If the environment variable is not set, since there is no sensible
        default - the published location and a local copy look nothing alike.
    """
    configured = os.environ.get(BASE_URL_VARIABLE)
    if not configured:
        raise RuntimeError(
            f"{BASE_URL_VARIABLE} is not set, so there is nowhere to read the Use Case 1 "
            "datasets from. Point it at the DTC-IS store, or at a directory of unpacked "
            ".zarr stores to work offline.",
        )
    return configured.rstrip("/")


def dataset_url(collection: str) -> str:
    """
    Give the full location of one collection's store.

    Parameters
    ----------
    collection : str
        A collection name, as listed in ``STORES``.

    Returns
    -------
    str
        Base location and store name joined.

    Raises
    ------
    KeyError
        If the collection is not one of the eight.
    """
    if collection not in STORES:
        raise KeyError(
            f"{collection!r} is not a Use Case 1 collection. Expected one of: "
            f"{', '.join(sorted(STORES))}",
        )
    return f"{base_url()}/{STORES[collection]}"


def open_dataset(collection: str) -> xr.Dataset:
    """
    Open one published collection.

    Parameters
    ----------
    collection : str
        A collection name, as listed in ``STORES``.

    Returns
    -------
    xarray.Dataset
        The store as written, with its CF attributes intact.
    """
    url = dataset_url(collection)
    logger.info("reading %s from %s", collection, url)
    try:
        return xr.open_zarr(url, consolidated=True)
    except (KeyError, ValueError):
        # Stores written by older converter versions carry no consolidated
        # metadata. Reading them is slower but otherwise identical.
        logger.debug("%s has no consolidated metadata", collection)
        return xr.open_zarr(url, consolidated=False)


def discharge(ds: xr.Dataset, *, measured_only: bool = False) -> pd.Series:
    """
    Take the gauge record out of its store as a daily series.

    Parameters
    ----------
    ds : xarray.Dataset
        The discharge collection, as :func:`open_dataset` returns it.
    measured_only : bool, optional
        True for the pressure-transducer record without its gap filling. The
        default is the gap-filled series, which is what the use case models.

    Returns
    -------
    pandas.Series
        Discharge in m3 s-1, indexed by date, gaps kept as NaN.
    """
    name = f"measured_{DISCHARGE_VARIABLE}" if measured_only else DISCHARGE_VARIABLE
    values = np.asarray(ds[name].values, dtype=float).ravel()
    return pd.Series(values, index=pd.DatetimeIndex(ds["time"].values),
                     name="discharge_m3s").sort_index()


def fields(ds: xr.Dataset, variable: str = "surface_runoff_amount") -> pd.DataFrame:
    """
    Take a gridded or node-wise variable out of its store as a table.

    Parameters
    ----------
    ds : xarray.Dataset
        A runoff, input or lake collection.
    variable : str, optional
        Which variable to read.

    Returns
    -------
    pandas.DataFrame
        Rows are timestamps, columns are positions. Gridded stores are
        flattened, so a column is one cell either way and the rest of the
        chain does not have to know which shape it came from.

    Raises
    ------
    KeyError
        If the variable is not in the store.
    """
    if variable not in ds.data_vars:
        raise KeyError(
            f"{variable!r} is not in this dataset. It holds: "
            f"{', '.join(sorted(str(v) for v in ds.data_vars))}",
        )
    array = ds[variable]
    spatial = [str(dimension) for dimension in array.dims if dimension != "time"]
    flattened = array.stack(position=spatial) if len(spatial) > 1 else array
    return pd.DataFrame(np.asarray(flattened.transpose("time", ...).values, dtype=float),
                        index=pd.DatetimeIndex(ds["time"].values))
