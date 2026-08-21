"""
uc2_mb_validation_helpers.py.

Helper functions for validating DTC-IS Greenland Ice Sheet mass-balance estimates against independent
observational references.

This module provides:
- Aggregation of the DTC-IS mass balance products served by the DTC Query API into annual totals [Gt/yr].
- Application of the trained DTC-IS mass-balance model to each year's own features, which turns the
  single-epoch what-if prediction into an annual series that can itself be validated.
- Retrieval and harmonisation of the two independent reference records used here:
  IMBIE-3 (Otosaka et al. 2026, monthly to 2023) and the GRACE/GRACE-FO gravimetric mass balance (GMB)
  time series distributed through GravIS (COST-G Level-3 ice-mass products).
- Year-by-year intercomparison tables, agreement statistics and validation figures.

All mass-balance quantities are expressed as gigatonnes per year [Gt/yr], with negative values denoting
mass loss from the ice sheet.

Dependencies: numpy, pandas, requests, xarray, matplotlib, cartopy.
The annual model prediction additionally needs cloudpickle, xgboost, scipy, pyproj and netCDF4, which are
imported only when it is used.

Authors: DTC Ice Sheets consortium, DTU Space
"""

import hashlib
import os
import tempfile
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr
from matplotlib.ticker import MaxNLocator

from dtc_is_notebook_helpers.uc2_plotting_helpers import MASS_BALANCE_COL_NAME, MASS_BALANCE_ERROR_COL_NAME

# 1 mm of global mean sea-level equivalent corresponds to roughly 361 Gt of ice.
GT_PER_MM_SLE = 361.0
# Density used by the DTC-IS mass-balance model to convert metres of ice equivalent into mass.
ICE_DENSITY_KG_M3 = 917.0

DOWNLOAD_TIMEOUT = 180  # seconds
DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "dtc_is_mb_validation"

# IMBIE-3: "Mass balance of the Greenland and Antarctic Ice Sheets from the 1970s to 2023", Otosaka et al.
# (2026), UK Polar Data Centre / NERC-BAS. The Greenland file is monthly and runs 1971-07 to 2023-12: a
# commented header, then a date, the total dM/dt [Gt/yr] and its 1-sigma, the cumulative mass change and its
# 1-sigma, and the same pair of columns again for the surface and dynamic components separately.
IMBIE_DOI = "10.5285/128c5e33-5224-4197-82f0-19dcc95b80a0"
IMBIE_GREENLAND_URL = (
    "https://ramadda.data.bas.ac.uk/repository/entry/get?entryid=synth%3A128c5e33-5224-4197-82f0-19dcc95b80a0"
    "%3AL2ltYmllM19ncmVlbmxhbmRfR3RfcGFydGl0aW9uZWQuY3N2"
)
IMBIE_FILENAME = "imbie3_greenland_Gt_partitioned.csv"

# The previous release, "Antarctic and Greenland Ice Sheet mass balance 1992-2020 for IPCC AR6", Shepherd et
# al. (2021). Its Greenland file stops at 2020.92 and is laid out with a decimal year rather than a date, but
# `load_imbie_annual` reads either, so passing this URL compares the DTC-IS against the AR6 record instead.
IMBIE_2021_DOI = "10.5285/77b64c55-7166-4a06-9def-2e400398e452"
IMBIE_2021_GREENLAND_URL = (
    "https://ramadda.data.bas.ac.uk/repository/entry/get?entryid=synth%3A77b64c55-7166-4a06-9def-2e400398e452"
    "%3AL2ltYmllX2dyZWVubGFuZF8yMDIxX0d0LmNzdg%3D%3D"
)

# GRACE/GRACE-FO gravimetric mass balance (GMB): basin-averaged Greenland ice-mass changes from the COST-G
# GravIS Level-3 ice-mass product (Sasgen, AWI). Same product family as the Antarctic mass balance served by
# the DTC Query API. Space-delimited ASCII: decimal year, ice-sheet total [Gt], per-basin totals [Gt],
# 1-sigma of the total [Gt], per-basin 1-sigma [Gt].
GRAVIS_DOI = "10.5880/COST-G.GRAVIS_02_L3_ICE"
GRAVIS_GREENLAND_URL = (
    "https://isdc-data.gfz.de/grace/GravIS/COST-G/Level-3/ICE/GIS/GRAVIS-3_COSTG_0200_GIS_BAVE_AWI_0001.asc"
)
GRAVIS_FILENAME = "gravis_costg_greenland_ice_mass.asc"

# Static artefacts published alongside the DTC Query API: the baseline feature grid the mass-balance module
# runs on, and the trained gradient-boosted model itself. Both are the files the DTC-IS dashboard consumes,
# and both are readable without a token.
STATIC_FILES_BASE_URL = (
    "http://dtc-query-api-bucket.s3-website.gra.io.cloud.ovh.net/sea_level_response/static_dash_files/"
)
FEATURE_GRID_URL = STATIC_FILES_BASE_URL + "ISMB_module_output_greenland.zarr/"
XGB_MODEL_URL = STATIC_FILES_BASE_URL + "xgb_model.pkl"
XGB_MODEL_FILENAME = "dtc_is_xgb_model.pkl"

# Raw inputs the per-year features are rebuilt from. The DTC Query API serves neither the ESA CCI surface
# elevation change nor the C3S surface temperature, so both are read from a local DataPrep tree; see
# `predict_annual_mass_balance` for how the two environment variables below point at it.
DATAPREP_DIR_ENV_VAR = "DTC_IS_DATAPREP_DIR"
LST_TABLE_ENV_VAR = "DTC_IS_LST_TABLE"
CCI_SEC_GLOB = "CCI_SEC_*dhdt_5km_*.nc"
LST_MONTHLY_SUBDIR = "c3s_ist_monthly"
# CryoSat-2 switched Greenland acquisition from LRM to SARIn here, which is what the RAmode feature encodes.
SARIN_START = datetime(2010, 10, 1)
SECONDS_PER_YEAR = 365.25 * 86400.0
# Sampling gap beyond which an elevation-change window is not treated as a full year.
ANNUAL_WINDOW_TOLERANCE = 0.02

# Column names of the annual series carried through the intercomparison.
SIMONSEN_COL_NAME = "simonsen_gt_yr"
MLMODEL_COL_NAME = "mlmodel_gt_yr"
IMBIE_COL_NAME = "imbie_gt_yr"
GMB_COL_NAME = "gmb_gt_yr"

SERIES_LABELS = {
    SIMONSEN_COL_NAME: "DTC-IS product (Simonsen et al. 2021)",
    MLMODEL_COL_NAME: "DTC-IS ML model, annual features",
    IMBIE_COL_NAME: "IMBIE-3",
    GMB_COL_NAME: "GMB (GRACE/GRACE-FO)",
}

SERIES_COLORS = {
    SIMONSEN_COL_NAME: "#7b3294",
    MLMODEL_COL_NAME: "#d94801",
    IMBIE_COL_NAME: "#4292c6",
    GMB_COL_NAME: "#1b7837",
}

# Pairings are coloured by the reference they are judged against, so several of them share a colour. These
# keep them apart by the series under evaluation: the marker in the scatter panel, the hatch in the bar panel.
SERIES_MARKERS = {SIMONSEN_COL_NAME: "D", MLMODEL_COL_NAME: "^", GMB_COL_NAME: "s", IMBIE_COL_NAME: "o"}
SERIES_HATCHES = {SIMONSEN_COL_NAME: None, MLMODEL_COL_NAME: "//", GMB_COL_NAME: "xx", IMBIE_COL_NAME: ".."}

# Every pairing that can be given the same statistics: the two DTC-IS series against each independent
# reference, the model against the product it was trained on, and the two references against one another.
# Reporting them all makes the DTC-IS residuals readable against how far the references sit from each other.
#   (key, series column, reference column, label)
ALL_COMPARISONS: tuple[tuple[str, str, str, str], ...] = (
    ("mlmodel_vs_imbie", MLMODEL_COL_NAME, IMBIE_COL_NAME, "ML model vs IMBIE"),
    ("mlmodel_vs_gmb", MLMODEL_COL_NAME, GMB_COL_NAME, "ML model vs GMB"),
    ("mlmodel_vs_simonsen", MLMODEL_COL_NAME, SIMONSEN_COL_NAME, "ML model vs DTC-IS"),
    ("simonsen_vs_imbie", SIMONSEN_COL_NAME, IMBIE_COL_NAME, "DTC-IS vs IMBIE"),
    ("simonsen_vs_gmb", SIMONSEN_COL_NAME, GMB_COL_NAME, "DTC-IS vs GMB"),
    ("gmb_vs_imbie", GMB_COL_NAME, IMBIE_COL_NAME, "GMB vs IMBIE"),
)


def annual_series_from_mass_balance_dataset(mass_balance_ds: xr.Dataset, series_name: str = "simonsen") -> pd.DataFrame:
    """
    Aggregate a DTC-IS mass balance GeoZarr into annual ice-sheet-wide totals.

    The precomputed products served by ``GET /mass-balance/{id}`` store the mass balance of every grid point
    for every time step in kg per point and year, so the ice-sheet total is the spatial sum. The uncertainty
    field is summed linearly rather than in quadrature: altimetry errors are spatially correlated, which makes
    the linear sum the conservative ice-sheet-wide bound.

    Parameters
    ----------
    mass_balance_ds : xr.Dataset
        Mass balance dataset as opened from the URL returned by the DTC Query API. Must contain a time
        dimension along with the standard mass balance and uncertainty variables.
    series_name : str
        Prefix given to the returned value columns, by default "simonsen", which yields the columns
        ``simonsen_gt_yr`` and ``simonsen_unc_gt_yr``.

    Returns
    -------
    pd.DataFrame
        Data frame with the columns ``year``, ``<series_name>_gt_yr`` and ``<series_name>_unc_gt_yr``.

    Raises
    ------
    KeyError
        If the dataset carries no time dimension.
    """
    if "time" not in mass_balance_ds.dims:
        raise KeyError("The mass balance dataset has no time dimension, so no annual series can be derived.")

    space_dims = [str(dim) for dim in mass_balance_ds[MASS_BALANCE_COL_NAME].dims if str(dim) != "time"]
    totals = mass_balance_ds[MASS_BALANCE_COL_NAME].sum(dim=space_dims, skipna=True).values / 1e12
    uncertainties = np.abs(mass_balance_ds[MASS_BALANCE_ERROR_COL_NAME]).sum(dim=space_dims, skipna=True).values / 1e12

    frame = pd.DataFrame(
        {
            "year": pd.to_datetime(mass_balance_ds["time"].values).year,
            f"{series_name}_gt_yr": np.asarray(totals, dtype=float),
            f"{series_name}_unc_gt_yr": np.asarray(uncertainties, dtype=float),
        }
    )
    return frame.groupby("year", as_index=False).mean()


def _download_to_cache(url: str, filename: str, cache_dir: Path | None = None) -> Path:
    """
    Download a file once and reuse the cached copy on later calls.

    The cached name carries a digest of the URL, so two releases of the same record cache side by side. Keying
    on the filename alone would let a call that overrides the default URL, such as reading the 2021 IMBIE
    release through :func:`load_imbie_annual`, serve its content back to every later call under the name of
    the default record.

    Parameters
    ----------
    url : str
        Location of the file to retrieve.
    filename : str
        Name the file is cached under, before the URL digest is added to it.
    cache_dir : Path | None
        Directory holding the cached files, by default a ``dtc_is_mb_validation`` folder in the system
        temporary directory.

    Returns
    -------
    Path
        Path of the cached file.
    """
    cache_dir = DEFAULT_CACHE_DIR if cache_dir is None else Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = Path(filename)
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    path = cache_dir / f"{name.stem}-{digest}{name.suffix}"
    if not path.exists():
        response = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


class TemporalFeaturesUnavailableError(RuntimeError):
    """Raised when the per-year model features cannot be rebuilt from the inputs reachable on this machine."""


class AnnualFeatureModel:
    """
    Apply the trained DTC-IS mass-balance model to the observed state of the ice sheet, year by year.

    The what-if endpoint of the DTC Query API runs the same gradient-boosted model, but only on one epoch: it
    reports ``years: null``, so it cannot produce a time series. This class fills that gap by loading the two
    public artefacts the module is built from -- the baseline feature grid and the trained model -- and
    refreshing the time-varying features for each requested calendar year before predicting:

    ==========  =========================================================================================
    Feature     Rebuilt from
    ==========  =========================================================================================
    RAdh        Elevation change accumulated over the year, from the ESA CCI surface elevation change cube
    RAmode      Altimeter acquisition mode, LRM before the CryoSat-2 SARIn transition in October 2010
    RAdist      Distance-to-track proxy, held at unity as the training feature processor does
    LST         Annual mean C3S surface temperature, from the annual point table or the monthly files
    ==========  =========================================================================================

    Every other feature -- ice velocity, elevation, slope, firn thickness and basal melt -- stays at its
    baseline value, exactly as in the training-time feature construction. Features that cannot be refreshed
    for a given year keep their baseline value too, and the year records which ones were actually renewed so
    a partially refreshed year can be recognised in the output.

    Neither the ESA CCI elevation change nor the C3S surface temperature is served by the DTC Query API, so
    both are read from a local DataPrep tree. Without it the class cannot be constructed.
    """

    def __init__(
        self,
        dataprep_dir: str | Path | None = None,
        lst_table_path: str | Path | None = None,
        feature_grid_url: str = FEATURE_GRID_URL,
        model_url: str = XGB_MODEL_URL,
        cache_dir: Path | None = None,
    ) -> None:
        """
        Load the baseline feature grid and the trained model, and locate the raw per-year inputs.

        Parameters
        ----------
        dataprep_dir : str | Path | None
            Directory holding the raw ESA CCI surface elevation change file and, optionally, a
            ``c3s_ist_monthly`` subdirectory of monthly surface-temperature files. When None, the
            ``DTC_IS_DATAPREP_DIR`` environment variable is used.
        lst_table_path : str | Path | None
            Pickled annual surface-temperature point table (columns ``X``, ``Y`` and one column per year).
            This is the temperature the model was trained on, so it is preferred over averaging the monthly
            files. When None, the ``DTC_IS_LST_TABLE`` environment variable is used.
        feature_grid_url : str
            Baseline feature grid, by default the published :data:`FEATURE_GRID_URL`.
        model_url : str
            Trained model bundle, by default the published :data:`XGB_MODEL_URL`.
        cache_dir : Path | None
            Directory the downloaded model is cached in, by default :data:`DEFAULT_CACHE_DIR`.

        Raises
        ------
        TemporalFeaturesUnavailableError
            If neither the DataPrep directory nor the annual temperature table can be found, since the
            prediction would then repeat the baseline epoch for every year rather than describe it.
        """
        self.dataprep_dir = _resolve_optional_path(dataprep_dir, DATAPREP_DIR_ENV_VAR)
        self.lst_table_path = _resolve_optional_path(lst_table_path, LST_TABLE_ENV_VAR)
        if self.dataprep_dir is None and self.lst_table_path is None:
            raise TemporalFeaturesUnavailableError(
                "The per-year model features need the raw ESA CCI elevation change and C3S surface "
                "temperature, neither of which the DTC Query API serves. Point "
                f"${DATAPREP_DIR_ENV_VAR} at a DataPrep directory holding a file matching "
                f"'{CCI_SEC_GLOB}', and optionally ${LST_TABLE_ENV_VAR} at the annual temperature table."
            )

        self._elevation_change: dict | None = None
        self._lst_table: pd.DataFrame | None = None
        self._lst_files: tuple[list[Path], np.ndarray] | None = None

        self.baseline, self.cell_area_m2 = _load_baseline_feature_grid(feature_grid_url)
        self.booster, self.feature_names = _load_xgb_model(model_url, cache_dir)
        missing = [name for name in self.feature_names if name not in self.baseline.columns and name != "decimal_year"]
        if missing:
            raise TemporalFeaturesUnavailableError(f"The baseline feature grid is missing model features: {missing}")

    def _load_elevation_change(self) -> dict:
        """Cumulative elevation-change cube from the ESA CCI file, projected onto EPSG:3413."""
        if self._elevation_change is not None:
            return self._elevation_change
        self._elevation_change = {}
        matches = sorted(self.dataprep_dir.glob(CCI_SEC_GLOB)) if self.dataprep_dir is not None else []
        if not matches:
            return self._elevation_change

        import pyproj  # noqa: PLC0415
        from netCDF4 import Dataset  # noqa: PLC0415

        with Dataset(matches[-1], "r") as source:
            source.set_auto_mask(False)
            raw = {name: source.variables[name][:] for name in source.variables}

        latitude = raw.get("lat", raw.get("Lat"))
        longitude = raw.get("lon", raw.get("Lon"))
        epochs = np.array([datetime(1970, 1, 1) + timedelta(seconds=int(t)) for t in raw["time"]])

        # The file stores cumulative elevation change as (time, y, x); the feature processor works on
        # (y, x, time), and references every epoch to the first one.
        cumulative = raw["ZZ"]
        if cumulative.ndim == 3 and cumulative.shape[0] == len(epochs):
            cumulative = np.transpose(cumulative, (1, 2, 0))
        if latitude.ndim == 2 and latitude.shape != cumulative.shape[:2]:
            latitude, longitude = latitude.T, longitude.T
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
        easting, northing = transformer.transform(longitude, latitude)

        self._elevation_change = {
            "cumulative": cumulative - cumulative[:, :, 0][:, :, None],
            "epochs": epochs,
            "x": easting,
            "y": northing,
            "source": matches[-1].name,
        }
        return self._elevation_change

    def _to_model_grid(self, point_x: np.ndarray, point_y: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Interpolate scattered values onto the model grid, linearly with a nearest-neighbour hole fill."""
        from scipy.interpolate import griddata  # noqa: PLC0415

        points = np.column_stack((np.ravel(point_x), np.ravel(point_y)))
        values = np.ravel(values)
        target = (self.baseline["X"].to_numpy(), self.baseline["Y"].to_numpy())
        gridded = griddata(points, values, target, method="linear")
        holes = ~np.isfinite(gridded)
        if holes.any():  # as the training-time feature processor does
            gridded[holes] = griddata(points, values, (target[0][holes], target[1][holes]), method="nearest")
        return gridded

    def _annual_elevation_change(self, year: int) -> np.ndarray | None:
        """Elevation change accumulated over the calendar year, on the model grid."""
        cube = self._load_elevation_change()
        if not cube:
            return None
        start, end = datetime(year, 1, 1), datetime(year + 1, 1, 1)
        inside = (cube["epochs"] >= start) & (cube["epochs"] <= end)
        if inside.sum() < 2:
            return None

        window = cube["cumulative"][:, :, inside]
        change = window[:, :, -1] - window[:, :, 0]
        # The model was trained on the change accumulated over a full year, so a window the epochs only
        # partly cover is rescaled to its one-year equivalent rather than reported short.
        epochs = cube["epochs"][inside]
        span_years = (epochs[-1] - epochs[0]).total_seconds() / SECONDS_PER_YEAR
        if span_years > 0 and abs(span_years - 1.0) > ANNUAL_WINDOW_TOLERANCE:
            change = change / span_years

        finite = np.isfinite(change)
        return self._to_model_grid(cube["x"][finite], cube["y"][finite], change[finite])

    def _annual_temperature(self, year: int) -> np.ndarray | None:
        """Annual mean surface temperature on the model grid, from the point table or the monthly files."""
        from_table = self._temperature_from_table(year)
        return from_table if from_table is not None else self._temperature_from_monthly_files(year)

    def _temperature_from_table(self, year: int) -> np.ndarray | None:
        """Annual mean surface temperature read out of the pickled point table."""
        if self._lst_table is None:
            table = pd.DataFrame()
            if self.lst_table_path is not None and self.lst_table_path.is_file():
                candidate = pd.read_pickle(self.lst_table_path)  # noqa: S301 - a table published with the model
                if isinstance(candidate, pd.DataFrame) and {"X", "Y"} <= set(candidate.columns):
                    table = candidate
            self._lst_table = table
        if self._lst_table.empty:
            return None

        column = next((c for c in self._lst_table.columns if str(c) == str(year)), None)
        if column is None:
            return None
        points = self._lst_table[["X", "Y", column]].dropna()
        if points.empty:
            return None
        return self._to_model_grid(points["X"].to_numpy(), points["Y"].to_numpy(), points[column].to_numpy(dtype=float))

    def _monthly_temperature_index(self) -> tuple[list[Path], np.ndarray]:
        """Index the monthly surface-temperature files by the year and month in their filename."""
        if self._lst_files is not None:
            return self._lst_files
        self._lst_files = ([], np.array([]))
        directory = self.dataprep_dir / LST_MONTHLY_SUBDIR if self.dataprep_dir is not None else None
        if directory is None or not directory.is_dir():
            return self._lst_files

        import re  # noqa: PLC0415

        stamp = re.compile(r"(\d{6})")
        paths, dates = [], []
        for path in sorted(directory.glob("*.nc")):
            match = stamp.search(path.name)
            if match:
                paths.append(path)
                dates.append(datetime.strptime(match.group(1), "%Y%m"))
        self._lst_files = (paths, np.array(dates))
        return self._lst_files

    def _temperature_from_monthly_files(self, year: int) -> np.ndarray | None:
        """Annual mean surface temperature averaged from the monthly C3S files covering the year."""
        paths, dates = self._monthly_temperature_index()
        if len(dates) == 0:
            return None
        start, end = datetime(year, 1, 1), datetime(year + 1, 1, 1)
        selected = [path for path, date in zip(paths, dates, strict=True) if start <= date < end]
        if not selected:
            return None

        import pyproj  # noqa: PLC0415

        stacked = xr.concat(
            [xr.open_dataset(path).expand_dims(id=[index]).load() for index, path in enumerate(selected, 1)],
            dim="id",
        ).mean(dim="id", skipna=True)
        stacked = stacked.assign_coords(lon=(((stacked.lon + 180) % 360) - 180)).sortby("lon")
        greenland = stacked.sel(lon=slice(-75, -10), lat=slice(59, 84))

        longitude, latitude = np.meshgrid(greenland["lon"].values, greenland["lat"].values)
        easting, northing = pyproj.Proj(pyproj.CRS.from_epsg(3413))(longitude, latitude)
        gridded = self._to_model_grid(easting, northing, greenland["surface_temperature"].values)
        holes = ~np.isfinite(gridded)
        if holes.any():
            gridded[holes] = np.nanmean(gridded)
        return gridded

    def build_features(self, year: int) -> tuple[pd.DataFrame, list[str]]:
        """
        Assemble the model feature table for one calendar year.

        Parameters
        ----------
        year : int
            Calendar year to build the features for.

        Returns
        -------
        tuple[pd.DataFrame, list[str]]
            The feature table on the model grid, and the names of the features that were actually refreshed
            for this year. Features absent from that list are held at their baseline value.
        """
        features = self.baseline.copy()
        refreshed: list[str] = []

        if "RAdh" in self.feature_names:
            elevation_change = self._annual_elevation_change(year)
            if elevation_change is not None:
                features["RAdh"] = elevation_change
                refreshed.append("RAdh")

        if "RAdist" in self.feature_names and "RAdist" in features.columns:
            features["RAdist"] = np.ones(len(features))  # a proxy, as in the training feature processor
            refreshed.append("RAdist")

        if "RAmode" in self.feature_names and datetime(year + 1, 1, 1) < SARIN_START:
            features["RAmode"] = 1.0  # LRM everywhere before the CryoSat-2 SARIn transition
            refreshed.append("RAmode")

        if "LST" in self.feature_names:
            temperature = self._annual_temperature(year)
            if temperature is not None:
                features["LST"] = temperature
                refreshed.append("LST")

        if "decimal_year" in self.feature_names:
            features["decimal_year"] = year + 0.5
            refreshed.append("decimal_year")

        return features, refreshed

    def predict_year(self, year: int) -> tuple[float, list[str]]:
        """
        Predict the ice-sheet-wide mass balance of one calendar year.

        Parameters
        ----------
        year : int
            Calendar year to predict.

        Returns
        -------
        tuple[float, list[str]]
            The ice-sheet total in Gt/yr, and the features that were refreshed for this year.
        """
        features, refreshed = self.build_features(year)
        matrix = features[self.feature_names].to_numpy(dtype=float)

        if hasattr(self.booster, "predict") and not hasattr(self.booster, "feature_names"):
            predicted = self.booster.predict(matrix)  # an sklearn-style estimator
        else:
            import xgboost  # noqa: PLC0415

            predicted = self.booster.predict(xgboost.DMatrix(matrix, feature_names=self.feature_names))
        total_gt_yr = float(np.nansum(predicted * self.cell_area_m2 * ICE_DENSITY_KG_M3) * 1e-12)
        return total_gt_yr, refreshed


def _resolve_optional_path(value: str | Path | None, env_var: str) -> Path | None:
    """Take a path from the argument, else from an environment variable, and keep it only if it exists."""
    if value is None:
        value = os.environ.get(env_var) or None
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.exists() else None


def _load_baseline_feature_grid(feature_grid_url: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Flatten the published feature grid into the on-ice point records the model is evaluated on."""
    grid = xr.open_dataset(feature_grid_url, engine="zarr").load()
    stacked = grid.assign(X=grid.x, Y=grid.y).stack(sample=("y", "x"))
    frame = pd.DataFrame(
        {
            str(name): np.asarray(stacked[name].values, dtype=float)
            for name in [*list(grid.data_vars), "X", "Y"]
            if stacked[name].ndim == 1
        }
    )
    area_column = next((name for name in ("grid_cell_area", "area", "Area") if name in frame.columns), None)
    if area_column is None:
        raise TemporalFeaturesUnavailableError(f"The feature grid at {feature_grid_url} carries no grid cell area.")

    on_ice = frame[area_column].notna() & frame["X"].notna() & frame["Y"].notna()
    frame = frame.loc[on_ice].reset_index(drop=True)
    return frame.drop(columns=[area_column]), frame[area_column].to_numpy(dtype=float)


def _load_xgb_model(model_url: str, cache_dir: Path | None = None) -> tuple[object, list[str]]:
    """Load the trained model bundle and the ordered feature names it expects."""
    import cloudpickle  # noqa: PLC0415

    local = Path(model_url).expanduser()
    path = local if local.is_file() else _download_to_cache(model_url, XGB_MODEL_FILENAME, cache_dir)
    bundle = cloudpickle.loads(path.read_bytes())  # noqa: S301 - the model published with the DTC-IS module
    if isinstance(bundle, dict):
        return bundle["model"], list(bundle["feature_names"])
    return bundle[0], list(bundle[1])


def predict_annual_mass_balance(
    years: Sequence[int],
    dataprep_dir: str | Path | None = None,
    lst_table_path: str | Path | None = None,
    series_name: str = "mlmodel",
    **model_kwargs: object,
) -> pd.DataFrame:
    """
    Predict the annual Greenland mass balance by applying the DTC-IS model to each year's own features.

    This is the temporal counterpart of the single-epoch ``POST /mass-balance/what-if`` prediction: the same
    trained model is evaluated once per calendar year on the observed state of that year, which turns the
    prediction into a series that can be validated against the annual reference records.

    Parameters
    ----------
    years : Sequence[int]
        Calendar years to predict.
    dataprep_dir : str | Path | None
        Directory holding the raw ESA CCI elevation change file, or None to read ``DTC_IS_DATAPREP_DIR``.
    lst_table_path : str | Path | None
        Pickled annual surface-temperature point table, or None to read ``DTC_IS_LST_TABLE``.
    series_name : str
        Prefix given to the returned value column, by default "mlmodel", which yields ``mlmodel_gt_yr``.
    **model_kwargs : object
        Further keyword arguments forwarded to :class:`AnnualFeatureModel`, such as ``feature_grid_url``,
        ``model_url`` or ``cache_dir``.

    Returns
    -------
    pd.DataFrame
        Data frame with the columns ``year``, ``<series_name>_gt_yr`` and ``<series_name>_source``, the last
        naming the features that were refreshed for that year.

    Raises
    ------
    TemporalFeaturesUnavailableError
        If the per-year features cannot be rebuilt on this machine, or if no requested year could be
        refreshed at all, which would leave the series repeating the baseline epoch.
    """
    model = AnnualFeatureModel(dataprep_dir=dataprep_dir, lst_table_path=lst_table_path, **model_kwargs)

    rows = []
    for year in years:
        total_gt_yr, refreshed = model.predict_year(int(year))
        rows.append(
            {
                "year": int(year),
                f"{series_name}_gt_yr": total_gt_yr,
                f"{series_name}_source": "+".join(refreshed) if refreshed else "baseline-only",
            }
        )

    frame = pd.DataFrame(rows, columns=["year", f"{series_name}_gt_yr", f"{series_name}_source"])
    time_varying = {"RAdh", "LST"}
    if frame.empty or not any(time_varying & set(str(source).split("+")) for source in frame[f"{series_name}_source"]):
        raise TemporalFeaturesUnavailableError(
            "None of the requested years could have its elevation change or surface temperature refreshed, "
            "so the prediction would repeat the baseline epoch. Check that "
            f"${DATAPREP_DIR_ENV_VAR} holds a file matching '{CCI_SEC_GLOB}' covering these years."
        )
    return frame


def load_imbie_annual(
    path: Path | None = None, url: str = IMBIE_GREENLAND_URL, cache_dir: Path | None = None
) -> pd.DataFrame:
    """
    Load the IMBIE Greenland record and aggregate it to annual mass balance [Gt/yr].

    The published file is monthly, so each calendar year is the mean of its twelve dM/dt values. The quoted
    1-sigma is averaged in the same way rather than reduced by the square root of the sample count: the
    monthly uncertainties are dominated by systematic terms and are near constant within a year, so averaging
    preserves the published error budget instead of inventing precision. Years the release only partly covers
    are dropped so that no annual value is biased by a truncated set of months.

    Both published layouts are read: IMBIE-3 dates its rows and prefixes the file with a commented header,
    while the 2021 release carries a bare decimal year. Where a release partitions the mass balance into its
    surface and dynamic components, the total is taken and the components are ignored.

    Parameters
    ----------
    path : Path | None
        Local copy of the IMBIE CSV. When None, the file is downloaded from *url* and cached.
    url : str
        Source of the IMBIE Greenland CSV, by default the IMBIE-3 record at the UK Polar Data Centre.
        :data:`IMBIE_2021_GREENLAND_URL` reads the earlier IPCC AR6 release instead.
    cache_dir : Path | None
        Directory used to cache the download, by default the module cache directory.

    Returns
    -------
    pd.DataFrame
        Data frame with the columns ``year``, ``imbie_gt_yr``, ``imbie_unc_gt_yr`` and ``imbie_n``, the last
        holding the number of monthly values that contributed to each year.

    Raises
    ------
    KeyError
        If the file carries neither a decimal year nor a date column, or no mass-balance rate.
    """
    path = _download_to_cache(url, IMBIE_FILENAME, cache_dir) if path is None else Path(path)

    frame = pd.read_csv(path, comment="#")
    columns = {str(name).lower(): name for name in frame.columns}
    # A release that partitions the mass balance repeats these column names for the surface and dynamic
    # components; only the total belongs in the comparison.
    partitioned = ("surface", "dynamic")
    rate_col = _first_column(
        columns, wanted=("mass balance",), unwanted=("cumulative", "uncertainty", "anomaly", *partitioned)
    )
    if rate_col is None:
        raise KeyError(f"{path} carries no mass-balance rate column.")
    unc_col = _first_column(columns, wanted=("uncertainty",), unwanted=("cumulative", *partitioned))

    frame = frame.rename(columns={rate_col: "rate"})
    frame["year"] = _calendar_year(frame, columns, path)
    aggregation = {"imbie_gt_yr": ("rate", "mean"), "imbie_n": ("rate", "size")}
    if unc_col is not None:
        frame = frame.rename(columns={unc_col: "uncertainty"})
        aggregation["imbie_unc_gt_yr"] = ("uncertainty", "mean")

    annual = frame.groupby("year", as_index=False).agg(**aggregation)
    if "imbie_unc_gt_yr" not in annual.columns:
        annual["imbie_unc_gt_yr"] = np.nan
    annual = annual[annual["imbie_n"] >= 12]
    return annual[["year", "imbie_gt_yr", "imbie_unc_gt_yr", "imbie_n"]].reset_index(drop=True)


def _first_column(columns: dict[str, str], wanted: tuple[str, ...], unwanted: tuple[str, ...]) -> str | None:
    """Name of the first column whose lowercased header holds every wanted word and no unwanted one."""
    for key, name in columns.items():
        if all(word in key for word in wanted) and not any(word in key for word in unwanted):
            return name
    return None


def _calendar_year(frame: pd.DataFrame, columns: dict[str, str], path: Path) -> pd.Series:
    """Calendar year of every row, from a decimal year column or a date column, whichever the release uses."""
    decimal_year_col = _first_column(columns, wanted=("year",), unwanted=("uncertainty",))
    if decimal_year_col is not None and pd.api.types.is_numeric_dtype(frame[decimal_year_col]):
        return frame[decimal_year_col].astype(float).astype(int)

    date_col = _first_column(columns, wanted=("date",), unwanted=())
    if date_col is not None:
        return pd.to_datetime(frame[date_col]).dt.year

    raise KeyError(f"{path} carries neither a decimal year nor a date column.")


def load_gravimetric_cumulative(
    path: Path | None = None, url: str = GRAVIS_GREENLAND_URL, cache_dir: Path | None = None
) -> pd.DataFrame:
    """
    Load a GRACE/GRACE-FO cumulative ice-mass series.

    Two layouts are understood: the GravIS basin-averaged ASCII product, which holds the ice-sheet total in
    the second column followed by the individual drainage basins and then the matching 1-sigma values, and a
    plain three-column file of decimal year, cumulative mass anomaly and 1-sigma.

    Parameters
    ----------
    path : Path | None
        Local copy of the GMB file. When None, the file is downloaded from *url* and cached.
    url : str
        Source of the GMB record, by default the COST-G GravIS Greenland ice-mass product.
    cache_dir : Path | None
        Directory used to cache the download, by default the module cache directory.

    Returns
    -------
    pd.DataFrame
        Data frame with the columns ``decimal_year``, ``mass_gt`` and ``sigma_gt``, sorted in time.

    Raises
    ------
    ValueError
        If the file holds fewer than three columns and can therefore not be interpreted.
    """
    path = _download_to_cache(url, GRAVIS_FILENAME, cache_dir) if path is None else Path(path)

    raw = np.loadtxt(path, comments="#")
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError(f"Expected at least three columns in {path}, found shape {raw.shape}.")

    # In the GravIS layout the columns are: time, ice-sheet total, one column per basin, 1-sigma of the total,
    # one 1-sigma column per basin. The number of basins therefore follows from the column count.
    n_basins = (raw.shape[1] - 3) // 2
    sigma_index = 2 + n_basins
    frame = pd.DataFrame(
        {"decimal_year": raw[:, 0], "mass_gt": raw[:, 1], "sigma_gt": raw[:, sigma_index]},
    )
    return frame.sort_values("decimal_year").reset_index(drop=True)


def annual_rates_from_cumulative(
    cumulative: pd.DataFrame,
    series_name: str = "gmb",
    min_samples: int = 6,
    max_edge_gap: float = 0.15,
    max_internal_gap: float = 0.35,
) -> pd.DataFrame:
    """
    Derive annual mass balance [Gt/yr] from a cumulative gravimetric mass series.

    The rate for year Y is the mass difference between 1 January Y and 1 January Y+1, linearly interpolated
    onto those two epochs. Differencing over a full 12-month cycle cancels the Greenland seasonal signal and
    matches the definition of annual mass balance used by the other series. A least-squares trend through a
    single calendar year is not equivalent: the cumulative curve peaks in spring and falls steeply through the
    melt season, so a straight line through January to December is dominated by that asymmetry.

    The uncertainty is the quadrature sum of the 1-sigma values at the two endpoints.

    Years the record cannot support are dropped: fewer than *min_samples* epochs inside the year, no epoch
    within *max_edge_gap* of either year boundary, an internal sampling gap longer than *max_internal_gap*, or
    a year boundary the record does not bracket, which would have the rate extrapolated from a truncated year.
    For GRACE/GRACE-FO this removes 2017 and 2018 automatically, the years spanning the end of GRACE and the
    start of GRACE-FO, along with the year the record currently ends in.

    Parameters
    ----------
    cumulative : pd.DataFrame
        Cumulative series with the columns ``decimal_year``, ``mass_gt`` and ``sigma_gt``.
    series_name : str
        Prefix given to the returned value columns, by default "gmb".
    min_samples : int
        Minimum number of epochs inside a calendar year for that year to be reported, by default 6.
    max_edge_gap : float
        Largest distance in years the nearest epoch may sit from a year boundary, by default 0.15.
    max_internal_gap : float
        Longest sampling gap in years tolerated inside a calendar year, by default 0.35.

    Returns
    -------
    pd.DataFrame
        Data frame with the columns ``year``, ``<series_name>_gt_yr``, ``<series_name>_unc_gt_yr`` and
        ``<series_name>_n``, the last holding the number of epochs inside the year.
    """
    ordered = cumulative.sort_values("decimal_year")
    times = ordered["decimal_year"].to_numpy(dtype=float)
    mass = ordered["mass_gt"].to_numpy(dtype=float)
    sigma = ordered["sigma_gt"].to_numpy(dtype=float)

    rows = []
    for year in range(int(np.floor(times.min())), int(np.floor(times.max())) + 1):
        if times.min() > year or times.max() < year + 1:
            continue
        inside = (times >= year) & (times < year + 1)
        edge_gaps = [float(np.min(np.abs(times - boundary))) for boundary in (year, year + 1)]
        window = (times >= year - max_edge_gap) & (times <= year + 1 + max_edge_gap)
        internal_gap = float(np.max(np.diff(times[window]))) if window.sum() > 1 else np.inf
        if inside.sum() < min_samples or max(edge_gaps) > max_edge_gap or internal_gap > max_internal_gap:
            continue

        start, end = float(np.interp(year, times, mass)), float(np.interp(year + 1, times, mass))
        sigma_start = float(np.interp(year, times, sigma))
        sigma_end = float(np.interp(year + 1, times, sigma))
        rows.append(
            {
                "year": year,
                f"{series_name}_gt_yr": end - start,
                f"{series_name}_unc_gt_yr": float(np.hypot(sigma_start, sigma_end)),
                f"{series_name}_n": int(inside.sum()),
            }
        )

    return pd.DataFrame(rows, columns=["year", f"{series_name}_gt_yr", f"{series_name}_unc_gt_yr", f"{series_name}_n"])


def residual_column_name(series: str, reference: str) -> str:
    """
    Name of the column holding the difference between two annual series.

    Parameters
    ----------
    series : str
        Column name of the series under evaluation, e.g. ``simonsen_gt_yr``.
    reference : str
        Column name of the reference series, e.g. ``imbie_gt_yr``.

    Returns
    -------
    str
        Name of the residual column, e.g. ``residual_simonsen_imbie_gt_yr``.
    """
    return f"residual_{series.removesuffix('_gt_yr')}_{reference.removesuffix('_gt_yr')}_gt_yr"


def build_intercomparison_table(
    annual_series: list[pd.DataFrame], start_year: int | None = None, end_year: int | None = None
) -> pd.DataFrame:
    """
    Join the annual series onto a common set of calendar years and add the pairwise differences.

    Parameters
    ----------
    annual_series : list[pd.DataFrame]
        Annual series to merge. Each frame must carry a ``year`` column; every other column is taken over
        unchanged.
    start_year : int | None
        First year of the comparison window. When None, the earliest year present in any series is used.
    end_year : int | None
        Last year of the comparison window. When None, the latest year present in any series is used.

    Returns
    -------
    pd.DataFrame
        One row per year of the comparison window, holding every series, the residual of each available
        pairing and the sea-level equivalent of each series in mm.

    Raises
    ------
    ValueError
        If no series carries any year at all.
    """
    populated = [frame for frame in annual_series if frame is not None and not frame.empty]
    years_present = [int(year) for frame in populated for year in frame["year"]]
    if not years_present:
        raise ValueError("None of the supplied annual series contains any year.")

    first = min(years_present) if start_year is None else start_year
    last = max(years_present) if end_year is None else end_year
    table = pd.DataFrame({"year": list(range(first, last + 1))})

    for frame in populated:
        merged = frame.copy()
        merged["year"] = pd.to_numeric(merged["year"], errors="coerce").astype("int64")
        table = table.merge(merged, on="year", how="left")

    for _, series, reference, _ in ALL_COMPARISONS:
        if series in table.columns and reference in table.columns:
            table[residual_column_name(series, reference)] = table[series] - table[reference]

    for column in (SIMONSEN_COL_NAME, MLMODEL_COL_NAME, IMBIE_COL_NAME, GMB_COL_NAME):
        if column in table.columns:
            table[f"{column.removesuffix('_gt_yr')}_mm_sle"] = -table[column] / GT_PER_MM_SLE

    return table


def available_comparisons(table: pd.DataFrame) -> tuple[tuple[str, str, str, str], ...]:
    """
    Select the pairings the intercomparison table can actually support.

    A pairing is dropped when either series is missing from the table or when fewer than two years hold a
    value for both of them.

    Parameters
    ----------
    table : pd.DataFrame
        Intercomparison table as returned by :func:`build_intercomparison_table`.

    Returns
    -------
    tuple[tuple[str, str, str, str], ...]
        The usable entries of :data:`ALL_COMPARISONS`.
    """
    usable = []
    for entry in ALL_COMPARISONS:
        _, series, reference, _ = entry
        if series not in table.columns or reference not in table.columns:
            continue
        if len(table.dropna(subset=[series, reference])) >= 2:
            usable.append(entry)
    return tuple(usable)


def compute_metrics(table: pd.DataFrame, series: str, reference: str) -> dict:
    """
    Compute the year-by-year agreement statistics of one annual series against another.

    Parameters
    ----------
    table : pd.DataFrame
        Intercomparison table as returned by :func:`build_intercomparison_table`.
    series : str
        Column name of the series under evaluation, e.g. ``simonsen_gt_yr``.
    reference : str
        Column name of the reference series, e.g. ``imbie_gt_yr``.

    Returns
    -------
    dict
        Statistics of the pairing, or an empty dictionary when the two series share no year.
    """
    paired = table.dropna(subset=[series, reference])
    if paired.empty:
        return {}

    observed = paired[reference].to_numpy(dtype=float)
    modelled = paired[series].to_numpy(dtype=float)
    residual = modelled - observed
    sum_of_squares = float(((observed - observed.mean()) ** 2).sum())
    multi_year = len(paired) > 1

    return {
        "series": series.removesuffix("_gt_yr"),
        "reference": reference.removesuffix("_gt_yr"),
        "n_years": int(len(paired)),
        "first_year": int(paired["year"].min()),
        "last_year": int(paired["year"].max()),
        "series_mean_gt_yr": float(modelled.mean()),
        "ref_mean_gt_yr": float(observed.mean()),
        "bias_gt_yr": float(residual.mean()),
        "mae_gt_yr": float(np.abs(residual).mean()),
        "rmse_gt_yr": float(np.sqrt((residual**2).mean())),
        "r2": float(1.0 - (residual**2).sum() / sum_of_squares) if sum_of_squares > 0 else np.nan,
        "pearson_r": float(np.corrcoef(observed, modelled)[0, 1]) if multi_year else np.nan,
        "cum_series_gt": float(modelled.sum()),
        "cum_ref_gt": float(observed.sum()),
        "cum_bias_mm_sle": float(-(modelled.sum() - observed.sum()) / GT_PER_MM_SLE),
    }


def compute_all_metrics(table: pd.DataFrame) -> dict[str, dict]:
    """
    Compute the agreement statistics of every pairing the intercomparison table supports.

    Parameters
    ----------
    table : pd.DataFrame
        Intercomparison table as returned by :func:`build_intercomparison_table`.

    Returns
    -------
    dict[str, dict]
        Statistics of each pairing, keyed by the comparison key of :data:`ALL_COMPARISONS`.
    """
    metrics = {}
    for key, series, reference, _ in available_comparisons(table):
        block = compute_metrics(table, series, reference)
        if block:
            metrics[key] = block
    return metrics


def metrics_summary_table(metrics: dict[str, dict]) -> pd.DataFrame:
    """
    Lay the agreement statistics out as a compact table, one column per pairing.

    Parameters
    ----------
    metrics : dict[str, dict]
        Statistics as returned by :func:`compute_all_metrics`.

    Returns
    -------
    pd.DataFrame
        Table of formatted statistics indexed by metric name, with one column per pairing.
    """
    labels = {key: label for key, _, _, label in ALL_COMPARISONS}
    rows = {
        "Years compared": lambda block: f"{block['n_years']} ({block['first_year']}-{block['last_year']})",
        "Mean A / B [Gt/yr]": lambda block: f"{block['series_mean_gt_yr']:.1f} / {block['ref_mean_gt_yr']:.1f}",
        "Bias A - B [Gt/yr]": lambda block: f"{block['bias_gt_yr']:+.1f}",
        "MAE / RMSE [Gt/yr]": lambda block: f"{block['mae_gt_yr']:.1f} / {block['rmse_gt_yr']:.1f}",
        "R2 / Pearson r": lambda block: f"{block['r2']:.3f} / {block['pearson_r']:.3f}",
        "Cumulative A / B [Gt]": lambda block: f"{block['cum_series_gt']:.0f} / {block['cum_ref_gt']:.0f}",
        "Cumulative bias [mm SLE]": lambda block: f"{block['cum_bias_mm_sle']:+.2f}",
    }
    table = {
        labels.get(key, key): {name: render(block) for name, render in rows.items()} for key, block in metrics.items()
    }
    return pd.DataFrame(table)


def format_intercomparison_table(table: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce the intercomparison table to the columns worth reading in a notebook.

    Only the annual series, their uncertainties and the residuals of the available pairings are kept, the
    columns are given readable headers, and the values are rounded to the precision the data supports.

    Parameters
    ----------
    table : pd.DataFrame
        Intercomparison table as returned by :func:`build_intercomparison_table`.

    Returns
    -------
    pd.DataFrame
        Display view of the table, indexed by year.
    """
    value_columns = [
        column
        for column in (
            SIMONSEN_COL_NAME,
            "simonsen_unc_gt_yr",
            MLMODEL_COL_NAME,
            IMBIE_COL_NAME,
            "imbie_unc_gt_yr",
            GMB_COL_NAME,
            "gmb_unc_gt_yr",
        )
        if column in table.columns
    ]
    residual_columns = [
        residual_column_name(series, reference)
        for _, series, reference, _ in available_comparisons(table)
        if residual_column_name(series, reference) in table.columns
    ]
    shown = table[["year", *value_columns, *residual_columns]].set_index("year")

    headers = {
        SIMONSEN_COL_NAME: "DTC-IS",
        "simonsen_unc_gt_yr": "DTC-IS 1σ",
        MLMODEL_COL_NAME: "ML model",
        IMBIE_COL_NAME: "IMBIE",
        "imbie_unc_gt_yr": "IMBIE 1σ",
        GMB_COL_NAME: "GMB",
        "gmb_unc_gt_yr": "GMB 1σ",
    }
    headers.update(
        {
            residual_column_name(series, reference): label.replace(" vs ", " - ")
            for _, series, reference, label in ALL_COMPARISONS
        }
    )

    return shown.rename(columns=headers).round(1)


def plot_mass_balance_prediction_map(
    prediction_ds: xr.Dataset, control_ds: xr.Dataset | None = None, plot_description_str: str = ""
) -> None:
    """
    Map the gridded mass-balance prediction returned by the DTC-IS what-if module.

    A second panel showing the anomaly with respect to the control run is added whenever a control dataset is
    supplied and the two differ, which is the case for every what-if scenario other than the default one.

    Parameters
    ----------
    prediction_ds : xr.Dataset
        Predicted mass balance on the model grid, holding a ``prediction`` variable in metres of ice
        equivalent per year with ``x`` and ``y`` coordinates in EPSG:3413.
    control_ds : xr.Dataset | None
        Control prediction on the same grid, by default None.
    plot_description_str : str
        Description of the run, used as the figure suptitle.
    """
    projection = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70)
    prediction = prediction_ds["prediction"].values
    anomaly = None if control_ds is None else prediction - control_ds["prediction"].values
    if anomaly is not None and not np.any(np.abs(np.nan_to_num(anomaly)) > 0):
        anomaly = None

    x_values = prediction_ds["x"].values
    y_values = prediction_ds["y"].values
    extent = [float(x_values.min()), float(x_values.max()), float(y_values.min()), float(y_values.max())]

    def _draw(ax: plt.Axes, values: np.ndarray, cmap: str, limit: float, title: str, label: str) -> None:
        """Draw one prediction field on a polar stereographic axis."""
        ax.set_extent(extent, crs=projection)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#ededed", zorder=0)
        ax.coastlines(resolution="50m", color="black", linewidth=0.6, zorder=3)
        mesh = ax.pcolormesh(
            x_values, y_values, values, cmap=cmap, vmin=-limit, vmax=limit, shading="nearest", transform=projection
        )
        ax.set_title(title, fontsize=12)
        plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.75, pad=0.04, label=label)

    n_panels = 1 if anomaly is None else 2
    fig, axes = plt.subplots(
        1, n_panels, figsize=(6.5 * n_panels, 7.5), subplot_kw={"projection": projection}, squeeze=False
    )
    prediction_limit = float(np.nanpercentile(np.abs(prediction), 99))
    # With a single panel the suptitle already names the field, so the panel title would only repeat it.
    prediction_title = "" if anomaly is None and plot_description_str else "Predicted mass balance"
    _draw(
        axes[0, 0],
        prediction,
        "RdBu",
        prediction_limit,
        prediction_title,
        "Mass balance (m ice eq. yr$^{-1}$)",
    )
    if anomaly is not None:
        anomaly_limit = max(float(np.nanpercentile(np.abs(anomaly), 99)), 1e-6)
        _draw(
            axes[0, 1],
            anomaly,
            "PuOr",
            anomaly_limit,
            "Scenario minus control",
            "Difference (m ice eq. yr$^{-1}$)",
        )

    if plot_description_str:
        fig.suptitle(plot_description_str, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def _plot_annual_panel(ax: plt.Axes, table: pd.DataFrame, what_if_total_gt_yr: float | None) -> None:
    """Draw the annual mass-balance series of every reference, the DTC-IS product and the model prediction."""
    if IMBIE_COL_NAME in table.columns:
        ax.bar(
            table["year"],
            table[IMBIE_COL_NAME],
            width=0.75,
            color="#9ecae1",
            edgecolor=SERIES_COLORS[IMBIE_COL_NAME],
            label=SERIES_LABELS[IMBIE_COL_NAME],
            yerr=table.get("imbie_unc_gt_yr"),
            error_kw={"ecolor": SERIES_COLORS[IMBIE_COL_NAME], "elinewidth": 1.0, "capsize": 2, "alpha": 0.8},
            zorder=2,
        )
    if SIMONSEN_COL_NAME in table.columns:
        product = table.dropna(subset=[SIMONSEN_COL_NAME]).sort_values("year")
        if "simonsen_unc_gt_yr" in product.columns:
            ax.fill_between(
                product["year"],
                product[SIMONSEN_COL_NAME] - product["simonsen_unc_gt_yr"],
                product[SIMONSEN_COL_NAME] + product["simonsen_unc_gt_yr"],
                color=SERIES_COLORS[SIMONSEN_COL_NAME],
                alpha=0.15,
                zorder=3,
            )
        ax.plot(
            product["year"],
            product[SIMONSEN_COL_NAME],
            color=SERIES_COLORS[SIMONSEN_COL_NAME],
            lw=1.8,
            marker="D",
            ms=4,
            zorder=5,
            label=SERIES_LABELS[SIMONSEN_COL_NAME],
        )
    if MLMODEL_COL_NAME in table.columns:
        modelled = table.dropna(subset=[MLMODEL_COL_NAME]).sort_values("year")
        ax.plot(
            modelled["year"],
            modelled[MLMODEL_COL_NAME],
            color=SERIES_COLORS[MLMODEL_COL_NAME],
            lw=1.8,
            marker="^",
            ms=5,
            zorder=5,
            label=SERIES_LABELS[MLMODEL_COL_NAME],
        )
    if GMB_COL_NAME in table.columns:
        gmb = table.dropna(subset=[GMB_COL_NAME])
        ax.errorbar(
            gmb["year"],
            gmb[GMB_COL_NAME],
            yerr=gmb.get("gmb_unc_gt_yr"),
            fmt="s",
            ms=5,
            lw=0,
            elinewidth=1.2,
            capsize=3,
            color=SERIES_COLORS[GMB_COL_NAME],
            ecolor=SERIES_COLORS[GMB_COL_NAME],
            zorder=6,
            label=SERIES_LABELS[GMB_COL_NAME],
        )
    if what_if_total_gt_yr is not None:
        ax.axhline(
            what_if_total_gt_yr,
            color="#d73027",
            ls="--",
            lw=1.4,
            label=f"DTC-IS prediction, present-day ({what_if_total_gt_yr:.0f} Gt/yr)",
        )
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mass balance (Gt yr$^{-1}$)")
    ax.set_title("(a) Annual mass balance", fontsize=11)
    ax.legend(fontsize=7.5, framealpha=0.9, ncol=2, loc="lower left")
    ax.grid(axis="y", alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _cumulative_curve(
    table: pd.DataFrame,
    column: str,
    anchor: float | None,
    tie_curve: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """
    Accumulate an annual series onto year boundaries and tie it to the common epoch.

    A series covering the anchor is tied to zero there, which is what makes the spread between the curves the
    disagreement accumulated during the common period. A series that only begins later cannot be tied that
    way, so it is offset to meet ``tie_curve`` at its own first year instead; without such a curve it is left
    starting from zero. The epoch it ended up tied at is returned alongside the curve.
    """
    series = table.dropna(subset=[column]).sort_values("year")
    if len(series) < 2:
        return None
    years = np.append(series["year"].to_numpy(dtype=float), float(series["year"].max()) + 1)
    cumulated = np.concatenate([[0.0], np.cumsum(series[column].to_numpy(dtype=float))])

    if anchor is not None and years[0] <= anchor <= years[-1]:
        return years, cumulated - np.interp(anchor, years, cumulated), anchor
    if tie_curve is not None:
        tie_years, tie_values = tie_curve
        if tie_years[0] <= years[0] <= tie_years[-1]:
            return years, cumulated + np.interp(years[0], tie_years, tie_values) - cumulated[0], float(years[0])
    return years, cumulated, float(years[0])


def _plot_cumulative_panel(ax: plt.Axes, table: pd.DataFrame, gmb_cumulative: pd.DataFrame | None) -> None:
    """Draw the cumulative mass change of every series, tied together at a common epoch."""
    anchor, tie_curve = None, None
    if gmb_cumulative is not None and not gmb_cumulative.empty:
        gmb_cumulative = gmb_cumulative.sort_values("decimal_year")
        anchor = float(gmb_cumulative["decimal_year"].iloc[0])
        tied_mass = gmb_cumulative["mass_gt"] - np.interp(
            anchor, gmb_cumulative["decimal_year"], gmb_cumulative["mass_gt"]
        )
        tie_curve = (gmb_cumulative["decimal_year"].to_numpy(dtype=float), tied_mass.to_numpy(dtype=float))

    for column in (IMBIE_COL_NAME, SIMONSEN_COL_NAME, MLMODEL_COL_NAME):
        if column not in table.columns:
            continue
        curve = _cumulative_curve(table, column, anchor, tie_curve)
        if curve is None:
            continue
        years, cumulated, tied_at = curve
        label = SERIES_LABELS[column]
        # A series that starts after the common epoch is tied elsewhere, which the legend has to say so the
        # offset between the curves is not read as a disagreement.
        if anchor is not None and abs(tied_at - anchor) > 0.5:
            label = f"{label}, tied at {tied_at:.0f}"
        ax.plot(years, cumulated, color=SERIES_COLORS[column], lw=2, label=label)

    if gmb_cumulative is not None and not gmb_cumulative.empty:
        mass = tied_mass
        # Break the line across sampling gaps longer than about four months, so that the GRACE to GRACE-FO
        # transition is not drawn as if it were observed.
        mass = mass.mask(gmb_cumulative["decimal_year"].diff().shift(-1) > 0.35)
        ax.plot(
            gmb_cumulative["decimal_year"],
            mass,
            color=SERIES_COLORS[GMB_COL_NAME],
            lw=1.6,
            label=SERIES_LABELS[GMB_COL_NAME],
        )
        ax.plot([anchor], [0], marker="o", ms=6, color="#4d4d4d", zorder=6)

    sea_level_axis = ax.secondary_yaxis(
        "right", functions=(lambda gt: -gt / GT_PER_MM_SLE, lambda mm: -mm * GT_PER_MM_SLE)
    )
    sea_level_axis.set_ylabel("Sea-level contribution (mm SLE)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative mass change (Gt)")
    title = "(b) Cumulative mass change" if anchor is None else f"(b) Cumulative mass change, tied at {anchor:.2f}"
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)


def _plot_scatter_panel(ax: plt.Axes, table: pd.DataFrame, metrics: dict[str, dict]) -> None:
    """Draw every pairing against the 1:1 line, with its regression line and statistics."""
    columns = [column for column in SERIES_COLORS if column in table.columns]
    finite = np.concatenate([table[column].dropna().to_numpy(dtype=float) for column in columns] + [np.array([0.0])])
    low, high = float(finite.min()) - 40, float(finite.max()) + 40
    ax.plot([low, high], [low, high], color="black", ls="--", lw=1, label="1:1")

    notes = []
    for key, series, reference, label in available_comparisons(table):
        paired = table.dropna(subset=[series, reference])
        color = SERIES_COLORS[reference]
        ax.scatter(
            paired[reference],
            paired[series],
            s=38,
            marker=SERIES_MARKERS[series],
            facecolors="none",
            edgecolors=color,
            linewidths=1.4,
            zorder=3,
            label=label,
        )
        slope, intercept = np.polyfit(paired[reference], paired[series], 1)
        ax.plot([low, high], [slope * low + intercept, slope * high + intercept], color=color, lw=1.3, alpha=0.85)
        block = metrics.get(key)
        if block:
            notes.append(
                f"{label}:  R² {block['r2']:5.2f}   RMSE {block['rmse_gt_yr']:3.0f}   "
                f"bias {block['bias_gt_yr']:+4.0f}   n {block['n_years']}"
            )
    if notes:
        ax.text(
            0.03,
            0.97,
            "\n".join(notes),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            family="monospace",
            bbox={"boxstyle": "round", "fc": "wheat", "alpha": 0.8},
        )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("B  (Gt yr$^{-1}$)")
    ax.set_ylabel("A  (Gt yr$^{-1}$)")
    ax.set_title("(c) Pairwise agreement, A vs B", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)


def _plot_residual_panel(ax: plt.Axes, table: pd.DataFrame, metrics: dict[str, dict]) -> None:
    """Draw the annual differences of every pairing, together with their mean."""
    comparisons = available_comparisons(table)
    bar_width = 0.8 / max(len(comparisons), 1)
    for offset, (key, series, reference, label) in enumerate(comparisons):
        color = SERIES_COLORS[reference]
        residual = table[residual_column_name(series, reference)].astype(float)
        position = table["year"] + (offset - (len(comparisons) - 1) / 2) * bar_width
        # The bars are coloured by the reference they are judged against, which several pairings share, so
        # the series under evaluation is what the hatch names.
        ax.bar(
            position,
            residual,
            width=bar_width,
            color=color,
            label=label,
            zorder=2,
            hatch=SERIES_HATCHES.get(series),
            edgecolor="white",
            linewidth=0.3,
        )
        block = metrics.get(key)
        if block:
            ax.axhline(block["bias_gt_yr"], color=color, ls="--", lw=1.1, alpha=0.9)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Year")
    ax.set_ylabel("A - B (Gt yr$^{-1}$)")
    ax.set_title("(d) Annual differences, dashed = mean", fontsize=11)
    ax.legend(fontsize=7.5, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_mb_validation_four_panels(
    table: pd.DataFrame,
    metrics: dict[str, dict],
    gmb_cumulative: pd.DataFrame | None = None,
    what_if_total_gt_yr: float | None = None,
    plot_description_str: str = "Greenland Ice Sheet annual mass balance",
) -> None:
    """
    Plot the temporal validation of the DTC-IS mass balance in the four-panel setup.

    The panels show (a) the annual series of every record with their uncertainties, (b) the cumulative mass
    change tied to a common epoch and its sea-level equivalent, (c) each pairing against the 1:1 line, and
    (d) the annual differences of each pairing around their mean.

    Parameters
    ----------
    table : pd.DataFrame
        Intercomparison table as returned by :func:`build_intercomparison_table`.
    metrics : dict[str, dict]
        Agreement statistics as returned by :func:`compute_all_metrics`.
    gmb_cumulative : pd.DataFrame | None
        Cumulative gravimetric series as returned by :func:`load_gravimetric_cumulative`, by default None.
        When given, the cumulative panel is tied to its first epoch and the raw curve is drawn alongside the
        accumulated annual series.
    what_if_total_gt_yr : float | None
        Ice-sheet total of the single-epoch DTC-IS mass-balance prediction [Gt/yr], drawn as a horizontal
        line in the annual panel, by default None. The annual model prediction, when the table carries one,
        is drawn as a series in its own right rather than through this argument.
    plot_description_str : str
        Description of the data being plotted. Used as the figure suptitle.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(plot_description_str, fontsize=13, fontweight="bold")

    _plot_annual_panel(axes[0, 0], table, what_if_total_gt_yr)
    _plot_cumulative_panel(axes[0, 1], table, gmb_cumulative)
    _plot_scatter_panel(axes[1, 0], table, metrics)
    _plot_residual_panel(axes[1, 1], table, metrics)

    fig.text(
        0.99,
        0.01,
        f"©DTC Ice Sheets | IMBIE-3: doi:{IMBIE_DOI} | GravIS COST-G: doi:{GRAVIS_DOI}",
        ha="right",
        va="bottom",
        fontsize=8,
        color="grey",
    )
    plt.tight_layout()
    plt.show()
