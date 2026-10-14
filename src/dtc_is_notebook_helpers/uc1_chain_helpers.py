"""
Use Case 1 - the DTC modules, run from the notebook rather than around it.

This is the end-to-end path: extracted snapshots in, GRF fitted on every one of
them, FTS run on the resulting matrix, coefficients out. Both modules are the
deployed ones - `use_case_1.grf_module` and David Parkes' `model.coeff` through
rpy2 - not reimplementations.

Two things make that practical in a notebook.

**Caching.** A GRF fit is one to three seconds a snapshot, so a full twenty-year
run is a few minutes and a re-run would be a few minutes again. Every matrix is
written to disk the first time and reloaded after, keyed on the inputs, so the
notebook is slow once and instant thereafter. `force=True` recomputes.

**Clipping.** INLA refuses a fit outright when any observation falls outside the
mesh - "rows in control.predictor contain only zeros". The extraction clips to a
bounding box, the mesh follows the catchment outline, so some points always fall
outside. They are removed per snapshot before fitting.

If the modules are not importable - no R, no INLA, the package not installed -
every entry point says so plainly and, where a cache exists, offers to use it
instead. That keeps the notebook openable on a laptop while still being the real
chain on a machine that has the environment.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Where a fitted field or a progress line goes. `print` in a notebook,
# a logger or a no-op anywhere else.
Progress = Callable[[str], None]

MESH_NAME = "watsonmesh"
PRIOR_SIGMA_1 = 1.0
PRIOR_SIGMA_2 = 0.01
MONTHS = ["May", "Jun", "Jul", "Aug", "Sep"]
_MONTH_NUMBER = {m: i for i, m in enumerate(MONTHS, start=5)}


class ModulesUnavailableError(RuntimeError):
    """The deployed modules cannot be imported in this environment."""


# --------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------

def check_environment() -> dict:
    """Report what is importable here. Called before anything heavy."""
    status = {}
    try:
        from use_case_1.grf_module.grf_module import GrfModule  # noqa: F401
        from use_case_1.grf_module.spde_mesh import SpdeMesh  # noqa: F401
        status["grf"] = True
    except Exception as error:                       # noqa: BLE001
        status["grf"] = False
        status["grf_error"] = f"{type(error).__name__}: {error}"
    try:
        import rpy2.robjects as ro  # noqa: F401
        status["fts"] = True
    except Exception as error:                       # noqa: BLE001
        status["fts"] = False
        status["fts_error"] = f"{type(error).__name__}: {error}"
    try:
        import xarray  # noqa: F401
        status["xarray"] = True
    except Exception:                                # noqa: BLE001
        status["xarray"] = False
    return status


def describe_environment(status: dict) -> str:
    """One readable block saying what will and will not run."""
    lines = []
    lines.append(f"GRF module    {'available' if status['grf'] else 'NOT available'}")
    if not status["grf"]:
        lines.append(f"              {status.get('grf_error', '')}")
    lines.append(f"FTS via rpy2  {'available' if status['fts'] else 'NOT available'}")
    if not status["fts"]:
        lines.append(f"              {status.get('fts_error', '')}")
    if status["grf"] and status["fts"]:
        lines.append("")
        lines.append("Both modules importable - the chain will run live.")
    else:
        lines.append("")
        lines.append("The chain cannot run live here. Cached module output will be")
        lines.append("used where it exists, and every cell says which it used.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Mesh clipping
# --------------------------------------------------------------------------

def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone-chain hull, anticlockwise. Written out to avoid a scipy dependency."""
    pts = np.unique(points, axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(sequence: np.ndarray) -> list[np.ndarray]:
        chain = []
        for p in sequence:
            while len(chain) >= 2:
                (ax, ay), (bx, by) = chain[-2], chain[-1]
                if (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax) > 0:
                    break
                chain.pop()
            chain.append(p)
        return chain[:-1]

    return np.array(half(pts) + half(pts[::-1]))


def mesh_mask(xy: np.ndarray, vertices: np.ndarray,
              margin_factor: float = 1.5) -> np.ndarray:
    """Points inside the mesh footprint.

    Two tests, because the outline is not convex. The hull removes anything
    outside the overall footprint; the distance test then removes points inside
    the hull but beyond the vertices, which is where an irregular catchment
    boundary bites.
    """
    hull = _convex_hull(vertices)
    edge = np.roll(hull, -1, axis=0) - hull
    rel = xy[:, None, :] - hull[None, :, :]
    cross = edge[None, :, 0] * rel[:, :, 1] - edge[None, :, 1] * rel[:, :, 0]
    inside_hull = (cross >= 0).all(axis=1)

    spacing_matrix = ((vertices[:, None, :] - vertices[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(spacing_matrix, np.inf)
    spacing = float(np.median(np.sqrt(spacing_matrix.min(axis=1))))

    distance = np.sqrt(((xy[:, None, :] - vertices[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    return inside_hull & (distance <= spacing * margin_factor)


# --------------------------------------------------------------------------
# GRF
# --------------------------------------------------------------------------

def snapshot_tags(directory: Path, value_suffix: str) -> list[str]:
    """Snapshot labels present in a directory, in chronological order."""
    listing = Path(directory) / "monthly_snapshots.csv"
    if listing.exists():
        return [line.strip() for line in listing.read_text().splitlines() if line.strip()]
    tags = [p.name[: -len(f"_{value_suffix}.csv")]
            for p in Path(directory).glob(f"*_{value_suffix}.csv")]
    return sorted(tags, key=_tag_sort_key)


def _tag_sort_key(tag: str) -> tuple[int, int]:
    match = re.match(r"^(\d{4})_([A-Za-z]{3})$", tag)
    if match:
        return int(match.group(1)), _MONTH_NUMBER.get(match.group(2), 0)
    match = re.match(r"^(\d{4})_(\d{2})(\d{2})$", tag)
    if match:
        return int(match.group(1)), int(match.group(2)) * 100 + int(match.group(3))
    return (0, 0)


# Confirmed against the deployed module on 14 September 2026: run_inla_model
# returns v, v_mean, v_uncertainty, v_uncertainty_mean. `v_uncertainty` is the
# per-node posterior spread; `v_uncertainty_mean` is the spread on the spatial
# mean, which the module computes itself and is the better of the two to use.
SD_NAMES = ("v_uncertainty", "v_sd", "v_stddev", "v_sigma", "sd", "stddev", "sigma")
MEAN_SD_NAMES = ("v_uncertainty_mean", "v_sd_mean", "uncertainty_mean")


def posterior_sd(fitted: xr.Dataset, variable: str = "v") -> np.ndarray | None:
    """Return the posterior standard deviation from a GRF result, if it has one.

    INLA returns a full posterior at every mesh node, not just its mean. Which
    of those summaries the module passes through has varied, so rather than
    assume a name this looks for the usual ones and for a `statistic`
    coordinate, and returns None when there is nothing to find - the caller
    then says so instead of inventing a number.
    """
    for name in SD_NAMES:
        if name in getattr(fitted, "data_vars", {}):
            return np.asarray(fitted[name].values).ravel()
    array = fitted[variable] if variable in getattr(fitted, "data_vars", {}) else None
    if array is not None:
        for dimension in ("statistic", "summary", "quantity"):
            if dimension in getattr(array, "coords", {}):
                labels = [str(v) for v in array[dimension].values]
                for wanted in ("sd", "stddev", "sigma"):
                    if wanted in labels:
                        return np.asarray(
                            array.sel({dimension: wanted}).values).ravel()
    return None


def posterior_mean_sd(fitted: xr.Dataset) -> float | None:
    """Return the module's own spread on the catchment mean, if it reports one.

    Averaging 295 node posteriors is not averaging 295 independent numbers, and
    getting that right needs the joint posterior. The module already has it and
    publishes the answer as `v_uncertainty_mean`, so use that in preference to
    any approximation made out here.
    """
    for name in MEAN_SD_NAMES:
        if name in getattr(fitted, "data_vars", {}):
            values = np.asarray(fitted[name].values).ravel()
            if values.size:
                return float(values[0])
    return None


def _cached_matrix(cache: Path, tags: Sequence[str],
                   progress: Progress) -> pd.DataFrame | None:
    """
    Read a cached fitted matrix, if one is there and describes these snapshots.

    Parameters
    ----------
    cache : pathlib.Path
        Where an earlier run wrote its matrix.
    tags : sequence of str
        The snapshots wanted, in order.
    progress : callable
        Where to send a line saying what happened.

    Returns
    -------
    pandas.DataFrame or None
        The cached matrix, or None when there is nothing usable.
    """
    if not cache.exists():
        return None
    cached = pd.read_csv(cache, index_col=0)
    if len(cached) == len(tags) and list(cached.index) == list(tags):
        progress(f"GRF: loaded {len(tags)} fitted fields from cache ({cache.name})")
        return cached
    progress(f"GRF: cache {cache.name} does not match these snapshots - refitting")
    return None


def _write_posterior_spread(sd_cache: Path, sd_rows: list[np.ndarray],
                            mean_sd: list[float | None], tags: Sequence[str],
                            progress: Progress) -> None:
    """
    Write the posterior spread beside the fitted matrix, when the module gave one.

    Parameters
    ----------
    sd_cache : pathlib.Path
        Where to write.
    sd_rows : list of numpy.ndarray
        One row of nodal standard deviations per snapshot.
    mean_sd : list of float or None
        The module's own spread on the spatial mean, per snapshot.
    tags : sequence of str
        Snapshot labels, used as the index.
    progress : callable
        Where to send a line saying what happened.
    """
    sd_cache.parent.mkdir(parents=True, exist_ok=True)
    if len(sd_rows) != len(tags):
        progress("GRF: the module returned no posterior spread, so none was "
                 "cached - uncertainty propagation will report it as missing")
        return
    frame = pd.DataFrame(np.array(sd_rows), index=pd.Index(tags, name="snapshot"))
    # The module's own spread on the spatial mean rides along as a named column,
    # so the propagation does not have to reconstruct it.
    complete = all(v is not None for v in mean_sd)
    if complete:
        frame["__mean_sd__"] = mean_sd
    frame.to_csv(sd_cache)
    progress(f"GRF: posterior spread cached to {sd_cache.name}"
             + ("" if complete else " (no spatial-mean spread reported)"))


def _snapshot_dataset(directory: Path, tag: str, value_suffix: str,
                      centroid_suffix: str, vertices: np.ndarray,
                      margin_factor: float) -> tuple[xr.Dataset, int, int]:
    """
    Read one snapshot and keep the points the mesh can take.

    Parameters
    ----------
    directory : pathlib.Path
        Directory of extracted snapshots.
    tag : str
        The snapshot label.
    value_suffix : str
        Suffix of the value file.
    centroid_suffix : str
        Suffix of the coordinate file.
    vertices : numpy.ndarray
        Mesh node positions, used to decide what is inside.
    margin_factor : float
        How far outside the hull a point may still sit.

    Returns
    -------
    tuple of (xarray.Dataset, int, int)
        The dataset the GRF module takes, how many points it holds, and how
        many the snapshot had.

    Raises
    ------
    ValueError
        If the coordinate and value files disagree in length, or too few
        points fall inside the mesh to fit.
    """
    xy = np.loadtxt(directory / f"{tag}_{centroid_suffix}.csv", delimiter=",")
    values = np.loadtxt(directory / f"{tag}_{value_suffix}.csv").ravel()
    if len(xy) != len(values):
        raise ValueError(f"{tag}: {len(xy)} coordinates against {len(values)} values")

    keep = np.isfinite(values) & mesh_mask(xy, vertices, margin_factor)
    if keep.sum() < 3:
        raise ValueError(f"{tag}: only {int(keep.sum())} usable points inside the mesh")

    dataset = xr.Dataset(
        {"v": (("time", "point_id"), values[keep][np.newaxis, :])},
        coords={"time": ("time", np.array(["2000-01-01"], dtype="datetime64[ns]")),
                "x": ("point_id", xy[keep, 0]),
                "y": ("point_id", xy[keep, 1])})
    return dataset, int(keep.sum()), len(values)


def run_grf(directory: str | Path, value_suffix: str, centroid_suffix: str = "centroid",
            cache: str | Path | None = None, tags: list[str] | None = None,
            margin_factor: float = 1.5, force: bool = False,
            sd_cache: str | Path | None = None,
            progress: Progress = print) -> pd.DataFrame:
    """Fit the GRF module to every snapshot. Returns (snapshot x mesh node).

    Reloads from `cache` when one exists and covers the same snapshots, so the
    notebook pays the fitting cost once. Raises `ModulesUnavailableError` when the
    module cannot be imported and no usable cache is present.

    When `sd_cache` is given the posterior standard deviation is written there
    too, which is what `uc1_hydro.propagate` needs to turn a validation band
    into a propagated uncertainty.
    """
    directory = Path(directory)
    tags = tags or snapshot_tags(directory, value_suffix)
    cache = Path(cache) if cache else None

    if cache and not force:
        cached = _cached_matrix(cache, tags, progress)
        if cached is not None:
            return cached

    try:
        from use_case_1.grf_module.grf_module import GrfModule
        from use_case_1.grf_module.spde_mesh import SpdeMesh
    except Exception as error:                       # noqa: BLE001
        raise ModulesUnavailableError(
            f"cannot import the GRF module ({type(error).__name__}: {error}). "
            "Run this notebook where use_case_1 and R/INLA are installed, or "
            "point `cache` at a matrix produced by an earlier run."
        ) from error

    grf = GrfModule()
    mesh = SpdeMesh(MESH_NAME, prior_sigma_1=PRIOR_SIGMA_1, prior_sigma_2=PRIOR_SIGMA_2)
    vertices = np.asarray(mesh.vertices)[:, :2]

    rows, sd_rows, mean_sd, started = [], [], [], time.time()
    for i, tag in enumerate(tags, start=1):
        dataset, used, total = _snapshot_dataset(
            directory, tag, value_suffix, centroid_suffix, vertices, margin_factor)
        fitted = grf.run_inla_model(dataset, spde_mesh=mesh)
        rows.append(np.asarray(fitted["v"].values).ravel()[:len(vertices)])
        spread = posterior_sd(fitted)
        if spread is not None:
            sd_rows.append(spread[:len(vertices)])
        mean_sd.append(posterior_mean_sd(fitted))

        if i == 1 or i % 10 == 0 or i == len(tags):
            elapsed = time.time() - started
            progress(f"GRF: {i}/{len(tags)} snapshots  ({elapsed:.0f}s elapsed, "
                     f"{elapsed / i:.1f}s each, {used}/{total} points used)")

    matrix = pd.DataFrame(np.array(rows), index=pd.Index(tags, name="snapshot"))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(cache)
        progress(f"GRF: cached to {cache.name}")

    if sd_cache:
        _write_posterior_spread(Path(sd_cache), sd_rows, mean_sd, tags, progress)
    return matrix


# --------------------------------------------------------------------------
# FTS
# --------------------------------------------------------------------------

def run_fts(matrix: pd.DataFrame, model_coeff_r: str | Path, delta: int = 5,
            p0: int = 2, factors: int = 3, cache: str | Path | None = None,
            force: bool = False, progress: Progress = print) -> pd.DataFrame:
    """Run David's `model.coeff()` on a GRF matrix. Returns (season x factor).

    `matrix` is snapshots by mesh nodes, as `run_grf` returns it; the module
    wants the transpose. `delta` is sub-periods per season - 5 for the monthly
    May-to-September snapshots.
    """
    cache = Path(cache) if cache else None
    n_seasons = matrix.shape[0] // delta

    if cache and cache.exists() and not force:
        cached = pd.read_csv(cache, index_col=0)
        if len(cached) == n_seasons:
            progress(f"FTS: loaded {len(cached)} seasons of coefficients from cache")
            return cached
        progress("FTS: cache does not match this matrix - recomputing")

    try:
        import rpy2.robjects as ro
    except Exception as error:                       # noqa: BLE001
        raise ModulesUnavailableError(
            f"cannot import rpy2 ({type(error).__name__}: {error}). The FTS module "
            "runs through R; install rpy2 with the hdftsa R package, or point `cache` at "
            "coefficients from an earlier run."
        ) from error

    values = matrix.to_numpy(dtype=float).T          # nodes x snapshots
    values = values[:, : n_seasons * delta]
    progress(f"FTS: {values.shape[0]} nodes x {values.shape[1]} snapshots "
             f"-> {n_seasons} seasons x {factors} factors")

    ro.r.source(str(model_coeff_r), encoding="UTF-8")
    model_coeff = ro.r["model.coeff"]
    r_matrix = ro.r.matrix(ro.FloatVector(values.flatten(order="C")),
                           nrow=values.shape[0], ncol=values.shape[1], byrow=True)
    result = model_coeff(data=r_matrix, delta=delta, p0=p0, L=factors, h=3)
    coefficients = np.array(
        result.rx2("model").rx2("model").rx2("model2")[0].rx2("coef"))

    frame = pd.DataFrame(coefficients,
                         columns=[f"factor_{i + 1}" for i in range(coefficients.shape[1])])
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache)
        progress(f"FTS: cached to {cache.name}")
    return frame


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def seasons_from_tags(tags: Sequence[str]) -> list[int]:
    """Return the distinct years a snapshot list covers, in order."""
    years = []
    for tag in tags:
        year = _tag_sort_key(str(tag))[0]
        if year and year not in years:
            years.append(year)
    return years


def year_of_tags(tags: Sequence[str]) -> list[int]:
    """Return the year of every snapshot, one entry per tag.

    `seasons_from_tags` returns the distinct seasons; this returns the season
    each individual snapshot belongs to, which is what a per-snapshot series
    needs to be grouped by.
    """
    return [_tag_sort_key(str(tag))[0] for tag in tags]


def matrix_to_frame(matrix: pd.DataFrame, log10: bool = True) -> pd.DataFrame:
    """GRF output with its snapshot labels turned into timestamps."""
    values = np.power(10.0, matrix.to_numpy(float)) if log10 else matrix.to_numpy(float)
    stamps = []
    for tag in matrix.index:
        year, key = _tag_sort_key(str(tag))
        if key >= 100:                                # daily label, MMDD
            stamps.append(pd.Timestamp(year=year, month=key // 100, day=key % 100))
        else:
            stamps.append(pd.Timestamp(year=year, month=key, day=1))
    return pd.DataFrame(values, index=pd.DatetimeIndex(stamps))
