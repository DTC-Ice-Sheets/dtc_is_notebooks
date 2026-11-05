"""Exploration visuals for the DTC super-resolution SMB emulator (single location mode).

Everything here works from ONE input: the dataset written by `run_postprocessing_loc`
(the "rcm-emulator" output of the API job). No local ERA5/RACMO re-download or
re-projection is needed.
"""

from __future__ import annotations

from datetime import date, datetime

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyproj
import seaborn as sns
import xarray as xr
from matplotlib.colors import TwoSlopeNorm
from plotly.subplots import make_subplots

POLAR = ccrs.Stereographic(central_latitude=-90.0, central_longitude=0.0, true_scale_latitude=-71.0)

# Must match run_postprocessing_loc (ERA5 pixels cropped from the model input tile)
CONTEXT_PAD_H = 16  # rows, top and bottom
CONTEXT_PAD_W = 8  # columns, left and right

REF_COLOR = "#0072b2"  # RACMO
EMU_COLOR = "#d55e00"  # emulator
SMB_CMAP = "RdBu"  # blue = net accumulation, red = net loss (same as the dashboard)
DIFF_CMAP = "PuOr"  # purple = emulator higher than RACMO, orange = lower
ERA5_UNITS = {"tp": "m", "t2m": "K", "d2m": "K", "sst": "K", "sp": "Pa", "msl": "Pa", "u10": "m/s", "v10": "m/s"}
SMB_UNITS = "kg m$^{-2}$"  # check against your product; the dashboard labels some SMB layers mm w.e.


# ============================================================================
# Loading
# ============================================================================


def output_paths(outputs: dict, node: str) -> list[str]:
    """Return the output path(s) of a job node as a list, whether the API gave a str or a list.

    Indexing a plain string (`path[0]`) silently returns its first CHARACTER, so always go through this.
    Raises a clear error when the workflow left a template expression (`{{= ... }}`) unrendered.
    """
    raw = outputs[node]["output_path"]
    paths = [raw] if isinstance(raw, str) else list(raw)
    unrendered = [p for p in paths if "{{" in p]
    if unrendered:
        raise ValueError(
            f"Output of '{node}' contains an unrendered workflow expression: {unrendered[0][:120]}... "
            "This is a server-side templating problem. Report the job id to the API maintainers."
        )
    return paths


def open_emulator_output(source: str | xr.Dataset) -> xr.Dataset:
    """Load the emulator output (zarr store, netCDF file or an already open Dataset) into memory."""
    if isinstance(source, xr.Dataset):
        ds = source
    else:
        path = str(source)
        try:
            ds = xr.open_zarr(path)
        except Exception:
            ds = xr.open_dataset(path)
    missing = {"predictions", "x_cropped"} - set(ds.data_vars)
    if missing:
        raise KeyError(f"Missing variables {sorted(missing)}; found {list(ds.data_vars)}")
    return ds.load()


# ============================================================================
# Data helpers
# ============================================================================


def reference_on_prediction_time(ds: xr.Dataset) -> xr.DataArray:
    """Return the RACMO reference on the emulator's `time` axis (NaN where RACMO has no matching month).

    `predictions` (ERA5 months) and `target_patches` (RACMO months) have DIFFERENT time axes, so comparing
    index i with index i is wrong whenever they start on different months or are stamped on different days.
    Matching by calendar month works for any day-of-month stamping. If RACMO is absent, return an all-NaN
    array with the predictions' shape.
    """
    pred_m = pd.DatetimeIndex(ds["time"].values).to_period("M")
    if "target_patches" not in ds or "time_r" not in ds:
        return xr.full_like(ds["predictions"], np.nan, dtype=np.float32).rename("racmo_reference")

    ref_m = pd.DatetimeIndex(ds["time_r"].values).to_period("M")
    if not ref_m.is_unique:
        raise ValueError("RACMO reference has several steps per month; resample it to monthly first.")
    idx = ref_m.get_indexer(pred_m)  # -1 where RACMO has no such month
    data = ds["target_patches"].values
    out = np.full((pred_m.size, *data.shape[1:]), np.nan, dtype="float32")
    ok = idx >= 0
    out[ok] = data[idx[ok]]
    return xr.DataArray(
        out,
        dims=("time", "y", "x"),
        coords={"time": ds["time"].values, "y": ds["y"].values, "x": ds["x"].values},
        name="racmo_reference",
    )


def point_series(ds: xr.Dataset, era5_channel: str = "tp") -> pd.DataFrame:
    """Monthly series at the requested location: emulator, RACMO and the ERA5 tile mean.

    The pixel is looked up by coordinates, not as the geometric tile centre: near the domain edge the tile
    window is shifted, so the centre pixel is then not the requested point.
    """
    x0, y0 = ds.attrs["x_3031"], ds.attrs["y_3031"]
    sel = {"x": x0, "y": y0, "method": "nearest"}
    era5 = ds["x_cropped"].sel(channels=era5_channel).mean(("y_era5", "x_era5"))
    df = pd.DataFrame(
        {
            "emulator": ds["predictions"].sel(**sel).to_series(),
            "racmo": reference_on_prediction_time(ds).sel(**sel).to_series(),
            f"era5_{era5_channel}": era5.to_series(),
        }
    )
    df.index = pd.DatetimeIndex(df.index)
    return df


def skill_metrics(df: pd.DataFrame) -> dict:
    """Bias, RMSE, MAE and correlation over the months where both emulator and RACMO exist."""
    both = df[["emulator", "racmo"]].dropna()
    if len(both) < 2:
        return {"n": len(both), "bias": np.nan, "rmse": np.nan, "mae": np.nan, "r": np.nan}
    err = both["emulator"] - both["racmo"]
    return {
        "n": len(both),
        "bias": float(err.mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(err.abs().mean()),
        "r": float(both["emulator"].corr(both["racmo"])),
    }


def annual_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Sum of the monthly values per year; NaN unless the year has all 12 months (no partial years)."""
    return pd.DataFrame({c: df[c].groupby(df.index.year).sum(min_count=12) for c in ("racmo", "emulator")})


def _time_index(ds: xr.Dataset, t) -> int:  # noqa: ANN001
    """Accept an integer index or anything pandas can parse as a date ('2021-07')."""
    if isinstance(t, (int, np.integer)):
        return int(t)
    return int(np.abs(ds["time"].values - np.datetime64(pd.Timestamp(t))).argmin())


def _extent(x, y, pad_cells=(0, 0)) -> tuple[float, float, float, float]:  # noqa: ANN001
    """Outer edges (xmin, xmax, ymin, ymax) of a regular grid from its cell CENTRES, optionally padded."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    dx, dy = np.abs(np.diff(x)).mean(), np.abs(np.diff(y)).mean()
    px, py = pad_cells[0] * dx, pad_cells[1] * dy
    return x.min() - dx / 2 - px, x.max() + dx / 2 + px, y.min() - dy / 2 - py, y.max() + dy / 2 + py


def _smb_norm(*arrays: np.ndarray) -> TwoSlopeNorm:
    """Colour norm shared by RACMO and emulator, with white at zero (the gain/loss divide)."""
    vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    lo, hi = np.nanpercentile(vals, [1, 99])
    return TwoSlopeNorm(vmin=min(lo, -1e-6), vcenter=0.0, vmax=max(hi, 1e-6))


def era5_label(channel: str) -> str:
    """Axis label for an ERA5 channel. `x_cropped` is the RAW model input (not standardised, NaN kept), so these are
    native ERA5 units (assumed: monthly means as delivered by the download step; `tp` is a daily accumulation in m).
    """
    return f"ERA5 {channel} [{ERA5_UNITS.get(channel, '?')}]  (raw input)"


def check_polar_projection(lon: float = -65.1279, lat: float = -75.0076) -> None:
    """Print where EPSG:3031 (pyproj) and the cartopy projections put a point, to verify the projection choice."""
    ref = pyproj.Transformer.from_crs(4326, 3031, always_xy=True).transform(lon, lat)
    print(f"pyproj EPSG:3031              : x={ref[0]:12.1f} y={ref[1]:12.1f}") # noqa: T201
    for name, proj in (("SouthPolarStereo()", ccrs.SouthPolarStereo()), ("POLAR (lat_ts=-71)", POLAR)):
        x, y = proj.transform_point(lon, lat, ccrs.PlateCarree())
        print(f"cartopy {name:20s}: x={x:12.1f} y={y:12.1f}  (dx={x - ref[0]:9.1f} m)") # noqa: T201


# ============================================================================
# Figure 1: one month, spatial comparison
# ============================================================================


def plot_tile_comparison(
    ds: xr.Dataset,
    t=0,  # noqa: ANN001 - int index or date string
    *,
    coastlines: bool = True,
    coast_resolution: str = "50m",
    overview_halfwidth_km: float = 1200,
    era5_channel: str = "tp",
    era5_label_: str | None = None,
    units: str = SMB_UNITS,
    figsize: tuple[float, float] = (21, 5.6),
) -> plt.Figure:
    """ERA5 input | RACMO reference | emulator | emulator - RACMO | where the tile is.

    RACMO and emulator share ONE colour scale centred on zero, so colours are directly comparable. The scale
    is fixed over the whole period, so the colours do not jump when you move through months.

    Parameters
    ----------
    ds : xr.Dataset
        The emulator output dataset.
    t : int or str, optional
        The time index or date string for the month to plot. Default is 0.
    coastlines : bool, optional
        Whether to draw coastlines on the map. Default is True.
    coast_resolution : str, optional
        The resolution of the coastlines. Default is "50m".
    overview_halfwidth_km : float, optional
        The half-width of the overview map in kilometers. Default is 1200.
    era5_channel : str, optional
        The ERA5 channel to show in the first map. Default is "tp" (total precipitation).
    era5_label_ : str or None, optional
        The label for the ERA5 channel. If None, a default label will be generated.
    units : str, optional
        The units for the SMB maps and time series. Default is "kg m$^{-2}$".
    figsize : tuple[float, float], optional
        The size of the figure in inches. Default is (21, 5.6).

    Returns
    -------
    plt.Figure
        The generated figure.
    """
    ti = _time_index(ds, t)
    when = pd.Timestamp(ds["time"].values[ti]).strftime("%Y-%m")
    ref_all = reference_on_prediction_time(ds)
    pred, ref = ds["predictions"].isel(time=ti), ref_all.isel(time=ti)
    has_ref = bool(np.isfinite(ref.values).any())
    diff = pred - ref
    era5 = ds["x_cropped"].sel(channels=era5_channel).isel(time=ti)
    x0, y0 = ds.attrs["x_3031"], ds.attrs["y_3031"]

    norm = _smb_norm(ds["predictions"].values, ref_all.values)
    dlim = float(np.nanpercentile(np.abs(diff.values), 98)) if has_ref else 1.0
    dnorm = TwoSlopeNorm(vmin=-max(dlim, 1e-6), vcenter=0.0, vmax=max(dlim, 1e-6))

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.1])
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_map = fig.add_subplot(gs[0, 4], projection=POLAR)

    def draw(ax, xc, yc, field, cmap, norm_, title, label):  # noqa: ANN001, ANN202
        im = ax.pcolormesh(xc / 1e3, yc / 1e3, field, cmap=cmap, norm=norm_, shading="nearest")
        ax.set_title(title, fontsize=13)
        ax.set_aspect("equal")
        ax.set_anchor("S")  # equal aspect shrinks the axes; keep them at the bottom so the colourbar stays attached
        ax.set_xlabel("x [km]  (EPSG:3031)")
        ax.plot(x0 / 1e3, y0 / 1e3, "*", ms=14, mfc="gold", mec="k", zorder=5)  # requested location
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.9, aspect=30).set_label(label)

    def no_reference(ax, title):  # noqa: ANN001, ANN202
        ax.set_title(title, fontsize=13)
        ax.text(0.5, 0.5, "No RACMO reference\nfor this month", transform=ax.transAxes, ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    draw(
        axes[0], ds["x_era5"].values, ds["y_era5"].values, era5.values, "viridis", None,
        "ERA5 input\n(model input tile)", era5_label_ or era5_label(era5_channel),
    )
    if has_ref:
        draw(axes[1], ds["x"].values, ds["y"].values, ref.values, SMB_CMAP, norm, "RACMO reference", f"SMB [{units}]")
    else:
        no_reference(axes[1], "RACMO reference")
    draw(axes[2], ds["x"].values, ds["y"].values, pred.values, SMB_CMAP, norm, "Emulator", f"SMB [{units}]")
    if has_ref:
        draw(
            axes[3], ds["x"].values, ds["y"].values, diff.values, DIFF_CMAP, dnorm,
            "Emulator - RACMO", f"bias [{units}]",
        )
    else:
        no_reference(axes[3], "Emulator - RACMO")
    axes[0].set_ylabel("y [km]")

    # --- location map: model output tile (blue), full ERA5 input footprint (orange), requested point ---
    era5_ext = _extent(ds["x_era5"], ds["y_era5"], pad_cells=(CONTEXT_PAD_W, CONTEXT_PAD_H))
    tile_ext = _extent(ds["x"], ds["y"])
    hw = overview_halfwidth_km * 1e3

    ax_map.set_extent([x0 - hw, x0 + hw, y0 - hw, y0 + hw], crs=POLAR)
    ax_map.coastlines(coast_resolution, linewidth=0.6)
    ax_map.gridlines(linewidth=0.3, alpha=0.5)
    scale, kw = 1.0, {"transform": POLAR}

    for ext, color in ((era5_ext, "coral"), (tile_ext, "steelblue")):
        ax_map.add_patch(
            mpatches.Rectangle(
                (ext[0] * scale, ext[2] * scale), (ext[1] - ext[0]) * scale, (ext[3] - ext[2]) * scale,
                ec=color, fc=color, alpha=0.3, lw=1.2, **kw,
            )
        )
    ax_map.plot(x0 * scale, y0 * scale, "*", ms=14, mfc="gold", mec="k", **kw)
    ax_map.set_title("Tile location", fontsize=13)
    ax_map.legend(
        handles=[
            mpatches.Patch(fc="steelblue", alpha=0.4, label="Emulator output tile"),
            mpatches.Patch(fc="coral", alpha=0.4, label="ERA5 input footprint"),
            plt.Line2D([], [], marker="*", ls="", mfc="gold", mec="k", ms=11, label="Requested location"),
        ],
        loc="lower left", fontsize=8,
    )

    fig.suptitle(
        f"Glaciated SMB emulator - {when}   ({ds.attrs['lat']:.3f}, {ds.attrs['lon']:.3f})", fontsize=16
    )
    return fig


# ============================================================================
# Figure 2: the requested location through time
# ============================================================================


def plot_point_dashboard(
    ds: xr.Dataset,
    era5_channel: str = "tp",
    units: str = SMB_UNITS,
    figsize: tuple[float, float] = (16, 10.5),
) -> plt.Figure:
    """Input, emulator vs RACMO, bias, and skill diagnostics at the requested location.

    Parameters
    ----------
    ds : xr.Dataset
        The emulator output dataset.
    era5_channel : str, optional
        The ERA5 channel to show in the first map. Default is "tp" (total precipitation).
    units : str, optional
        The units for the SMB maps and time series. Default is "kg m$^{-2}$".
    figsize : tuple[float, float], optional
        The size of the figure in inches. Default is (16, 10.5).

    Returns
    -------
    plt.Figure
        The generated figure.
    """
    df = point_series(ds, era5_channel)
    m = skill_metrics(df)
    both = df[["racmo", "emulator"]].dropna()

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.8, 1.7, 1.5])
    ax_in = fig.add_subplot(gs[0, :])
    ax_smb = fig.add_subplot(gs[1, :], sharex=ax_in)
    ax_sc = fig.add_subplot(gs[2, 0])
    ax_sea = fig.add_subplot(gs[2, 1])
    ax_ann = fig.add_subplot(gs[2, 2])

    # --- input: ERA5 tile mean ---
    col = f"era5_{era5_channel}"
    ax_in.plot(df.index, df[col], color="#009e73", lw=1.5)
    ax_in.fill_between(df.index, df[col], alpha=0.15, color="#009e73")
    ax_in.set_title(f"INPUT: {era5_label(era5_channel)}, mean over the input tile", fontsize=13)
    ax_in.tick_params(labelbottom=False)

    # --- output: emulator vs RACMO, same axes so the comparison is direct ---
    ax_smb.plot(df.index, df["racmo"], color=REF_COLOR, lw=1.8, label="RACMO reference")
    ax_smb.plot(df.index, df["emulator"], color=EMU_COLOR, lw=1.8, label="Emulator")
    if np.isfinite(m["rmse"]):
        ax_smb.fill_between(
            df.index,
            df["emulator"] - m["rmse"],
            df["emulator"] + m["rmse"],
            color=EMU_COLOR,
            alpha=0.15,
            label="Emulator \u00b1 RMSE",
        )
    ok = (df["racmo"].notna() & df["emulator"].notna()).to_numpy()
    ax_smb.fill_between(
        df.index, df["racmo"], df["emulator"], where=ok, interpolate=True, color="0.5", alpha=0.25, label="difference"
    )
    ax_smb.axhline(0, color="k", lw=0.6)
    valid = df["racmo"].dropna()
    if len(valid) and valid.index.max() < df.index.max():  # emulator keeps going after RACMO ends
        ax_smb.axvspan(valid.index.max(), df.index.max(), color="0.92", zorder=0)
        mid = valid.index.max() + (df.index.max() - valid.index.max()) / 2
        ax_smb.text(mid, 0.04, "emulator only:\nno RACMO reference",
                    transform=ax_smb.get_xaxis_transform(), ha="center", va="bottom", color="0.35")
    elif not len(valid):
        ax_smb.text(0.5, 0.9, "no RACMO reference in this period", transform=ax_smb.transAxes, ha="center")
    ax_smb.set_ylabel(f"SMB [{units}]")
    ax_smb.set_title("OUTPUT: monthly SMB at the requested location", fontsize=13)
    ax_smb.legend(loc="upper left", ncol=4, fontsize=9)

    # --- scatter, coloured by calendar month ---
    if len(both):
        sc = ax_sc.scatter(both["racmo"], both["emulator"],
                           c=both.index.month, cmap="twilight", s=28, edgecolor="k", lw=0.3)
        lo, hi = both.to_numpy().min(), both.to_numpy().max()
        ax_sc.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        fig.colorbar(sc, ax=ax_sc, label="month", ticks=[1, 4, 7, 10], shrink=0.8, pad=0.02)
        ax_sc.text(
            0.04, 0.96, f"n={m['n']}\nbias={m['bias']:+.1f}\nRMSE={m['rmse']:.1f}\nr={m['r']:.2f}",
            transform=ax_sc.transAxes, va="top", fontsize=9, bbox={"fc": "w", "alpha": 0.8, "ec": "0.7"},
        )  # fmt: skip
    ax_sc.set_xlabel("RACMO")
    ax_sc.set_ylabel("Emulator")
    ax_sc.set_title("Agreement (1:1 dashed)", fontsize=12)

    # --- seasonal cycle: both over the SAME months, plus the full-period emulator ---
    months = np.arange(1, 13)
    if len(both):
        ax_sea.plot(months, both["racmo"].groupby(both.index.month).mean().reindex(months),
                    color=REF_COLOR, marker="o", label="RACMO")
        ax_sea.plot(months, both["emulator"].groupby(both.index.month).mean().reindex(months),
                    color=EMU_COLOR, marker="o", label="Emulator (same months)")
    ax_sea.plot(months, df["emulator"].groupby(df.index.month).mean().reindex(months),
                color=EMU_COLOR, ls="--", label="Emulator (all months)")
    ax_sea.set_xticks(months, list("JFMAMJJASOND"))
    ax_sea.set_title("Seasonal cycle (mean)", fontsize=12)
    ax_sea.legend(fontsize=8)

    # --- annual totals: complete years only ---
    ann = annual_totals(df)
    if ann.notna().any().any():
        ann.rename(columns={"racmo": "RACMO", "emulator": "Emulator"}).plot.bar(
            ax=ax_ann, color=[REF_COLOR, EMU_COLOR], width=0.8, rot=0
        )
        ax_ann.legend(fontsize=8, loc="upper left")
        ax_ann.margins(y=0.25)  # head-room so the legend never covers a bar
    ax_ann.set_xlabel("")
    ax_ann.set_title("Annual totals (complete years)", fontsize=12)
    ax_ann.set_ylabel(f"[{units} / yr]  (sum of monthly values)")

    fig.suptitle(f"Emulator at ({ds.attrs['lat']:.3f}, {ds.attrs['lon']:.3f})", fontsize=16)
    return fig


# ============================================================================
# Figure 3: interactive explorer (plotly)
# ============================================================================

def build_explorer(
    ds: xr.Dataset,
    *,
    stride: int = 2,
    era5_channel: str = "tp",
    units: str = SMB_UNITS,
    height: int = 780,
)-> go.Figure:  # -> plotly.graph_objects.Figure
    """Month slider driving ERA5 precipitation, emulator and RACMO maps with synchronized time series.

    Parameters
    ----------
    ds : xr.Dataset
        The emulator output dataset.
    stride : int, optional
        Subsample the maps for speed. Default is 2 (every second pixel).
    era5_channel : str, optional
        The ERA5 channel to show in the first map. Default is "tp" (total precipitation).
    units : str, optional
        The units for the SMB maps and time series. Default is "kg m$^{-2}$".
    height : int, optional
        The height of the figure in pixels. Default is 780.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = point_series(ds, era5_channel)
    m = skill_metrics(df)
    ref = reference_on_prediction_time(ds)
    pred_a = ds["predictions"].values[:, ::stride, ::stride]
    ref_a = ref.values[:, ::stride, ::stride]
    xk, yk = ds["x"].values[::stride] / 1e3, ds["y"].values[::stride] / 1e3
    era5_a = ds["x_cropped"].sel(channels=era5_channel).values[:, ::stride, ::stride]
    era5_xk = ds["x_era5"].values[::stride] / 1e3
    era5_yk = ds["y_era5"].values[::stride] / 1e3
    px, py = ds.attrs["x_3031"] / 1e3, ds.attrs["y_3031"] / 1e3

    # Share a scale between the two SMB maps, even when RACMO is entirely absent.
    finite_smb = np.concatenate([pred_a[np.isfinite(pred_a)], ref_a[np.isfinite(ref_a)]])
    lim = max(float(np.nanpercentile(np.abs(finite_smb), 99)), 1e-6)
    finite_era5 = era5_a[np.isfinite(era5_a)]
    era5_min, era5_max = (
        (float(np.nanpercentile(finite_era5, 1)), float(np.nanpercentile(finite_era5, 99)))
        if finite_era5.size else (0.0, 1.0)
    )
    if era5_min == era5_max:
        era5_max = era5_min + 1e-6

    times = pd.DatetimeIndex(ds["time"].values)
    labels = list(times.strftime("%Y-%m"))
    i0 = 0
    era5_name = "ERA5 total precipitation" if era5_channel == "tp" else era5_label(era5_channel)

    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{}, {}, {}], [{"colspan": 3, "secondary_y": True}, None, None]],
        row_heights=[0.62, 0.38], vertical_spacing=0.14, horizontal_spacing=0.11,
        subplot_titles=(
            f"{era5_name} ({era5_channel})", "Emulator", "RACMO reference",
            "Requested location: monthly SMB and ERA5 precipitation",
        ),
    )
    smb_hm = {"x": xk, "y": yk, "hoverongaps": False}
    era5_hm = {"x": era5_xk, "y": era5_yk, "hoverongaps": False}
    # Trace indices are referenced by the animation frames below.
    fig.add_trace(go.Heatmap(z=era5_a[i0], coloraxis="coloraxis2", **era5_hm), 1, 1)  # 0
    fig.add_trace(go.Heatmap(z=pred_a[i0], coloraxis="coloraxis", **smb_hm), 1, 2)  # 1
    fig.add_trace(go.Heatmap(z=ref_a[i0], coloraxis="coloraxis", **smb_hm), 1, 3)  # 2
    fig.add_trace(
        go.Scatter(
            x=[float(np.mean(xk))], y=[float(np.mean(yk))], mode="text",
            text=["No RACMO reference for this month"],
            visible=not bool(np.isfinite(ref_a[i0]).any()),
            showlegend=False, hoverinfo="skip", textfont={"color": "dimgray", "size": 13},
        ),
        1, 3,
    )  # 3: shown for months without RACMO data
    racmo_name = "RACMO" if np.isfinite(ref_a).any() else "RACMO (unavailable)"
    fig.add_trace(
        go.Scatter(x=df.index, y=df["racmo"], mode="lines", name=racmo_name, line={"color": REF_COLOR}),
        2, 1, secondary_y=False,
    )  # 4
    fig.add_trace(
        go.Scatter(x=df.index, y=df["emulator"], mode="lines", name="Emulator", line={"color": EMU_COLOR}),
        2, 1, secondary_y=False,
    )  # 5
    era5_col = f"era5_{era5_channel}"
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df[era5_col], mode="lines", name=era5_name, line={"color": "#009e73"}
        ),
        2, 1, secondary_y=True,
    )  # 6
    for col, color, secondary_y in (
        ("racmo", REF_COLOR, False), ("emulator", EMU_COLOR, False), (era5_col, "#009e73", True)
    ):  # 7, 8, 9: selected-month markers
        fig.add_trace(
            go.Scatter(
                x=[times[i0]], y=[df[col].iloc[i0]], mode="markers", showlegend=False, hoverinfo="skip",
                marker={"size": 12, "color": color, "line": {"color": "black", "width": 1.5}},
            ),
            2, 1, secondary_y=secondary_y,
        )
    for j in (1, 2, 3):
        fig.add_trace(
            go.Scatter(
                x=[px], y=[py], mode="markers", showlegend=False, hoverinfo="skip",
                marker={"symbol": "star", "size": 13, "color": "gold", "line": {"color": "black", "width": 1}},
            ),
            1, j,
        )
    if np.isfinite(m["rmse"]):  # 13: static +/- RMSE band around the emulator curve (whole-period skill, not per-frame)
        band_x = list(df.index) + list(df.index[::-1])
        band_y = list(df["emulator"] + m["rmse"]) + list((df["emulator"] - m["rmse"])[::-1])
        fig.add_trace(
            go.Scatter(
                x=band_x, y=band_y, mode="lines", fill="toself",
                fillcolor="rgba(213, 94, 0, 0.15)", line={"color": "rgba(0, 0, 0, 0)"},
                name="Emulator \u00b1 RMSE", hoverinfo="skip",
            ),
            2, 1, secondary_y=False,
        )
    frames = [
        go.Frame(
            name=str(i),
            traces=[0, 1, 2, 3, 7, 8, 9],
            data=[
                go.Heatmap(z=era5_a[i]), go.Heatmap(z=pred_a[i]), go.Heatmap(z=ref_a[i]),
                go.Scatter(visible=not bool(np.isfinite(ref_a[i]).any())),
                go.Scatter(x=[times[i]], y=[df["racmo"].iloc[i]]),
                go.Scatter(x=[times[i]], y=[df["emulator"].iloc[i]]),
                go.Scatter(x=[times[i]], y=[df[era5_col].iloc[i]]),
            ],
        )
        for i in range(len(times))
    ]
    fig.frames = frames

    for j in (1, 2, 3):  # square pixels on the maps
        fig.update_xaxes(title_text="x [km]", constrain="domain", row=1, col=j)
        fig.update_yaxes(scaleanchor="x" if j == 1 else f"x{j}", scaleratio=1, constrain="domain", row=1, col=j)
    fig.update_yaxes(title_text="y [km]", row=1, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text=f"SMB [{units.replace('$^{-2}$', '-2')}]", row=2, col=1, secondary_y=False)
    fig.update_yaxes(
        title_text=f"{era5_name} [{ERA5_UNITS.get(era5_channel, '?')}]",
        row=2, col=1, secondary_y=True,
    )

    anim = {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}
    fig.update_layout(
        template="plotly_white", height=height, margin={"t": 90, "b": 40, "r": 100},
        coloraxis={"colorscale": SMB_CMAP, "cmin": -lim, "cmax": lim,
                   "colorbar": {"title": "SMB", "x": 1.0, "y": 0.76, "len": 0.45, "thickness": 12}},
        coloraxis2={"colorscale": "Viridis", "cmin": era5_min, "cmax": era5_max,
                    "colorbar": {"title": f"{era5_channel} [{ERA5_UNITS.get(era5_channel, '?')}]",
                                 "x": 0.28, "y": 0.76, "len": 0.45, "thickness": 12}},
        legend={"orientation": "h", "x": 0.0, "y": 0.0},
        sliders=[{
            "active": i0, "pad": {"t": 40}, "currentvalue": {"prefix": "Month: "},
            "steps": [{"method": "animate", "label": lab, "args": [[str(i)], anim]} for i, lab in enumerate(labels)],
        }],
        updatemenus=[{
            "type": "buttons", "direction": "left", "x": 0.0, "y": -0.22, "showactive": False,
            "buttons": [
                {"label": "Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 400, "redraw": True}, "fromcurrent": True,
                                 "transition": {"duration": 0}}]},
                {"label": "Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate",
                                   "transition": {"duration": 0}}]},
            ],
        }],
    )
    return fig


# --------------------------------------------------
# Figure 4: plots by single date
# --------------------------------------------------

def plot_single_date(out_ds: xr.Dataset, selected_date: datetime) -> mpl.figure.Figure:
    """Plot ERA5 input, emulator prediction, RACMO reference and overview map for a single date.

    Parameters
    ----------
    out_ds : xr.Dataset
        The emulator output dataset.
    selected_date : datetime
        The date for which to plot the data.
    """
    # check if the date within the dataset range
    date_min = out_ds.isel(time=0).time.values
    date_max = out_ds.isel(time=-1).time.values
    if not (date_min <= np.datetime64(selected_date) <= date_max):
        raise ValueError(f"Date {selected_date} is not in the dataset. Available dates: {out_ds['time'].values}")

    # select the nearest date in the dataset to the provided date
    era5 = out_ds["x_cropped"].sel(channels="tp").sel(time=selected_date, method="nearest").values
    predictions = out_ds["predictions"].sel(time=selected_date, method="nearest").values
    target = out_ds["target_patches"].sel(time_r=selected_date, method="nearest").values

    fig1, ax = plt.subplots(1, 4, figsize=(12, 5), subplot_kw={"projection": POLAR})

    vmin = np.min([np.min(predictions), np.min(target)])
    vmax = np.max([np.max(predictions), np.max(target)])

    vmin = np.min(target)
    vmax = np.max(target)

    im0 = ax[0].pcolormesh(out_ds.x_era5.values,
                        out_ds.y_era5.values,
                        era5,
                        cmap="viridis",
                        transform=POLAR)
    ax[0].set_title("ERA5 T.Precipitation \nnorm[mm/day]", fontsize=14)
    ax[0].coastlines("50m", linewidth=0.5)
    ax[0].set_extent([out_ds.x_era5.values.min(), out_ds.x_era5.values.max(),
                    out_ds.y_era5.values.min(), out_ds.y_era5.values.max()],
                    crs=POLAR)

    im1 = ax[1].pcolormesh(out_ds.x.values,
                            out_ds.y.values,
                            np.squeeze(target),
                            cmap="coolwarm",
                            transform=POLAR,
                            vmin=vmin, vmax=vmax)
    ax[1].set_title("RACMO reference \n[kg m-2]", fontsize=14)
    ax[1].coastlines("50m", linewidth=0.5)
    ax[1].set_extent([out_ds.x.values.min(), out_ds.x.values.max(),
                        out_ds.y.values.min(), out_ds.y.values.max()],
                        crs=POLAR)

    im2 = ax[2].pcolormesh(out_ds.x.values,
                        out_ds.y.values,
                        predictions,
                        cmap="coolwarm",
                        transform=POLAR,
                        vmin=vmin, vmax=vmax)
    ax[2].set_title("Predicted SMB \n[kg m-2]", fontsize=14)
    ax[2].coastlines("50m", linewidth=0.5)
    ax[2].set_extent([out_ds.x.values.min(), out_ds.x.values.max(),
                    out_ds.y.values.min(), out_ds.y.values.max()],
                    crs=POLAR)

    # ── Antartica overview ──────────────────────────────────────────────────────────
    def _extent(x, y, pad_cells=(0, 0)) -> tuple[float, float, float, float]:  # noqa: ANN001
        """Outer edges (xmin, xmax, ymin, ymax) of a regular grid from its cell CENTRES, optionally padded."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        dx, dy = np.abs(np.diff(x)).mean(), np.abs(np.diff(y)).mean()
        px, py = pad_cells[0] * dx, pad_cells[1] * dy
        return x.min() - dx / 2 - px, x.max() + dx / 2 + px, y.min() - dy / 2 - py, y.max() + dy / 2 + py



    context_pad_h = 16  # rows, top and bottom
    context_pad_w = 8  # columns, left and right
    polar = ccrs.Stereographic(central_latitude=-90.0, central_longitude=0.0, true_scale_latitude=-71.0)
    era5_ext = _extent(out_ds["x_era5"], out_ds["y_era5"], pad_cells=(context_pad_w, context_pad_h))
    tile_ext = _extent(out_ds["x"], out_ds["y"])
    hw = 2400 * 1e3
    x0, y0 = out_ds.attrs["x_3031"], out_ds.attrs["y_3031"]
    ax[3].set_extent([x0 - hw, x0 + hw, y0 - hw, y0 + hw], crs=polar)
    ax[3].coastlines("50m", linewidth=0.6)
    ax[3].gridlines(linewidth=0.3, alpha=0.5)
    scale, kw = 1.0, {"transform": polar}


    for ext, color in ((era5_ext, "coral"), (tile_ext, "steelblue")):
        ax[3].add_patch(mpatches.Rectangle(
            (ext[0] * scale, ext[2] * scale), (ext[1] - ext[0]) * scale, (ext[3] - ext[2]) * scale,
            ec=color, fc=color, alpha=0.3, lw=1.2, **kw)
            )
        ax[3].plot(x0 * scale, y0 * scale, "*", ms=14, mfc="gold", mec="k", **kw)
        ax[3].set_title("Tile location", fontsize=13)
        ax[3].legend(
            handles=[
                mpatches.Patch(fc="steelblue", alpha=0.4, label="Emulator output tile"),
                mpatches.Patch(fc="coral", alpha=0.4, label="ERA5 input footprint"),
                plt.Line2D([], [], marker="*", ls="", mfc="gold", mec="k", ms=11, label="Requested location"),
            ],
            loc="lower left", fontsize=8,
        )

    # colorbar under each plot, except the overview (4th) panel
    for im, axis in ((im0, ax[0]), (im1, ax[1]), (im2, ax[2])):
        fig1.colorbar(im, ax=axis, orientation="horizontal", pad=0.05, shrink=0.9)

    plt.suptitle(f"Glaciated SMB emulator - Antarctica - {date}", fontsize=18)
    plt.tight_layout()
    plt.close()
    return fig1


# -----------------------------------------------------------
# figure 5: validation
# -----------------------------------------------------------

def plot_validation(out_ds: xr.Dataset, selected_date: datetime) -> mpl.figure.Figure:
    """
    Plot the validation results for a given date.

    It plots the bias (predictions - target) and RMSE, as well as a violin plot comparing
    the distributions of predictions and target values.

    Parameters
    ----------
    out_ds : xr.Dataset
        The dataset containing the predictions and target values.
    selected_date : datetime
        The date for which to plot the validation results.

    Raises
    ------
    ValueError
        If the provided date is not within the range of the dataset.

    Returns
    -------
        The function generates and displays the plots for the specified date.
    """
    # check if the date within the dataset range
    date_min = out_ds.isel(time=0).time.values
    date_max = out_ds.isel(time=-1).time.values
    if not (date_min <= np.datetime64(selected_date) <= date_max):
        raise ValueError(f"Date {selected_date} is not in the dataset. Available dates: {out_ds['time'].values}")

    predictions = out_ds["predictions"].sel(time=selected_date, method="nearest").values
    target = out_ds["target_patches"].sel(time_r=selected_date, method="nearest").values

    bias = predictions - target
    diff_vmin = np.nanpercentile(bias, 2)
    diff_vmax = np.nanpercentile(bias, 98)

    rmse = np.sqrt(bias ** 2)
    rmse_max = np.nanpercentile(rmse, 98)

    fig2 = plt.figure(figsize=(10, 8))                 # taller, for 2 rows
    grid = fig2.add_gridspec(2, 2, height_ratios=[1, 0.8])  # tune height_ratios to taste

    ax = [
        fig2.add_subplot(grid[0, 0], projection=POLAR),
        fig2.add_subplot(grid[0, 1], projection=POLAR),
        fig2.add_subplot(grid[1, :]),   # spans both columns of row 2, no projection needed
    ]

    biaslim = max(abs(diff_vmin), abs(diff_vmax))
    im0 = ax[0].pcolormesh(out_ds["predictions"].x.values,
                        out_ds["predictions"].y.values,
                        bias,
                        cmap="PiYG",
                        transform=POLAR,
                        vmin=-biaslim, vmax=biaslim)
    ax[0].set_title("Bias: Predictions - Target", fontsize=14)
    ax[0].coastlines("50m", linewidth=0.5)
    ax[0].set_extent([out_ds["predictions"].x.values.min(), out_ds["predictions"].x.values.max(),
                    out_ds["predictions"].y.values.min(), out_ds["predictions"].y.values.max()],
                    crs=POLAR)

    im1 = ax[1].pcolormesh(out_ds["predictions"].x.values,
                        out_ds["predictions"].y.values,
                        rmse,
                        cmap="inferno_r",
                        vmin=0.0, vmax=rmse_max,
                        transform=POLAR)
    ax[1].set_title("RMSE", fontsize=14)
    ax[1].coastlines("50m", linewidth=0.5)
    ax[1].set_extent([out_ds["predictions"].x.values.min(), out_ds["predictions"].x.values.max(),
                    out_ds["predictions"].y.values.min(), out_ds["predictions"].y.values.max()],
                    crs=POLAR)

    # colorbar under each plot, except the overview (4th) panel
    for im, axis in ((im0, ax[0]), (im1, ax[1])):
        fig2.colorbar(im, ax=axis, orientation="vertical", pad=0.05, shrink=0.9)

    pred = np.asarray(predictions).ravel()
    ref = np.asarray(target).ravel()
    pred = pred[np.isfinite(pred)]
    ref = ref[np.isfinite(ref)]

    if pred.size == 0 or ref.size == 0:
        raise ValueError("Both distributions must contain at least one finite value.")

    order = ["Prediction", "Target"]
    colors = {"Prediction": "royalblue", "Target": "darkorange"}
    raincloud_data = pd.DataFrame({
        "value": np.concatenate([pred, ref]),
        "distribution": ["Prediction"] * pred.size + ["Target"] * ref.size,
    })

    sns.violinplot(
        data=raincloud_data,
        x="value",
        y="distribution",
        hue="distribution",
        order=order,
        hue_order=order,
        dodge=False,
        cut=0,
        inner=None,
        palette=colors,
        legend=False,
        ax=ax[2],
    )

    # Keep the upper half of each violin.
    for idx, body in enumerate(ax[2].collections[:2]):
        vertices = body.get_paths()[0].vertices
        vertices[:, 1] = np.clip(vertices[:, 1], idx, idx + 0.38)

    rng = np.random.default_rng(0)
    for idx, (values, name) in enumerate(zip([pred, ref], order, strict=True)):
        sample = rng.choice(values, size=min(1500, values.size), replace=False)
        ax[2].scatter(
            sample,
            idx - rng.uniform(0.02, 0.18, size=sample.size),
            color=colors[name],
            alpha=0.35,
            s=8,
            zorder=3,
        )

    # Add boxplots and a manageable sample of individual values.
    ax[2].boxplot(
        [pred, ref],
        positions=[0, 1],
        orientation="horizontal",
        widths=0.12,
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
    )
    ax[2].set_title("Predicted vs Target SMB Distribution", fontsize=14)

    fig2.suptitle(f"Validation - {out_ds.time.values[0].astype('datetime64[D]')}", fontsize=18)
    fig2.tight_layout()
    plt.close()

    return fig2


PLATE = ccrs.PlateCarree()

def select_location(
    lat: float = -75.0076,
    lon: float = -65.1279,
    *,
    figsize: tuple[float, float] = (7, 7),
) -> dict:
    """Interactive Antarctica map for picking a lat/lon by clicking.

    Parameters
    ----------
    lat : float, optional
        Initial latitude to display on the map (default is -75.0076).
    lon : float, optional
        Initial longitude to display on the map (default is -65.1279).
    figsize : tuple of float, optional
        Size of the figure in inches (default is (7, 7)).

    Returns
    -------
    dict
        A dictionary containing the selected latitude and longitude with keys 'lat' and 'lon'.
    """
    fig_sel = plt.figure(figsize=figsize)
    ax_sel = fig_sel.add_subplot(1, 1, 1, projection=POLAR)
    ax_sel.set_extent([-180, 180, -90, -60], crs=PLATE)
    ax_sel.coastlines("50m", linewidth=0.6)
    ax_sel.gridlines(linewidth=0.3, alpha=0.5, draw_labels=True)
    ax_sel.set_title("Click Antarctica to select a location")

    selected = {"lat": lat, "lon": lon}
    marker, = ax_sel.plot(lon, lat, "*", ms=14, mfc="gold", mec="k", transform=PLATE)
    label = ax_sel.text(0.02, 0.02, f"lat={lat:.4f}, lon={lon:.4f}", transform=ax_sel.transAxes,
                         bbox={ "fc": "white", "alpha": 0.8 })

    def on_click(event):  # noqa: ANN202, ANN001
        if event.inaxes is not ax_sel or event.xdata is None or event.ydata is None or event.button != 1:
            return
        click_lon, click_lat = PLATE.transform_point(event.xdata, event.ydata, ax_sel.projection)
        if not -90 <= click_lat <= -60:
            label.set_text("Click on Antarctica (south of 60S)")
            fig_sel.canvas.draw_idle()
            return
        click_lon = (click_lon + 180) % 360 - 180
        selected.update(lat=click_lat, lon=click_lon)
        marker.set_data([click_lon], [click_lat])
        label.set_text(f"lat={click_lat:.4f}, lon={click_lon:.4f}")
        print(f"Selected location: LAT={click_lat:.4f}, LON={click_lon:.4f}") # noqa: T201
        fig_sel.canvas.draw_idle()

    fig_sel.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    return selected
