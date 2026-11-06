"""
Exploration visuals for the DTC super-resolution SMB emulator (single location mode).

Everything here works from one input: the dataset written by `run_postprocessing_loc`
(the "rcm-emulator" output of the API job).
The code can be found here:
https://github.com/DTC-Ice-Sheets/dtc_super_resolution/blob/main/src/dtc_super_resolution/rcm_emulator_loc_entrypoint.py
"""

from __future__ import annotations

from datetime import datetime

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import xarray as xr
from plotly.subplots import make_subplots

POLAR = ccrs.Stereographic(central_latitude=-90.0, central_longitude=0.0, true_scale_latitude=-71.0)

# Must match run_postprocessing_loc (ERA5 pixels cropped from the model input tile)
CONTEXT_PAD_H = 16  # rows, top and bottom
CONTEXT_PAD_W = 8  # columns, left and right

REF_COLOR = "#0072b2"  # RACMO
EMU_COLOR = "#d55e00"  # emulator
SMB_CMAP = "RdBu"  # blue = net accumulation, red = net loss (same as the dashboard)

# all the ERA5 channels are standardised per tile
ERA5_UNITS = {"d2m": "K", "msl": "Pa","sp": "Pa",  "sst": "K", "t2m": "K", "tp": "m", "u10": "m/s", "v10": "m/s"}
SMB_UNITS = "mm w.e. yr-1"


# ============================================================================
# Loading
# ============================================================================


def find_output_path(outputs: dict, name: str) -> str:
    """
    Return the one job output path whose store name (last path component) contains `name`.

    Searches the outputs of every node, so it does not depend on node names or on the order of the paths,
    e.g. ``find_output_path(result.outputs, "era5_data")`` for ``.../era5_data.zarr``.

    Parameters
    ----------
    outputs : dict
        The ``outputs`` of a finished API job, mapping node names to ``{"output_path": str | list[str]}``.
    name : str
        The (part of the) store name to look for, e.g. ``"era5_data"``.

    Returns
    -------
    str
        The matching output path.

    Raises
    ------
    ValueError
        If no path, or more than one path, matches `name`.
    """
    candidates = []
    for value in outputs.values():
        if "output_path" in value:
            raw = value["output_path"]
            # A node can give a single path or a list of paths
            candidates += [raw] if isinstance(raw, str) else list(raw)
    matches = [p for p in candidates if name in p.split("?")[0].rstrip("/").rsplit("/", 1)[-1]]
    if len(matches) != 1:
        found = "none" if not matches else f"{len(matches)} ({matches})"
        raise ValueError(f"Expected exactly one output named like '{name}', found {found}. Outputs: {candidates}")
    return matches[0]


# ============================================================================
# Data helpers
# ============================================================================


def reference_on_prediction_time(ds: xr.Dataset) -> xr.DataArray:
    """
    Return the RACMO reference on the emulator's `time` axis (NaN where RACMO has no matching month).

    `predictions` (ERA5 months) and `target_patches` (RACMO months) have DIFFERENT time axes, so comparing
    index i with index i is wrong whenever they start on different months or are stamped on different days.
    Matching by calendar month works for any day-of-month stamping. If RACMO is absent, return an all-NaN
    array with the predictions' shape.

    Parameters
    ----------
    ds : xr.Dataset
        The emulator output dataset.

    Returns
    -------
    xr.DataArray
        The RACMO reference on the emulator's `time` axis.
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
    """
    Monthly series at the requested location: emulator, RACMO and the ERA5 tile mean.

    The pixel is looked up by coordinates, not as the geometric tile centre: near the domain edge the tile
    window is shifted, so the centre pixel is then not the requested point.

    Parameters
    ----------
    ds : xr.Dataset
        The emulator output dataset.
    era5_channel : str, optional
        The ERA5 channel to average over the tile. Default is "tp" (total precipitation).

    Returns
    -------
    pd.DataFrame
        The monthly series, with columns "emulator", "racmo" and "era5_<era5_channel>".
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
    """
    Bias, RMSE, MAE and correlation over the months where both emulator and RACMO exist.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly series with "emulator" and "racmo" columns, e.g. from `point_series`.

    Returns
    -------
    dict
        The metrics, with keys "n", "bias", "rmse", "mae" and "r".
    """
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


def _extent(x, y, pad_cells=(0, 0)) -> tuple[float, float, float, float]:  # noqa: ANN001
    """
    Outer edges (xmin, xmax, ymin, ymax) of a regular grid from its cell CENTRES, optionally padded.

    Parameters
    ----------
    x, y : array-like
        The cell-centre coordinates along each axis.
    pad_cells : tuple[int, int], optional
        The number of cells to pad on each side, as (x, y). Default is (0, 0).

    Returns
    -------
    tuple[float, float, float, float]
        The outer edges (xmin, xmax, ymin, ymax).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    dx, dy = np.abs(np.diff(x)).mean(), np.abs(np.diff(y)).mean()
    px, py = pad_cells[0] * dx, pad_cells[1] * dy
    return x.min() - dx / 2 - px, x.max() + dx / 2 + px, y.min() - dy / 2 - py, y.max() + dy / 2 + py


def era5_label(channel: str) -> str:
    """
    Axis label for an ERA5 channel.

    `x_cropped` is the RAW model input (not standardised, NaN kept), so these are
    native ERA5 units (assumed: monthly means as delivered by the download step; `tp` is a daily accumulation in m).

    Parameters
    ----------
    channel : str
        The ERA5 channel name, e.g. "tp".

    Returns
    -------
    str
        The axis label.
    """
    return f"ERA5 {channel} [{ERA5_UNITS.get(channel, '?')}]  (raw input)"


def _validate_selected_month(out_ds: xr.Dataset, selected_date: datetime) -> None:
    dataset_months = pd.DatetimeIndex(out_ds["time"].values).to_period("M")
    requested_month = pd.Period(selected_date, "M")

    if requested_month not in dataset_months:
        raise ValueError(
            f"Date {selected_date} is not in the dataset. "
            f"Available months: {sorted(set(dataset_months.astype(str)))}"
        )


# ============================================================================
# Figure 1: interactive explorer (plotly)
# ============================================================================

def build_explorer(
    ds: xr.Dataset,
    *,
    stride: int = 2,
    era5_channel: str = "tp",
    units: str = SMB_UNITS,
    height: int = 780,
)-> go.Figure:
    """
    Month slider driving ERA5 precipitation, emulator and RACMO maps with synchronized time series.

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
        The interactive explorer figure.
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
# Figure 2: plots by single date
# --------------------------------------------------

def plot_single_date(out_ds: xr.Dataset, selected_date: datetime) -> mpl.figure.Figure:
    """
    Plot ERA5 input, emulator prediction, RACMO reference and overview map for a single date.

    Parameters
    ----------
    out_ds : xr.Dataset
        The emulator output dataset.
    selected_date : datetime
        The date for which to plot the data.

    Returns
    -------
    mpl.figure.Figure
        The generated figure.
    """
    # check if the date within the dataset range
    _validate_selected_month(out_ds, selected_date)

    # select the nearest date in the dataset to the provided date
    era5 = out_ds["x_cropped"].sel(channels="tp").sel(time=np.datetime64(selected_date, "M"), method="nearest").values
    predictions_da = out_ds["predictions"].sel(time=np.datetime64(selected_date, "M"), method="nearest")
    target = reference_on_prediction_time(out_ds).sel(time=predictions_da.time).values
    predictions = predictions_da.values

    if not np.isfinite(target).any():
        print("No RACMO reference is available for the selected prediction month.") # noqa: T201

    fig1, ax = plt.subplots(1, 4, figsize=(12, 5), subplot_kw={"projection": POLAR})

    vmin = np.nanmin([np.nanmin(predictions), np.nanmin(target)])
    vmax = np.nanmax([np.nanmax(predictions), np.nanmax(target)])

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
    era5_ext = _extent(out_ds["x_era5"], out_ds["y_era5"], pad_cells=(CONTEXT_PAD_W, CONTEXT_PAD_H))
    tile_ext = _extent(out_ds["x"], out_ds["y"])
    hw = 2400 * 1e3
    x0, y0 = out_ds.attrs["x_3031"], out_ds.attrs["y_3031"]
    ax[3].set_extent([x0 - hw, x0 + hw, y0 - hw, y0 + hw], crs=POLAR)
    ax[3].coastlines("50m", linewidth=0.6)
    ax[3].gridlines(linewidth=0.3, alpha=0.5)
    scale, kw = 1.0, {"transform": POLAR}


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

    sel_date = np.datetime64(selected_date, "M")
    when = pd.Timestamp(out_ds["time"].sel(time=sel_date, method="nearest").values).strftime("%Y-%m")
    fig1.suptitle(f"Glaciated SMB emulator - Antarctica - {when}", fontsize=18)
    plt.tight_layout()
    plt.close()
    return fig1


# -----------------------------------------------------------
# Figure 3: validation
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
    mpl.figure.Figure
        The function generates and displays the plots for the specified date.
    """
    # check if the date within the dataset range
    _validate_selected_month(out_ds, selected_date)

    predictions_da = out_ds["predictions"].sel(time=np.datetime64(selected_date, "M"), method="nearest")
    predictions = predictions_da.values
    target = reference_on_prediction_time(out_ds).sel(time=predictions_da.time).values
    if not np.isfinite(target).any():
        raise ValueError("No RACMO reference is available for the selected prediction month.")

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
    ax[2].set_title(f"Predicted vs Target SMB Distribution [{SMB_UNITS}]", fontsize=14)

    sel_date = np.datetime64(selected_date, "M")
    when = pd.Timestamp(out_ds["time"].sel(time=sel_date, method="nearest").values).strftime("%Y-%m")
    fig2.suptitle(f"Validation - {when}", fontsize=18)
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
    """
    Interactive Antarctica map for picking a lat/lon by clicking.

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
