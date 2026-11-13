# Interactive explorer for SSMIS / ASCAT / UMelt: the counterpart of `build_explorer` in the emulator demonstrator.
# Run this cell once to define `build_umelt_explorer`; the next cell calls it.
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots

"""Functions to download data from the Data Download Module."""

import logging
from datetime import datetime
import pandas as pd
import json
import ipywidgets as widgets
from ipyleaflet import Map, DrawControl, basemaps, projections, ScaleControl, FullScreenControl
from shapely.geometry import shape, mapping
from shapely.ops import transform
from pyproj import Transformer

from dtc_query_client import ApiClient, DataDownloadRequest, DatasetID, Epsg4326PolygonJson, GenericApi
from dtc_query_client.helpers import wait_for_job
from shapely.geometry import MultiPolygon, Polygon, mapping
from datetime import date

# Set up logger for this module
logger = logging.getLogger(__name__)

# How each product is drawn: variable in the dataset, titles, units, colours.
UMELT_PRODUCTS = {
    "ssmis": {
        "var": "brightness_temperature", "name": "SSMIS", "unit": "K", "ylabel": "SSMIS brightness<br>temperature [K]",
        "colorscale": "RdPu", "reverse": False, "line": "#FF6B6B", "fixed_range": None,
    },
    "ascat": {
        "var": "surface_backwards_scattering_coefficient_of_radar_wave", "name": "ASCAT", "unit": "dB",
        "ylabel": "ASCAT backscatter<br>coefficient [dB]",
        "colorscale": "Viridis", "reverse": True, "line": "#30B64A", "fixed_range": None,
    },
    "umelt": {
        "var": "melt", "name": "UMelt", "unit": "melt presence", "ylabel": "UMelt mean<br>melt presence",
        "colorscale": "Bluered", "reverse": False, "line": "#45B7D1",
        "fixed_range": (0.0, 1.0),
    },
}  


def get_date_picker(desc="Select a date:", default_date=date(2018, 1, 1)):
    """Creates a widget to select a date between 2016 and 2021."""
    return widgets.DatePicker(
        value=default_date, 
        min=date(2016, 1, 1),
        max=date(2021, 12, 31),
        description=desc, 
        style={"description_width": "initial"}
    )


async def submit_data_download(
        client: ApiClient,
        dataset_id: DatasetID,
        start_time: datetime,
        end_time: datetime,
        epsg4326_polygon: Polygon | MultiPolygon,
        variables: list[str] | None = None
) -> str:
    """
    Submit a data download job to the DTC Query API and return the job ID.

    Parameters
    ----------
    dataset_id: DatasetID
        The ID of the dataset to download.
    start_time: datetime
        The start time of the data range to download.
    end_time: datetime
        The end time of the data range to download.
    epsg4326_polygon: Polygon | MultiPolygon
        The polygon in EPSG:4326 coordinates to define the area of interest.
    variables: list[str] | None
        The list of variables to download. If None, all variables are downloaded.

    Returns
    -------
    str
        The job ID of the submitted data download job.
    """
    response = await GenericApi(client).download_data(
        DataDownloadRequest(
            dataset_id=dataset_id,
            start_time=start_time,
            end_time=end_time,
            epsg4326_polygon_json=Epsg4326PolygonJson.from_dict(mapping(epsg4326_polygon)),
            variables=variables,
        )
    )
    logger.info("Data download job started with ID: %s", response.job_id)
    return response.job_id


async def run_data_download(
    client: ApiClient,
    dataset_id: DatasetID,
    start_time: datetime,
    end_time: datetime,
    epsg4326_polygon: Polygon | MultiPolygon,
    variables: list[str] | None = None,
) -> str:
    """Submit a data download job to the DTC Query API and return the resulting GeoZarr URL."""
    if end_time is None:
        end_time = start_time + pd.Timedelta(days=7)  # Default to a 1-week range if end_time is not provided

    if client is None:
        raise ValueError("An ApiClient instance must be provided.")

    job_id = await submit_data_download(client, dataset_id, start_time, end_time, epsg4326_polygon, variables)

    try:
        result = await wait_for_job(client, job_id)
    except RuntimeError:
        logger.exception("Data download job %s failed", job_id)
        raise
    print(f"Data download job {job_id} completed successfully.")
    return result.outputs["main"]["output_path"]


def generate_slider_dates(start_date: datetime, end_date: datetime, freq: str) -> list[datetime]:
    """Generate date/time steps at monthly intervals."""
    date_range = pd.date_range(start_date, end_date, freq=freq)
    return [d.to_pydatetime() for d in date_range]


# Equal-area projection for South Polar region (EASE-Grid 2.0 South) used for area calculation
_to_ea = Transformer.from_crs("EPSG:4326", "EPSG:6932", always_xy=True).transform

def polygon_selector():
    """Create a widget for drawing and selecting polygons on a map."""
    USE_ESRI = False   # set True if the NASA GIBS tiles don't load for you
    if USE_ESRI:
        base, crs = basemaps.Esri.AntarcticBasemap, projections.EPSG3031.ESRIBasemap
    else:
        base, crs = basemaps.NASAGIBS.BlueMarble3031, projections.EPSG3031.NASAGIBS

    m = Map(
        center=(-90, 0),
        zoom=1,
        crs=crs,
        basemap=base,
        scroll_wheel_zoom=True,
        layout=widgets.Layout(height="650px"),
    )
    m.add_control(ScaleControl(position="bottomleft"))
    m.add_control(FullScreenControl())

    # Draw control
    shape_style = {"color": "#e4572e", "fillColor": "#e4572e", "fillOpacity": 0.25, "weight": 2}
    dc = DrawControl(
        polygon={"shapeOptions": shape_style, "allowIntersection": False},
        rectangle={"shapeOptions": shape_style},
        polyline={}, circle={}, circlemarker={}, marker={},
        edit=True, remove=True,
    )
    m.add_control(dc)

    # --- State and widgets ---
    selected_polygon = None      # shapely geometry (lon/lat, EPSG:4326) of the current selection
    polygons = []                # all drawn polygons as shapely geometries

    picker = widgets.Dropdown(description="Polygon:", options=[], layout=widgets.Layout(width="300px"))
    clear_btn = widgets.Button(description="Clear all", icon="trash", button_style="warning")
    save_btn = widgets.Button(description="Save GeoJSON", icon="save", button_style="success")
    status = widgets.HTML()


    def _area_km2(geom):
        """Compute the area of a polygon in km² using an equal-area projection."""
        return transform(_to_ea, geom).area / 1e6


    def _select(_=None):
        """The polygon chosen in the dropdown becomes the current selection."""
        nonlocal selected_polygon
        idx = picker.value
        selected_polygon = polygons[idx] if (idx is not None and polygons) else None


    def _sync(*_):
        """Rebuild the polygon list from the draw control's current features."""
        nonlocal polygons
        previous, kept = len(polygons), selected_polygon
        polygons = [shape(f["geometry"]) for f in dc.data if (f.get("geometry") or {}).get("type") == "Polygon"]
        old = picker.value
        picker.unobserve(_select, names="value")
        picker.options = [(f"Polygon {i + 1}  ({_area_km2(g):,.0f} km²)", i) for i, g in enumerate(polygons)]
        if polygons:
            # Which polygon is selected afterwards: a polygon that was just drawn becomes the selection; otherwise the
            # one that was selected stays.
            same = next((i for i, g in enumerate(polygons) if kept is not None and g.equals(kept)), None)
            if len(polygons) > previous:
                picker.value = len(polygons) - 1
            elif same is not None:
                picker.value = same
            elif old is not None and old < len(polygons):    # the selected polygon was edited / deleted
                picker.value = old
            else:
                picker.value = len(polygons) - 1
        picker.observe(_select, names="value")
        _select()


    dc.observe(_sync, names="data")
    picker.observe(_select, names="value")


    def _clear(_):
        """Remove all polygons and reset the selection."""
        # Empty `data`: the browser redraws from it and _sync() runs through the observer above. 
        dc.data = []
        status.value = ""

    def _save(_):
        """Save the current selection to a GeoJSON file."""
        if selected_polygon is None:
            status.value = "<span style='color:#b00'>Draw a polygon first.</span>"
            return
        fc = {"type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {}, "geometry": mapping(selected_polygon)}]}
        with open("antarctica_selection.geojson", "w") as f:
            json.dump(fc, f)
        status.value = "Saved to <code>antarctica_selection.geojson</code>"

    clear_btn.on_click(_clear)
    save_btn.on_click(_save)
    _select()

    box = widgets.VBox([m, widgets.HBox([picker, clear_btn, save_btn, status])])
    # Expose the live selection without requiring callers 
    box.get_selected_polygon = lambda: selected_polygon
    return box

# ------------------------------------------------------------
# Interactive explorer
# ------------------------------------------------------------

@dataclass
class _Product:
    """One product prepared for plotting: lazy data, its regional-mean series and the valid time steps."""

    da: xr.DataArray  # (time, y, x), sorted by time, lazy
    times: pd.DatetimeIndex  
    mean: np.ndarray  
    valid: np.ndarray  
    k: int  # block-averaging factor used for the maps


def _prepare_product(ds: xr.Dataset, spec: dict, max_cells: int) -> _Product:
    """Check one dataset, order it in time and compute its regional-mean series."""
    name, var = spec["name"], spec["var"]
    if ds is None or var not in ds.data_vars:
        raise ValueError(f"{name}: variable '{var}' not found.")
    da = ds[var]
    
    if da.sizes["x"] == 0 or da.sizes["y"] == 0:
        raise ValueError(f"{name}: no grid cells inside the selected area")

    # Regional mean on the full-resolution data
    mean = da.mean(("y", "x"), skipna=True).compute()
    # Block-averaging factor for the maps: keep the number of cells below max_cells
    k = max(1, int(np.ceil(np.sqrt(da.sizes["y"] * da.sizes["x"] / max_cells))))
    return _Product(
        da=da,
        times=pd.DatetimeIndex(mean["time"].values),
        mean=mean.to_numpy().astype(float),
        valid=np.isfinite(mean.to_numpy()),
        k=k,
    )


def _reference_times(products: dict, reference: str, max_steps: int) -> tuple[pd.DatetimeIndex, str]:
    """Time axis that drives the slider: one product's valid steps, or the union of all of them."""
    valid = {key: p.times[p.valid] for key, p in products.items()}
    if reference == "auto":  # UMelt is the product on display, but it only exists in the melt season
        reference = "umelt" if len(valid["umelt"]) else "union"
    if reference == "union":
        ref = valid["ssmis"].union(valid["ascat"]).union(valid["umelt"])
    elif reference in valid:
        ref = valid[reference]
    else:
        raise ValueError(f"reference must be 'auto', 'union' or one of {list(valid)}; got {reference!r}")
    ref = ref.unique().sort_values()
    if len(ref) > max_steps:  # keep the figure light: evenly spaced subset
        ref = ref[np.unique(np.linspace(0, len(ref) - 1, max_steps).round().astype(int))]
    return ref, reference


def _as_of(ref_times: pd.DatetimeIndex, src_times: pd.DatetimeIndex, max_age: pd.Timedelta) -> tuple[np.ndarray, np.ndarray]:
    """For every reference time, it finds the nearest source.

    Returns (position in src_times, age). The position is -1 where there is no observation within `max_age`.
    """
    ref, src = ref_times.to_numpy(), src_times.to_numpy()
    pos = np.searchsorted(src, ref, side="right") - 1  # -1 = before the first observation (or no observation at all)
    age = np.full(len(ref), np.timedelta64("NaT", "ns"), dtype="timedelta64[ns]")
    has = pos >= 0
    age[has] = ref[has] - src[pos[has]]
    ok = has & (age <= max_age.to_timedelta64())  # NaT compares False
    return np.where(ok, pos, -1), age


def _break_gaps(times: pd.DatetimeIndex, values: np.ndarray, gap: pd.Timedelta) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Insert a NaN in the middle of every hole longer than `gap`, so a line is not drawn across it."""
    if len(times) < 2:  # noqa: PLR2004
        return times, values
    idx = np.flatnonzero(np.diff(times.to_numpy()) > gap.to_timedelta64())
    if idx.size == 0:
        return times, values
    mid = times.to_numpy()[idx] + (times.to_numpy()[idx + 1] - times.to_numpy()[idx]) / 2
    t = np.concatenate([times.to_numpy(), mid])
    v = np.concatenate([values, np.full(idx.size, np.nan)])
    order = np.argsort(t)
    return pd.DatetimeIndex(t[order]), v[order]


def build_umelt_explorer(
    ssmis_ds: xr.Dataset,
    ascat_ds: xr.Dataset,
    umelt_ds: xr.Dataset,
    *,
    reference: str = "auto",
    max_steps: int = 120,
    max_cells: int = 8_000,
    max_age: str = "72h",
    gap: str = "7D",
    height: int = 1100,
) -> go.Figure:
    """Time slider driving three maps (SSMIS, ASCAT, UMelt) and their regional-mean time series.

    The three products are sampled at different times (SSMIS about daily, ASCAT every 1-2 days, UMelt sub-daily but
    only in the melt season), so the slider runs on ONE reference time axis and every map shows the latest
    observation of its product up to that time. Each map title gives the time of the observation shown and how
    much older than the slider it is; in the time series a dot marks that observation.

    Parameters
    ----------
    ssmis_ds, ascat_ds, umelt_ds : xr.Dataset
        As returned by `xr.open_zarr` on the downloads (variables `brightness_temperature`,
        `surface_backwards_scattering_coefficient_of_radar_wave` and `melt`, dimensions time / y / x).
    reference : {"auto", "umelt", "ssmis", "ascat", "union"}
        Which time axis the slider follows. "auto" is UMelt when it has data, otherwise the union of all three.
    max_steps : int
        Longest slider (default 120); a longer reference axis is thinned to evenly spaced steps.
    max_cells : int
        Largest map (cells) per frame (default 8 000, about 90 x 90). Bigger grids are block-averaged, so a binary
        melt map becomes a melt fraction. Every frame is embedded in the figure, so its size grows with
        steps x cells: roughly 4 kB per step per 1 000 cells, per map. Raise it for finer UMelt detail.
    max_age : str
        A product with no observation this recent before the slider time is shown blank ("no observation").
    gap : str
        Holes longer than this are not bridged by the time-series lines (e.g. UMelt in winter).
    height : int
        Figure height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        `fig.show()` in a notebook, `fig.write_html("explorer.html")` to share. `fig.layout.meta` records the
        reference axis, number of steps, block-averaging factors and the approximate size in MB.
    """
    max_age, gap = pd.Timedelta(max_age), pd.Timedelta(gap)
    sources = {"ssmis": ssmis_ds, "ascat": ascat_ds, "umelt": umelt_ds}
    prods = {key: _prepare_product(ds, UMELT_PRODUCTS[key], max_cells) for key, ds in sources.items()}
    ref_times, ref_used = _reference_times(prods, reference, max_steps)
    n = len(ref_times)

    # --- for every slider step: which observation of each product is shown, and the map arrays for those ---
    shown = {}
    for key, p in prods.items():
        src_idx = np.flatnonzero(p.valid)  # positions of the valid steps in p.da
        pos, age = _as_of(ref_times, p.times[src_idx], max_age)
        obs_idx = np.full(n, -1)  # for every slider step: index in p.da of the observation shown (-1 = none)
        obs_idx[pos >= 0] = src_idx[pos[pos >= 0]]
        needed = np.unique(obs_idx[obs_idx >= 0])  # distinct source frames used by the slider
        sub = p.da.isel(time=needed)
        if p.k > 1:  # block-average rather than subsample: a 0/1 melt map becomes a melt fraction
            sub = sub.coarsen(y=p.k, x=p.k, boundary="trim").mean()  # NaNs are skipped for float data
        obs_time = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        obs_time[obs_idx >= 0] = p.times.to_numpy()[obs_idx[obs_idx >= 0]]
        shown[key] = {
            "frames": sub.to_numpy().astype("float32"),
            "frame_of_step": np.where(obs_idx >= 0, np.searchsorted(needed, obs_idx), -1),
            "obs_time": obs_time,
            "age": age,
            "x": sub["x"].to_numpy() / 1e3,
            "y": sub["y"].to_numpy() / 1e3,
        }

    # float32 maps are embedded base64-encoded, which adds about 40% on top of the raw arrays
    approx_mb = n * sum(s["frames"].shape[1] * s["frames"].shape[2] * 4 for s in shown.values()) / 1e6 * 1.4
    if approx_mb > 50:  # noqa: PLR2004
        warnings.warn(
            f"The figure will embed about {approx_mb:.0f} MB ({n} steps). Lower max_cells or max_steps "
            "if it is slow to open.",
            stacklevel=2,
        )

    # --- colour limits: fixed over all frames so colours stay comparable while moving the slider ---
    limits = {}
    for key, s in shown.items():
        fixed = UMELT_PRODUCTS[key]["fixed_range"]
        finite = s["frames"][np.isfinite(s["frames"])]
        lo, hi = fixed or ((float(np.percentile(finite, 1)), float(np.percentile(finite, 99))) if finite.size else (0.0, 1.0))
        limits[key] = (lo, hi if hi > lo else lo + 1e-6)

    # --- figure skeleton: 3 maps on top, one full-width time-series row per product below ---
    fig = make_subplots(
        rows=4, cols=3,
        specs=[[{}, {}, {}], [{"colspan": 3}, None, None], [{"colspan": 3}, None, None], [{"colspan": 3}, None, None]],
        row_heights=[0.45, 0.18, 0.18, 0.19], vertical_spacing=0.065, horizontal_spacing=0.09,
        subplot_titles=[UMELT_PRODUCTS[k]["name"] for k in prods] + ["", "", ""],
    )  
    # Trace indices (the animation frames refer to them): 0-2 maps, 3-5 series, 6-8 observation dots, 9-11 time lines,
    # 12-14 invisible colour-bar keepers (added further down)
    for j, (key, s) in enumerate(shown.items(), start=1):
        spec = UMELT_PRODUCTS[key]
        z0 = s["frames"][s["frame_of_step"][0]] if s["frame_of_step"][0] >= 0 else np.full(s["frames"].shape[1:], np.nan, "float32")
        fig.add_trace(
            go.Heatmap(
                x=s["x"], y=s["y"], z=z0, coloraxis=f"coloraxis{j if j > 1 else ''}", hoverongaps=False,
                hovertemplate=f"x %{{x:.1f}} km<br>y %{{y:.1f}} km<br>%{{z:.3g}} {spec['unit']}<extra>{spec['name']}</extra>",
            ),
            row=1, col=j,
        )
    for r, (key, p) in enumerate(prods.items(), start=2):
        spec = UMELT_PRODUCTS[key]
        t_line, v_line = _break_gaps(p.times[p.valid], p.mean[p.valid], gap)
        fig.add_trace(
            go.Scatter(x=t_line, y=v_line, mode="lines", line={"color": spec["line"], "width": 1.4}, connectgaps=False,
                       hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.3g}<extra>" + spec["name"] + "</extra>"),
            row=r, col=1,
        )
    marker = {"size": 11, "line": {"color": "black", "width": 1.5}}
    obs_value = {}
    for r, (key, p) in enumerate(prods.items(), start=2):
        s = shown[key]
        # mean value of the observation each step shows (looked up by exact timestamp in the full series)
        lookup = pd.Series(p.mean, index=p.times)
        obs_value[key] = np.array([lookup.get(pd.Timestamp(t), np.nan) if not pd.isna(t) else np.nan for t in s["obs_time"]])
        fig.add_trace(
            go.Scatter(x=[s["obs_time"][0]], y=[obs_value[key][0]], mode="markers", hoverinfo="skip",
                       marker={**marker, "color": UMELT_PRODUCTS[key]["line"]}),
            row=r, col=1,
        )
    ranges = {}
    for r, (key, p) in enumerate(prods.items(), start=2):
        finite = p.mean[p.valid]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        pad = 0.08 * (hi - lo or 1.0)
        ranges[key] = (lo - pad, hi + pad)
        fig.add_trace(
            go.Scatter(x=[ref_times[0]] * 2, y=list(ranges[key]), mode="lines", hoverinfo="skip",
                       line={"color": "red", "width": 1.2, "dash": "dot"}),
            row=r, col=1,
        )

    # Traces 12-14: an invisible one-row heatmap per colour axis. A map with no observation is all NaN, and plotly then
    # drops its colour bar; these keep the bars on screen (opacity 0: nothing is drawn, no hover).
    for j, key in enumerate(prods, start=1):
        xs, ys = shown[key]["x"], shown[key]["y"]
        fig.add_trace(
            go.Heatmap(x=[xs[0], xs[0] + 1e-3], y=[ys[0]], z=[list(limits[key])], opacity=0, hoverinfo="skip",
                       coloraxis=f"coloraxis{j if j > 1 else ''}"),
            row=1, col=j,
        )

    # --- animation frames: only the maps, the dots, the time lines and the three map titles change ---
    base_annotations = [a.to_plotly_json() for a in fig.layout.annotations]

    def title(key: str, i: int) -> str:
        """Map title: product name, time of the observation shown and its age relative to the slider."""
        name, s = UMELT_PRODUCTS[key]["name"], shown[key]
        if s["frame_of_step"][i] < 0:
            limit = f"{max_age / pd.Timedelta(hours=1):g} h"
            return f"<b>{name}</b><br><span style='color:firebrick'>no observation within {limit}</span>"
        hours = s["age"][i] / np.timedelta64(1, "h")
        colour = "dimgray" if hours <= 24 else ("darkorange" if hours <= 48 else "firebrick")  # noqa: PLR2004
        older = f" ({hours:.0f} h earlier)" if hours >= 0.5 else ""  # noqa: PLR2004
        return f"<b>{name}</b><br><span style='color:{colour}'>{pd.Timestamp(s['obs_time'][i]):%Y-%m-%d %H:%M}{older}</span>"

    frames = []
    for i in range(n):
        maps = []
        for key, s in shown.items():
            fi = s["frame_of_step"][i]
            maps.append(go.Heatmap(z=s["frames"][fi] if fi >= 0 else np.full(s["frames"].shape[1:], np.nan, "float32")))
        dots = [go.Scatter(x=[shown[k]["obs_time"][i]], y=[obs_value[k][i]]) for k in prods]
        lines = [go.Scatter(x=[ref_times[i]] * 2) for _ in prods]
        annotations = [dict(a) for a in base_annotations]
        for a, key in zip(annotations[:3], prods):
            a["text"] = title(key, i)
        frames.append(go.Frame(name=str(i), traces=[0, 1, 2, 6, 7, 8, 9, 10, 11], data=maps + dots + lines, layout={"annotations": annotations}))
    fig.frames = frames
    for a, key in zip(fig.layout.annotations[:3], prods):
        a.text = title(key, 0)

    # --- axes ---
    for j, key in enumerate(prods, start=1):  # square pixels on the maps
        # fixed ranges (the extent of the grid): a map with no observation is all NaN, and without them plotly
        # would lose the axis range and collapse the panel
        xs, ys = shown[key]["x"], shown[key]["y"]
        dx = float(np.abs(np.diff(xs)).mean()) if len(xs) > 1 else 1.0
        dy = float(np.abs(np.diff(ys)).mean()) if len(ys) > 1 else 1.0
        fig.update_xaxes(title_text="x [km]", constrain="domain", range=[xs.min() - dx / 2, xs.max() + dx / 2], row=1, col=j)
        fig.update_yaxes(scaleanchor="x" if j == 1 else f"x{j}", scaleratio=1, constrain="domain",
                         range=[ys.min() - dy / 2, ys.max() + dy / 2], row=1, col=j)
    fig.update_yaxes(title_text="y [km]", row=1, col=1)
    for r, key in enumerate(prods, start=2):
        fig.update_yaxes(title_text=UMELT_PRODUCTS[key]["ylabel"], range=ranges[key], row=r, col=1)
        fig.update_xaxes(matches="x4", showticklabels=(r == 4), row=r, col=1)  # noqa: PLR2004
    fig.update_yaxes(range=[0, max(ranges["umelt"][1], 1e-3)], row=4, col=1)  # a melt fraction cannot be negative
    fig.update_xaxes(title_text="Time", row=4, col=1)

    # --- colour bars to the right of each map, positioned from the actual subplot domains ---
    y_dom = fig.layout.yaxis.domain
    coloraxes = {}
    for j, key in enumerate(prods, start=1):
        spec, x_dom = UMELT_PRODUCTS[key], fig.layout[f"xaxis{j if j > 1 else ''}"].domain
        coloraxes[f"coloraxis{j if j > 1 else ''}"] = {
            "colorscale": spec["colorscale"], "reversescale": spec["reverse"], "cmin": limits[key][0], "cmax": limits[key][1],
            "colorbar": {"title": {"text": spec["unit"]}, "x": x_dom[1] + 0.005, "xanchor": "left",
                         "y": sum(y_dom) / 2, "len": 0.8 * (y_dom[1] - y_dom[0]), "thickness": 12},
        } 

    # --- slider and play button ---
    anim = {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}
    labels = ref_times.strftime("%Y-%m-%d" if (ref_times == ref_times.normalize()).all() else "%Y-%m-%d %H:%M")
    fig.update_layout(
        **coloraxes,
        template="plotly_white", height=height, showlegend=False,
        margin={"t": 100, "b": 200, "l": 80, "r": 120},
        meta={"reference": ref_used, "steps": n, "coarsen": {k: p.k for k, p in prods.items()}, "approx_mb": round(approx_mb, 1)},
        sliders=[{
            "active": 0, "pad": {"t": 70}, "x": 0.12, "len": 0.88,
            "currentvalue": {"prefix": "Time: ", "font": {"size": 14, "color": "#2a3f5f"}},
            "steps": [{"method": "animate", "label": lab, "args": [[str(i)], anim]} for i, lab in enumerate(labels)],
            # with many steps the tick labels would overlap: hide them, the current value is shown above the slider
            **({"font": {"color": "rgba(0,0,0,0)"}, "ticklen": 0, "tickcolor": "rgba(0,0,0,0)"} if n > 25 else {}),  # noqa: PLR2004
        }],
        updatemenus=[{
            "type": "buttons", "direction": "left", "x": 0.0, "y": -0.145, "showactive": False,
            "buttons": [
                {"label": "Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 150, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]},
                {"label": "Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]},
            ],
        }],
    )  
    return fig



