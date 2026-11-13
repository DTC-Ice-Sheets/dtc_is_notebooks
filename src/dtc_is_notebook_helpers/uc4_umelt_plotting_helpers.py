"""Helper functions for plotting UMelt data."""

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from dtc_query_client import ApiClient, DataDownloadRequest, DatasetID, Epsg4326PolygonJson, GenericApi
from dtc_query_client.helpers import wait_for_job
from ipyleaflet import DrawControl, FullScreenControl, Map, ScaleControl, basemaps, projections
from plotly.subplots import make_subplots
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform

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



def get_date_picker(desc: str = "Select a date:", default_date: date = date(2018, 1, 1)) -> widgets.DatePicker:
    """Create a widget to select a date between 2016 and 2021."""
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
    print(f"Data download job {job_id} completed successfully.") # noqa: T201
    return result.outputs["main"]["output_path"]


# Equal-area projection for South Polar region (EASE-Grid 2.0 South) used for area calculation
_to_ea = Transformer.from_crs("EPSG:4326", "EPSG:6932", always_xy=True).transform


def _area_km2(geom: Polygon) -> float:
    """Compute the area of a polygon in km² using an equal-area projection."""
    return transform(_to_ea, geom).area / 1e6


def _make_map(use_esri: bool) -> Map:
    """Make a map of Antarctica (polar stereographic, EPSG:3031) with scale and full-screen controls."""
    if use_esri:
        base, crs = basemaps.Esri.AntarcticBasemap, projections.EPSG3031.ESRIBasemap
    else:
        base, crs = basemaps.NASAGIBS.BlueMarble3031, projections.EPSG3031.NASAGIBS
    m = Map(center=(-90, 0), zoom=1, crs=crs, basemap=base, scroll_wheel_zoom=True,
            layout=widgets.Layout(height="650px"))
    m.add_control(ScaleControl(position="bottomleft"))
    m.add_control(FullScreenControl())
    return m


def _make_draw_control() -> DrawControl:
    """Drawing tools: polygons and rectangles only."""
    shape_style = {"color": "#e4572e", "fillColor": "#e4572e", "fillOpacity": 0.25, "weight": 2}
    return DrawControl(
        polygon={"shapeOptions": shape_style, "allowIntersection": False},
        rectangle={"shapeOptions": shape_style},
        polyline={}, circle={}, circlemarker={}, marker={},
        edit=True, remove=True,
    )


def _polygons_from(features: list[dict]) -> list[Polygon]:
    """Return the drawn polygons as shapely geometries (lon/lat). Features that are not polygons are ignored."""
    return [shape(f["geometry"]) for f in features if (f.get("geometry") or {}).get("type") == "Polygon"]


def _index_to_select(
    polygons: list[Polygon], previous_count: int, kept: Polygon | None, old_index: int | None
) -> int | None:
    """Which polygon is selected after the list of polygons changed (None when there are none).

    1. a polygon that was just drawn becomes the selection;
    2. otherwise the polygon that was selected stays (found by its geometry, since deleting another shifts the indices);
    3. if it was edited or deleted, the selection keeps its position in the list.
    """
    if not polygons:
        return None
    if len(polygons) > previous_count:
        return len(polygons) - 1
    for i, polygon in enumerate(polygons):
        if kept is not None and polygon.equals(kept):
            return i
    if old_index is not None and old_index < len(polygons):
        return old_index
    return len(polygons) - 1


def _write_geojson(polygon: Polygon, path: str) -> None:
    """Save one polygon as a GeoJSON FeatureCollection."""
    feature = {"type": "Feature", "properties": {}, "geometry": mapping(polygon)}
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": [feature]}, f)


class PolygonSelector(widgets.VBox):
    """Create a map to draw polygons on, a dropdown to choose which one is selected, and Clear / Save buttons.

    `get_selected_polygon()` returns the selection as a shapely geometry in lon/lat (EPSG:4326), or None.
    """

    def __init__(self, use_esri: bool = False) -> None:
        super().__init__()
        self._polygons: list[Polygon] = []  # all drawn polygons
        self._selected: Polygon | None = None  # the polygon that is passed on to the next step

        self._draw = _make_draw_control()
        self._map = _make_map(use_esri)
        self._map.add_control(self._draw)
        self._picker = widgets.Dropdown(description="Polygon:", options=[], layout=widgets.Layout(width="300px"))
        self._status = widgets.HTML()
        clear_btn = widgets.Button(description="Clear all", icon="trash", button_style="warning")
        save_btn = widgets.Button(description="Save GeoJSON", icon="save", button_style="success")

        self._draw.observe(self._sync, names="data")
        self._picker.observe(self._select, names="value")
        clear_btn.on_click(self._clear)
        save_btn.on_click(self._save)

        self.children = [self._map, widgets.HBox([self._picker, clear_btn, save_btn, self._status])]

    def get_selected_polygon(self) -> Polygon | None:
        """Return the polygon chosen in the dropdown (lon/lat, EPSG:4326), or None if nothing is drawn."""
        return self._selected

    def _sync(self, *_: object) -> None:
        """Rebuild the polygon list and the dropdown from the draw control's features."""
        previous, kept, old_index = len(self._polygons), self._selected, self._picker.value
        self._polygons = _polygons_from(self._draw.data)

        self._picker.unobserve(self._select, names="value")  # no selection events while the options are rebuilt
        labels = [f"Polygon {i + 1}  ({_area_km2(g):,.0f} km²)" for i, g in enumerate(self._polygons)]
        self._picker.options = [(label, i) for i, label in enumerate(labels)]
        index = _index_to_select(self._polygons, previous, kept, old_index)
        if index is not None:
            self._picker.value = index
        self._picker.observe(self._select, names="value")
        self._select()

    def _select(self, _: object = None) -> None:
        """Link the selection to the polygon chosen in the dropdown."""
        index = self._picker.value
        self._selected = None if index is None else self._polygons[index]

    def _clear(self, _: widgets.Button) -> None:
        """Remove all polygons."""
        self._draw.data = []
        self._status.value = ""

    def _save(self, _: widgets.Button) -> None:
        """Save the selected polygon to a GeoJSON file."""
        if self._selected is None:
            self._status.value = "<span style='color:#b00'>Draw a polygon first.</span>"
        else:
            _write_geojson(self._selected, "selected_polygon.geojson")
            self._status.value = "Saved to <code>selected_polygon.geojson</code>"


def polygon_selector(use_esri: bool = False) -> widgets.VBox:
    """Create a widget for drawing and selecting polygons on a map.

    Set `use_esri=True` if the NASA GIBS tiles don't load for you.
    """
    return PolygonSelector(use_esri)


# ------------------------------------------------------------
# Interactive explorer
# ------------------------------------------------------------
#
# The figure has 3 maps on top and one time-series row per product below. Its traces are added in this order, and the
# animation frames update them by position:
#   0-2 maps | 3-5 regional-mean lines | 6-8 observation dots | 9-11 time lines | 12-14 colour-bar keepers
ANIMATED_TRACES = [0, 1, 2, 6, 7, 8, 9, 10, 11]  # everything that moves with the slider
_ANIMATE_NOW = {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}


@dataclass
class _Product:
    """Define one product prepared for plotting."""

    da: xr.DataArray  # (time, y, x), lazy
    times: pd.DatetimeIndex
    mean: np.ndarray  # regional mean per time step, computed on the full-resolution data
    valid: np.ndarray  # True where the time step holds data
    k: int  # block-averaging factor of the maps


@dataclass
class _Shown:
    """Define what one product displays on the slider."""

    frames: np.ndarray  # (n_frames, ny, nx) block-averaged maps
    frame_of_step: np.ndarray  # slider step -> index into `frames` (-1 = no observation within max_age)
    obs_time: np.ndarray  # slider step -> time of the observation shown (NaT = none)
    obs_value: np.ndarray  # slider step -> regional mean of that observation (NaN = none)
    age: np.ndarray  # slider step -> how much older than the slider time that observation is
    x: np.ndarray  # map x coordinates [km]
    y: np.ndarray  # map y coordinates [km]

    def map_at(self, step: int) -> np.ndarray:
        """Return the map shown at a slider step."""
        i = self.frame_of_step[step]
        return self.frames[i] if i >= 0 else np.full(self.frames.shape[1:], np.nan, "float32")


def _suffix(n: int) -> str:
    """Return the axes number suffix."""
    return "" if n == 1 else str(n)


def _prepare_product(ds: xr.Dataset, spec: dict, max_cells: int) -> _Product:
    """Prepare a product for plotting.

    Computes its regional-mean series and the block-averaging factor of its maps.
    """
    da = ds[spec["var"]]
    if da.sizes["x"] == 0 or da.sizes["y"] == 0:
        raise ValueError(f"{spec['name']}: no grid cells inside the selected area")
    mean = da.mean(("y", "x"), skipna=True).compute()  # on the full-resolution data
    return _Product(
        da=da,
        times=pd.DatetimeIndex(mean["time"].values),
        mean=mean.to_numpy().astype(float),
        valid=np.isfinite(mean.to_numpy()),
        k=max(1, int(np.ceil(np.sqrt(da.sizes["y"] * da.sizes["x"] / max_cells)))),
    )


def _reference_times(products: dict, max_steps: int) -> pd.DatetimeIndex:
    """Define the time axis that drives the slider."""
    ref = pd.DatetimeIndex(np.concatenate([p.times[p.valid] for p in products.values()])).unique().sort_values()
    if len(ref) > max_steps:  # keep the figure light: evenly spaced subset
        ref = ref[np.unique(np.linspace(0, len(ref) - 1, max_steps).round().astype(int))]
    return ref


def _as_of(ref_times: pd.DatetimeIndex,
           src_times: pd.DatetimeIndex,
           max_age: pd.Timedelta) -> tuple[np.ndarray, np.ndarray]:
    """Find the latest source time at or before it for each reference time frame.

    Returns (position in src_times, age). The position is -1 where there is no observation within `max_age`.
    """
    ref, src = ref_times.to_numpy(), src_times.to_numpy()
    pos = np.searchsorted(src, ref, side="right") - 1  # -1 = before the first observation (or no observation at all)
    age = np.full(len(ref), np.timedelta64("NaT", "ns"), dtype="timedelta64[ns]")
    has = pos >= 0
    age[has] = ref[has] - src[pos[has]]
    ok = has & (age <= max_age.to_timedelta64())  # NaT compares False
    return np.where(ok, pos, -1), age


def _observations_shown(p: _Product, ref_times: pd.DatetimeIndex, max_age: pd.Timedelta) -> _Shown:
    """Define which observation of this product is displayed for each time step."""
    valid_idx = np.flatnonzero(p.valid)  # positions of the valid steps in p.da
    pos, age = _as_of(ref_times, p.times[valid_idx], max_age)
    obs_idx = np.full(len(ref_times), -1)  # position in p.da of the observation shown at each step (-1 = none)
    obs_idx[pos >= 0] = valid_idx[pos[pos >= 0]]
    has_obs = obs_idx >= 0

    needed = np.unique(obs_idx[has_obs])  # the distinct source frames the slider uses
    maps = p.da.isel(time=needed).coarsen(y=p.k, x=p.k, boundary="trim").mean()  # block average: NaNs are skipped

    obs_time = np.full(len(obs_idx), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    obs_time[has_obs] = p.times.to_numpy()[obs_idx[has_obs]]
    obs_value = np.full(len(obs_idx), np.nan)
    obs_value[has_obs] = p.mean[obs_idx[has_obs]]
    return _Shown(
        frames=maps.to_numpy().astype("float32"),
        frame_of_step=np.where(has_obs, np.searchsorted(needed, obs_idx), -1),
        obs_time=obs_time,
        obs_value=obs_value,
        age=age,
        x=maps["x"].to_numpy() / 1e3,
        y=maps["y"].to_numpy() / 1e3,
    )


def _break_gaps(times: pd.DatetimeIndex, values: np.ndarray, gap: pd.Timedelta) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Insert a NaN in the middle of every hole longer than `gap`, so a line is not drawn across it."""
    t = times.to_numpy()
    before = np.flatnonzero(np.diff(t) > gap.to_timedelta64())  # the step before each hole
    middle = t[before] + (t[before + 1] - t[before]) / 2
    all_t = np.concatenate([t, middle])
    all_v = np.concatenate([values, np.full(len(before), np.nan)])
    order = np.argsort(all_t)
    return pd.DatetimeIndex(all_t[order]), all_v[order]


def _color_limits(spec: dict, frames: np.ndarray) -> tuple[float, float]:
    """Define the colour limits, fixed over all frames so that colours stay comparable while moving the slider."""
    finite = frames[np.isfinite(frames)]
    if spec["fixed_range"]:
        lo, hi = spec["fixed_range"]
    elif finite.size:
        lo, hi = float(finite.min()), float(finite.max())
    else:  # a product that is never observed
        lo, hi = 0.0, 1.0
    return lo, hi if hi > lo else lo + 1e-6


def _series_range(values: np.ndarray) -> tuple[float, float]:
    """Return the Y range of a regional-mean series, with some head-room so the observation dots are not clipped."""
    lo, hi = (float(values.min()), float(values.max())) if values.size else (0.0, 1.0)
    pad = 0.08 * (hi - lo or 1.0)
    return lo - pad, hi + pad


# Functions to build the figure


def _make_figure_skeleton(names: list[str]) -> go.Figure:
    """Make the skeleton of the figure with 3 maps on top, then one full-width time-series row per product."""
    full_row = [{"colspan": 3}, None, None]
    return make_subplots(
        rows=4, cols=3,
        specs=[[{}, {}, {}], full_row, full_row, full_row],
        row_heights=[0.45, 0.18, 0.18, 0.19], vertical_spacing=0.065, horizontal_spacing=0.09,
        subplot_titles=names + ["", "", ""],
    )


def _add_map_traces(fig: go.Figure, shown: dict) -> None:
    """Add the map traces for the first slider step."""
    for col, (key, s) in enumerate(shown.items(), start=1):
        spec = UMELT_PRODUCTS[key]
        hover = f"x %{{x:.1f}} km<br>y %{{y:.1f}} km<br>%{{z:.3g}} {spec['unit']}<extra>{spec['name']}</extra>"
        heatmap = go.Heatmap(x=s.x, y=s.y, z=s.map_at(0), coloraxis=f"coloraxis{_suffix(col)}",
                             hoverongaps=False, hovertemplate=hover)
        fig.add_trace(heatmap, row=1, col=col)


def _add_series_traces(fig: go.Figure, products: dict, shown: dict, ranges: dict, first_time: pd.Timestamp,
                       gap: pd.Timedelta) -> None:
    """Add series for traces 3-11.

    It includes the regional-mean lines, then the observation dots, then the red time lines.
    """
    dot_style = {"size": 11, "line": {"color": "black", "width": 1.5}}
    lines, dots, time_lines = [], [], []
    for key, p in products.items():
        spec, s = UMELT_PRODUCTS[key], shown[key]
        t, v = _break_gaps(p.times[p.valid], p.mean[p.valid], gap)
        lines.append(go.Scatter(
            x=t, y=v, mode="lines", line={"color": spec["line"], "width": 1.4}, connectgaps=False,
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.3g}<extra>" + spec["name"] + "</extra>",
        ))
        dots.append(go.Scatter(
            x=[s.obs_time[0]], y=[s.obs_value[0]], mode="markers", hoverinfo="skip",
            marker={**dot_style, "color": spec["line"]},
        ))
        time_lines.append(go.Scatter(
            x=[first_time] * 2, y=list(ranges[key]), mode="lines", hoverinfo="skip",
            line={"color": "red", "width": 1.2, "dash": "dot"},
        ))
    for group in (lines, dots, time_lines):
        for row, trace in enumerate(group, start=2):
            fig.add_trace(trace, row=row, col=1)


def _add_colorbar_keepers(fig: go.Figure, shown: dict, limits: dict) -> None:
    """Add an invisible heatmap per colour axis for traces 12-14.

    A map with no observation is all NaN, and plotly then drops its colour bar. These keep the bars on screen
    (opacity 0: nothing is drawn, no hover).
    """
    for col, (key, s) in enumerate(shown.items(), start=1):
        keeper = go.Heatmap(x=[s.x[0], s.x[0] + 1e-3], y=[s.y[0]], z=[list(limits[key])], opacity=0, hoverinfo="skip",
                            coloraxis=f"coloraxis{_suffix(col)}")
        fig.add_trace(keeper, row=1, col=col)


# Titles and animation frames

def _map_title(key: str, s: _Shown, step: int, max_age: pd.Timedelta) -> str:
    """Define the title of a map for a slider step."""
    name = UMELT_PRODUCTS[key]["name"]
    msg = f"<b>{name}</b><br><span style='color:firebrick'>no data within {max_age / pd.Timedelta(hours=1):g} h</span>"
    if s.frame_of_step[step] < 0:
        return msg
    return f"<b>{name}</b><br><span style='color:dimgray'>{pd.Timestamp(s.obs_time[step]):%Y-%m-%d %H:%M}</span>"


def _annotations_with_titles(base: list[dict], titles: list[str]) -> list[dict]:
    """Build the annotations for the animation frames, with the map titles updated for each step."""
    annotations = [dict(a) for a in base]
    for annotation, title in zip(annotations, titles, strict=True):
        annotation["text"] = title
    return annotations


def _make_frames(fig: go.Figure, shown: dict, ref_times: pd.DatetimeIndex, max_age: pd.Timedelta) -> list[go.Frame]:
    """Make the animation frames for the figure."""
    base = [a.to_plotly_json() for a in fig.layout.annotations]
    frames = []
    for step in range(len(ref_times)):
        maps = [go.Heatmap(z=s.map_at(step)) for s in shown.values()]
        dots = [go.Scatter(x=[s.obs_time[step]], y=[s.obs_value[step]]) for s in shown.values()]
        time_lines = [go.Scatter(x=[ref_times[step]] * 2) for _ in shown]
        titles = [_map_title(key, s, step, max_age) for key, s in shown.items()]
        frames.append(go.Frame(
            name=str(step), traces=ANIMATED_TRACES, data=maps + dots + time_lines,
            layout={"annotations": _annotations_with_titles(base, titles)},
        ))
    return frames


# Axes, colour bars, slider

def _padded_range(coords: np.ndarray) -> list[float]:
    """Compute the axes range."""
    half_cell = np.abs(np.diff(coords)).mean() / 2 if len(coords) > 1 else 0.5
    return [coords.min() - half_cell, coords.max() + half_cell]


def _style_axes(fig: go.Figure, shown: dict, ranges: dict) -> None:
    """Style the axes of the figure."""
    # Fixed map ranges (the grid extent): a map with no observation is all NaN, and without them plotly would lose
    # the axis range and collapse the panel.
    for col, s in enumerate(shown.values(), start=1):
        fig.update_xaxes(title_text="x [km]", constrain="domain", range=_padded_range(s.x), row=1, col=col)
        fig.update_yaxes(scaleanchor="x" + _suffix(col), scaleratio=1, constrain="domain",
                         range=_padded_range(s.y), row=1, col=col)
    fig.update_yaxes(title_text="y [km]", row=1, col=1)

    for row, key in enumerate(shown, start=2):
        fig.update_yaxes(title_text=UMELT_PRODUCTS[key]["ylabel"], range=ranges[key], row=row, col=1)
        fig.update_xaxes(matches="x4", showticklabels=(row == 4), row=row, col=1)  # noqa: PLR2004
    fig.update_yaxes(range=[0, max(ranges["umelt"][1], 1e-3)], row=4, col=1)  # a melt fraction cannot be negative
    fig.update_xaxes(title_text="Time", row=4, col=1)


def _coloraxes(fig: go.Figure, limits: dict) -> dict:
    """Define axes colorscales, limits and colour bars."""
    y_dom = fig.layout.yaxis.domain
    coloraxes = {}
    for col, (key, (cmin, cmax)) in enumerate(limits.items(), start=1):
        spec = UMELT_PRODUCTS[key]
        x_right = fig.layout["xaxis" + _suffix(col)].domain[1]
        coloraxes["coloraxis" + _suffix(col)] = {
            "colorscale": spec["colorscale"], "reversescale": spec["reverse"], "cmin": cmin, "cmax": cmax,
            "colorbar": {"title": {"text": spec["unit"]}, "x": x_right + 0.005, "xanchor": "left",
                         "y": sum(y_dom) / 2, "len": 0.8 * (y_dom[1] - y_dom[0]), "thickness": 12},
        }
    return coloraxes


def _slider(ref_times: pd.DatetimeIndex) -> dict:
    """Return the time slider."""
    only_dates = (ref_times == ref_times.normalize()).all()
    labels = ref_times.strftime("%Y-%m-%d" if only_dates else "%Y-%m-%d %H:%M")
    steps = [{"method": "animate", "label": label, "args": [[str(i)], _ANIMATE_NOW]} for i, label in enumerate(labels)]
    slider = {
        "active": 0, "pad": {"t": 70}, "x": 0.12, "len": 0.88,
        "currentvalue": {"prefix": "Time: ", "font": {"size": 14, "color": "#2a3f5f"}},
        "steps": steps,
    }
    if len(ref_times) > 25:  # noqa: PLR2004
        # with many steps the tick labels would overlap: hide them, the current value is shown above the slider
        slider.update({"font": {"color": "rgba(0,0,0,0)"}, "ticklen": 0, "tickcolor": "rgba(0,0,0,0)"})
    return slider


def _play_buttons() -> dict:
    """Return the play / Pause buttons."""
    play = {"frame": {"duration": 150, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}
    pause = {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}
    return {
        "type": "buttons", "direction": "left", "x": 0.0, "y": -0.145, "showactive": False,
        "buttons": [{"label": "Play", "method": "animate", "args": [None, play]},
                    {"label": "Pause", "method": "animate", "args": [[None], pause]}],
    }



def build_umelt_explorer(
    ssmis_ds: xr.Dataset,
    ascat_ds: xr.Dataset,
    umelt_ds: xr.Dataset,
    *,
    max_steps: int = 120,
    max_cells: int = 8_000,
    max_age: str = "72h",
    gap: str = "7D",
    height: int = 1100,
) -> go.Figure:
    """Build an interactive explorer for the datasets (SSMIS, ASCAT, UMelt) and their regional-mean time series.

    The three products are sampled at different times (SSMIS about daily, ASCAT every 1-2 days, UMelt sub-daily but
    only in the melt season), so the slider runs on ONE reference time axis and every map shows the latest
    observation of its product up to that time. Each map title gives the time of the observation shown and how
    much older than the slider it is; in the time series a dot marks that observation.

    Parameters
    ----------
    ssmis_ds, ascat_ds, umelt_ds : xr.Dataset
        As returned by `xr.open_zarr` on the downloads (variables `brightness_temperature`,
        `surface_backwards_scattering_coefficient_of_radar_wave` and `melt`, dimensions time / y / x).
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
        number of steps and the block-averaging factors that were used.
    """
    max_age, gap = pd.Timedelta(max_age), pd.Timedelta(gap)

    # 1. data: one prepared product per dataset, the slider's time axis, and what each product shows at every step
    sources = {"ssmis": ssmis_ds, "ascat": ascat_ds, "umelt": umelt_ds}
    products = {key: _prepare_product(ds, UMELT_PRODUCTS[key], max_cells) for key, ds in sources.items()}
    ref_times = _reference_times(products, max_steps)
    shown = {key: _observations_shown(p, ref_times, max_age) for key, p in products.items()}
    limits = {key: _color_limits(UMELT_PRODUCTS[key], s.frames) for key, s in shown.items()}
    ranges = {key: _series_range(p.mean[p.valid]) for key, p in products.items()}

    # 2. figure: traces, then the animation, then the look
    fig = _make_figure_skeleton([UMELT_PRODUCTS[key]["name"] for key in products])
    _add_map_traces(fig, shown)
    _add_series_traces(fig, products, shown, ranges, ref_times[0], gap)
    _add_colorbar_keepers(fig, shown, limits)
    fig.frames = _make_frames(fig, shown, ref_times, max_age)
    for annotation, (key, s) in zip(fig.layout.annotations, shown.items(), strict=True):
        annotation.text = _map_title(key, s, 0, max_age)

    _style_axes(fig, shown, ranges)
    fig.update_layout(
        **_coloraxes(fig, limits),
        template="plotly_white", height=height, showlegend=False,
        margin={"t": 100, "b": 200, "l": 80, "r": 120},
        meta={"reference": "union", "steps": len(ref_times), "coarsen": {key: p.k for key, p in products.items()}},
        sliders=[_slider(ref_times)],
        updatemenus=[_play_buttons()],
    )
    return fig
