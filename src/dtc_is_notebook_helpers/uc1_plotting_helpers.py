"""
Use Case 1 - Hydropower. The figures behind the demonstrator notebook.

Two kinds of helper live here: the building blocks for drawing a fitted field
over the catchment and for labelling a melt-season axis, and the notebook's
larger figures. Each figure function returns its figure rather than showing
it, so the notebook decides when it is drawn.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from matplotlib.axes import Axes
from matplotlib.collections import Collection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

from dtc_is_notebook_helpers import uc1_hydro_helpers as uc1

# House style: measured is always blue, modelled always orange.
MEASURED = "#2a78d6"
MODELLED = "#eb6834"
ALTERNATIVE = "#1baf7a"
# One colour per forcing, so that a forcing keeps its colour from figure to figure.
FORCING_COLOURS = {"RACMO": MODELLED, "MAR": ALTERNATIVE, "RACMO55": "#8a6bbf"}
# The observed record first, then one colour per scenario.
SCENARIO_COLOURS = (MEASURED, MODELLED, ALTERNATIVE, "#4a3aa7", "#e34948", "#eda100")
# The lake record is clipped below this cover.
LAKE_FLOOR = 1e-4

DISCHARGE_LABEL = "discharge (m$^3$/s)"


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def basin_path(rings: list[np.ndarray], scale: float = 1.0) -> MplPath:
    """
    Join the outline rings into one matplotlib path, holes included.

    Parameters
    ----------
    rings : list of numpy.ndarray
        The outline, as `load_basin` returns it.
    scale : float, optional
        Factor applied to every coordinate.

    Returns
    -------
    matplotlib.path.Path
        The outline as one path.
    """
    return MplPath.make_compound_path(
        *[MplPath(np.asarray(ring, dtype=float) * scale, closed=True)
          for ring in rings])


def mesh_edges(ax: Axes, vertices: np.ndarray, triangles: np.ndarray,
               rings: list[np.ndarray] | None = None, colour: str = "white",
               alpha: float = 0.55) -> list:
    """
    Draw a mesh's triangulation over whatever is already on the axes.

    `mesh_map` draws a field and can draw the triangulation with it, but only
    where the field sits on the mesh itself. A field returned by the deployed
    module arrives interpolated onto a regular grid, and the mesh is still
    worth showing: the fit was done on it, and the detail in the colour is
    interpolation between its nodes rather than measurement.

    Coordinates are divided by a thousand, to match `mesh_map`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    vertices : numpy.ndarray
        Node coordinates in metres, with x and y in the first two columns.
    triangles : numpy.ndarray
        Triangles as node index triples.
    rings : list of numpy.ndarray, optional
        The outline to clip to, as `load_basin` returns it. No clipping by
        default.
    colour : str, optional
        Colour of the edges.
    alpha : float, optional
        Opacity of the edges.

    Returns
    -------
    list of matplotlib.lines.Line2D
        The drawn edges.
    """
    xy = np.asarray(vertices, dtype=float)[:, :2] / 1000.0
    triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1],
                                       np.asarray(triangles, dtype=int))
    drawn = ax.triplot(triangulation, color=colour, lw=0.35, alpha=alpha, zorder=3)
    if rings:
        path = basin_path(rings, scale=1 / 1000.0)
        for artist in drawn:
            artist.set_clip_path(PathPatch(path, transform=ax.transData,
                                           facecolor="none", edgecolor="none"))
    return drawn


def mesh_map(ax: Axes, vertices: np.ndarray, values: np.ndarray,
             rings: list[np.ndarray] | None = None,
             triangles: np.ndarray | None = None, cmap: object = "viridis",
             edges: bool = False, edge_colour: str = "white",
             edge_alpha: float = 0.55, outline: str = "#55606e",
             margin_km: float = 4.0) -> Collection:
    """
    Draw a fitted field over the mesh, clipped to the catchment.

    The field is continuous and the mesh nodes are only where it is evaluated,
    so it is drawn as a surface rather than as points. Clipping to the outline
    keeps the picture to the catchment the use case is about; without it the
    picture covers roughly twice the area, most of it the mesh's margin.

    Pass `triangles` - the module's own triangulation, as node index triples -
    to draw the mesh the fit actually used. Without it the nodes are
    re-triangulated here, which matches on this mesh but is a reconstruction.
    `edges=True` draws the triangulation over the field, which is worth it
    where the point of the figure is that everything lands on one mesh. The
    edges have to stay legible without competing with the field, so the colour
    follows the ramp underneath: white over a mostly dark one, grey over a
    mostly pale one.

    Coordinates are divided by a thousand, so the axes are in kilometres.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    vertices : numpy.ndarray
        Node coordinates in metres, with x and y in the first two columns.
    values : numpy.ndarray
        The field's value at each node.
    rings : list of numpy.ndarray, optional
        The outline to clip to, as `load_basin` returns it. No clipping by
        default.
    triangles : numpy.ndarray, optional
        Triangles as node index triples.
    cmap : str or matplotlib.colors.Colormap, optional
        Colour map for the field.
    edges : bool, optional
        Whether to draw the triangulation over the field.
    edge_colour : str, optional
        Colour of the edges.
    edge_alpha : float, optional
        Opacity of the edges.
    outline : str, optional
        Colour of the catchment outline.
    margin_km : float, optional
        Space left around the outline, in kilometres.

    Returns
    -------
    matplotlib.collections.Collection
        The drawn surface, for a colour bar.
    """
    xy = np.asarray(vertices, dtype=float)[:, :2] / 1000.0
    triangulation = mtri.Triangulation(
        xy[:, 0], xy[:, 1],
        None if triangles is None else np.asarray(triangles, dtype=int))
    surface = ax.tripcolor(triangulation, np.asarray(values, dtype=float),
                           cmap=cmap, shading="gouraud")
    ax.set_aspect("equal")
    drawn = [surface]
    if edges:
        drawn.extend(ax.triplot(triangulation, color=edge_colour, lw=0.35,
                                alpha=edge_alpha, zorder=3))
    if rings:
        path = basin_path(rings, scale=1 / 1000.0)
        for artist in drawn:
            artist.set_clip_path(PathPatch(path, transform=ax.transData,
                                           facecolor="none", edgecolor="none"))
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=outline,
                               lw=1.0, zorder=5))
        x, y = path.vertices[:, 0], path.vertices[:, 1]
        ax.set_xlim(x.min() - margin_km, x.max() + margin_km)
        ax.set_ylim(y.min() - margin_km, y.max() + margin_km)
    return surface


def month_axis(ax: Axes, months: tuple = uc1.MELT_MONTHS) -> None:
    """
    Label an axis measured in days of the year with month names instead.

    A melt season is read in months rather than in the 121st to the 266th day
    of the year, and the day a step happens to start on means nothing to a
    reader. The ticks sit on the first of each month, so the space between two
    of them is the month itself.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes whose x axis is in days of the year.
    months : tuple of int, optional
        The months to label.
    """
    stamps = [pd.Timestamp(2015, month, 1) for month in (*months, months[-1] + 1)]
    ax.set_xticks([stamp.dayofyear for stamp in stamps])
    ax.set_xticklabels([stamp.strftime("%b") for stamp in stamps])
    ax.set_xlabel("")


def headroom(ax: Axes, fraction: float = 0.22) -> None:
    """
    Leave space above the data, so a legend at the top does not sit on it.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to adjust.
    fraction : float, optional
        Space to add, as a fraction of the current y range.
    """
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, bottom + (top - bottom) * (1.0 + fraction))


def label_points(ax: Axes, x: object, y: object, labels: object,
                 fontsize: int = 8, colour: str = "#666666") -> None:
    """
    Name every point, nudged so that the names do not land on each other.

    Each label is tried in a few positions around its point and keeps the
    first that is clear of the labels already placed. Where a cluster leaves
    nowhere free, the last position is kept rather than the label dropped.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes the points are drawn on.
    x, y : array-like
        Coordinates of the points.
    labels : iterable
        One label per point.
    fontsize : int, optional
        Size of the labels.
    colour : str, optional
        Colour of the labels.
    """
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    offsets = [(4, 3), (4, -11), (-4, 3), (-4, -11), (0, 10), (0, -16)]
    placed = []
    for xi, yi, text in zip(np.asarray(x), np.asarray(y), labels, strict=False):
        note = ax.annotate(str(text), (xi, yi), xytext=offsets[0],
                           textcoords="offset points", fontsize=fontsize,
                           color=colour)
        for dx, dy in offsets:
            note.set_position((dx, dy))
            note.set_ha("left" if dx >= 0 else "right")
            box = note.get_window_extent(renderer)
            if not any(box.overlaps(other) for other in placed):
                break
        placed.append(box)


def collapsed(frame: pd.DataFrame, summary: str, decimals: int = 2) -> None:
    """
    Show a table behind a click, so detail is available but not in the way.

    Parameters
    ----------
    frame : pandas.DataFrame
        The table to show.
    summary : str
        Text on the line that is clicked to open it.
    decimals : int, optional
        Decimal places to round to.
    """
    display(HTML(f"<details><summary style='cursor:pointer'>{summary}</summary>"
                 f"{frame.round(decimals).to_html()}</details>"))


# --------------------------------------------------------------------------
# The notebook's figures
# --------------------------------------------------------------------------

def plot_runoff_map(fields: pd.DataFrame, points: np.ndarray, rings: list[np.ndarray],
                    mesh_vertices: np.ndarray, mesh_triangles: np.ndarray,
                    forcing: str) -> Figure:
    """
    Map the mean July runoff field over the catchment, with the module's mesh.

    Parameters
    ----------
    fields : pandas.DataFrame
        Fitted runoff, time steps by grid cell.
    points : numpy.ndarray
        Coordinates of each grid cell, in metres.
    rings : list of numpy.ndarray
        The catchment outline, as `uc1_hydro_helpers.load_basin` returns it.
    mesh_vertices, mesh_triangles : numpy.ndarray
        The module's mesh, as `uc1_hydro_helpers.load_mesh` returns it.
    forcing : str
        Name of the forcing, for the title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    july = fields[fields.index.month == 7].mean(axis=0).to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.grid(False)

    # The field is continuous and the columns are only where it is evaluated, so
    # draw it as a surface, clipped to the catchment. Without the clip the picture
    # covers roughly twice the area, most of it the mesh's margin.
    intensity = np.log10(july)
    surface = mesh_map(ax, points, intensity, rings, None, cmap="YlOrRd")
    # The module's own triangulation is drawn over the surface. The field arrives
    # interpolated onto a grid, but it was fitted on these few hundred nodes, and
    # the detail in the colour is interpolation between them, not measurement.
    mesh_edges(ax, mesh_vertices, mesh_triangles, rings)
    ax.set_xlabel("kilometres easting")
    ax.set_ylabel("kilometres northing")
    ax.set_title(f"GRF output: mean July field, {forcing}", loc="left")
    bar = fig.colorbar(surface, ax=ax)
    ticks = [t for t in range(-2, 5) if np.nanmin(intensity) <= t <= np.nanmax(intensity)]
    bar.set_ticks(ticks)
    bar.set_ticklabels([f"{10.0 ** t:g}" for t in ticks])
    bar.set_label("mean July runoff (mm w.e.)")
    return fig


def plot_lake_season(lake_fields: pd.DataFrame, lake_points: np.ndarray,
                     lake_in_basin: np.ndarray, rings: list[np.ndarray],
                     mesh_vertices: np.ndarray, mesh_triangles: np.ndarray) -> Figure:
    """
    Map the mean July lake cover, and draw the catchment mean through each season.

    Parameters
    ----------
    lake_fields : pandas.DataFrame
        Fitted lake cover, time steps by grid cell.
    lake_points : numpy.ndarray
        Coordinates of each grid cell, in metres.
    lake_in_basin : numpy.ndarray
        True for each grid cell inside the catchment.
    rings : list of numpy.ndarray
        The catchment outline, as `uc1_hydro_helpers.load_basin` returns it.
    mesh_vertices, mesh_triangles : numpy.ndarray
        The module's mesh, as `uc1_hydro_helpers.load_mesh` returns it.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    july = lake_fields[lake_fields.index.month == 7]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.0), layout="constrained",
                                      gridspec_kw={"width_ratios": [1, 1.1]})

    # Lake cover spans four orders of magnitude. On a linear scale nine nodes
    # in ten fall into the bottom tenth of the colour range and the map reads
    # as blank, so both panels use log10 - the scale the module fits on - with
    # the labels written as percentages.
    lake_cmap = LinearSegmentedColormap.from_list(
        "lake", plt.cm.Blues(np.linspace(0.18, 1.0, 256)))
    intensity = np.log10(july.mean(axis=0).to_numpy())

    # The GRF is a continuous surface and the grid is only where it is read
    # back, so draw it as one rather than as dots, clipped to the same
    # catchment outline the runoff map uses.
    left.grid(False)
    # The mesh is drawn here as it is on the runoff map, in grey rather than
    # white: most of this ramp is pale, and white edges disappear into it.
    field = mesh_map(left, lake_points, intensity, rings, None, cmap=lake_cmap)
    mesh_edges(left, mesh_vertices, mesh_triangles, rings,
               colour="#4b5563", alpha=0.4)
    left.set_xlabel("kilometres easting")
    left.set_ylabel("kilometres northing")
    left.set_title("Average July lake cover", loc="left", pad=10)
    bar = fig.colorbar(field, ax=left, orientation="horizontal",
                       fraction=0.055, pad=0.06, aspect=34)
    ticks = [t for t in (-4, -3, -2, -1, 0) if np.nanmin(intensity) <= t <= np.nanmax(intensity)]
    bar.set_ticks(ticks)
    bar.set_ticklabels([f"{100 * 10.0 ** t:g}%" for t in ticks])
    bar.set_label("share of a cell covered by lakes", size=10)
    bar.ax.tick_params(length=0)
    bar.outline.set_visible(False)

    # Eight seasons drawn equally is a thicket, so the individual years stay
    # recessive, the median season carries the shape, and only the extremes
    # are named.
    catchment = uc1.catchment_mean(lake_fields, lake_in_basin)
    lake_by_year = dict(list(catchment.groupby(catchment.index.year)))
    for n, season in enumerate(lake_by_year.values()):
        right.plot(season.index.month, season.to_numpy(), lw=1.3,
                   color="#b8c4d2", zorder=2,
                   label="each melt season" if n == 0 else None)
    fitted_months = sorted({t.month for t in catchment.index})
    # Aligned on the month rather than on position: a season the module could
    # not fit right through is short, and lining those up by position would
    # pair one year's September with another's August.
    typical = pd.DataFrame({
        year: pd.Series(season.to_numpy(), index=season.index.month)
        for year, season in lake_by_year.items()
    }).reindex(fitted_months).median(axis=1)
    right.plot(typical.index, typical.to_numpy(), lw=2.8, color=MEASURED, marker="o",
               ms=6, zorder=5, label="median season")
    # The biggest and smallest lake seasons, named at their peak rather than at
    # the end, where every year converges on the floor and the labels collide.
    peaks = {year: season.max() for year, season in lake_by_year.items()}
    extremes = (max(peaks, key=peaks.get), min(peaks, key=peaks.get))
    for n, year in enumerate(extremes):
        season = lake_by_year[year]
        right.plot(season.index.month, season.to_numpy(), lw=1.8,
                   color="#5b6b7f", zorder=4,
                   label="biggest and smallest season" if n == 0 else None)
        top = season.idxmax()
        right.annotate(str(year), (top.month, season.loc[top]),
                       xytext=(0, 8), textcoords="offset points", fontsize=9,
                       color="#5b6b7f", ha="center")

    # Most Septembers sit on the floor of the record, where the lakes have
    # drained and the mapping is clipped at a cover of 0.01%. Marking it says
    # why so many seasons end on the same value. The label goes in the key
    # rather than on the line, which is where the season curves are.
    right.axhline(LAKE_FLOOR, color="#9aa3ad", lw=1, ls=(0, (4, 3)), zorder=1,
                  label="floor of the lake record")

    right.set_yscale("log")
    right.set_ylim(6e-5, 0.25)
    right.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
    right.set_yticklabels(["0.01%", "0.1%", "1%", "10%"])
    right.set_xlim(min(fitted_months) - 0.25, max(fitted_months) + 0.25)
    right.set_xticks(fitted_months)
    right.set_xticklabels([pd.Timestamp(2020, m, 1).strftime("%b")
                           for m in fitted_months])
    right.set_ylabel("catchment mean lake cover")
    right.set_title("Lake cover through each melt season", loc="left", pad=10)
    right.legend(frameon=False, fontsize=9, loc="upper left")
    return fig


def plot_discharge_record(discharge: pd.Series, observed: pd.DataFrame) -> Figure:
    """
    Draw the daily gauge record, and every melt season on one axis.

    Parameters
    ----------
    discharge : pandas.Series
        Daily gauge discharge, indexed by date.
    observed : pandas.DataFrame
        Mean discharge, years by step, as `uc1_hydro_helpers.step_matrix` returns it.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.4))

    left.plot(discharge.index, discharge.values, color=MEASURED, lw=0.7)
    left.set_xlabel("year")
    left.set_ylabel(DISCHARGE_LABEL)
    left.set_title("Daily discharge, full record", loc="left")

    shades = plt.cm.Blues(np.linspace(0.35, 1.0, len(observed)))
    for shade, (_, row) in zip(shades, observed.iterrows(), strict=False):
        right.plot(row.index, row.values, color=shade, lw=1.1)
    month_axis(right)
    right.set_ylabel(DISCHARGE_LABEL)
    right.set_title(f"Melt seasons overlaid, {observed.index.min()}–{observed.index.max()}"
                    " (darker = later)", loc="left")
    fig.tight_layout()
    return fig


def plot_predictor_candidates(candidates: dict[str, pd.Series],
                              measured_total: pd.Series) -> Figure:
    """
    Scatter each candidate seasonal number against the measured seasonal volume.

    Parameters
    ----------
    candidates : dict of str to pandas.Series
        One value per season for each candidate, indexed by year.
    measured_total : pandas.Series
        Measured melt-season volume in km3, indexed by year.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, one panel per candidate.
    """
    fig, axes = plt.subplots(1, len(candidates), figsize=(6 * len(candidates), 4.8),
                             squeeze=False)
    for ax, (name, series) in zip(axes[0], candidates.items(), strict=False):
        shared = series.index.intersection(measured_total.index)
        x, y = series.loc[shared], measured_total.loc[shared]
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        slope, intercept = np.polyfit(x, y, 1)
        ax.scatter(x, y, s=52, color=MEASURED, edgecolor="white", linewidth=1.2, zorder=3)
        label_points(ax, x, y, shared)
        grid = np.linspace(x.min(), x.max(), 50)
        ax.plot(grid, intercept + slope * grid, color=MODELLED, lw=2)
        ax.set_xlabel(f"{name} (from the fitted GRF fields)")
        ax.set_ylabel("measured melt-season discharge (km$^3$)")
        ax.set_title(f"{name}: r$^2$ = {r2:.2f}, n = {len(shared)}", loc="left")
    fig.tight_layout()
    return fig


def plot_prediction_band(band: pd.DataFrame, measured: pd.Series, predicted_label: str,
                         measured_label: str, title: str) -> Figure:
    """
    Draw one predicted season with its 68% and 95% bands, against the measurement.

    Parameters
    ----------
    band : pandas.DataFrame
        ``predicted``, ``lower_68``, ``upper_68``, ``lower_95`` and ``upper_95``
        for each step, as a model's ``bands`` returns them.
    measured : pandas.Series
        Measured discharge for each step of the same season.
    predicted_label, measured_label : str
        Legend entries for the two lines.
    title : str
        Title of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    fig, ax = plt.subplots()
    ax.fill_between(band.index, band["lower_95"], band["upper_95"], color=MODELLED,
                    alpha=0.15, lw=0, label="95% prediction interval")
    ax.fill_between(band.index, band["lower_68"], band["upper_68"], color=MODELLED,
                    alpha=0.18, lw=0, label="68% prediction interval")
    ax.plot(band.index, band["predicted"], color=MODELLED, lw=2.4, marker="o", ms=5,
            label=predicted_label)
    ax.plot(measured.index, measured.to_numpy(), color=MEASURED, lw=2, label=measured_label)
    month_axis(ax)
    ax.set_ylabel(DISCHARGE_LABEL)
    headroom(ax)
    ax.legend(frameon=False, ncol=3)
    ax.set_title(title, loc="left")
    return fig


def plot_lake_contribution(contribution: pd.DataFrame, seasons: pd.Index) -> Figure:
    """
    Bar the held-out error of each combination of runoff and lake inputs.

    Parameters
    ----------
    contribution : pandas.DataFrame
        One row per combination, with a ``held-out RMSE (m3/s)`` column.
    seasons : pandas.Index
        The seasons the comparison covers, for the title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.set_axisbelow(True)
    bars = ax.bar(contribution.index, contribution["held-out RMSE (m3/s)"],
                  color=[MODELLED, ALTERNATIVE, MEASURED, "#8a6bbf", "#5b6b7f"], width=0.55)
    for bar, value in zip(bars, contribution["held-out RMSE (m3/s)"], strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 4, f"{value:.0f}",
                ha="center", fontsize=10)
    ax.set_ylabel("held-out error (m$^3$/s)")
    ax.set_title(f"What each input contributes, {seasons.min()}"
                 f"\u2013{seasons.max()} (lower is better)", loc="left")
    ax.set_ylim(0, contribution["held-out RMSE (m3/s)"].max() * 1.18)
    fig.tight_layout()
    return fig


def plot_totals_and_peaks(table: pd.DataFrame, threshold_m3s: float) -> Figure:
    """
    Compare measured and modelled annual totals and annual peaks.

    Parameters
    ----------
    table : pandas.DataFrame
        Indexed by year, with ``total, measured (km3)``, ``total, modelled (km3)``,
        ``peak, measured (m3/s)`` and ``peak, modelled (m3/s)`` columns.
    threshold_m3s : float
        The operating threshold, in m3/s.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.2))
    width = 0.4
    years = table.index.to_numpy()
    left.bar(years - width / 2, table["total, measured (km3)"], width, color=MEASURED,
             label="measured")
    left.bar(years + width / 2, table["total, modelled (km3)"], width, color=MODELLED,
             label="modelled")
    left.set_xticks(years[::2])
    left.set_ylabel("melt-season total (km$^3$)")
    left.set_title("Annual totals", loc="left")
    left.set_axisbelow(True)
    headroom(left)
    left.legend(frameon=False, ncol=2)

    right.plot(years, table["peak, measured (m3/s)"], color=MEASURED, lw=2, marker="o",
               ms=5, label="measured")
    right.plot(years, table["peak, modelled (m3/s)"], color=MODELLED, lw=2, marker="o",
               ms=5, label="modelled")
    right.axhline(threshold_m3s, color="#888", ls=":", lw=1.6,
                  label=f"threshold {threshold_m3s} m$^3$/s")
    right.set_xticks(years[::2])
    right.set_ylabel("peak flow (m$^3$/s)")
    right.set_title("Annual peaks", loc="left")
    headroom(right)
    right.legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    return fig


def plot_forcing_comparison(totals: pd.DataFrame, mean_seasons: pd.DataFrame,
                            labels: dict[str, str]) -> Figure:
    """
    Compare the forcings on the seasons they share: annual totals, and the mean season.

    Parameters
    ----------
    totals : pandas.DataFrame
        Melt-season totals in km3, years by forcing, with a ``measured`` column.
    mean_seasons : pandas.DataFrame
        Mean discharge in m3/s, steps by forcing, with a ``measured`` column.
    labels : dict of str to str
        How each forcing is named in the legend.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    forcings = [name for name in totals.columns if name != "measured"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.2))
    left.plot(totals.index, totals["measured"], color=MEASURED, lw=2.4, marker="o", ms=5,
              label="measured")
    for name in forcings:
        left.plot(totals.index, totals[name], color=FORCING_COLOURS.get(name), lw=1.8,
                  marker="o", ms=4, label=f"from {labels[name]}")
    left.set_xticks(list(totals.index)[::2])
    left.set_ylabel("melt-season total (km$^3$)")
    left.set_title("Same seasons, three runoff datasets", loc="left")
    headroom(left, 0.38)
    left.legend(frameon=False, ncol=2, fontsize=9, loc="upper left")

    for name in forcings:
        right.plot(mean_seasons.index, mean_seasons[name], color=FORCING_COLOURS.get(name),
                   lw=2.4, marker="o", ms=4, label=f"from {labels[name]}")
    right.plot(mean_seasons.index, mean_seasons["measured"], color=MEASURED, lw=2.4,
               label="measured")
    month_axis(right)
    right.set_ylabel(DISCHARGE_LABEL)
    right.set_title("Mean season", loc="left")
    headroom(right)
    right.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    return fig


def plot_scenarios(runs: dict[str, pd.DataFrame]) -> Figure:
    """
    Draw every season of every scenario, and the mean season under each.

    Parameters
    ----------
    runs : dict of str to pandas.DataFrame
        Discharge in m3/s, years by step, for each scenario. The observed record
        is expected first.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for colour, (name, run_profiles) in zip(SCENARIO_COLOURS, runs.items(), strict=False):
        for _, row in run_profiles.iterrows():
            left.plot(row.index, row.values, color=colour, lw=0.7, alpha=0.28)
        right.plot(run_profiles.columns, run_profiles.mean(axis=0), color=colour,
                   lw=2.6, marker="o", ms=4, label=name)

    month_axis(left)
    left.set_ylabel(DISCHARGE_LABEL)
    left.set_title("Every season, every scenario", loc="left")
    month_axis(right)
    right.set_title("Mean season under each scenario", loc="left")
    right.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    return fig


def plot_scenario_summary(summary: pd.DataFrame) -> Figure:
    """
    Bar the annual total and the peak flow under each scenario.

    Parameters
    ----------
    summary : pandas.DataFrame
        One row per scenario, as `uc1_hydro_helpers.compare` returns it.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    colours = list(SCENARIO_COLOURS[:len(summary)])
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.2))
    left.set_axisbelow(True)
    right.set_axisbelow(True)
    positions = np.arange(len(summary))

    left.bar(positions, summary["mean total (km3)"], color=colours)
    left.errorbar(positions, summary["mean total (km3)"],
                  yerr=[summary["mean total (km3)"] - summary["driest (km3)"],
                        summary["wettest (km3)"] - summary["mean total (km3)"]],
                  fmt="none", ecolor="#444", capsize=5, lw=1.4)
    left.set_xticks(positions)
    left.set_xticklabels(summary.index, rotation=20, ha="right")
    left.set_ylabel("melt-season total (km$^3$)")
    left.set_title("Annual total: mean, with driest and wettest season", loc="left")

    right.bar(positions, summary["mean peak (m3/s)"], color=colours)
    right.set_xticks(positions)
    right.set_xticklabels(summary.index, rotation=20, ha="right")
    right.set_ylabel("peak flow (m$^3$/s)")
    right.set_title("Peak flow: mean across seasons", loc="left")
    fig.tight_layout()
    return fig


def plot_uncertainty_band(fitting: pd.DataFrame, total: pd.DataFrame, measured: pd.Series,
                          year: int) -> Figure:
    """
    Draw one season's band with fitting uncertainty alone, and with harmonisation added.

    Parameters
    ----------
    fitting, total : pandas.DataFrame
        ``lower``, ``predicted`` and ``upper`` for each step, as
        `uc1_hydro_helpers.propagate` returns them, without and with the
        harmonisation spread.
    measured : pandas.Series
        Measured discharge for each step of the same season.
    year : int
        The season, for the legend and the title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    fig, ax = plt.subplots()
    ax.fill_between(total.index, total["lower"], total["upper"], color=MODELLED,
                    alpha=0.16, lw=0, label="fitting + harmonisation")
    ax.fill_between(fitting.index, fitting["lower"], fitting["upper"],
                    color=MODELLED, alpha=0.30, lw=0, label="fitting only")
    ax.plot(total.index, total["predicted"], color=MODELLED, lw=2.2, marker="o", ms=4)
    ax.plot(measured.index, measured.to_numpy(), color=MEASURED, lw=2,
            label=f"{year} measured")
    month_axis(ax)
    ax.set_ylabel(DISCHARGE_LABEL)
    headroom(ax)
    ax.legend(frameon=False, ncol=3)
    ax.set_title(f"{year}: both sources of uncertainty", loc="left")
    return fig


def plot_spread_sensitivity(fractions: Sequence[float], widths: Sequence[float]) -> Figure:
    """
    Draw how wide the band would be for a range of assumed harmonisation spreads.

    Parameters
    ----------
    fractions : sequence of float
        Assumed spreads, as a fraction of the seasonal number.
    widths : sequence of float
        The mean band width for each, in m3/s.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot([f * 100 for f in fractions], widths, color=MODELLED, lw=2.4,
            marker="o", ms=7)
    for fraction, width in zip(fractions, widths, strict=False):
        ax.annotate(f"{width:.0f}", (fraction * 100, width), fontsize=10,
                    color=MODELLED, xytext=(0, 9), textcoords="offset points",
                    ha="center")
    ax.set_xlabel("harmonisation spread, as a % of the seasonal number")
    ax.set_ylabel("mean band width (m$^3$/s)")
    ax.set_ylim(min(widths) * 0.95, max(widths) * 1.08)
    ax.set_title("Sensitivity: what the missing term would do", loc="left")
    return fig


def plot_baseline_errors(baselines: dict[str, pd.DataFrame]) -> Figure:
    """
    Draw each approach's error against the gauge, season by season.

    Parameters
    ----------
    baselines : dict of str to pandas.DataFrame
        An observed/predicted frame for each approach, as
        `uc1_hydro_helpers.leave_one_year_out` returns it.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for colour, (name, v) in zip(["#7a8088", ALTERNATIVE, MODELLED, "#8a6bbf"],
                                 baselines.items(), strict=False):
        error = ((v["predicted"] - v["observed"]) ** 2).mean(axis=1) ** 0.5
        ax.plot(error.index, error.values, color=colour, lw=2, marker="o", ms=5,
                label=f"{name} — {uc1.profile_skill(v)['rmse_m3s']:.0f} m$^3$/s overall")
    first = next(iter(baselines.values()))
    ax.set_xticks(list(first["observed"].index)[::2])
    ax.set_ylabel("weekly error against the gauge (m$^3$/s)")
    ax.set_title("How wrong each approach is, season by season", loc="left")
    headroom(ax)
    ax.legend(frameon=False, fontsize=10)
    return fig
