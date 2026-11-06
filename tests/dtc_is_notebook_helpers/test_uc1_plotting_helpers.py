"""Tests for the Use Case 1 plotting helpers.

The building blocks are tested for what they do to the axes. The notebook's
figures are run on a small synthetic chain - a known seasonal shape scaled by a
known per-year factor - and checked for the parts a reader relies on.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from dtc_is_notebook_helpers import uc1_hydro_helpers as uc1
from dtc_is_notebook_helpers import uc1_plotting_helpers as uc1_plot

FIRST_YEAR, LAST_YEAR = 2006, 2015


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def rings(basin_csv) -> list[np.ndarray]:
    """Give the test outline in metres, as the helpers expect it."""
    return [ring * 1000.0 for ring in uc1.load_basin(basin_csv)]


@pytest.fixture
def discharge() -> pd.Series:
    """Daily discharge: one Gaussian melt season a year, scaled by the year's level."""
    dates = pd.date_range(f"{FIRST_YEAR}-01-01", f"{LAST_YEAR}-12-31", freq="D")
    shape = np.exp(-((dates.dayofyear.to_numpy() - 200) ** 2) / (2 * 30.0**2))
    level = 1.0 + 0.1 * (dates.year.to_numpy() - FIRST_YEAR)
    return pd.Series(600.0 * level * shape, index=dates)


@pytest.fixture
def predictors() -> pd.Series:
    years = range(FIRST_YEAR, LAST_YEAR + 1)
    return pd.Series({year: 1.0 + 0.1 * (year - FIRST_YEAR) for year in years}, name="catchment sum")


@pytest.fixture
def observed(discharge) -> pd.DataFrame:
    return uc1.step_matrix(discharge)


@pytest.fixture
def model(discharge, predictors) -> uc1.ProfileModel:
    return uc1.ProfileModel.fit(discharge, predictors)


@pytest.fixture
def grid_points() -> np.ndarray:
    """Give a 6 x 6 grid of cell centres over the test outline, in metres."""
    x, y = np.meshgrid(np.linspace(500, 9500, 6), np.linspace(500, 9500, 6))
    return np.column_stack([x.ravel(), y.ravel()])


@pytest.fixture
def mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array([[0.0, 0.0], [10000.0, 0.0], [10000.0, 10000.0], [0.0, 10000.0]])
    return vertices, np.array([[0, 1, 2], [0, 2, 3]])


def _monthly_fields(points: np.ndarray, first: int, last: int, low: float, high: float) -> pd.DataFrame:
    """Positive melt-season fields, one row per month, one column per cell."""
    months = pd.date_range(f"{first}-05-01", f"{last}-09-01", freq="MS")
    months = months[months.month.isin(uc1.MELT_MONTHS)]
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.uniform(low, high, size=(len(months), len(points))), index=months)


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def test_mesh_map_clips_the_surface_to_the_outline(rings):
    rng = np.random.default_rng(0)
    nodes = rng.uniform(-5000, 15000, size=(40, 2))
    _, ax = plt.subplots()
    surface = uc1_plot.mesh_map(ax, nodes, rng.normal(size=40), rings)
    assert surface.get_clip_path() is not None
    assert ax.get_xlim() == pytest.approx((-4.0, 14.0))   # the outline, plus the margin


def test_mesh_map_draws_the_triangulation_it_is_given(rings, mesh):
    vertices, triangles = mesh
    _, ax = plt.subplots()
    uc1_plot.mesh_map(ax, vertices, np.arange(4.0), rings, triangles, edges=True)
    assert ax.lines, "edges=True should draw the mesh"
    assert all(line.get_clip_path() is not None for line in ax.lines)


def test_mesh_map_without_an_outline_draws_the_whole_mesh():
    rng = np.random.default_rng(0)
    nodes = rng.uniform(0, 10000, size=(30, 2))
    _, ax = plt.subplots()
    surface = uc1_plot.mesh_map(ax, nodes, rng.normal(size=30))
    assert surface.get_clip_path() is None


def test_mesh_edges_are_clipped_to_the_outline(rings, mesh):
    vertices, triangles = mesh
    _, ax = plt.subplots()
    drawn = uc1_plot.mesh_edges(ax, vertices, triangles, rings)
    assert drawn
    assert all(artist.get_clip_path() is not None for artist in drawn)


def test_headroom_only_lifts_the_top_of_the_axis():
    _, ax = plt.subplots()
    ax.plot([0, 1], [0.0, 10.0])
    bottom, top = ax.get_ylim()
    uc1_plot.headroom(ax, 0.5)
    assert ax.get_ylim()[0] == pytest.approx(bottom)
    assert ax.get_ylim()[1] == pytest.approx(bottom + (top - bottom) * 1.5)


def test_label_points_moves_a_label_that_would_sit_on_another():
    _, ax = plt.subplots()
    # Fixed limits, and two points a pixel or so apart, so the first position
    # one label takes is certain to be in the way of the next.
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.scatter([5.0, 5.02], [5.0, 5.0])
    uc1_plot.label_points(ax, [5.0, 5.02], [5.0, 5.0], ["2019", "2020"])
    first, second = [note.get_position() for note in ax.texts]
    assert first != second
    assert [note.get_text() for note in ax.texts] == ["2019", "2020"]


def test_month_axis_puts_month_names_on_a_day_of_year_axis():
    _, ax = plt.subplots()
    ax.plot([121, 200, 273], [1.0, 2.0, 1.0])
    uc1_plot.month_axis(ax)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["May", "Jun", "Jul", "Aug", "Sep", "Oct"]
    assert list(ax.get_xticks()) == [121, 152, 182, 213, 244, 274]


def test_collapsed_puts_the_table_behind_a_summary(monkeypatch):
    shown = []
    monkeypatch.setattr(uc1_plot, "display", shown.append)
    uc1_plot.collapsed(pd.DataFrame({"a": [1.234]}), "Show it", decimals=1)
    html = shown[0].data
    assert html.startswith("<details><summary")
    assert "Show it" in html
    assert "1.2" in html and "1.234" not in html


# --------------------------------------------------------------------------
# The notebook's figures
# --------------------------------------------------------------------------

@pytest.fixture
def runs(model, predictors) -> dict[str, pd.DataFrame]:
    scenario = uc1.Scenario.from_perturbation("Warmer", predictors, weight=0.8)
    return {"Observed record": model.profiles(predictors), "Warmer": scenario.run(model)}


@pytest.fixture
def forcing_totals_and_seasons(model, predictors, observed) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles = {"RACMO55": model.profiles(predictors), "RACMO": model.profiles(predictors * 1.1),
                "MAR": model.profiles(predictors * 0.9)}
    totals = pd.DataFrame({name: uc1.summarise(p, 7)["total_km3"] for name, p in profiles.items()})
    totals["measured"] = uc1.summarise(observed, 7)["total_km3"]
    seasons = pd.DataFrame({name: p.mean(axis=0) for name, p in profiles.items()})
    seasons["measured"] = observed.mean(axis=0)
    return totals, seasons


FIGURES = {
    "runoff_map": (
        lambda f: uc1_plot.plot_runoff_map(
            _monthly_fields(f["grid_points"], 2006, 2008, 1.0, 500.0), f["grid_points"],
            f["rings"], *f["mesh"], "RACMO55"),
        2),     # the map and its colour bar
    "lake_season": (
        lambda f: uc1_plot.plot_lake_season(
            _monthly_fields(f["grid_points"], 2014, 2016, 1e-4, 0.1), f["grid_points"],
            np.ones(len(f["grid_points"]), dtype=bool), f["rings"], *f["mesh"]),
        3),     # the map, its colour bar, and the seasons
    "discharge_record": (
        lambda f: uc1_plot.plot_discharge_record(f["discharge"], f["observed"]), 2),
    "predictor_candidates": (
        lambda f: uc1_plot.plot_predictor_candidates(
            {"catchment sum": f["predictors"], "other": f["predictors"] ** 2},
            pd.Series({y: uc1.annual_total(row, 7) for y, row in f["observed"].iterrows()})),
        2),     # one panel per candidate
    "prediction_band": (
        lambda f: uc1_plot.plot_prediction_band(
            f["model"].bands(np.array([1.5])), f["observed"].loc[2010],
            "2010 predicted", "2010 measured", "2010"),
        1),
    "lake_contribution": (
        lambda f: uc1_plot.plot_lake_contribution(
            pd.DataFrame({"held-out RMSE (m3/s)": [235.0, 275.0, 206.0, 174.0, 175.0]},
                         index=["a", "b", "c", "d", "e"]),
            pd.Index([2014, 2021])),
        1),
    "totals_and_peaks": (
        lambda f: uc1_plot.plot_totals_and_peaks(
            pd.DataFrame({"total, measured (km3)": [1.0, 2.0], "total, modelled (km3)": [1.1, 1.9],
                          "peak, measured (m3/s)": [900.0, 1300.0],
                          "peak, modelled (m3/s)": [800.0, 1100.0]}, index=[2010, 2011]),
            1200),
        2),
    "forcing_comparison": (
        lambda f: uc1_plot.plot_forcing_comparison(
            *f["forcing_totals_and_seasons"],
            {"RACMO55": "RACMO 5.5 km", "RACMO": "RACMO 2 km", "MAR": "MAR 5 km"}),
        2),
    "scenarios": (lambda f: uc1_plot.plot_scenarios(f["runs"]), 2),
    "scenario_summary": (lambda f: uc1_plot.plot_scenario_summary(uc1.compare(f["runs"], 7)), 2),
    "uncertainty_band": (
        lambda f: uc1_plot.plot_uncertainty_band(
            uc1.propagate(f["model"], np.array([1.5]), 0.0),
            uc1.propagate(f["model"], np.array([1.5]), 0.1),
            f["observed"].loc[2010], 2010),
        1),
    "spread_sensitivity": (
        lambda _: uc1_plot.plot_spread_sensitivity([0.0, 0.05, 0.2], [400.0, 420.0, 520.0]), 1),
    "baseline_errors": (
        lambda f: uc1_plot.plot_baseline_errors(
            {"model": uc1.leave_one_year_out(f["discharge"], f["predictors"]),
             "naive": uc1.leave_one_year_out(f["discharge"], f["predictors"] ** 2)}),
        1),
}


@pytest.fixture
def chain(rings, grid_points, mesh, discharge, observed, predictors, model, runs,
          forcing_totals_and_seasons) -> dict:
    """Everything the figure functions are drawn from, by name."""
    return {"rings": rings, "grid_points": grid_points, "mesh": mesh, "discharge": discharge,
            "observed": observed, "predictors": predictors, "model": model, "runs": runs,
            "forcing_totals_and_seasons": forcing_totals_and_seasons}


@pytest.mark.parametrize(("draw", "n_axes"), FIGURES.values(), ids=FIGURES.keys())
def test_plot_functions_return_a_drawn_figure(chain, draw, n_axes):
    figure = draw(chain)
    assert isinstance(figure, Figure)
    assert len(figure.axes) == n_axes
    assert all(ax.has_data() or ax.get_label() == "<colorbar>" for ax in figure.axes)


def test_plot_prediction_band_labels_both_bands_and_both_lines(model, observed):
    figure = uc1_plot.plot_prediction_band(model.bands(np.array([1.5])), observed.loc[2010],
                                           "2010 predicted", "2010 measured", "2010")
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert labels == ["95% prediction interval", "68% prediction interval",
                      "2010 predicted", "2010 measured"]


def test_plot_forcing_comparison_keeps_each_forcing_in_its_colour(forcing_totals_and_seasons):
    labels = {"RACMO55": "RACMO 5.5 km", "RACMO": "RACMO 2 km", "MAR": "MAR 5 km"}
    figure = uc1_plot.plot_forcing_comparison(*forcing_totals_and_seasons, labels)
    for ax in figure.axes:
        colours = {line.get_label(): line.get_color() for line in ax.lines}
        assert colours["measured"] == uc1_plot.MEASURED
        for key, label in labels.items():
            assert colours[f"from {label}"] == uc1_plot.FORCING_COLOURS[key]


def test_plot_lake_season_names_the_biggest_and_smallest_season(grid_points, rings, mesh):
    fields = _monthly_fields(grid_points, 2014, 2016, 1e-4, 0.01)
    fields.loc[fields.index.year == 2015] *= 10.0       # 2015 the biggest season
    fields.loc[fields.index.year == 2016] /= 10.0       # 2016 the smallest
    figure = uc1_plot.plot_lake_season(fields, grid_points, np.ones(len(grid_points), dtype=bool),
                                       rings, *mesh)
    seasons_panel = figure.axes[1]
    assert sorted(text.get_text() for text in seasons_panel.texts) == ["2015", "2016"]
