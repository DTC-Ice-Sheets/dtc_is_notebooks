"""Tests for the Use Case 1 hydropower helpers.

The data here is synthetic on purpose: a known seasonal shape scaled by a known
per-year factor. That makes the expected answers arithmetic rather than
whatever the real record happens to contain, so a failure points at the code.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dtc_is_notebook_helpers import uc1_hydro_helpers as uc1

FIRST_YEAR, LAST_YEAR = 2006, 2015
PEAK_DOY, PEAK_WIDTH, PEAK_M3S = 200, 30.0, 600.0


def _levels() -> dict[int, float]:
    return {year: 1.0 + 0.1 * (year - FIRST_YEAR) for year in range(FIRST_YEAR, LAST_YEAR + 1)}


@pytest.fixture
def discharge() -> pd.Series:
    """Daily discharge: one Gaussian melt season a year, scaled by the year's level."""
    dates = pd.date_range(f"{FIRST_YEAR}-01-01", f"{LAST_YEAR}-12-31", freq="D")
    shape = np.exp(-((dates.dayofyear.to_numpy() - PEAK_DOY) ** 2) / (2 * PEAK_WIDTH**2))
    level = np.array([_levels()[date.year] for date in dates])
    return pd.Series(PEAK_M3S * level * shape, index=dates, name="discharge_m3s")


@pytest.fixture
def predictors() -> pd.Series:
    """One predictor per season, exactly the factor the discharge was scaled by."""
    return pd.Series(_levels(), name="catchment sum")


@pytest.fixture
def model(discharge, predictors) -> uc1.ProfileModel:
    return uc1.ProfileModel.fit(discharge, predictors)


def test_step_matrix_covers_every_season(discharge):
    steps = uc1.step_matrix(discharge, step_days=7)
    assert list(steps.index) == list(range(FIRST_YEAR, LAST_YEAR + 1))
    assert steps.notna().to_numpy().all()


def test_a_longer_step_gives_fewer_columns(discharge):
    weekly = uc1.step_matrix(discharge, step_days=7)
    fortnightly = uc1.step_matrix(discharge, step_days=14)
    assert fortnightly.shape[1] < weekly.shape[1]


def test_the_fit_recovers_a_relationship_it_was_built_from(model):
    # The discharge is exactly linear in the predictor, so every step should fit
    # it to machine precision rather than merely well.
    assert np.nanmedian(model.r2) > 0.999
    assert model.years.min() == FIRST_YEAR
    assert model.years.max() == LAST_YEAR


def test_a_profile_scales_with_its_predictor(model):
    one = uc1.annual_total(model.profile(1.0), model.step_days)
    two = uc1.annual_total(model.profile(2.0), model.step_days)
    # Twice the predictor is not twice the discharge - there is an intercept -
    # but it must be strictly larger, and the step is the predictor difference.
    assert two > one
    half = uc1.annual_total(model.profile(1.5), model.step_days)
    assert one < half < two


def test_a_profile_is_never_negative(model):
    assert (model.profile(-5.0) >= 0).all()


def test_the_band_widens_away_from_the_fitted_range(model):
    inside = model.band(np.array([1.5]))
    outside = model.band(np.array([5.0]))
    width_in = (inside["upper"] - inside["lower"]).mean()
    width_out = (outside["upper"] - outside["lower"]).mean()
    assert width_out > width_in


def test_a_wider_coverage_gives_a_wider_band(model):
    bands = model.bands(np.array([1.5]))
    assert (bands["upper_95"] >= bands["upper_68"]).all()
    assert (bands["lower_95"] <= bands["lower_68"]).all()


def test_leave_one_year_out_holds_each_season_back(discharge, predictors):
    validation = uc1.leave_one_year_out(discharge, predictors)
    assert set(validation.index) == set(range(FIRST_YEAR, LAST_YEAR + 1))
    assert set(validation.columns.get_level_values(0)) == {"observed", "predicted"}
    skill = uc1.profile_skill(validation)
    assert skill["r2"] > 0.99
    assert skill["rmse_m3s"] >= 0


def test_summarise_reports_one_row_per_season(model, predictors):
    table = uc1.summarise(model.profiles(predictors), model.step_days)
    assert len(table) == len(predictors)
    # Later seasons were built larger, so the totals must increase.
    assert table["total_km3"].is_monotonic_increasing


def test_the_peak_lands_in_the_season_it_was_built_in(model):
    peak, start = uc1.annual_peak(model.profile(1.5))
    assert peak > 0
    assert PEAK_DOY - 14 <= start <= PEAK_DOY + 14


def test_a_higher_threshold_is_exceeded_less_often(model, predictors):
    profiles = model.profiles(predictors)
    low = uc1.steps_above(profiles, 100.0)
    high = uc1.steps_above(profiles, 500.0)
    assert (low >= high).all()


def test_catchment_mean_weights_sum_to_one():
    fields = pd.DataFrame(np.array([[1.0, 3.0], [2.0, 4.0]]))
    np.testing.assert_allclose(uc1.catchment_mean(fields).to_numpy(), [2.0, 3.0])
    weighted = uc1.catchment_mean(fields, weights=np.array([0.25, 0.75]))
    np.testing.assert_allclose(weighted.to_numpy(), [2.5, 3.5])


def test_seasonal_predictor_drops_a_half_covered_season():
    dates = pd.date_range("2020-05-01", "2021-08-15", freq="D")
    runoff = pd.Series(1.0, index=dates)
    seasonal = uc1.seasonal_predictor(runoff)
    # 2021 stops mid-August, so it is not a comparable season and must go.
    assert list(seasonal.index) == [2020]


def test_propagation_widens_the_band_and_the_budget_agrees(model):
    without = uc1.propagate(model, 1.5, 0.0, draws=400)
    with_sd = uc1.propagate(model, 1.5, 0.08, draws=400)
    assert (with_sd["upper"] - with_sd["lower"]).mean() > (without["upper"] - without["lower"]).mean()

    budget = uc1.uncertainty_budget(model, 1.5, 0.08, draws=400)
    assert budget.loc["fitting + harmonisation"].iloc[0] > budget.loc["fitting only"].iloc[0]
    assert 0 <= budget.loc["fitting + harmonisation"].iloc[1] <= 1


def test_multiplicative_factor_is_the_antilog_of_the_spread():
    factor = uc1.multiplicative_factor(pd.Series([1.0, 1.0]))
    np.testing.assert_allclose(factor, 10.0)


def test_catchment_sd_is_larger_when_errors_are_correlated():
    sd_fields = pd.DataFrame(np.full((2, 100), 0.1))
    correlated = uc1.catchment_sd(sd_fields, correlated=True)
    independent = uc1.catchment_sd(sd_fields, correlated=False)
    assert (correlated > independent).all()


def test_a_perturbed_scenario_is_repeatable(discharge, predictors, model):
    first = uc1.Scenario.from_perturbation("warm", predictors, weight=0.8, seed=1)
    again = uc1.Scenario.from_perturbation("warm", predictors, weight=0.8, seed=1)
    pd.testing.assert_series_equal(first.values, again.values)
    assert "warm" in str(first)
    assert len(first.run(model)) == len(first.values)


def test_compare_puts_every_scenario_in_the_table(model, predictors):
    runs = {"observed": model.profiles(predictors),
            "warm": uc1.Scenario.from_perturbation("warm", predictors, weight=0.9).run(model)}
    table = uc1.compare(runs, model.step_days)
    assert set(table.index) >= set(runs)


def test_runoff_as_discharge_is_the_naive_baseline(discharge):
    monthly = pd.Series(100.0, index=pd.date_range("2006-06-15", periods=12, freq="MS"))
    out = uc1.runoff_as_discharge(monthly, discharge)
    assert len(out) > 0
    assert (out.to_numpy() >= 0).all()


def test_runoff_as_discharge_shifts_the_season_for_travel_time(discharge):
    months = pd.date_range(f"{FIRST_YEAR}-05-01", periods=5, freq="MS")
    monthly = pd.Series([0.0, 0.0, 100.0, 0.0, 0.0], index=months)

    straight = uc1.runoff_as_discharge(monthly, discharge, lag_steps=0)
    lagged = uc1.runoff_as_discharge(monthly, discharge, lag_steps=1)

    peak = straight["predicted"].loc[FIRST_YEAR].to_numpy().argmax()
    assert lagged["predicted"].loc[FIRST_YEAR].to_numpy().argmax() == peak + 1


# --------------------------------------------------------------------------
# The shape model. Same synthetic season, but the monthly curve is what drives
# it, so the tests are about the curve getting through rather than the total.
# --------------------------------------------------------------------------

@pytest.fixture
def monthly(discharge) -> pd.Series:
    """Monthly runoff shaped like the discharge it should predict."""
    months = pd.date_range(f"{FIRST_YEAR}-05-01", f"{LAST_YEAR}-09-01", freq="MS")
    months = months[months.month.isin(uc1.MELT_MONTHS)]
    shape = np.exp(-((months.dayofyear.to_numpy() - PEAK_DOY) ** 2) / (2 * 40.0**2))
    level = np.array([_levels()[stamp.year] for stamp in months])
    return pd.Series(120.0 * level * shape, index=months)


def test_shape_model_recovers_a_season_it_was_built_from(discharge, monthly):
    model = uc1.ShapeModel.fit(discharge, monthly)
    # Five monthly points interpolated across 22 steps will not match a smooth
    # Gaussian season exactly, so this is a floor rather than a target.
    assert model.r2 > 0.6
    assert model.lag in (0, 1, 2, 3)
    assert len(model.starts) == 22


def test_shape_model_profiles_cover_every_season(discharge, monthly):
    model = uc1.ShapeModel.fit(discharge, monthly)
    profiles = model.profiles(model.years)
    assert list(profiles.index) == list(model.years)
    assert (profiles.to_numpy() >= 0).all()


def test_shape_model_bands_widen_with_coverage(discharge, monthly):
    model = uc1.ShapeModel.fit(discharge, monthly)
    bands = model.bands(FIRST_YEAR + 1, levels=(0.68, 0.95))
    assert (bands["upper_95"] >= bands["upper_68"]).all()
    assert (bands["lower_95"] <= bands["lower_68"]).all()


def test_shape_model_takes_an_extra_predictor_per_season(discharge, monthly):
    extras = pd.DataFrame({"lake": _levels()}, index=list(_levels()))
    model = uc1.ShapeModel.fit(discharge, monthly, extras=extras)
    assert model.extra_names == ["lake"]
    assert len(model.extra_coefficients) == 1


def test_shape_model_uses_an_extras_row_it_is_handed(discharge, monthly):
    # A held-out season knows its own extra predictors even though the model
    # was fitted without them. If the row passed here were ignored, the fitted
    # model would quietly fall back to the training mean and the prediction
    # would not depend on the extra predictor at all - which is exactly what a
    # leave-one-year-out test of that predictor must not do.
    years = list(_levels())
    extras = pd.DataFrame({"lake": _levels()}, index=years)
    held = years[-1]
    model = uc1.ShapeModel.fit(discharge[discharge.index.year != held], monthly,
                               extras=extras.drop(index=held))
    low = model.profile(held, extras=extras.loc[[held]])
    high = model.profile(held, extras=extras.loc[[held]] * 5)
    assert not np.allclose(low.to_numpy(), high.to_numpy())


def test_shape_leave_one_year_out_holds_each_season_back(discharge, monthly):
    out = uc1.shape_leave_one_year_out(discharge, monthly)
    assert set(out.columns.get_level_values(0)) == {"observed", "predicted"}
    assert len(out) == LAST_YEAR - FIRST_YEAR + 1
    assert uc1.profile_skill(out)["r2"] > 0.5


def test_season_shape_is_zero_outside_the_melt_season(discharge, monthly):
    flow = uc1.monthly_flow(monthly)
    curve = uc1.season_shape(flow, FIRST_YEAR, 7, 0)
    assert len(curve) == 22
    assert np.isfinite(curve).all()
    assert curve.min() >= 0.0


def test_model_str_names_the_predictor(model):
    assert "ProfileModel" in str(model)
    assert "catchment sum" in str(model)


# --------------------------------------------------------------------------
# The catchment outline. A unit square with a square hole in it, so that what
# is inside and what is not is arithmetic rather than a judgement.
# --------------------------------------------------------------------------

def _basin_csv(path) -> object:
    rows = [(0, 0.0, 0.0), (0, 10.0, 0.0), (0, 10.0, 10.0), (0, 0.0, 10.0), (0, 0.0, 0.0),
            (1, 4.0, 4.0), (1, 6.0, 4.0), (1, 6.0, 6.0), (1, 4.0, 6.0), (1, 4.0, 4.0)]
    pd.DataFrame(rows, columns=["ring", "x", "y"]).to_csv(path, index=False)
    return path


def test_load_basin_reads_the_exterior_first(tmp_path):
    rings = uc1.load_basin(_basin_csv(tmp_path / "basin.csv"))
    assert len(rings) == 2
    assert rings[0].shape == (5, 2)
    assert rings[0][:, 0].max() == 10.0


def test_basin_nodes_keeps_the_inside_and_drops_the_hole(tmp_path):
    rings = uc1.load_basin(_basin_csv(tmp_path / "basin.csv"))
    nodes = np.array([[5.0, 2.0],      # inside
                      [1.0, 1.0],      # inside
                      [5.0, 5.0],      # in the hole
                      [20.0, 5.0]])    # outside altogether
    inside = uc1.basin_nodes(nodes, rings)
    assert list(inside) == [True, True, False, False]


def test_catchment_mean_can_be_taken_over_part_of_the_mesh():
    fields = pd.DataFrame([[1.0, 1.0, 100.0], [2.0, 2.0, 200.0]],
                          index=pd.to_datetime(["2014-07-01", "2014-08-01"]))
    mask = np.array([True, True, False])
    np.testing.assert_allclose(uc1.catchment_mean(fields, mask).to_numpy(), [1.0, 2.0])


def test_mesh_map_clips_the_surface_to_the_outline(tmp_path):
    rings = uc1.load_basin(_basin_csv(tmp_path / "basin.csv"))
    rings = [ring * 1000.0 for ring in rings]      # the helper works in metres
    rng = np.random.default_rng(0)
    nodes = rng.uniform(-5000, 15000, size=(40, 2))
    figure, ax = plt.subplots()
    surface = uc1.mesh_map(ax, nodes, rng.normal(size=40), rings)
    assert surface.get_clip_path() is not None
    assert ax.get_xlim() == pytest.approx((-4.0, 14.0))   # the outline, plus the margin
    plt.close(figure)


def test_mesh_map_draws_the_triangulation_it_is_given(tmp_path):
    rings = [ring * 1000.0 for ring in uc1.load_basin(_basin_csv(tmp_path / "basin.csv"))]
    nodes = np.array([[0.0, 0.0], [10000.0, 0.0], [10000.0, 10000.0], [0.0, 10000.0]])
    figure, ax = plt.subplots()
    uc1.mesh_map(ax, nodes, np.arange(4.0), rings,
                 np.array([[0, 1, 2], [0, 2, 3]]), edges=True)
    assert ax.lines, "edges=True should draw the mesh"
    assert all(line.get_clip_path() is not None for line in ax.lines)
    plt.close(figure)


def test_headroom_only_lifts_the_top_of_the_axis():
    figure, ax = plt.subplots()
    ax.plot([0, 1], [0.0, 10.0])
    bottom, top = ax.get_ylim()
    uc1.headroom(ax, 0.5)
    assert ax.get_ylim()[0] == pytest.approx(bottom)
    assert ax.get_ylim()[1] == pytest.approx(bottom + (top - bottom) * 1.5)
    plt.close(figure)


def test_label_points_moves_a_label_that_would_sit_on_another():
    figure, ax = plt.subplots()
    # Fixed limits, and two points a pixel or so apart, so the first position
    # one label takes is certain to be in the way of the next.
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.scatter([5.0, 5.02], [5.0, 5.0])
    uc1.label_points(ax, [5.0, 5.02], [5.0, 5.0], ["2019", "2020"])
    first, second = [note.get_position() for note in ax.texts]
    assert first != second
    assert [note.get_text() for note in ax.texts] == ["2019", "2020"]
    plt.close(figure)


def test_month_axis_puts_month_names_on_a_day_of_year_axis():
    figure, ax = plt.subplots()
    ax.plot([121, 200, 273], [1.0, 2.0, 1.0])
    uc1.month_axis(ax)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["May", "Jun", "Jul", "Aug", "Sep", "Oct"]
    assert list(ax.get_xticks()) == [121, 152, 182, 213, 244, 274]
    plt.close(figure)


def test_mesh_map_without_an_outline_draws_the_whole_mesh():
    rng = np.random.default_rng(0)
    nodes = rng.uniform(0, 10000, size=(30, 2))
    figure, ax = plt.subplots()
    surface = uc1.mesh_map(ax, nodes, rng.normal(size=30))
    assert surface.get_clip_path() is None
    plt.close(figure)


# --------------------------------------------------------------------------
# Quantiles and the shape model
# --------------------------------------------------------------------------

def test_the_t_quantile_approaches_the_normal_one_as_dof_grows():
    near_normal = uc1._t_quantile(0.975, 100_000)
    assert abs(near_normal - 1.96) < 0.01
    # Few degrees of freedom must give a wider interval, never a narrower one.
    assert uc1._t_quantile(0.975, 3) > near_normal


def test_the_normal_quantile_is_symmetric():
    assert abs(uc1._normal_quantile(0.5)) < 1e-9
    assert uc1._normal_quantile(0.975) == pytest.approx(-uc1._normal_quantile(0.025), abs=1e-6)


def test_monthly_flow_converts_millimetres_to_cubic_metres_a_second():
    index = pd.DatetimeIndex(["2014-07-01"])
    flow = uc1.monthly_flow(pd.Series([31.0], index=index), area_m2=86_400_000.0)
    # 31 mm over that area in 31 days works out at exactly 1 m3/s.
    np.testing.assert_allclose(flow.to_numpy(), [1.0])


def test_season_shape_profiles_returns_an_observed_predicted_frame(discharge):
    months = pd.date_range(f"{FIRST_YEAR}-05-01", f"{LAST_YEAR}-09-01", freq="MS")
    months = months[months.month.isin(uc1.MELT_MONTHS)]
    shape = np.exp(-((months.dayofyear.to_numpy() - PEAK_DOY) ** 2) / (2 * 40.0**2))
    level = np.array([_levels()[stamp.year] for stamp in months])
    monthly = pd.Series(120.0 * level * shape, index=months)

    out = uc1.season_shape_profiles(monthly, discharge)

    assert set(out.columns.get_level_values(0)) == {"observed", "predicted"}
    assert len(out) > 0
    assert uc1.profile_skill(out)["r2"] > 0.5


def test_catchment_mean_ignores_columns_with_no_weight():
    # A field interpolated onto a grid is undefined outside the mesh. Those
    # columns carry no weight, and must not take the average with them.
    frame = pd.DataFrame([[1.0, 3.0, np.nan], [2.0, 4.0, np.nan]])
    weights = np.array([1.0, 1.0, 0.0])

    series = uc1.catchment_mean(frame, weights)

    np.testing.assert_allclose(series.to_numpy(), [2.0, 3.0])


def test_catchment_mean_keeps_a_missing_value_that_does_carry_weight():
    # A gap inside the catchment is a real gap and should stay visible.
    frame = pd.DataFrame([[1.0, np.nan]])
    series = uc1.catchment_mean(frame, np.array([1.0, 1.0]))
    assert np.isnan(series.to_numpy()[0])


def test_catchment_mean_refuses_weights_that_are_all_zero():
    with pytest.raises(ValueError, match="nothing to average"):
        uc1.catchment_mean(pd.DataFrame([[1.0, 2.0]]), np.zeros(2))
