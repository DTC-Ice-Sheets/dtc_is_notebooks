"""Tests for uc4_plotting_helpers."""

from __future__ import annotations

from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

matplotlib.use("Agg")
from matplotlib.backend_bases import MouseEvent  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from dtc_is_notebook_helpers import uc4_plotting_helpers as h  # noqa: E402

# ----------------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------------

N_TIME = 24  # emulator months: 2020-01 .. 2021-12 (stamped on the 15th)
RACMO_START = 0  # month offset (from 2020-01) of the first RACMO step
N_RACMO = 18  # RACMO months (stamped on the 1st): 2020-01 .. 2021-06
NY, NX = 5, 6  # output tile (RACMO grid)
NY_E, NX_E = 4, 3  # ERA5 input tile
BIAS = -0.5  # emulator - RACMO, by construction
PIXEL_J, PIXEL_I = 1, 2  # pixel that the requested location falls into


def _value(m, j, i):
    """Field value for month offset `m` and pixel (j, i). Mixed sign, unique per cell."""
    return (m - 10) + 0.1 * j + 0.01 * i


def make_dataset(n_time=N_TIME, n_racmo=N_RACMO, racmo_start=RACMO_START, pred_day=15) -> xr.Dataset:
    """Build a dataset with the same layout as the emulator output.

    Emulator = value(m); RACMO = value(m) - BIAS, so emulator - RACMO == BIAS for every matching month,
    even when RACMO starts on a different month than the emulator. The emulator months are stamped on day
    `pred_day` of each month (the real output is stamped on the 1st).
    """
    x = 100_000 + 5_500 * np.arange(NX, dtype=float)
    y = -500_000 - 5_500 * np.arange(NY, dtype=float)  # descending, like a raster
    x_e = 90_000 + 27_500 * np.arange(NX_E, dtype=float)
    y_e = -480_000 - 27_500 * np.arange(NY_E, dtype=float)

    t_pred = (pd.date_range("2020-01-01", periods=n_time, freq="MS") + pd.Timedelta(days=pred_day - 1)).to_numpy()
    t_ref = pd.date_range("2020-01-01", periods=n_racmo, freq="MS").shift(racmo_start, freq="MS").to_numpy()
    t_pred, t_ref = t_pred.astype("datetime64[ns]"), t_ref.astype("datetime64[ns]")

    jj, ii = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    pred = np.stack([_value(m, jj, ii) for m in range(n_time)]).astype("float32")
    ref = np.stack([_value(m + racmo_start, jj, ii) - BIAS for m in range(n_racmo)]).astype("float32")

    je, ie = np.meshgrid(np.arange(NY_E), np.arange(NX_E), indexing="ij")
    tp = np.stack([m + 0.001 * (je * NX_E + ie) for m in range(n_time)])
    t2m = np.stack([250.0 + m + 0 * je for m in range(n_time)])
    era5 = np.stack([tp, t2m], axis=1).astype("float32")  # (time, channels, y_era5, x_era5)

    return xr.Dataset(
        {
            "predictions": (("time", "y", "x"), pred),
            "target_patches": (("time_r", "y", "x"), ref),
            "x_cropped": (("time", "channels", "y_era5", "x_era5"), era5),
        },
        coords={
            "time": t_pred,
            "time_r": t_ref,
            "x": x,
            "y": y,
            "channels": ["tp", "t2m"],
            "x_era5": x_e,
            "y_era5": y_e,
        },
        attrs={
            "lon": -65.1279,
            "lat": -75.0076,
            "x_3031": float(x[PIXEL_I] + 100.0),  # slightly off the cell centre: must snap to the nearest pixel
            "y_3031": float(y[PIXEL_J] - 50.0),
        },
    )


@pytest.fixture
def ds() -> xr.Dataset:
    """Return a synthetic dataset with the same layout as the emulator output."""
    return make_dataset()


@pytest.fixture
def ds_no_racmo(ds) -> xr.Dataset:
    """Return a synthetic dataset without RACMO."""
    return ds.drop_vars(["target_patches", "time_r"])


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all figures after each test to avoid memory leaks."""
    yield
    plt.close("all")


@pytest.fixture
def offline_coastlines(monkeypatch):
    """`GeoAxes.coastlines` downloads Natural Earth shapefiles; replace it with a no-op."""
    from cartopy.mpl.geoaxes import GeoAxes

    monkeypatch.setattr(GeoAxes, "coastlines", lambda self, *a, **k: None)


# ----------------------------------------------------------------------------
# find_output_path
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("outputs", "name", "expected"),
    [
        (
            {"a": {"output_path": ["s3://b/rcm_data.zarr", "s3://b/era5_data.zarr/"]}},
            "era5_data",
            "s3://b/era5_data.zarr/",
        ),
        ({"a": {"output_path": ["s3://b/rcm_data.zarr", "s3://b/era5_data.zarr"]}}, "rcm_data", "s3://b/rcm_data.zarr"),
        (
            {"a": {"output_path": "s3://b/rcm-emulator-results.zarr"}},
            "rcm-emulator-results",
            "s3://b/rcm-emulator-results.zarr",
        ),
        (
            {"a": {"output_path": ["s3://era5_data/rcm_data.zarr", "s3://b/era5_data.zarr"]}},
            "era5_data",
            "s3://b/era5_data.zarr",
        ),
        (
            {"a": {"output_path": "https://h/era5_data.zarr?sig=rcm_data"}},
            "era5_data",
            "https://h/era5_data.zarr?sig=rcm_data",
        ),
        (
            {"a": {"output_path": "s3://b/era5_data.zarr"}, "b": {"status": "done"}},
            "era5_data",
            "s3://b/era5_data.zarr",
        ),
        ({"a": {"output_path": "https://h/era5_data.zarr?sig=rcm_data"}}, "rcm_data", ValueError),
        ({"a": {"output_path": ["s3://b/rcm_data.zarr", "s3://b/era5_data.zarr"]}}, "_data", ValueError),
    ],
)
def test_find_output_path(outputs, name, expected):
    """Test that `find_output_path` picks the one output whose store name matches, and raises otherwise."""
    if expected is ValueError:
        with pytest.raises(ValueError, match="Expected exactly one output"):
            h.find_output_path(outputs, name)
    else:
        assert h.find_output_path(outputs, name) == expected


# ----------------------------------------------------------------------------
# reference_on_prediction_time
# ----------------------------------------------------------------------------


def test_reference_on_prediction_time_shape_dims_name_and_dtype(ds):
    """Test that the function returns a DataArray with the correct shape, dimensions, name, and dtype."""
    ref = h.reference_on_prediction_time(ds)
    assert ref.dims == ("time", "y", "x")
    assert ref.shape == ds["predictions"].shape
    assert ref.name == "racmo_reference"
    assert ref.dtype == np.float32


def test_reference_on_prediction_time_time_axis_is_the_emulators(ds):
    """Test that the time axis of the returned DataArray matches the emulator's time axis."""
    ref = h.reference_on_prediction_time(ds)
    np.testing.assert_array_equal(ref["time"].values, ds["time"].values)


def test_reference_on_prediction_time_months_match_despite_different_day_stamping(ds):
    """Test that the function correctly matches months despite different day stamping."""
    # emulator stamped on the 15th, RACMO on the 1st: matching must be by calendar month
    ref = h.reference_on_prediction_time(ds)
    for t in range(N_RACMO):
        np.testing.assert_allclose(ref.values[t], ds["target_patches"].values[t])


def test_reference_on_prediction_time_nan_where_racmo_has_no_month(ds):
    """Test that the function returns NaN for months where RACMO has no data."""
    ref = h.reference_on_prediction_time(ds)
    assert np.isnan(ref.values[N_RACMO:]).all()
    assert np.isfinite(ref.values[:N_RACMO]).all()


def test_reference_on_prediction_time_racmo_starting_later_is_aligned_by_month_not_index():
    """Test that the function aligns by month, not by index, when RACMO starts later."""
    start = 3  # RACMO begins in 2020-04
    ds = make_dataset(racmo_start=start, n_racmo=10)
    ref = h.reference_on_prediction_time(ds)
    assert np.isnan(ref.values[:start]).all()
    for t in range(start, start + 10):
        expected = _value(t, 0, 0) - BIAS  # value is defined per calendar month
        assert ref.values[t, 0, 0] == pytest.approx(expected, abs=1e-5)
    assert np.isnan(ref.values[start + 10 :]).all()


def test_reference_on_prediction_time_missing_target_patches_gives_all_nan(ds_no_racmo):
    """Test that the function returns all NaN when target_patches is missing."""
    ref = h.reference_on_prediction_time(ds_no_racmo)
    assert ref.shape == ds_no_racmo["predictions"].shape
    assert np.isnan(ref.values).all()
    assert ref.name == "racmo_reference"


def test_reference_on_prediction_time_target_patches_without_time_r_gives_all_nan(ds):
    """Test that the function returns all NaN when target_patches is present but time_r is missing."""
    ref = h.reference_on_prediction_time(ds.drop_vars("time_r"))
    assert np.isnan(ref.values).all()


def test_reference_on_prediction_time_time_r_without_target_patches_gives_all_nan(ds):
    """Test that the function returns all NaN when time_r is present but target_patches is missing."""
    ref = h.reference_on_prediction_time(ds.drop_vars("target_patches"))
    assert np.isnan(ref.values).all()


def test_reference_on_prediction_time_several_steps_in_one_month_raises(ds):
    """Test that the function raises a ValueError when there are several steps in one month."""
    t = ds["time_r"].values.copy()
    t[1] = t[0] + np.timedelta64(5, "D")  # same month as t[0]
    with pytest.raises(ValueError, match="several steps per month"):
        h.reference_on_prediction_time(ds.assign_coords(time_r=t))


def test_reference_on_prediction_time_input_dataset_is_not_modified(ds):
    """Test that the function does not modify the input dataset."""
    before = ds["target_patches"].values.copy()
    h.reference_on_prediction_time(ds)
    np.testing.assert_array_equal(ds["target_patches"].values, before)


# ----------------------------------------------------------------------------
# point_series
# ----------------------------------------------------------------------------


def test_point_series_columns_and_index(ds):
    """Test that the function returns a DataFrame with the correct columns and index."""
    df = h.point_series(ds)
    assert list(df.columns) == ["emulator", "racmo", "era5_tp"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == N_TIME


def test_point_series_emulator_taken_from_nearest_pixel(ds):
    """Test that the emulator values are taken from the nearest pixel to the requested coordinates."""
    df = h.point_series(ds)
    expected = ds["predictions"].values[:, PIXEL_J, PIXEL_I]
    np.testing.assert_allclose(df["emulator"].to_numpy(), expected, rtol=1e-5)


def test_point_series_racmo_is_aligned_and_nan_after_it_ends(ds):
    """Test that the RACMO values are aligned with the emulator and NaN after RACMO ends."""
    df = h.point_series(ds)
    assert df["racmo"].iloc[:N_RACMO].notna().all()
    assert df["racmo"].iloc[N_RACMO:].isna().all()
    np.testing.assert_allclose(
        df["racmo"].iloc[:N_RACMO].to_numpy(),
        ds["target_patches"].values[:, PIXEL_J, PIXEL_I],
        rtol=1e-5,
    )


def test_point_series_era5_column_is_tile_mean(ds):
    """Test that the ERA5 column is the mean of the tile for the specified channel."""
    df = h.point_series(ds)
    expected = ds["x_cropped"].sel(channels="tp").mean(("y_era5", "x_era5")).values
    np.testing.assert_allclose(df["era5_tp"].to_numpy(), expected, rtol=1e-5)


def test_point_series_other_era5_channel(ds):
    """Test that the function can return a different ERA5 channel when specified."""
    df = h.point_series(ds, era5_channel="t2m")
    assert "era5_t2m" in df.columns
    assert "era5_tp" not in df.columns
    assert df["era5_t2m"].iloc[0] == pytest.approx(250.0)


def test_point_series_unknown_channel_raises(ds):
    """Test that the function raises a KeyError when an unknown ERA5 channel is specified."""
    with pytest.raises(KeyError):
        h.point_series(ds, era5_channel="nope")


def test_point_series_without_racmo_column_is_all_nan(ds_no_racmo):
    """Test that the RACMO column is all NaN when target_patches is missing."""
    df = h.point_series(ds_no_racmo)
    assert df["racmo"].isna().all()
    assert df["emulator"].notna().all()


# ----------------------------------------------------------------------------
# skill_metrics
# ----------------------------------------------------------------------------


def test_skill_metrics_perfect_agreement():
    """Test that the skill metrics are all zero when the emulator and RACMO are identical."""
    s = pd.Series([1.0, -2.0, 3.0, 5.0])
    m = h.skill_metrics(pd.DataFrame({"emulator": s, "racmo": s}))
    assert m["n"] == 4
    assert m["bias"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["r"] == pytest.approx(1.0)


def test_skill_metrics_constant_offset():
    """Test that the skill metrics reflect a constant offset between the emulator and RACMO."""
    racmo = pd.Series([1.0, 2.0, 4.0, 8.0])
    m = h.skill_metrics(pd.DataFrame({"emulator": racmo + 2.0, "racmo": racmo}))
    assert m["bias"] == pytest.approx(2.0)
    assert m["rmse"] == pytest.approx(2.0)
    assert m["mae"] == pytest.approx(2.0)
    assert m["r"] == pytest.approx(1.0)


def test_skill_metrics_known_values():
    """Test that the skill metrics are calculated correctly for known values."""
    emu, ref = np.array([1.0, 2.0, 4.0]), np.array([1.0, 3.0, 2.0])
    err = emu - ref  # [0, -1, 2]
    m = h.skill_metrics(pd.DataFrame({"emulator": emu, "racmo": ref}))
    assert m["n"] == 3
    assert m["bias"] == pytest.approx(err.mean())
    assert m["rmse"] == pytest.approx(np.sqrt((err**2).mean()))
    assert m["mae"] == pytest.approx(np.abs(err).mean())
    assert m["r"] == pytest.approx(np.corrcoef(emu, ref)[0, 1])


def test_skill_metrics_sign_convention_bias_is_emulator_minus_racmo():
    """Test that the bias is calculated as emulator minus RACMO."""
    m = h.skill_metrics(pd.DataFrame({"emulator": [1.0, 2.0], "racmo": [3.0, 5.0]}))
    assert m["bias"] < 0


def test_skill_metrics_perfect_anticorrelation():
    """Test that the skill metrics show perfect anticorrelation."""
    m = h.skill_metrics(pd.DataFrame({"emulator": [1.0, 2.0, 3.0], "racmo": [3.0, 2.0, 1.0]}))
    assert m["r"] == pytest.approx(-1.0)


def test_skill_metrics_nan_rows_are_dropped_pairwise():
    """Test that rows with NaN in either column are dropped pairwise."""
    df = pd.DataFrame({"emulator": [1.0, 2.0, np.nan, 4.0, 9.0], "racmo": [1.0, np.nan, 3.0, 4.0, 9.0]})
    m = h.skill_metrics(df)
    assert m["n"] == 3  # rows 0, 3, 4
    assert m["bias"] == pytest.approx(0.0)


@pytest.mark.parametrize("n_valid", [0, 1])
def test_skill_metrics_fewer_than_two_pairs_returns_nans(n_valid):
    """Test that the skill metrics return NaN when there are fewer than two valid pairs."""
    df = pd.DataFrame({"emulator": [1.0, 2.0, 3.0], "racmo": [np.nan] * 3})
    df.iloc[:n_valid, 1] = 1.0
    m = h.skill_metrics(df)
    assert m["n"] == n_valid
    assert all(np.isnan(m[k]) for k in ("bias", "rmse", "mae", "r"))


def test_skill_metrics_on_synthetic_dataset(ds):
    """Test the skill metrics on a synthetic dataset."""
    m = h.skill_metrics(h.point_series(ds))
    assert m["n"] == N_RACMO  # only months with both series
    assert m["bias"] == pytest.approx(BIAS, abs=1e-4)
    assert m["rmse"] == pytest.approx(abs(BIAS), abs=1e-4)
    assert m["mae"] == pytest.approx(abs(BIAS), abs=1e-4)
    assert m["r"] == pytest.approx(1.0)


# ----------------------------------------------------------------------------
# era5_label
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(("channel", "unit"), list(h.ERA5_UNITS.items()))
def test_era5_label_known_channels_show_their_units(channel, unit):
    """Test that known channels show their units."""
    label = h.era5_label(channel)
    assert channel in label
    assert f"[{unit}]" in label


def test_era5_label_unknown_channel_shows_question_mark():
    """Test that unknown channels show a question mark in the label."""
    assert "[?]" in h.era5_label("foo")


def test_era5_label_mentions_raw_input():
    """Test that the label mentions 'raw input'."""
    assert "raw input" in h.era5_label("tp")


# ----------------------------------------------------------------------------
# POLAR projection
# ----------------------------------------------------------------------------


def test_polar_projection_matches_epsg3031():
    """`POLAR` (true-scale latitude -71) must agree with EPSG:3031 to well under a metre."""
    import cartopy.crs as ccrs
    import pyproj

    lon, lat = -65.1279, -75.0076
    ref = pyproj.Transformer.from_crs(4326, 3031, always_xy=True).transform(lon, lat)
    x, y = h.POLAR.transform_point(lon, lat, ccrs.PlateCarree())
    assert (x, y) == pytest.approx(ref, abs=1.0)


def test_cartopy_default_south_polar_stereo_does_not_match_epsg3031():
    """Why POLAR exists: cartopy's default `SouthPolarStereo()` is ~40 km off at this latitude."""
    import cartopy.crs as ccrs
    import pyproj

    lon, lat = -65.1279, -75.0076
    ref = pyproj.Transformer.from_crs(4326, 3031, always_xy=True).transform(lon, lat)
    x, _ = ccrs.SouthPolarStereo().transform_point(lon, lat, ccrs.PlateCarree())
    assert abs(x - ref[0]) > 10_000


# ----------------------------------------------------------------------------
# build_explorer
# ----------------------------------------------------------------------------


def test_build_explorer_returns_plotly_figure(ds):
    """Test that the function returns a Plotly Figure."""
    assert isinstance(h.build_explorer(ds), go.Figure)


def test_build_explorer_one_frame_and_slider_step_per_month(ds):
    """Test that the figure has one frame and one slider step for each month in the dataset."""
    fig = h.build_explorer(ds)
    assert len(fig.frames) == N_TIME
    steps = fig.layout.sliders[0].steps
    assert len(steps) == N_TIME
    assert steps[0].label == "2020-01"
    assert steps[-1].label == "2021-12"


def test_build_explorer_frames_target_the_dynamic_traces(ds):
    """Test that each frame targets the correct dynamic traces in the figure."""
    fig = h.build_explorer(ds)
    assert all(list(f.traces) == [0, 1, 2, 3, 7, 8, 9] for f in fig.frames)
    assert [f.name for f in fig.frames][:3] == ["0", "1", "2"]


def test_build_explorer_stride_subsamples_maps(ds):
    """Test that the stride parameter subsamples the maps correctly, reducing their resolution."""
    fig = h.build_explorer(ds, stride=2)
    assert np.asarray(fig.data[1].z).shape == (int(np.ceil(NY / 2)), int(np.ceil(NX / 2)))
    fig1 = h.build_explorer(ds, stride=1)
    assert np.asarray(fig1.data[1].z).shape == (NY, NX)


def test_build_explorer_map_axes_are_in_km(ds):
    """Test that the map axes are in kilometers, not meters, by checking the x-coordinates of the mesh traces."""
    fig = h.build_explorer(ds, stride=1)
    np.testing.assert_allclose(np.asarray(fig.data[1].x), ds["x"].values / 1e3)


def test_build_explorer_smb_colour_axis_is_symmetric_about_zero(ds):
    """Test that the SMB colour axis is symmetric about zero, with cmin = -cmax and cmax > 0."""
    ca = h.build_explorer(ds).layout.coloraxis
    assert ca.cmin == pytest.approx(-ca.cmax)
    assert ca.cmax > 0


def test_build_explorer_placeholder_visible_when_racmo_absent(ds_no_racmo):
    """Test that the placeholder trace is visible when RACMO reference data is absent."""
    fig = h.build_explorer(ds_no_racmo)
    assert fig.data[3].visible is True
    assert "No RACMO reference" in fig.data[3].text[0]


def test_build_explorer_placeholder_toggles_per_frame(ds):
    """Test that the placeholder trace visibility toggles correctly per frame."""
    fig = h.build_explorer(ds)
    visible = [f.data[3].visible for f in fig.frames]
    assert visible[:N_RACMO] == [False] * N_RACMO
    assert visible[N_RACMO:] == [True] * (N_TIME - N_RACMO)


def test_build_explorer_racmo_trace_name(ds, ds_no_racmo):
    """Test that the RACMO trace has the correct name when RACMO data is present and a placeholder otherwise."""
    assert h.build_explorer(ds).data[4].name == "RACMO"
    assert h.build_explorer(ds_no_racmo).data[4].name == "RACMO (unavailable)"


def test_build_explorer_rmse_band_present_only_with_racmo(ds, ds_no_racmo):
    """Test that the RMSE band trace is present when RACMO data is available and absent when it is not."""
    names = [t.name for t in h.build_explorer(ds).data]
    assert "Emulator \u00b1 RMSE" in names
    names = [t.name for t in h.build_explorer(ds_no_racmo).data]
    assert "Emulator \u00b1 RMSE" not in names


def test_build_explorer_time_series_traces_hold_the_point_series(ds):
    """Test that the time series traces in the figure hold the same data as returned by point_series()."""
    fig = h.build_explorer(ds)
    df = h.point_series(ds)
    np.testing.assert_allclose(np.asarray(fig.data[5].y, dtype=float), df["emulator"].to_numpy())
    np.testing.assert_allclose(np.asarray(fig.data[6].y, dtype=float), df["era5_tp"].to_numpy())


def test_build_explorer_marker_follows_the_frame(ds):
    """Test that the marker in the time series plot follows the frame."""
    fig = h.build_explorer(ds)
    df = h.point_series(ds)
    frame = fig.frames[5]
    assert frame.data[5].y[0] == pytest.approx(df["emulator"].iloc[5])  # emulator marker
    assert pd.Timestamp(frame.data[5].x[0]) == df.index[5]


def test_build_explorer_constant_era5_gets_nonzero_colour_range(ds):
    """Test that when the ERA5 tile is constant, the colour axis still has a nonzero range."""
    ds["x_cropped"][:] = 3.0
    ca2 = h.build_explorer(ds).layout.coloraxis2
    assert ca2.cmax > ca2.cmin


def test_build_explorer_requested_location_star_in_each_map(ds):
    """Test that a star marker is present in each map at the requested location."""
    fig = h.build_explorer(ds)
    stars = [t for t in fig.data if isinstance(t, go.Scatter) and t.marker.symbol == "star"]
    assert len(stars) == 3
    assert all(s.x[0] == pytest.approx(ds.attrs["x_3031"] / 1e3) for s in stars)


@pytest.mark.parametrize("drop_racmo", [False, True])
def test_build_explorer_maps_cover_the_whole_tile(ds, drop_racmo):
    """Test that each map's axes span its tile, also when the RACMO map has no data to autoscale on."""
    if drop_racmo:
        ds = ds.drop_vars(["target_patches", "time_r"])
    layout = h.build_explorer(ds, stride=1).layout
    grids = {"": ("x_era5", "y_era5"), "2": ("x", "y"), "3": ("x", "y")}  # ERA5, emulator, RACMO
    for suffix, (xname, yname) in grids.items():
        x_km, y_km = ds[xname].values / 1e3, ds[yname].values / 1e3
        x_range, y_range = layout[f"xaxis{suffix}"].range, layout[f"yaxis{suffix}"].range
        assert x_range[0] < x_km.min() and x_range[1] > x_km.max()
        assert y_range[0] < y_km.min() and y_range[1] > y_km.max()


def test_build_explorer_figure_is_json_serialisable(ds):
    """Test that the figure can be serialized to JSON, which is important for web applications."""
    assert h.build_explorer(ds).to_json()


def test_build_explorer_works_with_nan_racmo_everywhere_and_in_pieces(ds):
    """Test that the function works correctly when RACMO data is NaN in some places."""
    ds["target_patches"][3] = np.nan  # one RACMO month entirely NaN
    fig = h.build_explorer(ds)
    assert fig.frames[3].data[3].visible is True


# ----------------------------------------------------------------------------
# plot_single_date / plot_validation
# ----------------------------------------------------------------------------

DATE_FUNCS = [h.plot_single_date, h.plot_validation]
DATE_FUNC_IDS = ["plot_single_date", "plot_validation"]


@pytest.mark.parametrize("func", DATE_FUNCS, ids=DATE_FUNC_IDS)
def test_date_plots_returns_a_figure(ds, func, offline_coastlines):
    """Test that the function returns a matplotlib Figure for a valid date."""
    # NB: both are annotated `-> None` but return the figure.
    assert isinstance(func(ds, datetime(2020, 6, 15)), Figure)


@pytest.mark.parametrize("func", DATE_FUNCS, ids=DATE_FUNC_IDS)
def test_date_plots_date_before_dataset_raises(ds, func, offline_coastlines):
    """Test that the function raises a ValueError when the date is before the dataset's range."""
    with pytest.raises(ValueError, match="not in the dataset"):
        func(ds, datetime(2019, 1, 1))


@pytest.mark.parametrize("func", DATE_FUNCS, ids=DATE_FUNC_IDS)
def test_date_plots_date_after_dataset_raises(ds, func, offline_coastlines):
    """Test that the function raises a ValueError when the date is after the dataset's range."""
    with pytest.raises(ValueError, match="not in the dataset"):
        func(ds, datetime(2030, 1, 1))


@pytest.mark.parametrize("func", DATE_FUNCS, ids=DATE_FUNC_IDS)
def test_date_plots_first_date_is_accepted(ds, func, offline_coastlines):
    """Test that the function accepts the first date in the dataset."""
    assert isinstance(func(ds, datetime(2020, 1, 16)), Figure)


@pytest.mark.parametrize("func", DATE_FUNCS, ids=DATE_FUNC_IDS)
def test_date_plots_last_date_is_accepted(func, offline_coastlines):
    """Test that the function accepts the last date in the dataset.

    Uses a dataset whose RACMO record covers every month: `plot_validation` needs a reference for the
    selected month, and the default fixture's RACMO ends in 2021-06.
    """
    full = make_dataset(n_racmo=N_TIME)
    assert isinstance(func(full, datetime(2021, 12, 14)), Figure)


def test_plot_validation_raises_for_month_without_racmo(ds, offline_coastlines):
    """Test that plot_validation refuses a month that has no RACMO reference."""
    with pytest.raises(ValueError, match="No RACMO reference"):
        h.plot_validation(ds, datetime(2021, 12, 14))


def test_plot_validation_raises_without_racmo_in_dataset(ds_no_racmo, offline_coastlines):
    """Test that plot_validation refuses a dataset that has no RACMO reference at all."""
    with pytest.raises(ValueError, match="No RACMO reference"):
        h.plot_validation(ds_no_racmo, datetime(2020, 6, 15))


def test_plot_single_date_has_four_map_panels_and_three_colourbars(ds, offline_coastlines):
    """Test that the figure returned by plot_single_date has four map panels and three colourbars."""
    fig = h.plot_single_date(ds, datetime(2020, 6, 15))
    assert len(fig.axes) == 7


MONTH_CASES = [
    (datetime(2020, 3, 21), "2020-03"),  # after mid-month: must not jump to the next month
    (datetime(2020, 3, 1), "2020-03"),
    (datetime(2020, 1, 1), "2020-01"),  # first day of the first month
    (datetime(2021, 12, 31), "2021-12"),  # last day of the last month
]


@pytest.mark.parametrize("pred_day", [1, 15])
@pytest.mark.parametrize(("date", "month"), MONTH_CASES)
@pytest.mark.parametrize("func", DATE_FUNCS, ids=DATE_FUNC_IDS)
def test_date_plots_use_the_month_of_the_date(func, date, month, pred_day):
    """Test that both date plots show the calendar month of the requested date, for any day-of-month stamping."""
    full = make_dataset(n_racmo=N_TIME, pred_day=pred_day)  # RACMO for every month, so validation always works
    assert func(full, date)._suptitle.get_text().endswith(month)


@pytest.mark.parametrize("pred_day", [1, 15])
@pytest.mark.parametrize(("date", "month"), MONTH_CASES)
def test_plot_single_date_draws_the_predictions_of_the_month(date, month, pred_day):
    """Test that plot_single_date draws the predictions of the requested calendar month."""
    ds = make_dataset(pred_day=pred_day)
    fig = h.plot_single_date(ds, date)
    expected = ds["predictions"].isel(time=list(pd.DatetimeIndex(ds["time"].values).strftime("%Y-%m")).index(month))
    np.testing.assert_array_equal(np.asarray(fig.axes[2].collections[0].get_array()), expected.values)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize(
    ("drop_racmo", "date"),
    [(False, datetime(2021, 12, 1)), (True, datetime(2020, 6, 1))],
    ids=["month-after-racmo-ends", "no-racmo-in-dataset"],
)
def test_plot_single_date_without_racmo_shows_placeholder(ds, drop_racmo, date):
    """Test that a month without RACMO gets a placeholder and a colour range from the prediction only."""
    if drop_racmo:
        ds = ds.drop_vars(["target_patches", "time_r"])
    fig = h.plot_single_date(ds, date)
    assert any("No RACMO reference" in t.get_text() for t in fig.axes[1].texts)
    predictions = np.asarray(fig.axes[2].collections[0].get_array())
    norm = fig.axes[2].collections[0].norm
    assert (norm.vmin, norm.vmax) == pytest.approx((predictions.min(), predictions.max()))


def test_plot_single_date_with_racmo_has_no_placeholder(ds):
    """Test that a month with RACMO data does not show the placeholder."""
    fig = h.plot_single_date(ds, datetime(2020, 6, 1))
    assert not any("No RACMO reference" in t.get_text() for t in fig.axes[1].texts)


def test_plot_single_date_draws_location_star_once(ds):
    """Test that the tile-location panel draws the requested-location star once."""
    fig = h.plot_single_date(ds, datetime(2020, 6, 1))
    assert len(fig.axes[3].lines) == 1


def test_plot_validation_bias_panel_is_symmetric_about_zero(ds):
    """Test that the bias panel in the validation plot is symmetric about zero, with vmin = -vmax."""
    fig = h.plot_validation(ds, datetime(2020, 6, 15))
    mesh = fig.axes[0].collections[0]
    assert mesh.norm.vmin == pytest.approx(-mesh.norm.vmax)


def test_plot_validation_bias_is_predictions_minus_target(ds):
    """Test that the bias panel in the validation plot is computed as predictions minus target."""
    fig = h.plot_validation(ds, datetime(2020, 6, 15))
    drawn = np.asarray(fig.axes[0].collections[0].get_array())
    # 2020-06 emulator month index 5; RACMO matches it by construction
    np.testing.assert_allclose(drawn, BIAS, atol=1e-4)


def test_plot_validation_rmse_panel_starts_at_zero(ds):
    """Test that the RMSE panel in the validation plot has a colour scale starting at zero."""
    fig = h.plot_validation(ds, datetime(2020, 6, 15))
    rmse_mesh = next(c for ax in fig.axes for c in ax.collections if c.get_cmap().name == "inferno_r")
    assert rmse_mesh.norm.vmin == 0.0


def test_plot_validation_ignores_nan_in_distributions(ds):
    """Test that the validation plot ignores NaN values when computing distributions."""
    ds["predictions"][5, 0, 0] = np.nan
    ds["target_patches"][5, 1, 1] = np.nan
    assert isinstance(h.plot_validation(ds, datetime(2020, 6, 15)), Figure)


def test_plot_validation_with_no_finite_values_raises(ds):
    """Test that the validation plot raises a ValueError when there are no finite values."""
    # RACMO stays finite so the RACMO-availability check passes; the distribution check must then trigger.
    ds["predictions"][:] = np.nan
    with pytest.raises(ValueError, match="at least one finite value"):
        h.plot_validation(ds, datetime(2020, 6, 15))


# ----------------------------------------------------------------------------
# select_location
# ----------------------------------------------------------------------------


@pytest.fixture
def no_show_no_coast(monkeypatch):
    """Prevent the tests from showing a matplotlib window and from trying to draw coastlines."""
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)


def _click_at(fig, ax, lon, lat, button=1):
    """Fire a matplotlib click at the display position of (lon, lat)."""
    import cartopy.crs as ccrs

    fig.canvas.draw()
    data = ax.projection.transform_point(lon, lat, ccrs.PlateCarree())
    px, py = ax.transData.transform(data)
    fig.canvas.callbacks.process("button_press_event", MouseEvent("button_press_event", fig.canvas, px, py, button))


def test_select_location_returns_initial_location_without_clicks(no_show_no_coast):
    """Test that the function returns the initial location when no clicks are made."""
    assert h.select_location() == {"lat": -75.0076, "lon": -65.1279}


def test_select_location_custom_initial_location(no_show_no_coast):
    """Test that the function returns the custom initial location when specified."""
    assert h.select_location(lat=-80.0, lon=10.0) == {"lat": -80.0, "lon": 10.0}


def test_select_location_left_click_updates_selection_and_prints(capsys, no_show_no_coast):
    """Test that a left click updates the selection and prints the new location."""
    sel = h.select_location()
    fig = plt.gcf()
    _click_at(fig, fig.axes[0], lon=30.0, lat=-75.0)
    assert sel["lat"] == pytest.approx(-75.0, abs=0.5)
    assert sel["lon"] == pytest.approx(30.0, abs=1.0)
    assert "Selected location" in capsys.readouterr().out


def test_select_location_longitude_is_wrapped_to_pm180(no_show_no_coast):
    """Test that the longitude is wrapped to the range [-180, 180] when a click is made."""
    sel = h.select_location()
    fig = plt.gcf()
    _click_at(fig, fig.axes[0], lon=170.0, lat=-75.0)
    assert -180 <= sel["lon"] <= 180
    assert sel["lon"] == pytest.approx(170.0, abs=1.0)


def test_select_location_click_outside_the_axes_is_ignored(no_show_no_coast):
    """Test that a click outside the axes does not change the selection."""
    sel = h.select_location()
    fig = plt.gcf()
    fig.canvas.draw()
    fig.canvas.callbacks.process("button_press_event", MouseEvent("button_press_event", fig.canvas, 1, 1, 1))
    assert sel == {"lat": -75.0076, "lon": -65.1279}


def test_select_location_click_north_of_60s_is_rejected_with_hint(no_show_no_coast):
    """Test that a click north of 60S is rejected and a hint is displayed on the plot."""
    sel = h.select_location()
    fig = plt.gcf()
    ax = fig.axes[0]
    fig.canvas.draw()
    # far corner of the (square) axes lies outside the 60S circle
    px, py = ax.transData.transform((ax.get_xlim()[1], ax.get_ylim()[1]))
    px, py = px - 2, py - 2
    fig.canvas.callbacks.process("button_press_event", MouseEvent("button_press_event", fig.canvas, px, py, 1))
    assert sel == {"lat": -75.0076, "lon": -65.1279}
    assert any("south of 60S" in t.get_text() for t in ax.texts)
