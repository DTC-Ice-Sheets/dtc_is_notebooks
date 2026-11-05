# ruff: noqa: S101, PLR2004, ANN, D, SLF001
"""Tests for uc4_plotting_helpers.

Run with:  pytest test_uc4_plotting_helpers.py -q

Notes
-----
* A tiny synthetic dataset mimics the output of `run_postprocessing_loc`; no real data or network is needed.
* Matplotlib runs on the Agg backend and cartopy's `coastlines` is stubbed (it would download Natural Earth data).
* Tests marked ``xfail(strict=True)`` document suspected bugs in the module. When a bug is fixed the test
  XPASSes, strict mode turns that into a failure, and the marker can simply be removed.
"""

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
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
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


def make_dataset(n_time=N_TIME, n_racmo=N_RACMO, racmo_start=RACMO_START) -> xr.Dataset:
    """Build a dataset with the same layout as the emulator output.

    Emulator = value(m); RACMO = value(m) - BIAS, so emulator - RACMO == BIAS for every matching month,
    even when RACMO starts on a different month than the emulator.
    """
    x = 100_000 + 5_500 * np.arange(NX, dtype=float)
    y = -500_000 - 5_500 * np.arange(NY, dtype=float)  # descending, like a raster
    x_e = 90_000 + 27_500 * np.arange(NX_E, dtype=float)
    y_e = -480_000 - 27_500 * np.arange(NY_E, dtype=float)

    t_pred = (pd.date_range("2020-01-01", periods=n_time, freq="MS") + pd.Timedelta(days=14)).to_numpy()
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


def _texts(fig: Figure) -> list[str]:
    """All text strings drawn in a figure (axes titles, annotations, labels)."""
    out = [fig._suptitle.get_text()] if fig._suptitle is not None else []
    for ax in fig.axes:
        out += [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
        out += [t.get_text() for t in ax.texts]
    return out


# ----------------------------------------------------------------------------
# open_emulator_output
# ----------------------------------------------------------------------------


class TestOpenEmulatorOutput:
    """Test the `open_emulator_output` function."""

    def test_accepts_open_dataset(self, ds):
        """Test that the function accepts an open xarray Dataset."""
        out = h.open_emulator_output(ds)
        assert isinstance(out, xr.Dataset)
        xr.testing.assert_identical(out, ds)

    @pytest.mark.parametrize("missing", ["predictions", "x_cropped"])
    def test_missing_required_variable_raises(self, ds, missing):
        """Test that the function raises a KeyError when a required variable is missing."""
        with pytest.raises(KeyError, match=missing):
            h.open_emulator_output(ds.drop_vars(missing))

    def test_target_patches_is_optional(self, ds_no_racmo):
        """Test that the function does not raise when target_patches is missing."""
        assert "target_patches" not in h.open_emulator_output(ds_no_racmo)

    def test_reads_zarr_store(self, ds, tmp_path):
        """Test that the function can read a zarr store."""
        pytest.importorskip("zarr")
        path = tmp_path / "out.zarr"
        ds.to_zarr(path)
        out = h.open_emulator_output(str(path))
        np.testing.assert_allclose(out["predictions"].values, ds["predictions"].values)

    def test_nonexistent_path_raises(self, tmp_path):
        """Test that the function raises an exception when the path does not exist."""
        with pytest.raises(Exception):  # noqa: B017, PT011 - backend-specific (FileNotFoundError / ValueError)
            h.open_emulator_output(str(tmp_path / "nope.nc"))


# ----------------------------------------------------------------------------
# reference_on_prediction_time
# ----------------------------------------------------------------------------


class TestReferenceOnPredictionTime:
    """Test the `reference_on_prediction_time` function."""

    def test_shape_dims_name_and_dtype(self, ds):
        """Test that the function returns a DataArray with the correct shape, dimensions, name, and dtype."""
        ref = h.reference_on_prediction_time(ds)
        assert ref.dims == ("time", "y", "x")
        assert ref.shape == ds["predictions"].shape
        assert ref.name == "racmo_reference"
        assert ref.dtype == np.float32

    def test_time_axis_is_the_emulators(self, ds):
        """Test that the time axis of the returned DataArray matches the emulator's time axis."""
        ref = h.reference_on_prediction_time(ds)
        np.testing.assert_array_equal(ref["time"].values, ds["time"].values)

    def test_months_match_despite_different_day_stamping(self, ds):
        """Test that the function correctly matches months despite different day stamping."""
        # emulator stamped on the 15th, RACMO on the 1st: matching must be by calendar month
        ref = h.reference_on_prediction_time(ds)
        for t in range(N_RACMO):
            np.testing.assert_allclose(ref.values[t], ds["target_patches"].values[t])

    def test_nan_where_racmo_has_no_month(self, ds):
        """Test that the function returns NaN for months where RACMO has no data."""
        ref = h.reference_on_prediction_time(ds)
        assert np.isnan(ref.values[N_RACMO:]).all()
        assert np.isfinite(ref.values[:N_RACMO]).all()

    def test_racmo_starting_later_is_aligned_by_month_not_index(self):
        """Test that the function aligns by month, not by index, when RACMO starts later."""
        start = 3  # RACMO begins in 2020-04
        ds = make_dataset(racmo_start=start, n_racmo=10)
        ref = h.reference_on_prediction_time(ds)
        assert np.isnan(ref.values[:start]).all()
        for t in range(start, start + 10):
            expected = _value(t, 0, 0) - BIAS  # value is defined per calendar month
            assert ref.values[t, 0, 0] == pytest.approx(expected, abs=1e-5)
        assert np.isnan(ref.values[start + 10 :]).all()

    def test_missing_target_patches_gives_all_nan(self, ds_no_racmo):
        """Test that the function returns all NaN when target_patches is missing."""
        ref = h.reference_on_prediction_time(ds_no_racmo)
        assert ref.shape == ds_no_racmo["predictions"].shape
        assert np.isnan(ref.values).all()
        assert ref.name == "racmo_reference"

    def test_target_patches_without_time_r_gives_all_nan(self, ds):
        """Test that the function returns all NaN when target_patches is present but time_r is missing."""
        ref = h.reference_on_prediction_time(ds.drop_vars("time_r"))
        assert np.isnan(ref.values).all()

    def test_time_r_without_target_patches_gives_all_nan(self, ds):
        """Test that the function returns all NaN when time_r is present but target_patches is missing."""
        ref = h.reference_on_prediction_time(ds.drop_vars("target_patches"))
        assert np.isnan(ref.values).all()

    def test_several_steps_in_one_month_raises(self, ds):
        """Test that the function raises a ValueError when there are several steps in one month."""
        t = ds["time_r"].values.copy()
        t[1] = t[0] + np.timedelta64(5, "D")  # same month as t[0]
        with pytest.raises(ValueError, match="several steps per month"):
            h.reference_on_prediction_time(ds.assign_coords(time_r=t))

    def test_input_dataset_is_not_modified(self, ds):
        """Test that the function does not modify the input dataset."""
        before = ds["target_patches"].values.copy()
        h.reference_on_prediction_time(ds)
        np.testing.assert_array_equal(ds["target_patches"].values, before)


# ----------------------------------------------------------------------------
# point_series
# ----------------------------------------------------------------------------


class TestPointSeries:
    """Test the `point_series` function."""

    def test_columns_and_index(self, ds):
        """Test that the function returns a DataFrame with the correct columns and index."""
        df = h.point_series(ds)
        assert list(df.columns) == ["emulator", "racmo", "era5_tp"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == N_TIME

    def test_emulator_taken_from_nearest_pixel(self, ds):
        """Test that the emulator values are taken from the nearest pixel to the requested coordinates."""
        df = h.point_series(ds)
        expected = ds["predictions"].values[:, PIXEL_J, PIXEL_I]
        np.testing.assert_allclose(df["emulator"].to_numpy(), expected, rtol=1e-5)

    def test_racmo_is_aligned_and_nan_after_it_ends(self, ds):
        """Test that the RACMO values are aligned with the emulator and NaN after RACMO ends."""
        df = h.point_series(ds)
        assert df["racmo"].iloc[:N_RACMO].notna().all()
        assert df["racmo"].iloc[N_RACMO:].isna().all()
        np.testing.assert_allclose(
            df["racmo"].iloc[:N_RACMO].to_numpy(),
            ds["target_patches"].values[:, PIXEL_J, PIXEL_I],
            rtol=1e-5,
        )

    def test_era5_column_is_tile_mean(self, ds):
        """Test that the ERA5 column is the mean of the tile for the specified channel."""
        df = h.point_series(ds)
        expected = ds["x_cropped"].sel(channels="tp").mean(("y_era5", "x_era5")).values
        np.testing.assert_allclose(df["era5_tp"].to_numpy(), expected, rtol=1e-5)

    def test_other_era5_channel(self, ds):
        """Test that the function can return a different ERA5 channel when specified."""
        df = h.point_series(ds, era5_channel="t2m")
        assert "era5_t2m" in df.columns
        assert "era5_tp" not in df.columns
        assert df["era5_t2m"].iloc[0] == pytest.approx(250.0)

    def test_unknown_channel_raises(self, ds):
        """Test that the function raises a KeyError when an unknown ERA5 channel is specified."""
        with pytest.raises(KeyError):
            h.point_series(ds, era5_channel="nope")

    def test_without_racmo_column_is_all_nan(self, ds_no_racmo):
        """Test that the RACMO column is all NaN when target_patches is missing."""
        df = h.point_series(ds_no_racmo)
        assert df["racmo"].isna().all()
        assert df["emulator"].notna().all()


# ----------------------------------------------------------------------------
# skill_metrics
# ----------------------------------------------------------------------------


class TestSkillMetrics:
    """Test the `skill_metrics` function."""

    def test_perfect_agreement(self):
        """Test that the skill metrics are all zero when the emulator and RACMO are identical."""
        s = pd.Series([1.0, -2.0, 3.0, 5.0])
        m = h.skill_metrics(pd.DataFrame({"emulator": s, "racmo": s}))
        assert m["n"] == 4
        assert m["bias"] == pytest.approx(0.0)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["mae"] == pytest.approx(0.0)
        assert m["r"] == pytest.approx(1.0)

    def test_constant_offset(self):
        """Test that the skill metrics reflect a constant offset between the emulator and RACMO."""
        racmo = pd.Series([1.0, 2.0, 4.0, 8.0])
        m = h.skill_metrics(pd.DataFrame({"emulator": racmo + 2.0, "racmo": racmo}))
        assert m["bias"] == pytest.approx(2.0)
        assert m["rmse"] == pytest.approx(2.0)
        assert m["mae"] == pytest.approx(2.0)
        assert m["r"] == pytest.approx(1.0)

    def test_known_values(self):
        """Test that the skill metrics are calculated correctly for known values."""
        emu, ref = np.array([1.0, 2.0, 4.0]), np.array([1.0, 3.0, 2.0])
        err = emu - ref  # [0, -1, 2]
        m = h.skill_metrics(pd.DataFrame({"emulator": emu, "racmo": ref}))
        assert m["n"] == 3
        assert m["bias"] == pytest.approx(err.mean())
        assert m["rmse"] == pytest.approx(np.sqrt((err**2).mean()))
        assert m["mae"] == pytest.approx(np.abs(err).mean())
        assert m["r"] == pytest.approx(np.corrcoef(emu, ref)[0, 1])

    def test_sign_convention_bias_is_emulator_minus_racmo(self):
        """Test that the bias is calculated as emulator minus RACMO."""
        m = h.skill_metrics(pd.DataFrame({"emulator": [1.0, 2.0], "racmo": [3.0, 5.0]}))
        assert m["bias"] < 0

    def test_perfect_anticorrelation(self):
        """Test that the skill metrics show perfect anticorrelation."""
        m = h.skill_metrics(pd.DataFrame({"emulator": [1.0, 2.0, 3.0], "racmo": [3.0, 2.0, 1.0]}))
        assert m["r"] == pytest.approx(-1.0)

    def test_nan_rows_are_dropped_pairwise(self):
        """Test that rows with NaN in either column are dropped pairwise."""
        df = pd.DataFrame(
            {"emulator": [1.0, 2.0, np.nan, 4.0, 9.0], "racmo": [1.0, np.nan, 3.0, 4.0, 9.0]}
        )
        m = h.skill_metrics(df)
        assert m["n"] == 3  # rows 0, 3, 4
        assert m["bias"] == pytest.approx(0.0)

    @pytest.mark.parametrize("n_valid", [0, 1])
    def test_fewer_than_two_pairs_returns_nans(self, n_valid):
        """Test that the skill metrics return NaN when there are fewer than two valid pairs."""
        df = pd.DataFrame({"emulator": [1.0, 2.0, 3.0], "racmo": [np.nan] * 3})
        df.iloc[:n_valid, 1] = 1.0
        m = h.skill_metrics(df)
        assert m["n"] == n_valid
        assert all(np.isnan(m[k]) for k in ("bias", "rmse", "mae", "r"))

    def test_on_synthetic_dataset(self, ds):
        """Test the skill metrics on a synthetic dataset."""
        m = h.skill_metrics(h.point_series(ds))
        assert m["n"] == N_RACMO  # only months with both series
        assert m["bias"] == pytest.approx(BIAS, abs=1e-4)
        assert m["rmse"] == pytest.approx(abs(BIAS), abs=1e-4)
        assert m["mae"] == pytest.approx(abs(BIAS), abs=1e-4)
        assert m["r"] == pytest.approx(1.0)


# ----------------------------------------------------------------------------
# annual_totals
# ----------------------------------------------------------------------------


class TestAnnualTotals:
    """Test the `annual_totals` function."""

    @staticmethod
    def _df(start, periods, racmo=1.0, emulator=2.0):
        """Create a DataFrame with the given start date, periods, and values."""
        idx = pd.date_range(start, periods=periods, freq="MS")
        return pd.DataFrame({"racmo": racmo, "emulator": emulator}, index=idx)

    def test_complete_years_are_summed(self):
        """Test that complete years are summed correctly."""
        out = h.annual_totals(self._df("2020-01-01", 24))
        assert list(out.index) == [2020, 2021]
        assert out["racmo"].tolist() == [12.0, 12.0]
        assert out["emulator"].tolist() == [24.0, 24.0]

    def test_partial_year_is_nan(self):
        """Test that a partial year is returned as NaN."""
        out = h.annual_totals(self._df("2020-01-01", 18))
        assert out.loc[2020, "racmo"] == 12.0
        assert np.isnan(out.loc[2021, "racmo"])
        assert np.isnan(out.loc[2021, "emulator"])

    def test_missing_month_inside_year_gives_nan(self):
        """Test that a missing month inside a year results in NaN for that year."""
        df = self._df("2020-01-01", 12)
        df.iloc[5, df.columns.get_loc("racmo")] = np.nan
        out = h.annual_totals(df)
        assert np.isnan(out.loc[2020, "racmo"])
        assert out.loc[2020, "emulator"] == 24.0  # other column unaffected

    def test_columns_are_racmo_and_emulator_only(self):
        """Test that the returned DataFrame has only the 'racmo' and 'emulator' columns."""
        df = self._df("2020-01-01", 12)
        df["era5_tp"] = 5.0
        assert list(h.annual_totals(df).columns) == ["racmo", "emulator"]

    def test_year_not_starting_in_january_is_not_complete(self):
        """Test that a year not starting in January is considered incomplete and returns NaN."""
        out = h.annual_totals(self._df("2020-07-01", 12))  # Jul-Dec 2020 + Jan-Jun 2021
        assert out.isna().all().all()

    def test_on_synthetic_dataset(self, ds):
        """Test the annual totals on a synthetic dataset."""
        out = h.annual_totals(h.point_series(ds))
        assert out["emulator"].notna().all()  # emulator has two full years
        assert out["racmo"].notna().tolist() == [True, False]  # RACMO ends in June 2021


# ----------------------------------------------------------------------------
# _smb_norm
# ----------------------------------------------------------------------------


class TestSmbNorm:
    """Test the `_smb_norm` function."""

    def test_centre_is_zero_and_bounds_straddle_it(self):
        """Test that the TwoSlopeNorm is centered at zero and the bounds straddle it."""
        norm = h._smb_norm(np.linspace(-100, 100, 1001))
        assert isinstance(norm, TwoSlopeNorm)
        assert norm.vcenter == 0.0
        assert norm.vmin == pytest.approx(-98, abs=0.5)
        assert norm.vmax == pytest.approx(98, abs=0.5)

    def test_zero_maps_to_middle_of_colormap(self):
        """Test that zero maps to the middle of the colormap (0.5)."""
        norm = h._smb_norm(np.linspace(-10, 40, 501))
        assert float(norm(0.0)) == pytest.approx(0.5)

    def test_nan_and_inf_are_ignored(self):
        """Test that NaN and Inf values are ignored when computing the normalization bounds."""
        clean = np.linspace(-10, 10, 201)
        dirty = np.concatenate([clean, [np.nan, np.inf, -np.inf]])
        a, b = h._smb_norm(clean), h._smb_norm(dirty)
        assert (a.vmin, a.vmax) == pytest.approx((b.vmin, b.vmax))

    def test_several_arrays_are_pooled(self):
        """Test that several arrays are pooled together to compute the normalization bounds."""
        norm = h._smb_norm(np.linspace(-5, 5, 101), np.linspace(-50, 50, 101))
        assert norm.vmax > 40

    def test_all_positive_data_still_has_valid_norm(self):
        """Test that all-positive data still results in a valid TwoSlopeNorm with vmin < 0 < vmax."""
        norm = h._smb_norm(np.linspace(1, 10, 50))
        assert norm.vmin < 0 < norm.vmax  # TwoSlopeNorm requires vmin < vcenter < vmax

    def test_all_negative_data_still_has_valid_norm(self):
        """Test that all-negative data still results in a valid TwoSlopeNorm with vmin < 0 < vmax."""
        norm = h._smb_norm(np.linspace(-10, -1, 50))
        assert norm.vmin < 0 < norm.vmax

    def test_all_zero_data_still_has_valid_norm(self):
        """Test that all-zero data still results in a valid TwoSlopeNorm with vmin < 0 < vmax."""
        norm = h._smb_norm(np.zeros((4, 4)))
        assert norm.vmin < 0 < norm.vmax


# ----------------------------------------------------------------------------
# era5_label
# ----------------------------------------------------------------------------


class TestEra5Label:
    """Test the `era5_label` function."""

    @pytest.mark.parametrize(("channel", "unit"), list(h.ERA5_UNITS.items()))
    def test_known_channels_show_their_units(self, channel, unit):
        """Test that known channels show their units."""
        label = h.era5_label(channel)
        assert channel in label
        assert f"[{unit}]" in label

    def test_unknown_channel_shows_question_mark(self):
        """Test that unknown channels show a question mark in the label."""
        assert "[?]" in h.era5_label("foo")

    def test_mentions_raw_input(self):
        """Test that the label mentions 'raw input'."""
        assert "raw input" in h.era5_label("tp")


# ----------------------------------------------------------------------------
# check_polar_projection
# ----------------------------------------------------------------------------


class TestCheckPolarProjection:
    """Test the `check_polar_projection` function."""

    def test_prints_one_line_per_projection(self, capsys):
        """Test that the function prints one line per projection and includes the expected information."""
        h.check_polar_projection()
        lines = capsys.readouterr().out.strip().splitlines()
        assert len(lines) == 3
        assert lines[0].startswith("pyproj EPSG:3031")
        assert "SouthPolarStereo()" in lines[1]
        assert "POLAR" in lines[2]

    def test_module_polar_projection_matches_epsg3031(self):
        """`POLAR` (true-scale latitude -71) must agree with EPSG:3031 to well under a metre."""
        import cartopy.crs as ccrs
        import pyproj

        lon, lat = -65.1279, -75.0076
        ref = pyproj.Transformer.from_crs(4326, 3031, always_xy=True).transform(lon, lat)
        x, y = h.POLAR.transform_point(lon, lat, ccrs.PlateCarree())
        assert (x, y) == pytest.approx(ref, abs=1.0)

    def test_cartopy_default_south_polar_stereo_does_not_match_epsg3031(self):
        """Why POLAR exists: cartopy's default `SouthPolarStereo()` is ~40 km off at this latitude."""
        import cartopy.crs as ccrs
        import pyproj

        lon, lat = -65.1279, -75.0076
        ref = pyproj.Transformer.from_crs(4326, 3031, always_xy=True).transform(lon, lat)
        x, _ = ccrs.SouthPolarStereo().transform_point(lon, lat, ccrs.PlateCarree())
        assert abs(x - ref[0]) > 10_000


# ----------------------------------------------------------------------------
# plot_tile_comparison
# ----------------------------------------------------------------------------


class TestPlotTileComparison:
    """Test the `plot_tile_comparison` function."""

    def test_returns_figure_without_cartopy_path(self, ds):
        """Test that the function returns a matplotlib Figure when coastlines are not requested."""
        fig = h.plot_tile_comparison(ds, coastlines=False)
        assert isinstance(fig, Figure)

    def test_returns_figure_with_cartopy_path(self, ds):
        """Test that the function returns a matplotlib Figure with GeoAxes when coastlines are requested."""
        fig = h.plot_tile_comparison(ds, coastlines=True)
        assert isinstance(fig, Figure)
        assert any(type(a).__name__ == "GeoAxes" for a in fig.axes)

    def test_suptitle_has_month_and_coordinates(self, ds):
        """Test that the figure's suptitle includes the month and coordinates."""
        fig = h.plot_tile_comparison(ds, 2, coastlines=False)
        title = fig._suptitle.get_text()
        assert "2020-03" in title
        assert "-75.008" in title
        assert "-65.128" in title

    def test_integer_and_date_string_select_the_same_month(self, ds):
        """Test that selecting a month by integer index or by date string yields the same suptitle."""
        a = h.plot_tile_comparison(ds, 5, coastlines=False)._suptitle.get_text()
        b = h.plot_tile_comparison(ds, "2020-06-15", coastlines=False)._suptitle.get_text()
        assert a == b

    def test_no_placeholder_when_reference_exists(self, ds):
        """Test that no placeholder text is shown when RACMO reference data exists for the selected month."""
        texts = _texts(h.plot_tile_comparison(ds, 0, coastlines=False))
        assert not any("No RACMO reference" in t for t in texts)

    def test_placeholder_for_month_without_racmo(self, ds):
        """Test that a placeholder text is shown for months where RACMO reference data does not exist."""
        texts = _texts(h.plot_tile_comparison(ds, N_RACMO + 2, coastlines=False))
        assert sum("No RACMO reference" in t for t in texts) == 2  # reference panel and difference panel

    def test_placeholder_when_racmo_is_absent_entirely(self, ds_no_racmo):
        """Test that a placeholder text is shown when RACMO reference data is absent entirely."""
        fig = h.plot_tile_comparison(ds_no_racmo, coastlines=False)
        assert sum("No RACMO reference" in t for t in _texts(fig)) == 2

    def test_custom_units_reach_the_colourbars(self, ds):
        """Test that custom units specified in the function call appear in the colourbar labels."""
        fig = h.plot_tile_comparison(ds, coastlines=False, units="mm w.e.")
        assert any("mm w.e." in t for t in _texts(fig))

    def test_custom_era5_label(self, ds):
        """Test that a custom label for the ERA5 panel is used when specified."""
        fig = h.plot_tile_comparison(ds, coastlines=False, era5_label_="MY LABEL")
        assert "MY LABEL" in _texts(fig)

    def test_other_era5_channel_uses_its_units(self, ds):
        """Test that specifying a different ERA5 channel results in the correct units appearing in the colourbar."""
        fig = h.plot_tile_comparison(ds, coastlines=False, era5_channel="t2m")
        assert any("t2m" in t and "[K]" in t for t in _texts(fig))

    def test_colour_scale_is_fixed_over_time(self, ds):
        """Moving through months must not rescale the colours (norm built from the whole period)."""

        def limits(t):
            fig = h.plot_tile_comparison(ds, t, coastlines=False)
            meshes = [c for ax in fig.axes for c in ax.collections if c.get_cmap().name == h.SMB_CMAP]
            return {(c.norm.vmin, c.norm.vmax) for c in meshes}

        assert limits(0) == limits(10)

    def test_smb_maps_share_one_norm(self, ds):
        """All SMB maps in the figure must share the same norm object for consistent colour mapping."""
        fig = h.plot_tile_comparison(ds, 0, coastlines=False)
        # RACMO map, emulator map and their colourbars all reference the very same norm object
        norms = {id(c.norm): c.norm for ax in fig.axes for c in ax.collections if c.get_cmap().name == h.SMB_CMAP}
        assert len(norms) == 1
        (norm,) = norms.values()
        assert norm.vmin < 0 < norm.vmax

    def test_constant_zero_fields_do_not_crash(self, ds):
        """Test that the function does not crash when all fields are constant zero."""
        ds["predictions"][:] = 0.0
        ds["target_patches"][:] = 0.0
        assert isinstance(h.plot_tile_comparison(ds, coastlines=False), Figure)

    def test_nan_in_era5_tile_does_not_crash(self, ds):
        """Test that the function does not crash when there are NaN values in the ERA5 tile."""
        ds["x_cropped"][0, 0, 0, 0] = np.nan
        assert isinstance(h.plot_tile_comparison(ds, 0, coastlines=False), Figure)

    def test_custom_figsize(self, ds):
        """Test that the function respects a custom figure size."""
        fig = h.plot_tile_comparison(ds, coastlines=False, figsize=(10, 4))
        assert tuple(fig.get_size_inches()) == (10, 4)


# ----------------------------------------------------------------------------
# plot_point_dashboard
# ----------------------------------------------------------------------------


class TestPlotPointDashboard:
    """Test the plot_point_dashboard function for creating a dashboard figure for a specific point."""

    def test_returns_figure(self, ds):
        """Test that the function returns a matplotlib Figure."""
        assert isinstance(h.plot_point_dashboard(ds), Figure)

    def test_suptitle_has_coordinates(self, ds):
        """Test that the figure's suptitle includes the coordinates of the point."""
        title = h.plot_point_dashboard(ds)._suptitle.get_text()
        assert "-75.008" in title
        assert "-65.128" in title

    def test_panel_titles(self, ds):
        """Test that the figure contains the expected panel titles."""
        titles = _texts(h.plot_point_dashboard(ds))
        assert any(t.startswith("INPUT:") for t in titles)
        assert "OUTPUT: monthly SMB at the requested location" in titles
        assert "Agreement (1:1 dashed)" in titles
        assert "Seasonal cycle (mean)" in titles
        assert "Annual totals (complete years)" in titles

    def test_metrics_annotation_in_scatter(self, ds):
        """Test that the scatter plot contains an annotation with the skill metrics."""
        texts = _texts(h.plot_point_dashboard(ds))
        note = next(t for t in texts if t.startswith(f"n={N_RACMO}"))
        assert "bias=-0.5" in note
        assert "RMSE=0.5" in note
        assert "r=1.00" in note

    def test_emulator_only_shading_when_racmo_ends_early(self, ds):
        """Test that the figure contains a shaded area indicating 'emulator only' when RACMO ends early."""
        assert any("emulator only" in t for t in _texts(h.plot_point_dashboard(ds)))

    def test_no_emulator_only_label_when_racmo_covers_everything(self):
        """Test that the figure does not contain an 'emulator only' label when RACMO covers the entire period."""
        ds = make_dataset(n_racmo=N_TIME)
        assert not any("emulator only" in t for t in _texts(h.plot_point_dashboard(ds)))

    def test_without_racmo(self, ds_no_racmo):
        """Test that the figure contains a placeholder text when RACMO reference data is absent."""
        fig = h.plot_point_dashboard(ds_no_racmo)
        texts = _texts(fig)
        assert "no RACMO reference in this period" in texts
        assert not any(t.startswith("n=") for t in texts)  # no scatter stats

    def test_short_record_without_complete_year(self, ds):
        """Test that the function handles a short record without a complete year."""
        short = ds.isel(time=slice(0, 6))
        assert isinstance(h.plot_point_dashboard(short), Figure)

    def test_single_month(self, ds):
        """Test that the function handles a single month of data."""
        assert isinstance(h.plot_point_dashboard(ds.isel(time=slice(0, 1))), Figure)

    def test_other_channel_and_units(self, ds):
        """Test that specifying a different ERA5 channel and units results in the correct labels in the figure."""
        fig = h.plot_point_dashboard(ds, era5_channel="t2m", units="mm w.e.")
        texts = _texts(fig)
        assert any("t2m" in t for t in texts)
        assert any("mm w.e." in t for t in texts)

    def test_annual_bar_chart_has_complete_years_only(self, ds):
        """Test that the annual bar chart only includes complete years and matches the computed annual totals."""
        fig = h.plot_point_dashboard(ds)
        ann_ax = next(ax for ax in fig.axes if ax.get_title() == "Annual totals (complete years)")
        heights = [p.get_height() for p in ann_ax.patches]  # column by column: RACMO 2020, 2021, Emulator 2020, 2021
        ann = h.annual_totals(h.point_series(ds))
        assert len(heights) == 4
        assert heights[0] == pytest.approx(ann.loc[2020, "racmo"])
        assert heights[1] == 0  # incomplete RACMO year is NaN, which pandas draws as a zero-height bar
        assert heights[2] == pytest.approx(ann.loc[2020, "emulator"])
        assert heights[3] == pytest.approx(ann.loc[2021, "emulator"])


# ----------------------------------------------------------------------------
# build_explorer
# ----------------------------------------------------------------------------


class TestBuildExplorer:
    """Test the build_explorer function for creating an interactive Plotly figure."""

    def test_returns_plotly_figure(self, ds):
        """Test that the function returns a Plotly Figure."""
        assert isinstance(h.build_explorer(ds), go.Figure)

    def test_one_frame_and_slider_step_per_month(self, ds):
        """Test that the figure has one frame and one slider step for each month in the dataset."""
        fig = h.build_explorer(ds)
        assert len(fig.frames) == N_TIME
        steps = fig.layout.sliders[0].steps
        assert len(steps) == N_TIME
        assert steps[0].label == "2020-01"
        assert steps[-1].label == "2021-12"

    def test_frames_target_the_dynamic_traces(self, ds):
        """Test that each frame targets the correct dynamic traces in the figure."""
        fig = h.build_explorer(ds)
        assert all(list(f.traces) == [0, 1, 2, 3, 7, 8, 9] for f in fig.frames)
        assert [f.name for f in fig.frames][:3] == ["0", "1", "2"]

    def test_stride_subsamples_maps(self, ds):
        """Test that the stride parameter subsamples the maps correctly, reducing their resolution."""
        fig = h.build_explorer(ds, stride=2)
        assert np.asarray(fig.data[1].z).shape == (int(np.ceil(NY / 2)), int(np.ceil(NX / 2)))
        fig1 = h.build_explorer(ds, stride=1)
        assert np.asarray(fig1.data[1].z).shape == (NY, NX)

    def test_map_axes_are_in_km(self, ds):
        """Test that the map axes are in kilometers, not meters, by checking the x-coordinates of the mesh traces."""
        fig = h.build_explorer(ds, stride=1)
        np.testing.assert_allclose(np.asarray(fig.data[1].x), ds["x"].values / 1e3)

    def test_smb_colour_axis_is_symmetric_about_zero(self, ds):
        """Test that the SMB colour axis is symmetric about zero, with cmin = -cmax and cmax > 0."""
        ca = h.build_explorer(ds).layout.coloraxis
        assert ca.cmin == pytest.approx(-ca.cmax)
        assert ca.cmax > 0

    def test_placeholder_visible_when_racmo_absent(self, ds_no_racmo):
        """Test that the placeholder trace is visible when RACMO reference data is absent."""
        fig = h.build_explorer(ds_no_racmo)
        assert fig.data[3].visible is True
        assert "No RACMO reference" in fig.data[3].text[0]

    def test_placeholder_toggles_per_frame(self, ds):
        """Test that the placeholder trace visibility toggles correctly per frame."""
        fig = h.build_explorer(ds)
        visible = [f.data[3].visible for f in fig.frames]
        assert visible[:N_RACMO] == [False] * N_RACMO
        assert visible[N_RACMO:] == [True] * (N_TIME - N_RACMO)

    def test_racmo_trace_name(self, ds, ds_no_racmo):
        """Test that the RACMO trace has the correct name when RACMO data is present and a placeholder otherwise."""
        assert h.build_explorer(ds).data[4].name == "RACMO"
        assert h.build_explorer(ds_no_racmo).data[4].name == "RACMO (unavailable)"

    def test_rmse_band_present_only_with_racmo(self, ds, ds_no_racmo):
        """Test that the RMSE band trace is present when RACMO data is available and absent when it is not."""
        names = [t.name for t in h.build_explorer(ds).data]
        assert "Emulator \u00b1 RMSE" in names
        names = [t.name for t in h.build_explorer(ds_no_racmo).data]
        assert "Emulator \u00b1 RMSE" not in names

    def test_time_series_traces_hold_the_point_series(self, ds):
        """Test that the time series traces in the figure hold the same data as returned by point_series()."""
        fig = h.build_explorer(ds)
        df = h.point_series(ds)
        np.testing.assert_allclose(np.asarray(fig.data[5].y, dtype=float), df["emulator"].to_numpy())
        np.testing.assert_allclose(np.asarray(fig.data[6].y, dtype=float), df["era5_tp"].to_numpy())

    def test_marker_follows_the_frame(self, ds):
        """Test that the marker in the time series plot follows the frame."""
        fig = h.build_explorer(ds)
        df = h.point_series(ds)
        frame = fig.frames[5]
        assert frame.data[5].y[0] == pytest.approx(df["emulator"].iloc[5])  # emulator marker
        assert pd.Timestamp(frame.data[5].x[0]) == df.index[5]

    def test_constant_era5_gets_nonzero_colour_range(self, ds):
        """Test that when the ERA5 tile is constant, the colour axis still has a nonzero range."""
        ds["x_cropped"][:] = 3.0
        ca2 = h.build_explorer(ds).layout.coloraxis2
        assert ca2.cmax > ca2.cmin

    def test_requested_location_star_in_each_map(self, ds):
        """Test that a star marker is present in each map at the requested location."""
        fig = h.build_explorer(ds)
        stars = [t for t in fig.data if isinstance(t, go.Scatter) and t.marker.symbol == "star"]
        assert len(stars) == 3
        assert all(s.x[0] == pytest.approx(ds.attrs["x_3031"] / 1e3) for s in stars)

    def test_figure_is_json_serialisable(self, ds):
        """Test that the figure can be serialized to JSON, which is important for web applications."""
        assert h.build_explorer(ds).to_json()

    def test_works_with_nan_racmo_everywhere_and_in_pieces(self, ds):
        """Test that the function works correctly when RACMO data is NaN in some places."""
        ds["target_patches"][3] = np.nan  # one RACMO month entirely NaN
        fig = h.build_explorer(ds)
        assert fig.frames[3].data[3].visible is True


# ----------------------------------------------------------------------------
# plot_single_date / plot_validation
# ----------------------------------------------------------------------------

DATE_FUNCS = [h.plot_single_date, h.plot_validation]
DATE_FUNC_IDS = ["plot_single_date", "plot_validation"]


@pytest.mark.usefixtures("offline_coastlines")
class TestSingleDateAndValidation:
    """Test the plot_single_date and plot_validation functions, which both take a date as input."""

    @pytest.mark.parametrize("func", DATE_FUNCS, ids=DATE_FUNC_IDS)
    def test_returns_a_figure(self, ds, func):
        """Test that the function returns a matplotlib Figure for a valid date."""
        # NB: both are annotated `-> None` but return the figure.
        assert isinstance(func(ds, datetime(2020, 6, 15)), Figure)

    @pytest.mark.parametrize("func", DATE_FUNCS, ids=DATE_FUNC_IDS)
    def test_date_before_dataset_raises(self, ds, func):
        """Test that the function raises a ValueError when the date is before the dataset's range."""
        with pytest.raises(ValueError, match="not in the dataset"):
            func(ds, datetime(2019, 1, 1))

    @pytest.mark.parametrize("func", DATE_FUNCS, ids=DATE_FUNC_IDS)
    def test_date_after_dataset_raises(self, ds, func):
        """Test that the function raises a ValueError when the date is after the dataset's range."""
        with pytest.raises(ValueError, match="not in the dataset"):
            func(ds, datetime(2030, 1, 1))

    @pytest.mark.parametrize("func", DATE_FUNCS, ids=DATE_FUNC_IDS)
    def test_first_date_is_accepted(self, ds, func):
        """Test that the function accepts the first date in the dataset."""
        assert isinstance(func(ds, datetime(2020, 1, 16)), Figure)

    @pytest.mark.parametrize("func", DATE_FUNCS, ids=DATE_FUNC_IDS)
    def test_last_date_is_accepted(self, ds, func):
        """Test that the function accepts the last date in the dataset."""
        assert isinstance(func(ds, datetime(2021, 12, 15)), Figure)

    @pytest.mark.parametrize("func", DATE_FUNCS, ids=DATE_FUNC_IDS)
    def test_figure_is_closed_so_notebooks_do_not_show_it_twice(self, ds, func):
        """Test that the function closes the figure after returning it, so that notebooks do not show it twice."""
        fig = func(ds, datetime(2020, 6, 15))
        assert not plt.fignum_exists(fig.number)

    def test_single_date_has_four_map_panels_and_three_colourbars(self, ds):
        """Test that the figure returned by plot_single_date has four map panels and three colourbars."""
        fig = h.plot_single_date(ds, datetime(2020, 6, 15))
        assert len(fig.axes) == 7

    def test_single_date_uses_nearest_time_step(self, ds):
        """Test that plot_single_date uses the nearest time step to the requested date."""
        # 2020-06-20 is closest to the 15 June step; both calls must draw identical predictions.
        a = h.plot_single_date(ds, datetime(2020, 6, 20))
        b = h.plot_single_date(ds, datetime(2020, 6, 15))
        pa = a.axes[2].collections[0].get_array()
        pb = b.axes[2].collections[0].get_array()
        np.testing.assert_array_equal(np.asarray(pa), np.asarray(pb))

    def test_validation_bias_panel_is_symmetric_about_zero(self, ds):
        """Test that the bias panel in the validation plot is symmetric about zero, with vmin = -vmax."""
        fig = h.plot_validation(ds, datetime(2020, 6, 15))
        mesh = fig.axes[0].collections[0]
        assert mesh.norm.vmin == pytest.approx(-mesh.norm.vmax)

    def test_validation_bias_is_predictions_minus_target(self, ds):
        """Test that the bias panel in the validation plot is computed as predictions minus target."""
        fig = h.plot_validation(ds, datetime(2020, 6, 15))
        drawn = np.asarray(fig.axes[0].collections[0].get_array())
        # 2020-06 emulator month index 5; RACMO matches it by construction
        np.testing.assert_allclose(drawn, BIAS, atol=1e-4)

    def test_validation_rmse_panel_starts_at_zero(self, ds):
        """Test that the RMSE panel in the validation plot has a colour scale starting at zero."""
        fig = h.plot_validation(ds, datetime(2020, 6, 15))
        rmse_mesh = next(c for ax in fig.axes for c in ax.collections if c.get_cmap().name == "inferno_r")
        assert rmse_mesh.norm.vmin == 0.0


    def test_validation_ignores_nan_in_distributions(self, ds):
        """Test that the validation plot ignores NaN values when computing distributions."""
        ds["predictions"][5, 0, 0] = np.nan
        ds["target_patches"][5, 1, 1] = np.nan
        assert isinstance(h.plot_validation(ds, datetime(2020, 6, 15)), Figure)

    def test_validation_with_no_finite_values_raises(self, ds):
        """Test that the validation plot raises a ValueError when there are no finite values."""
        ds["predictions"][:] = np.nan
        ds["target_patches"][:] = np.nan
        with pytest.raises(ValueError):  # noqa: PT011
            h.plot_validation(ds, datetime(2020, 6, 15))

# ----------------------------------------------------------------------------
# select_location
# ----------------------------------------------------------------------------


class TestSelectLocation:
    """Test the interactive location selection function `select_location()`."""
    @pytest.fixture(autouse=True)
    def _no_show_no_coast(self, monkeypatch):
        """Prevent the tests from showing a matplotlib window and from trying to draw coastlines."""
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        monkeypatch.setattr(plt, "coastlines", lambda *a, **k: None)

    @staticmethod
    def _click(fig, ax, lon, lat, button=1):
        """Fire a matplotlib click at the display position of (lon, lat)."""
        import cartopy.crs as ccrs

        fig.canvas.draw()
        data = ax.projection.transform_point(lon, lat, ccrs.PlateCarree())
        px, py = ax.transData.transform(data)
        fig.canvas.callbacks.process("button_press_event", MouseEvent("button_press_event", fig.canvas, px, py, button))

    def test_returns_initial_location_without_clicks(self):
        """Test that the function returns the initial location when no clicks are made."""
        assert h.select_location() == {"lat": -75.0076, "lon": -65.1279}

    def test_custom_initial_location(self):
        """Test that the function returns the custom initial location when specified."""
        assert h.select_location(lat=-80.0, lon=10.0) == {"lat": -80.0, "lon": 10.0}

    def test_left_click_updates_selection_and_prints(self, capsys):
        """Test that a left click updates the selection and prints the new location."""
        sel = h.select_location()
        fig = plt.gcf()
        self._click(fig, fig.axes[0], lon=30.0, lat=-75.0)
        assert sel["lat"] == pytest.approx(-75.0, abs=0.5)
        assert sel["lon"] == pytest.approx(30.0, abs=1.0)
        assert "Selected location" in capsys.readouterr().out

    def test_longitude_is_wrapped_to_pm180(self):
        """Test that the longitude is wrapped to the range [-180, 180] when a click is made."""
        sel = h.select_location()
        fig = plt.gcf()
        self._click(fig, fig.axes[0], lon=170.0, lat=-75.0)
        assert -180 <= sel["lon"] <= 180
        assert sel["lon"] == pytest.approx(170.0, abs=1.0)

    def test_click_outside_the_axes_is_ignored(self):
        """Test that a click outside the axes does not change the selection."""
        sel = h.select_location()
        fig = plt.gcf()
        fig.canvas.draw()
        fig.canvas.callbacks.process("button_press_event", MouseEvent("button_press_event", fig.canvas, 1, 1, 1))
        assert sel == {"lat": -75.0076, "lon": -65.1279}

    def test_click_north_of_60s_is_rejected_with_hint(self):
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

