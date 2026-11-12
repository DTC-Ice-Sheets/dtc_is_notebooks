"""
Tests for plotting_helpers.py in notebooks.helpers.

We expect the plotting functions to run without errors, but we do not validate the visual output in these tests.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from dtc_is_notebook_helpers import uc2_plotting_helpers


@pytest.fixture
def example_mean_mass_balance_dataset(example_mass_balance_dataset: xr.Dataset) -> xr.Dataset:
    return uc2_plotting_helpers.compute_mean_mass_balance_over_time_window(example_mass_balance_dataset)


@pytest.mark.parametrize("dataset_value", ["GrIS", "AIS", "custom"])
def test_plot_scaled_mean_mass_balance_runs_no_error(example_mean_mass_balance_dataset: xr.Dataset, dataset_value: str):
    # Should run without error and produce a plot for each dataset_value
    with patch("matplotlib.pyplot.show"):
        uc2_plotting_helpers.plot_scaled_mean_mass_balance(
            mean_mb_ds=example_mean_mass_balance_dataset,
            dataset_value=dataset_value,
            plot_description_str=f"Test plot for {dataset_value}",
            scale=2.0,
        )


def test_plot_scaled_mean_mass_balance_changes_with_scale_factor(example_mean_mass_balance_dataset: xr.Dataset):
    # Helper: render the plot and return the pixel buffer
    def render_plot(scale_factor):
        with patch("matplotlib.pyplot.show"):  # prevent actual display
            uc2_plotting_helpers.plot_scaled_mean_mass_balance(
                mean_mb_ds=example_mean_mass_balance_dataset,
                dataset_value="GrIS",
                plot_description_str="Test plot",
                scale=scale_factor,
            )
            fig = plt.gcf()
            fig.canvas.draw()  # force render
            # Convert figure to an RGB array
            image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            plt.close(fig)
        return image

    # Render two versions with different scale factors
    img1 = render_plot(scale_factor=1.0)
    img2 = render_plot(scale_factor=10.0)

    # Check that the resulting images are not identical
    diff = np.abs(img1.astype(int) - img2.astype(int)).sum()
    assert diff > 0, "Plot output did not change with different scale factors"


def test_plot_scaled_mean_mass_balance_invalid_dataset_value(example_mean_mass_balance_dataset: xr.Dataset):
    # Should raise KeyError if dataset_value is not in config
    with pytest.raises(KeyError):
        with patch("matplotlib.pyplot.show"):
            uc2_plotting_helpers.plot_scaled_mean_mass_balance(
                mean_mb_ds=example_mean_mass_balance_dataset,
                dataset_value="invalid",
                plot_description_str="Invalid dataset",
            )


def test_plot_scaled_mean_mass_balance_infer_from_lat(example_mean_mass_balance_dataset: xr.Dataset):
    # Should succeed without raising when infer_from_lat=True, even for an unknown dataset_value
    with patch("matplotlib.pyplot.show"):
        uc2_plotting_helpers.plot_scaled_mean_mass_balance(
            mean_mb_ds=example_mean_mass_balance_dataset,
            dataset_value="unknown_region",
            plot_description_str="Inferred dataset",
            infer_from_lat=True,
        )


def test_plot_global_slr_three_panels_runs_no_error(
    example_global_slr_dataset: xr.Dataset, example_mean_mass_balance_dataset: xr.Dataset
):
    # Should run without error and produce a plot
    with patch("matplotlib.pyplot.show"):
        uc2_plotting_helpers.plot_global_slr_three_panels(
            global_slr_ds=example_global_slr_dataset,
            mean_mass_balance_ds=example_mean_mass_balance_dataset,
            plot_description_str="Test global SLR plot",
        )


def test_plot_slr_location_four_panels_runs_no_error(example_annual_slr_dataset: xr.Dataset):
    # Should run without error and produce a plot for a valid location
    lat0 = float(example_annual_slr_dataset["y"].values[0])
    lon0 = float(example_annual_slr_dataset["x"].values[0])
    with patch("matplotlib.pyplot.show"):
        uc2_plotting_helpers.plot_slr_location_four_panels(
            annual_slr_ds=example_annual_slr_dataset,
            lat0=lat0,
            lon0=lon0,
        )


def test_plot_slr_location_four_panels_changes_with_location(example_annual_slr_dataset: xr.Dataset):
    # Should run without error and produce a plot for a valid location
    lat0 = float(example_annual_slr_dataset["y"].values[0])
    lon0 = float(example_annual_slr_dataset["x"].values[0])
    print(lat0, lon0)
    with patch("matplotlib.pyplot.show"):
        uc2_plotting_helpers.plot_slr_location_four_panels(
            annual_slr_ds=example_annual_slr_dataset,
            lat0=lat0,
            lon0=lon0,
        )

    # Helper: render the plot and return the pixel buffer
    def render_plot(lon, lat):
        with patch("matplotlib.pyplot.show"):  # prevent actual display
            uc2_plotting_helpers.plot_slr_location_four_panels(
                annual_slr_ds=example_annual_slr_dataset,
                lat0=lat,
                lon0=lon,
            )
            fig = plt.gcf()
            plt.suptitle("Normalised Title", fontsize=15)
            fig.canvas.draw()  # force render
            # fig.savefig(f"loc_{lat}_{lon}.png", dpi=150, bbox_inches="tight")
            # Convert figure to an RGB array
            image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            plt.close(fig)
        return image

    # Render two versions with different scale factors
    img1 = render_plot(lon=100, lat=50)
    img2 = render_plot(lon=100, lat=0)

    # Check that the resulting images are not identical
    diff = np.abs(img1.astype(int) - img2.astype(int)).sum()
    assert diff > 0, "Plot output did not change with different scale factors"


def test_plot_slr_location_four_panels_nearest_grid_cell_selection(example_annual_slr_dataset: xr.Dataset):
    # Should select the nearest grid cell to the provided lat/lon
    lats = example_annual_slr_dataset["y"].values
    lons = example_annual_slr_dataset["x"].values
    # Pick a location slightly offset from the first grid cell
    lat0 = float(lats[0]) - 0.01
    lon0 = float(lons[0]) + 0.01
    with patch("matplotlib.pyplot.show"):
        # Should not raise
        uc2_plotting_helpers.plot_slr_location_four_panels(
            annual_slr_ds=example_annual_slr_dataset,
            lat0=lat0,
            lon0=lon0,
        )


@pytest.mark.parametrize(
    "lat0, lon0",
    [
        (999.0, 0.0),  # invalid latitude
        (0.0, 999.0),  # invalid longitude
        (-91.0, 0.0),  # latitude just below valid range
        (91.0, 0.0),  # latitude just above valid range
        (0.0, -181.0),  # longitude just below valid range
        (0.0, 181.0),  # longitude just above valid range
    ],
)
def test_plot_slr_location_four_panels_invalid_location(example_annual_slr_dataset: xr.Dataset, lat0, lon0):
    with pytest.raises(ValueError):
        with patch("matplotlib.pyplot.show"):
            uc2_plotting_helpers.plot_slr_location_four_panels(
                annual_slr_ds=example_annual_slr_dataset,
                lat0=lat0,
                lon0=lon0,
            )


@pytest.fixture
def masked_slr_dataset() -> xr.Dataset:
    # 3x3 grid (y: -10, 0, 10; x: -10, 0, 10) where only the cells in column x=10 hold data
    data = np.full((2, 3, 3), np.nan)
    data[:, :, 2] = 1.0
    return xr.Dataset(
        {"sdot": (("time", "y", "x"), data)},
        coords={"time": [0, 1], "y": [-10.0, 0.0, 10.0], "x": [-10.0, 0.0, 10.0]},
    )


@pytest.mark.parametrize(
    "lat0, lon0, expected_lat, expected_lon, expected_requested_valid",
    [
        (0.0, 10.0, 0.0, 10.0, True),
        (0.0, 0.0, 0.0, 10.0, False),
        (-9.0, -9.0, -10.0, 10.0, False),
    ],
    ids=["requested-cell-valid", "masked-redirects-to-neighbour", "masked-corner-redirects"],
)
def test_find_nearest_valid_cell_selects_nearest_non_nan_cell(
    masked_slr_dataset: xr.Dataset, lat0, lon0, expected_lat, expected_lon, expected_requested_valid
):
    lat, lon, dist_km, requested_valid = uc2_plotting_helpers.find_nearest_valid_cell(masked_slr_dataset, lat0, lon0)
    assert (lat, lon) == (expected_lat, expected_lon)
    assert requested_valid is expected_requested_valid
    assert (dist_km == pytest.approx(0.0)) is expected_requested_valid


def test_find_nearest_valid_cell_handles_longitude_wraparound(masked_slr_dataset: xr.Dataset):
    # -179 is 11 degrees from 170 across the antimeridian, so 170 is the nearest valid cell
    ds = masked_slr_dataset.assign_coords(x=[-170.0, 0.0, 170.0])
    ds["sdot"][:, :, 1] = 1.0  # x=0 is also valid, and closer than 170 if wraparound were ignored
    _, lon, _, _ = uc2_plotting_helpers.find_nearest_valid_cell(ds, 0.0, -179.0)
    assert lon == 170.0


def test_find_nearest_valid_cell_no_spatial_variable_raises():
    ds = xr.Dataset({"sdot": (("time",), [1.0, 2.0])}, coords={"time": [0, 1]})
    with pytest.raises(ValueError, match="y/x"):
        uc2_plotting_helpers.find_nearest_valid_cell(ds, 0.0, 0.0)
