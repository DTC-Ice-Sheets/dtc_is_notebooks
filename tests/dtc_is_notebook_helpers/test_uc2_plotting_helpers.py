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
