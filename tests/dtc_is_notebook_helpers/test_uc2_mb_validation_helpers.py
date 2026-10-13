"""
Tests for uc2_mb_validation_helpers.py in dtc_is_notebook_helpers.

The reference records are represented by small synthetic files, so no test reaches the network. The plotting
functions are expected to run without errors, but their visual output is not validated here.
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dtc_is_notebook_helpers import uc2_mb_validation_helpers as mbval
from dtc_is_notebook_helpers.uc2_plotting_helpers import MASS_BALANCE_COL_NAME, MASS_BALANCE_ERROR_COL_NAME


@pytest.fixture
def example_product_dataset() -> xr.Dataset:
    """Three points over four years, each point losing 1e12 kg (1 Gt) per year."""
    years = pd.to_datetime([f"{year}-01-01" for year in (2010, 2011, 2012, 2013)])
    mass_balance = np.full((3, 4), -1e12)
    mass_balance[:, 1] = -2e12  # 2011 loses twice as much
    return xr.Dataset(
        {
            MASS_BALANCE_COL_NAME: (["point", "time"], mass_balance),
            MASS_BALANCE_ERROR_COL_NAME: (["point", "time"], np.full((3, 4), 1e11)),
        },
        coords={"point": [0, 1, 2], "time": years},
    )


@pytest.fixture
def example_imbie_csv(tmp_path: Path) -> Path:
    """Monthly IMBIE-style CSV covering two complete years and one truncated year."""
    rows = []
    for year, rate in ((2010, -100.0), (2011, -200.0)):
        for month in range(12):
            rows.append({"Year": year + month / 12, "Mass balance (Gt/yr)": rate, "Mass balance uncertainty": 50.0})
    for month in range(6):
        rows.append({"Year": 2012 + month / 12, "Mass balance (Gt/yr)": -300.0, "Mass balance uncertainty": 50.0})
    path = tmp_path / "imbie.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def example_imbie3_csv(tmp_path: Path) -> Path:
    """IMBIE-3 style CSV: a commented header, dated rows, and the mass balance split into its components."""
    rows = []
    for year, rate in ((2010, -100.0), (2011, -200.0)):
        for month in range(1, 13):
            rows.append(
                {
                    "Date": f"{year}-{month:02d}-01",
                    "Mass balance (Gt/yr)": rate,
                    "Mass balance uncertainty (Gt/yr)": 50.0,
                    "Cumulative mass balance anomaly (Gt)": rate * month / 12,
                    "Cumulative mass balance anomaly uncertainty (Gt)": 20.0,
                    # The partitioned components must not be mistaken for the total.
                    "Surface mass balance anomaly (Gt/yr)": rate - 30.0,
                    "Surface mass balance anomaly uncertainty (Gt/yr)": 40.0,
                    "Dynamics mass balance anomaly (Gt/yr)": -30.0,
                    "Dynamics mass balance anomaly uncertainty (Gt/yr)": 10.0,
                }
            )
    # A truncated final year, as the release has at its start.
    rows.extend(
        {"Date": f"2012-{month:02d}-01", "Mass balance (Gt/yr)": -300.0, "Mass balance uncertainty (Gt/yr)": 50.0}
        for month in range(1, 7)
    )
    path = tmp_path / "imbie3.csv"
    header = "# id: GB/NERC/BAS/PDC/02074\n# title: a commented IMBIE-3 header\n# data_type: Greenland Gt\n"
    path.write_text(header + pd.DataFrame(rows).to_csv(index=False))
    return path


@pytest.fixture
def example_gravis_file(tmp_path: Path) -> Path:
    """GravIS-style ASCII file: time, total, two basins, 1-sigma of the total, two basin 1-sigmas."""
    times = 2010.0 + np.arange(36) / 12.0
    total = -100.0 * (times - 2010.0)  # a steady loss of 100 Gt/yr
    columns = np.column_stack(
        [
            times,
            total,
            total / 2,
            total / 2,
            np.full_like(times, 20.0),
            np.full_like(times, 5.0),
            np.full_like(times, 5.0),
        ]
    )
    path = tmp_path / "gravis.asc"
    np.savetxt(path, columns, header="a GravIS style header line")
    return path


@pytest.fixture
def example_annual_series() -> list[pd.DataFrame]:
    """One DTC-IS product series and two reference series sharing the years 2010 to 2013."""
    years = [2010, 2011, 2012, 2013]
    simonsen = pd.DataFrame(
        {"year": years, mbval.SIMONSEN_COL_NAME: [-100.0, -220.0, -180.0, -300.0], "simonsen_unc_gt_yr": [40.0] * 4}
    )
    imbie = pd.DataFrame(
        {"year": years, mbval.IMBIE_COL_NAME: [-120.0, -200.0, -150.0, -280.0], "imbie_unc_gt_yr": [50.0] * 4}
    )
    gmb = pd.DataFrame(
        {"year": years, mbval.GMB_COL_NAME: [-110.0, -210.0, -190.0, -260.0], "gmb_unc_gt_yr": [20.0] * 4}
    )
    return [simonsen, imbie, gmb]


@pytest.fixture
def example_comparison_table(example_annual_series: list[pd.DataFrame]) -> pd.DataFrame:
    return mbval.build_intercomparison_table(example_annual_series)


@pytest.fixture
def example_mlmodel_series() -> pd.DataFrame:
    """Annual model prediction, starting a year later than the other series."""
    return pd.DataFrame(
        {
            "year": [2011, 2012, 2013],
            mbval.MLMODEL_COL_NAME: [-260.0, -240.0, -320.0],
            "mlmodel_source": ["RAdh+LST"] * 3,
        }
    )


@pytest.fixture
def example_feature_grid(tmp_path: Path) -> str:
    """Baseline feature grid in the layout published alongside the API, written to a local zarr store."""
    x = np.linspace(-3e5, 3e5, 4)
    y = np.linspace(-3.0e6, -2.4e6, 3)
    shape = (y.size, x.size)
    area = np.full(shape, 5000.0**2)
    area[0, 0] = np.nan  # an off-ice cell, which the loader drops
    grid = xr.Dataset(
        {
            "RAdh": (["y", "x"], np.full(shape, -0.5)),
            "RAdist": (["y", "x"], np.ones(shape)),
            "RAmode": (["y", "x"], np.full(shape, 2.0)),
            "LST": (["y", "x"], np.full(shape, 260.0)),
            "elev": (["y", "x"], np.full(shape, 2000.0)),
            "grid_cell_area": (["y", "x"], area),
        },
        coords={"x": x, "y": y},
    )
    store = tmp_path / "features.zarr"
    grid.to_zarr(store)
    return str(store)


class FakeBooster:
    """An sklearn-style estimator standing in for the trained model bundle."""

    def __init__(self, metres_per_year: float = 1.0):
        self.metres_per_year = metres_per_year

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        """Scale with the elevation-change feature, so a year whose RAdh was refreshed predicts differently."""
        return self.metres_per_year * matrix[:, 0]


@pytest.fixture
def example_annual_model(example_feature_grid: str, tmp_path: Path) -> mbval.AnnualFeatureModel:
    """Model whose grid and booster are real but whose raw per-year inputs are stubbed out."""
    dataprep = tmp_path / "DataPrep"
    dataprep.mkdir()
    with patch.object(mbval, "_load_xgb_model", return_value=(FakeBooster(), ["RAdh", "RAdist", "RAmode", "LST"])):
        model = mbval.AnnualFeatureModel(dataprep_dir=dataprep, feature_grid_url=example_feature_grid)
    model._annual_elevation_change = lambda year: np.full(len(model.baseline), -0.1 * (year - 2009))  # noqa: SLF001
    model._annual_temperature = lambda year: np.full(len(model.baseline), 260.0)  # noqa: SLF001
    return model


@pytest.fixture
def example_prediction_dataset() -> xr.Dataset:
    """Build a small prediction grid in EPSG:3413 metres, as returned by the what-if module."""
    x = np.linspace(-3e5, 3e5, 6)
    y = np.linspace(-3.0e6, -2.4e6, 5)
    prediction = np.linspace(-2.0, 0.5, x.size * y.size).reshape(y.size, x.size)
    return xr.Dataset({"prediction": (["y", "x"], prediction)}, coords={"x": x, "y": y})


def test_annual_series_from_mass_balance_dataset(example_product_dataset: xr.Dataset):
    annual = mbval.annual_series_from_mass_balance_dataset(example_product_dataset)
    assert list(annual.columns) == ["year", "simonsen_gt_yr", "simonsen_unc_gt_yr"]
    assert annual["year"].tolist() == [2010, 2011, 2012, 2013]
    # Three points losing 1 Gt each, and twice that in 2011.
    np.testing.assert_allclose(annual["simonsen_gt_yr"], [-3.0, -6.0, -3.0, -3.0])
    # Uncertainties are summed linearly over the ice sheet.
    np.testing.assert_allclose(annual["simonsen_unc_gt_yr"], [0.3] * 4)


def test_annual_series_from_mass_balance_dataset_custom_series_name(example_product_dataset: xr.Dataset):
    annual = mbval.annual_series_from_mass_balance_dataset(example_product_dataset, series_name="dtcis")
    assert list(annual.columns) == ["year", "dtcis_gt_yr", "dtcis_unc_gt_yr"]


def test_annual_series_from_mass_balance_dataset_without_time(example_product_dataset: xr.Dataset):
    with pytest.raises(KeyError, match="no time dimension"):
        mbval.annual_series_from_mass_balance_dataset(example_product_dataset.isel(time=0))


def test_download_to_cache_downloads_once(tmp_path: Path):
    with patch.object(mbval.requests, "get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = b"payload"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        first = mbval._download_to_cache("https://example.invalid/file.csv", "file.csv", tmp_path)
        second = mbval._download_to_cache("https://example.invalid/file.csv", "file.csv", tmp_path)

    assert first == second
    assert first.parent == tmp_path
    assert first.name.startswith("file-") and first.suffix == ".csv"
    assert first.read_bytes() == b"payload"
    mock_get.assert_called_once()


def test_download_to_cache_separates_urls(tmp_path: Path):
    # Two releases of the same record ask to be cached under the same name; each must keep its own content.
    with patch.object(mbval.requests, "get") as mock_get:
        mock_get.side_effect = [
            MagicMock(content=b"imbie-3", raise_for_status=MagicMock(return_value=None)),
            MagicMock(content=b"imbie-2021", raise_for_status=MagicMock(return_value=None)),
        ]
        newest = mbval._download_to_cache("https://example.org/imbie3", "imbie.csv", tmp_path)  # noqa: SLF001
        previous = mbval._download_to_cache("https://example.org/imbie2021", "imbie.csv", tmp_path)  # noqa: SLF001

    assert newest != previous
    assert newest.read_bytes() == b"imbie-3"
    assert previous.read_bytes() == b"imbie-2021"
    assert newest.suffix == previous.suffix == ".csv"


def test_load_imbie_annual(example_imbie_csv: Path):
    annual = mbval.load_imbie_annual(path=example_imbie_csv)
    # The truncated 2012 is dropped, as no annual mean can be formed from six months.
    assert annual["year"].tolist() == [2010, 2011]
    np.testing.assert_allclose(annual["imbie_gt_yr"], [-100.0, -200.0])
    np.testing.assert_allclose(annual["imbie_unc_gt_yr"], [50.0, 50.0])
    assert annual["imbie_n"].tolist() == [12, 12]


def test_load_imbie_annual_imbie3_layout(example_imbie3_csv: Path):
    annual = mbval.load_imbie_annual(path=example_imbie3_csv)
    # The commented header is skipped, the dates give the calendar year, and the truncated 2012 is dropped.
    assert annual["year"].tolist() == [2010, 2011]
    # The total, not the surface (-130, -230) or dynamic (-30) component.
    np.testing.assert_allclose(annual["imbie_gt_yr"], [-100.0, -200.0])
    np.testing.assert_allclose(annual["imbie_unc_gt_yr"], [50.0, 50.0])


def test_load_imbie_annual_without_a_time_column(tmp_path: Path):
    path = tmp_path / "imbie_timeless.csv"
    pd.DataFrame([{"Mass balance (Gt/yr)": -100.0}] * 12).to_csv(path, index=False)
    with pytest.raises(KeyError, match="neither a decimal year nor a date"):
        mbval.load_imbie_annual(path=path)


def test_load_imbie_annual_without_a_rate_column(tmp_path: Path):
    path = tmp_path / "imbie_rateless.csv"
    pd.DataFrame([{"Date": "2010-01-01", "Cumulative mass balance anomaly (Gt)": -100.0}]).to_csv(path, index=False)
    with pytest.raises(KeyError, match="no mass-balance rate"):
        mbval.load_imbie_annual(path=path)


def test_load_imbie_annual_without_uncertainty_column(tmp_path: Path):
    rows = [{"Year": 2010 + month / 12, "Mass balance (Gt/yr)": -100.0} for month in range(12)]
    path = tmp_path / "imbie_no_unc.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    annual = mbval.load_imbie_annual(path=path)
    assert annual["imbie_unc_gt_yr"].isna().all()


def test_load_imbie_annual_downloads_when_no_path_given(example_imbie_csv: Path, tmp_path: Path):
    with patch.object(mbval, "_download_to_cache", return_value=example_imbie_csv) as mock_download:
        annual = mbval.load_imbie_annual(cache_dir=tmp_path)
    mock_download.assert_called_once()
    assert not annual.empty


def test_load_gravimetric_cumulative_gravis_layout(example_gravis_file: Path):
    cumulative = mbval.load_gravimetric_cumulative(path=example_gravis_file)
    assert list(cumulative.columns) == ["decimal_year", "mass_gt", "sigma_gt"]
    # The 1-sigma of the ice-sheet total sits after the per-basin columns, not next to the total.
    np.testing.assert_allclose(cumulative["sigma_gt"].unique(), [20.0])
    assert cumulative["decimal_year"].is_monotonic_increasing


def test_load_gravimetric_cumulative_three_column_layout(tmp_path: Path):
    path = tmp_path / "simple.dat"
    np.savetxt(path, np.column_stack([[2010.5, 2010.1], [-50.0, -10.0], [3.0, 2.0]]))

    cumulative = mbval.load_gravimetric_cumulative(path=path)
    np.testing.assert_allclose(cumulative["decimal_year"], [2010.1, 2010.5])
    np.testing.assert_allclose(cumulative["mass_gt"], [-10.0, -50.0])
    np.testing.assert_allclose(cumulative["sigma_gt"], [2.0, 3.0])


def test_load_gravimetric_cumulative_rejects_narrow_file(tmp_path: Path):
    path = tmp_path / "narrow.dat"
    np.savetxt(path, np.column_stack([[2010.0, 2010.1], [-50.0, -10.0]]))

    with pytest.raises(ValueError, match="at least three columns"):
        mbval.load_gravimetric_cumulative(path=path)


def test_annual_rates_from_cumulative(example_gravis_file: Path):
    cumulative = mbval.load_gravimetric_cumulative(path=example_gravis_file)
    annual = mbval.annual_rates_from_cumulative(cumulative)

    assert annual["year"].tolist() == [2010, 2011]
    np.testing.assert_allclose(annual["gmb_gt_yr"], [-100.0, -100.0])
    # The 1-sigma values of the two year boundaries are added in quadrature.
    np.testing.assert_allclose(annual["gmb_unc_gt_yr"], [np.hypot(20.0, 20.0)] * 2)
    assert annual["gmb_n"].tolist() == [12, 12]


def test_annual_rates_from_cumulative_drops_years_with_a_sampling_gap(example_gravis_file: Path):
    cumulative = mbval.load_gravimetric_cumulative(path=example_gravis_file)
    # Remove the second half of 2011, mimicking the break between GRACE and GRACE-FO.
    gapped = cumulative[(cumulative["decimal_year"] < 2011.4) | (cumulative["decimal_year"] > 2011.9)]

    annual = mbval.annual_rates_from_cumulative(gapped)
    assert annual["year"].tolist() == [2010]


def test_annual_rates_from_cumulative_custom_series_name(example_gravis_file: Path):
    cumulative = mbval.load_gravimetric_cumulative(path=example_gravis_file)
    annual = mbval.annual_rates_from_cumulative(cumulative, series_name="grace")
    assert list(annual.columns) == ["year", "grace_gt_yr", "grace_unc_gt_yr", "grace_n"]


def test_residual_column_name():
    assert mbval.residual_column_name("simonsen_gt_yr", "imbie_gt_yr") == "residual_simonsen_imbie_gt_yr"


def test_build_intercomparison_table(example_comparison_table: pd.DataFrame):
    table = example_comparison_table
    assert table["year"].tolist() == [2010, 2011, 2012, 2013]
    np.testing.assert_allclose(table["residual_simonsen_imbie_gt_yr"], [20.0, -20.0, -30.0, -20.0])
    np.testing.assert_allclose(table["residual_gmb_imbie_gt_yr"], [10.0, -10.0, -40.0, 20.0])
    # Mass loss is reported as a positive sea-level contribution.
    np.testing.assert_allclose(table["simonsen_mm_sle"], np.array([100.0, 220.0, 180.0, 300.0]) / mbval.GT_PER_MM_SLE)


def test_build_intercomparison_table_window(example_annual_series: list[pd.DataFrame]):
    table = mbval.build_intercomparison_table(example_annual_series, start_year=2011, end_year=2015)
    assert table["year"].tolist() == [2011, 2012, 2013, 2014, 2015]
    assert table.loc[table["year"] > 2013, mbval.SIMONSEN_COL_NAME].isna().all()


def test_build_intercomparison_table_ignores_empty_series(example_annual_series: list[pd.DataFrame]):
    table = mbval.build_intercomparison_table([*example_annual_series, pd.DataFrame(columns=["year"])])
    assert mbval.GMB_COL_NAME in table.columns


def test_build_intercomparison_table_without_any_year():
    with pytest.raises(ValueError, match="does not contain|contains any year|any year"):
        mbval.build_intercomparison_table([pd.DataFrame(columns=["year"])])


def test_available_comparisons(example_comparison_table: pd.DataFrame):
    keys = [key for key, _, _, _ in mbval.available_comparisons(example_comparison_table)]
    assert keys == ["simonsen_vs_imbie", "simonsen_vs_gmb", "gmb_vs_imbie"]


def test_available_comparisons_without_a_reference(example_annual_series: list[pd.DataFrame]):
    simonsen, imbie, _ = example_annual_series
    table = mbval.build_intercomparison_table([simonsen, imbie])
    keys = [key for key, _, _, _ in mbval.available_comparisons(table)]
    assert keys == ["simonsen_vs_imbie"]


def test_compute_metrics(example_comparison_table: pd.DataFrame):
    metrics = mbval.compute_metrics(example_comparison_table, mbval.SIMONSEN_COL_NAME, mbval.IMBIE_COL_NAME)
    assert metrics["n_years"] == 4
    assert metrics["first_year"] == 2010
    assert metrics["last_year"] == 2013
    assert metrics["bias_gt_yr"] == pytest.approx(-12.5)
    assert metrics["mae_gt_yr"] == pytest.approx(22.5)
    assert metrics["rmse_gt_yr"] == pytest.approx(np.sqrt((20**2 + 20**2 + 30**2 + 20**2) / 4))
    assert metrics["r2"] < 1.0
    assert metrics["pearson_r"] == pytest.approx(0.9, abs=0.1)
    assert metrics["cum_bias_mm_sle"] == pytest.approx(50.0 / mbval.GT_PER_MM_SLE)


def test_compute_metrics_of_a_series_against_itself(example_comparison_table: pd.DataFrame):
    table = example_comparison_table.copy()
    table["copy_gt_yr"] = table[mbval.IMBIE_COL_NAME]
    metrics = mbval.compute_metrics(table, "copy_gt_yr", mbval.IMBIE_COL_NAME)
    assert metrics["bias_gt_yr"] == pytest.approx(0.0)
    assert metrics["rmse_gt_yr"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_compute_metrics_without_overlap(example_comparison_table: pd.DataFrame):
    table = example_comparison_table.copy()
    table[mbval.GMB_COL_NAME] = np.nan
    assert mbval.compute_metrics(table, mbval.SIMONSEN_COL_NAME, mbval.GMB_COL_NAME) == {}


def test_compute_all_metrics(example_comparison_table: pd.DataFrame):
    metrics = mbval.compute_all_metrics(example_comparison_table)
    assert set(metrics) == {"simonsen_vs_imbie", "simonsen_vs_gmb", "gmb_vs_imbie"}


def test_metrics_summary_table(example_comparison_table: pd.DataFrame):
    summary = mbval.metrics_summary_table(mbval.compute_all_metrics(example_comparison_table))
    assert list(summary.columns) == ["DTC-IS vs IMBIE", "DTC-IS vs GMB", "GMB vs IMBIE"]
    assert summary.loc["Years compared", "DTC-IS vs IMBIE"] == "4 (2010-2013)"
    assert summary.loc["Bias A - B [Gt/yr]", "DTC-IS vs IMBIE"] == "-12.5"


def test_format_intercomparison_table(example_comparison_table: pd.DataFrame):
    shown = mbval.format_intercomparison_table(example_comparison_table)
    assert shown.index.name == "year"
    assert list(shown.columns) == [
        "DTC-IS",
        "DTC-IS 1σ",
        "IMBIE",
        "IMBIE 1σ",
        "GMB",
        "GMB 1σ",
        "DTC-IS - IMBIE",
        "DTC-IS - GMB",
        "GMB - IMBIE",
    ]
    assert shown.loc[2010, "DTC-IS - IMBIE"] == pytest.approx(20.0)


def test_plot_mass_balance_prediction_map_runs_no_error(example_prediction_dataset: xr.Dataset):
    with patch("matplotlib.pyplot.show"):
        mbval.plot_mass_balance_prediction_map(example_prediction_dataset, plot_description_str="Control run")


def test_plot_mass_balance_prediction_map_with_a_scenario(example_prediction_dataset: xr.Dataset):
    scenario = example_prediction_dataset.copy(deep=True)
    scenario["prediction"] = scenario["prediction"] - 0.2
    with patch("matplotlib.pyplot.show"):
        mbval.plot_mass_balance_prediction_map(scenario, example_prediction_dataset, "Warming scenario")


def test_plot_mass_balance_prediction_map_skips_an_empty_anomaly(example_prediction_dataset: xr.Dataset):
    # A scenario identical to the control has nothing to show in a second panel.
    with (
        patch("matplotlib.pyplot.show"),
        patch("matplotlib.pyplot.subplots", wraps=mbval.plt.subplots) as mock_subplots,
    ):
        mbval.plot_mass_balance_prediction_map(example_prediction_dataset, example_prediction_dataset)
    assert mock_subplots.call_args.args[1] == 1


def test_plot_mb_validation_four_panels_runs_no_error(example_comparison_table: pd.DataFrame):
    metrics = mbval.compute_all_metrics(example_comparison_table)
    epochs = 2010.0 + np.arange(48) / 12.0
    cumulative = pd.DataFrame({"decimal_year": epochs, "mass_gt": -100.0 * (epochs - 2010.0), "sigma_gt": 20.0})
    with patch("matplotlib.pyplot.show"):
        mbval.plot_mb_validation_four_panels(
            example_comparison_table,
            metrics,
            gmb_cumulative=cumulative,
            what_if_total_gt_yr=-431.0,
            plot_description_str="Test validation figure",
        )


def test_plot_mb_validation_four_panels_without_optional_inputs(example_annual_series: list[pd.DataFrame]):
    simonsen, imbie, _ = example_annual_series
    table = mbval.build_intercomparison_table([simonsen, imbie])
    with patch("matplotlib.pyplot.show"):
        mbval.plot_mb_validation_four_panels(table, mbval.compute_all_metrics(table))


@pytest.fixture
def example_cci_file(tmp_path: Path) -> Path:
    """ESA CCI style cumulative elevation change over Greenland, on a coarse lon/lat grid."""
    from netCDF4 import Dataset

    dataprep = tmp_path / "DataPrep"
    dataprep.mkdir()
    path = dataprep / "CCI_SEC_TEST_ZZ_dhdt_5km_012010_012012.nc"
    longitude, latitude = np.meshgrid(np.linspace(-60.0, -30.0, 5), np.linspace(65.0, 80.0, 5))
    # Epochs at 2010.0, 2010.5 and 2012.0, each half a metre lower than the last.
    epochs = [
        (datetime(2010, 1, 1) - datetime(1970, 1, 1)).total_seconds(),
        (datetime(2010, 7, 1) - datetime(1970, 1, 1)).total_seconds(),
        (datetime(2012, 1, 1) - datetime(1970, 1, 1)).total_seconds(),
    ]
    cumulative = np.stack([np.zeros_like(longitude), np.full_like(longitude, -0.5), np.full_like(longitude, -1.0)])

    with Dataset(path, "w") as source:
        source.createDimension("time", len(epochs))
        source.createDimension("y", longitude.shape[0])
        source.createDimension("x", longitude.shape[1])
        source.createVariable("time", "f8", ("time",))[:] = epochs
        source.createVariable("lat", "f8", ("y", "x"))[:] = latitude
        source.createVariable("lon", "f8", ("y", "x"))[:] = longitude
        source.createVariable("ZZ", "f8", ("time", "y", "x"))[:] = cumulative
    return path


@pytest.fixture
def example_lst_table(tmp_path: Path) -> Path:
    """Annual surface-temperature point table, in EPSG:3413 metres with one column per year."""
    point_x, point_y = np.meshgrid(np.linspace(-4e5, 4e5, 4), np.linspace(-3.1e6, -2.3e6, 4))
    path = tmp_path / "c3slst.pkl"
    pd.DataFrame(
        {
            "X": point_x.ravel(),
            "Y": point_y.ravel(),
            "2010": np.full(point_x.size, 261.0),
            "2011": np.full(point_x.size, 262.0),
        }
    ).to_pickle(path)
    return path


@pytest.fixture
def example_monthly_lst_dir(tmp_path: Path) -> Path:
    """DataPrep tree whose monthly C3S surface-temperature files cover one calendar year."""
    dataprep = tmp_path / "DataPrep"
    monthly = dataprep / mbval.LST_MONTHLY_SUBDIR
    monthly.mkdir(parents=True)
    longitude = np.linspace(-75.0, -10.0, 8)
    latitude = np.linspace(59.0, 84.0, 8)
    for month in range(1, 13):
        xr.Dataset(
            {"surface_temperature": (["lat", "lon"], np.full((latitude.size, longitude.size), 250.0 + month))},
            coords={"lon": longitude, "lat": latitude},
        ).to_netcdf(monthly / f"c3s_ist_2011{month:02d}.nc")
    return dataprep


def _model_over(
    feature_grid_url: str, dataprep_dir: Path | None, lst_table_path: Path | None = None
) -> mbval.AnnualFeatureModel:
    """Build a model over the given raw inputs, with the trained bundle stubbed out."""
    with patch.object(mbval, "_load_xgb_model", return_value=(FakeBooster(), ["RAdh", "RAdist", "RAmode", "LST"])):
        return mbval.AnnualFeatureModel(
            dataprep_dir=dataprep_dir, lst_table_path=lst_table_path, feature_grid_url=feature_grid_url
        )


def test_build_intercomparison_table_with_the_model_series(
    example_annual_series: list[pd.DataFrame], example_mlmodel_series: pd.DataFrame
):
    table = mbval.build_intercomparison_table([*example_annual_series, example_mlmodel_series])
    assert np.isnan(table.loc[table["year"] == 2010, mbval.MLMODEL_COL_NAME]).all()
    np.testing.assert_allclose(table["residual_mlmodel_simonsen_gt_yr"].dropna(), [-40.0, -60.0, -20.0])
    np.testing.assert_allclose(table["residual_mlmodel_imbie_gt_yr"].dropna(), [-60.0, -90.0, -40.0])


def test_available_comparisons_with_the_model_series(
    example_annual_series: list[pd.DataFrame], example_mlmodel_series: pd.DataFrame
):
    table = mbval.build_intercomparison_table([*example_annual_series, example_mlmodel_series])
    keys = [key for key, _, _, _ in mbval.available_comparisons(table)]
    assert keys == [
        "mlmodel_vs_imbie",
        "mlmodel_vs_gmb",
        "mlmodel_vs_simonsen",
        "simonsen_vs_imbie",
        "simonsen_vs_gmb",
        "gmb_vs_imbie",
    ]


def test_format_intercomparison_table_with_the_model_series(
    example_annual_series: list[pd.DataFrame], example_mlmodel_series: pd.DataFrame
):
    table = mbval.build_intercomparison_table([*example_annual_series, example_mlmodel_series])
    shown = mbval.format_intercomparison_table(table)
    assert "ML model" in shown.columns
    assert "ML model - DTC-IS" in shown.columns
    # The provenance column is carried through the table but is not part of the display view.
    assert "mlmodel_source" in table.columns
    assert "mlmodel_source" not in shown.columns


def test_cumulative_curve_ties_a_covering_series_to_zero_at_the_anchor(example_comparison_table: pd.DataFrame):
    years, cumulated, tied_at = mbval._cumulative_curve(  # noqa: SLF001
        example_comparison_table, mbval.SIMONSEN_COL_NAME, anchor=2011.0
    )
    assert tied_at == 2011.0
    assert cumulated[np.argmin(np.abs(years - 2011.0))] == pytest.approx(0.0)


def test_cumulative_curve_ties_a_late_series_to_the_reference_curve(
    example_annual_series: list[pd.DataFrame], example_mlmodel_series: pd.DataFrame
):
    table = mbval.build_intercomparison_table([*example_annual_series, example_mlmodel_series])
    tie_curve = (np.array([2010.0, 2011.0, 2012.0]), np.array([0.0, -150.0, -300.0]))
    years, cumulated, tied_at = mbval._cumulative_curve(  # noqa: SLF001
        table, mbval.MLMODEL_COL_NAME, anchor=2010.0, tie_curve=tie_curve
    )
    # The series starts in 2011, so it cannot be tied at the 2010 anchor and meets the reference curve instead.
    assert tied_at == 2011.0
    assert years[0] == 2011.0
    assert cumulated[0] == pytest.approx(-150.0)
    assert cumulated[-1] == pytest.approx(-150.0 - 820.0)


def test_cumulative_curve_without_anything_to_tie_to(
    example_annual_series: list[pd.DataFrame], example_mlmodel_series: pd.DataFrame
):
    table = mbval.build_intercomparison_table([*example_annual_series, example_mlmodel_series])
    _, cumulated, tied_at = mbval._cumulative_curve(table, mbval.MLMODEL_COL_NAME, anchor=2000.0)  # noqa: SLF001
    assert tied_at == 2011.0
    assert cumulated[0] == pytest.approx(0.0)


def test_cumulative_curve_of_a_single_year(example_comparison_table: pd.DataFrame):
    single = example_comparison_table[example_comparison_table["year"] == 2011]
    assert mbval._cumulative_curve(single, mbval.SIMONSEN_COL_NAME, anchor=None) is None  # noqa: SLF001


def test_plot_mb_validation_four_panels_with_the_model_series(
    example_annual_series: list[pd.DataFrame], example_mlmodel_series: pd.DataFrame
):
    table = mbval.build_intercomparison_table([*example_annual_series, example_mlmodel_series])
    epochs = 2010.0 + np.arange(48) / 12.0
    cumulative = pd.DataFrame({"decimal_year": epochs, "mass_gt": -100.0 * (epochs - 2010.0), "sigma_gt": 20.0})
    with patch("matplotlib.pyplot.show"):
        mbval.plot_mb_validation_four_panels(table, mbval.compute_all_metrics(table), gmb_cumulative=cumulative)


def test_resolve_optional_path_prefers_the_argument(tmp_path: Path):
    assert mbval._resolve_optional_path(tmp_path, "DTC_IS_UNSET_FOR_TESTS") == tmp_path  # noqa: SLF001


def test_resolve_optional_path_falls_back_to_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DTC_IS_TEST_PATH", str(tmp_path))
    assert mbval._resolve_optional_path(None, "DTC_IS_TEST_PATH") == tmp_path  # noqa: SLF001


def test_resolve_optional_path_drops_a_path_that_does_not_exist(tmp_path: Path):
    assert mbval._resolve_optional_path(tmp_path / "absent", "DTC_IS_UNSET_FOR_TESTS") is None  # noqa: SLF001
    assert mbval._resolve_optional_path(None, "DTC_IS_UNSET_FOR_TESTS") is None  # noqa: SLF001


def test_annual_feature_model_without_any_raw_input(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(mbval.DATAPREP_DIR_ENV_VAR, raising=False)
    monkeypatch.delenv(mbval.LST_TABLE_ENV_VAR, raising=False)
    with pytest.raises(mbval.TemporalFeaturesUnavailableError, match="DTC_IS_DATAPREP_DIR"):
        mbval.AnnualFeatureModel()


def test_annual_feature_model_rejects_a_grid_missing_model_features(example_feature_grid: str, tmp_path: Path):
    dataprep = tmp_path / "DataPrep"
    dataprep.mkdir()
    with (
        patch.object(mbval, "_load_xgb_model", return_value=(FakeBooster(), ["RAdh", "iv"])),
        pytest.raises(mbval.TemporalFeaturesUnavailableError, match="missing model features"),
    ):
        mbval.AnnualFeatureModel(dataprep_dir=dataprep, feature_grid_url=example_feature_grid)


def test_load_baseline_feature_grid_drops_cells_without_an_area(example_feature_grid: str):
    baseline, area = mbval._load_baseline_feature_grid(example_feature_grid)  # noqa: SLF001
    assert len(baseline) == len(area) == 11  # 12 cells, one of them off-ice
    assert "grid_cell_area" not in baseline.columns
    assert {"X", "Y", "RAdh", "LST"} <= set(baseline.columns)
    np.testing.assert_allclose(area, 5000.0**2)


def test_load_baseline_feature_grid_without_an_area_variable(tmp_path: Path):
    store = tmp_path / "arealess.zarr"
    xr.Dataset({"RAdh": (["y", "x"], np.zeros((2, 2)))}, coords={"x": [0.0, 1.0], "y": [0.0, 1.0]}).to_zarr(store)
    with pytest.raises(mbval.TemporalFeaturesUnavailableError, match="no grid cell area"):
        mbval._load_baseline_feature_grid(str(store))  # noqa: SLF001


def test_build_features_refreshes_only_the_time_varying_ones(example_annual_model: mbval.AnnualFeatureModel):
    features, refreshed = example_annual_model.build_features(2012)
    assert refreshed == ["RAdh", "RAdist", "LST"]  # 2012 is past the SARIn transition, so RAmode is not touched
    np.testing.assert_allclose(features["RAdh"], -0.3)
    np.testing.assert_allclose(features["elev"], 2000.0)  # a static feature, left alone


def test_build_features_sets_the_acquisition_mode_before_the_sarin_transition(
    example_annual_model: mbval.AnnualFeatureModel,
):
    features, refreshed = example_annual_model.build_features(2008)
    assert "RAmode" in refreshed
    np.testing.assert_allclose(features["RAmode"], 1.0)


def test_predict_year_integrates_the_grid_to_a_total(example_annual_model: mbval.AnnualFeatureModel):
    total_gt_yr, refreshed = example_annual_model.predict_year(2012)
    # 11 cells of 25 km2 each losing 0.3 m of ice equivalent, at 917 kg/m3.
    assert total_gt_yr == pytest.approx(-11 * 25e6 * 0.3 * mbval.ICE_DENSITY_KG_M3 * 1e-12)
    assert "RAdh" in refreshed


def test_predict_annual_mass_balance(example_annual_model: mbval.AnnualFeatureModel):
    with patch.object(mbval, "AnnualFeatureModel", return_value=example_annual_model):
        frame = mbval.predict_annual_mass_balance([2011, 2012])
    assert list(frame.columns) == ["year", "mlmodel_gt_yr", "mlmodel_source"]
    assert frame["year"].tolist() == [2011, 2012]
    assert frame["mlmodel_source"].tolist() == ["RAdh+RAdist+LST"] * 2
    # A steeper elevation change in 2012 has to give a larger loss than 2011.
    assert frame["mlmodel_gt_yr"].iloc[1] < frame["mlmodel_gt_yr"].iloc[0] < 0


def test_predict_annual_mass_balance_custom_series_name(example_annual_model: mbval.AnnualFeatureModel):
    with patch.object(mbval, "AnnualFeatureModel", return_value=example_annual_model):
        frame = mbval.predict_annual_mass_balance([2011], series_name="xgb")
    assert list(frame.columns) == ["year", "xgb_gt_yr", "xgb_source"]


def test_predict_annual_mass_balance_rejects_a_baseline_only_series(example_annual_model: mbval.AnnualFeatureModel):
    # Neither elevation change nor temperature available: every year would repeat the baseline epoch.
    example_annual_model._annual_elevation_change = lambda year: None  # noqa: SLF001
    example_annual_model._annual_temperature = lambda year: None  # noqa: SLF001
    with (
        patch.object(mbval, "AnnualFeatureModel", return_value=example_annual_model),
        pytest.raises(mbval.TemporalFeaturesUnavailableError, match="repeat the baseline epoch"),
    ):
        mbval.predict_annual_mass_balance([2011, 2012])


def test_to_model_grid_interpolates_and_fills_holes(example_annual_model: mbval.AnnualFeatureModel):
    # Two source points either side of the grid, so every target is inside the hull along x but not along y.
    source_x = np.array([-4e5, 4e5, -4e5, 4e5])
    source_y = np.array([-3.1e6, -3.1e6, -2.3e6, -2.3e6])
    gridded = example_annual_model._to_model_grid(source_x, source_y, np.array([0.0, 10.0, 0.0, 10.0]))  # noqa: SLF001
    assert np.isfinite(gridded).all()
    assert gridded.min() >= 0.0
    assert gridded.max() <= 10.0


def test_annual_elevation_change_rescales_a_partial_year(
    example_feature_grid: str, example_cci_file: Path, tmp_path: Path
):
    model = _model_over(example_feature_grid, example_cci_file.parent)
    # 2010 is covered by two epochs about half a year apart and 0.5 m down, so the annual rate is about 1 m.
    np.testing.assert_allclose(model._annual_elevation_change(2010), -1.0, rtol=0.02)  # noqa: SLF001


def test_annual_elevation_change_needs_two_epochs_in_the_year(example_feature_grid: str, example_cci_file: Path):
    model = _model_over(example_feature_grid, example_cci_file.parent)
    assert model._annual_elevation_change(2020) is None  # noqa: SLF001


def test_annual_elevation_change_without_a_cci_file(example_feature_grid: str, tmp_path: Path):
    empty = tmp_path / "EmptyDataPrep"
    empty.mkdir()
    model = _model_over(example_feature_grid, empty)
    assert model._annual_elevation_change(2010) is None  # noqa: SLF001


def test_temperature_from_a_point_table(example_feature_grid: str, example_lst_table: Path):
    model = _model_over(example_feature_grid, None, lst_table_path=example_lst_table)
    np.testing.assert_allclose(model._annual_temperature(2011), 262.0, rtol=1e-6)  # noqa: SLF001
    assert model._annual_temperature(1999) is None  # noqa: SLF001 - a year the table does not carry


def test_temperature_from_a_table_that_is_not_one(example_feature_grid: str, tmp_path: Path):
    path = tmp_path / "not_a_table.pkl"
    pd.DataFrame({"something": [1, 2]}).to_pickle(path)
    model = _model_over(example_feature_grid, None, lst_table_path=path)
    assert model._annual_temperature(2011) is None  # noqa: SLF001


def test_load_xgb_model_from_a_local_bundle(tmp_path: Path):
    import cloudpickle

    path = tmp_path / "model.pkl"
    path.write_bytes(cloudpickle.dumps({"model": FakeBooster(), "feature_names": ["RAdh", "LST"]}))
    booster, feature_names = mbval._load_xgb_model(str(path))  # noqa: SLF001
    assert feature_names == ["RAdh", "LST"]
    assert isinstance(booster, FakeBooster)


def test_load_xgb_model_from_a_tuple_bundle(tmp_path: Path):
    import cloudpickle

    path = tmp_path / "model.pkl"
    path.write_bytes(cloudpickle.dumps((FakeBooster(), ["RAdh"])))
    _, feature_names = mbval._load_xgb_model(str(path))  # noqa: SLF001
    assert feature_names == ["RAdh"]


def test_temperature_falls_back_to_the_monthly_files(example_feature_grid: str, example_monthly_lst_dir: Path):
    model = _model_over(example_feature_grid, example_monthly_lst_dir)
    # No point table, so the twelve monthly files are averaged: 251 through 262 K.
    np.testing.assert_allclose(model._annual_temperature(2011), 256.5, rtol=1e-6)  # noqa: SLF001


def test_temperature_from_monthly_files_of_a_year_that_is_not_covered(
    example_feature_grid: str, example_monthly_lst_dir: Path
):
    model = _model_over(example_feature_grid, example_monthly_lst_dir)
    assert model._annual_temperature(2015) is None  # noqa: SLF001


def test_temperature_without_any_monthly_directory(example_feature_grid: str, tmp_path: Path):
    empty = tmp_path / "EmptyDataPrep"
    empty.mkdir()
    model = _model_over(example_feature_grid, empty)
    assert model._annual_temperature(2011) is None  # noqa: SLF001


def test_build_features_supplies_the_decimal_year(example_feature_grid: str, tmp_path: Path):
    dataprep = tmp_path / "DataPrep"
    dataprep.mkdir()
    with patch.object(mbval, "_load_xgb_model", return_value=(FakeBooster(), ["RAdh", "decimal_year"])):
        model = mbval.AnnualFeatureModel(dataprep_dir=dataprep, feature_grid_url=example_feature_grid)
    features, refreshed = model.build_features(2012)
    assert "decimal_year" in refreshed
    np.testing.assert_allclose(features["decimal_year"], 2012.5)


def test_predict_year_with_a_real_xgboost_booster(example_feature_grid: str, tmp_path: Path):
    import xgboost

    dataprep = tmp_path / "DataPrep"
    dataprep.mkdir()
    feature_names = ["RAdh", "LST"]
    training = xgboost.DMatrix(
        np.array([[-1.0, 260.0], [-0.5, 261.0], [0.0, 262.0]]),
        label=np.array([-1.0, -0.5, 0.0]),
        feature_names=feature_names,
    )
    booster = xgboost.train({"max_depth": 2, "eta": 1.0}, training, num_boost_round=2)

    with patch.object(mbval, "_load_xgb_model", return_value=(booster, feature_names)):
        model = mbval.AnnualFeatureModel(dataprep_dir=dataprep, feature_grid_url=example_feature_grid)
    model._annual_elevation_change = lambda year: np.full(len(model.baseline), -1.0)  # noqa: SLF001
    model._annual_temperature = lambda year: np.full(len(model.baseline), 260.0)  # noqa: SLF001

    total_gt_yr, refreshed = model.predict_year(2012)
    assert refreshed == ["RAdh", "LST"]
    assert total_gt_yr < 0  # the booster was trained to map a metre of thinning onto a loss
