"""Tests for uc4_umelt_plotting_helpers."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Polygon, box

from dtc_is_notebook_helpers import uc4_umelt_plotting_helpers as h


def test_get_date_picker_uses_the_description_and_default_date():
    """Test that the date picker shows the given description and default date."""
    picker = h.get_date_picker(desc="Start date:", default_date=date(2019, 2, 3))
    assert picker.description == "Start date:"
    assert picker.value == date(2019, 2, 3)


def test_area_km2_of_a_one_degree_box_near_the_pole():
    """Test the area of a 1 x 1 degree box at 70-71S, about 111 km x 37 km."""
    assert h._area_km2(box(0, -71, 1, -70)) == pytest.approx(4_100, rel=0.05)


def test_polygons_from_keeps_only_polygons():
    """Test that only polygon features are converted; lines and features without geometry are ignored."""
    square = {"type": "Polygon", "coordinates": [[[0, -80], [1, -80], [1, -79], [0, -79], [0, -80]]]}
    features = [
        {"geometry": square},
        {"geometry": {"type": "LineString", "coordinates": [[0, -80], [1, -79]]}},
        {"geometry": None},
        {},
    ]
    polygons = h._polygons_from(features)
    assert len(polygons) == 1
    assert polygons[0].equals(Polygon(square["coordinates"][0]))


A, B, C = box(0, -80, 1, -79), box(2, -80, 3, -79), box(4, -80, 5, -79)


@pytest.mark.parametrize(
    ("polygons", "previous_count", "kept", "old_index", "expected"),
    [
        ([], 1, A, 0, None),  # nothing left
        ([A, B, C], 2, A, 0, 2),  # a new polygon was drawn: select it
        ([B, C], 3, C, 2, 1),  # another polygon was deleted: keep the same polygon at its new index
        ([A, C], 3, B, 1, 1),  # the selected polygon was deleted: keep the position
        ([A], 2, B, 1, 0),  # ... unless that position no longer exists: select the last one
    ],
    ids=["empty", "new-polygon", "other-deleted", "selected-deleted", "selected-deleted-at-end"],
)
def test_index_to_select(polygons, previous_count, kept, old_index, expected):
    """Test which polygon is selected after the list of drawn polygons changes."""
    assert h._index_to_select(polygons, previous_count, kept, old_index) == expected


@pytest.mark.parametrize(("n", "expected"), [(1, ""), (2, "2"), (12, "12")])
def test_suffix(n, expected):
    """Test the plotly axis-name suffix: none for the first axis, the number otherwise."""
    assert h._suffix(n) == expected


def test_as_of_takes_the_latest_observation_within_max_age():
    """Test that each reference time gets the latest source time at or before it, if recent enough."""
    src = pd.DatetimeIndex(["2018-01-01", "2018-01-03"])
    ref = pd.DatetimeIndex(["2017-12-31", "2018-01-01", "2018-01-02", "2018-01-03", "2018-01-10"])
    pos, age = h._as_of(ref, src, pd.Timedelta("2D"))
    np.testing.assert_array_equal(pos, [-1, 0, 0, 1, -1])  # before the first, exact, 1 day old, exact, too old
    assert pd.isna(age[0])
    assert pd.Timedelta(age[2]) == pd.Timedelta("1D")
    assert pd.Timedelta(age[4]) == pd.Timedelta("7D")


@pytest.fixture
def make_product():
    """Return a function making a product from its times and which of them hold data."""

    def make(times: list[str], valid: list[bool]) -> h._Product:
        da = xr.DataArray(np.zeros((len(times), 1, 1)), dims=("time", "y", "x"))
        return h._Product(
            da=da, times=pd.DatetimeIndex(times), mean=np.zeros(len(times)), valid=np.array(valid, dtype=bool), k=1
        )

    return make


def test_reference_times_are_the_sorted_unique_valid_times(make_product):
    """Test that the slider times are the times holding data in any product, without duplicates."""
    products = {
        "ssmis": make_product(["2018-01-03", "2018-01-01"], [True, True]),
        "ascat": make_product(["2018-01-01", "2018-01-02"], [True, False]),
    }
    assert list(h._reference_times(products, max_steps=10)) == list(pd.DatetimeIndex(["2018-01-01", "2018-01-03"]))


def test_reference_times_raises_when_no_product_has_data(make_product):
    """Test that a clear error is raised instead of an IndexError later on when nothing holds data."""
    products = {"ssmis": make_product(["2018-01-01"], [False]), "ascat": make_product([], [])}
    with pytest.raises(ValueError, match="No SSMIS, ASCAT or UMelt data"):
        h._reference_times(products, max_steps=10)


def test_break_gaps_inserts_nan_in_long_holes_only():
    """Test that a NaN is inserted in the middle of holes longer than `gap`, and nowhere else."""
    times = pd.DatetimeIndex(["2018-01-01", "2018-01-02", "2018-01-12"])
    t, v = h._break_gaps(times, np.array([1.0, 2.0, 3.0]), pd.Timedelta("7D"))
    assert list(t) == list(pd.DatetimeIndex(["2018-01-01", "2018-01-02", "2018-01-07", "2018-01-12"]))
    np.testing.assert_array_equal(v, [1.0, 2.0, np.nan, 3.0])


@pytest.mark.parametrize(
    ("fixed_range", "frames", "expected"),
    [
        ((0.0, 1.0), np.array([[5.0, 7.0]]), (0.0, 1.0)),  # a fixed range wins over the data
        (None, np.array([[np.nan, 2.0], [5.0, np.nan]]), (2.0, 5.0)),  # finite min/max of the data
        (None, np.full((2, 2), np.nan), (0.0, 1.0)),  # never observed
        (None, np.full((2, 2), 3.0), (3.0, 3.0 + 1e-6)),  # constant data: limits must still differ
    ],
    ids=["fixed", "data", "all-nan", "constant"],
)
def test_color_limits(fixed_range, frames, expected):
    """Test the colour limits of a product's maps."""
    assert h._color_limits({"fixed_range": fixed_range}, frames) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("values", "expected"),
    [(np.array([0.0, 10.0]), (-0.8, 10.8)), (np.array([]), (-0.08, 1.08)), (np.array([2.0]), (1.92, 2.08))],
    ids=["padded", "empty", "single-value"],
)
def test_series_range(values, expected):
    """Test that the series range is the data range with 8% head-room (or a default for no data)."""
    assert h._series_range(values) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("coords", "expected"),
    [(np.array([0.0, 2.0, 4.0]), [-1.0, 5.0]), (np.array([4.0, 2.0, 0.0]), [-1.0, 5.0]), (np.array([3.0]), [2.5, 3.5])],
    ids=["ascending", "descending", "single-cell"],
)
def test_padded_range(coords, expected):
    """Test that the axis range runs from the outer cell edges (half a cell beyond the outermost centres)."""
    assert h._padded_range(coords) == pytest.approx(expected)


def test_plot_antarctica_panel_shows_and_closes_the_map(monkeypatch):
    """Test that the map is shown with its title, size and colour limits, and closed afterwards."""
    from cartopy.mpl.geoaxes import GeoAxes

    monkeypatch.setattr(GeoAxes, "coastlines", lambda self, *a, **k: None)  # avoid downloading Natural Earth
    shown = []
    monkeypatch.setattr(h.plt, "show", lambda: shown.append(h.plt.gcf()))
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3), dims=("y", "x"), coords={"y": [-1.0e6, -1.1e6], "x": [0.0, 1.0e5, 2.0e5]}
    )

    h.plot_antarctica_panel(data, "SSMIS", "RdPu", figsize=(5, 5))

    (fig,) = shown
    ax = fig.axes[0]
    assert ax.get_title() == "SSMIS"
    assert tuple(fig.get_size_inches()) == (5, 5)
    mesh = ax.collections[0]
    assert (mesh.norm.vmin, mesh.norm.vmax) == (0.0, 5.0)
    assert mesh.get_cmap().name == "RdPu"
    assert not h.plt.fignum_exists(fig.number)
