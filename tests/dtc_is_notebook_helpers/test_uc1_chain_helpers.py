"""Tests for the Use Case 1 module runners.

The GRF and FTS modules themselves need R, INLA and the `use_case_1` package,
none of which is available in CI. What is tested here is everything around
them: the clipping, the snapshot bookkeeping, the caches, and that a missing
module is reported rather than crashed on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dtc_is_notebook_helpers import uc1_chain_helpers as chain

NODE_SPACING = 1000.0


@pytest.fixture
def vertices() -> np.ndarray:
    """Build a 5 x 5 square of mesh nodes, 1 km apart."""
    grid_x, grid_y = np.meshgrid(np.arange(5) * NODE_SPACING, np.arange(5) * NODE_SPACING)
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])


def _fitted(values: np.ndarray, sd: np.ndarray | None = None) -> xr.Dataset:
    """Build a GRF-shaped result, with or without a posterior spread."""
    data = {"v": (("point_id",), values)}
    if sd is not None:
        data["v_sd"] = (("point_id",), sd)
    return xr.Dataset(data)


# --------------------------------------------------------------------------
# Environment reporting
# --------------------------------------------------------------------------

def test_check_environment_answers_for_every_module():
    status = chain.check_environment()
    assert set(status) >= {"grf", "fts", "xarray"}
    assert all(isinstance(status[key], bool) for key in ("grf", "fts", "xarray"))


def test_describe_environment_says_the_chain_cannot_run_when_it_cannot():
    text = chain.describe_environment(
        {"grf": False, "grf_error": "no use_case_1", "fts": False, "fts_error": "no rpy2"})
    assert "NOT available" in text
    assert "cannot run live" in text
    assert "no use_case_1" in text


def test_describe_environment_says_the_chain_will_run_when_it_will():
    text = chain.describe_environment({"grf": True, "fts": True})
    assert "will run live" in text
    assert "NOT available" not in text


# --------------------------------------------------------------------------
# Clipping to the mesh
# --------------------------------------------------------------------------

def test_a_point_on_a_node_is_kept(vertices):
    assert chain.mesh_mask(vertices[:1], vertices).all()


def test_a_distant_point_is_dropped(vertices):
    far = np.array([[100 * NODE_SPACING, 0.0]])
    assert not chain.mesh_mask(far, vertices).any()


def test_a_point_just_outside_the_footprint_is_dropped(vertices):
    # Three node spacings beyond the corner: inside no triangle, and further
    # than the margin allows.
    outside = np.array([[-3 * NODE_SPACING, -3 * NODE_SPACING]])
    assert not chain.mesh_mask(outside, vertices).any()


def test_a_looser_margin_keeps_more(vertices):
    # Punch the centre node out, so the centre of the mesh is inside the hull
    # but one full node spacing from the nearest node. That is the gap the
    # margin controls, and it is what an irregular catchment boundary looks
    # like from the inside.
    holed = vertices[~np.all(vertices == 2 * NODE_SPACING, axis=1)]
    centre = np.array([[2 * NODE_SPACING, 2 * NODE_SPACING]])

    assert not chain.mesh_mask(centre, holed, margin_factor=0.9).any()
    assert chain.mesh_mask(centre, holed, margin_factor=1.1).all()


# --------------------------------------------------------------------------
# Snapshot bookkeeping
# --------------------------------------------------------------------------

def test_snapshot_tags_come_back_in_season_order(tmp_path):
    for tag in ("2015_Jun", "2014_Sep", "2014_May", "2015_May"):
        (tmp_path / f"{tag}_logru.csv").write_text("1.0\n")

    assert chain.snapshot_tags(tmp_path, "logru") == ["2014_May", "2014_Sep", "2015_May", "2015_Jun"]


def test_an_explicit_listing_wins_over_the_directory(tmp_path):
    (tmp_path / "2014_May_logru.csv").write_text("1.0\n")
    (tmp_path / "monthly_snapshots.csv").write_text("2014_Jul\n2014_Aug\n")

    assert chain.snapshot_tags(tmp_path, "logru") == ["2014_Jul", "2014_Aug"]


def test_seasons_and_years_describe_the_same_tags():
    tags = ["2014_Jun", "2014_Jul", "2015_Jun"]
    assert chain.seasons_from_tags(tags) == [2014, 2015]
    assert chain.year_of_tags(tags) == [2014, 2014, 2015]


def test_matrix_to_frame_dates_monthly_snapshots():
    matrix = pd.DataFrame(np.zeros((2, 3)), index=["2014_Jun", "2014_Jul"])
    frame = chain.matrix_to_frame(matrix)
    assert list(frame.index) == [pd.Timestamp("2014-06-01"), pd.Timestamp("2014-07-01")]
    # log10 is the default, so zeros come back as ones.
    assert (frame.to_numpy() == 1.0).all()


def test_matrix_to_frame_can_be_told_the_values_are_not_logs():
    matrix = pd.DataFrame(np.zeros((1, 2)), index=["2014_Jun"])
    assert (chain.matrix_to_frame(matrix, log10=False).to_numpy() == 0.0).all()


# --------------------------------------------------------------------------
# Reading the posterior out of a fitted field
# --------------------------------------------------------------------------

def test_posterior_sd_is_found_when_the_module_reports_one():
    fitted = _fitted(np.array([1.0, 2.0]), sd=np.array([0.1, 0.2]))
    np.testing.assert_allclose(chain.posterior_sd(fitted), [0.1, 0.2])


def test_posterior_sd_is_none_when_there_is_nothing_to_find():
    assert chain.posterior_sd(_fitted(np.array([1.0, 2.0]))) is None


def test_posterior_mean_sd_is_none_when_the_module_is_silent():
    assert chain.posterior_mean_sd(_fitted(np.array([1.0, 2.0]))) is None


# --------------------------------------------------------------------------
# Caches. These are the paths that run without the modules installed.
# --------------------------------------------------------------------------

def test_a_matching_cache_is_used(tmp_path):
    tags = ["2014_May", "2014_Jun"]
    cache = tmp_path / "grf.csv"
    pd.DataFrame(np.ones((2, 3)), index=pd.Index(tags, name="snapshot")).to_csv(cache)

    said = []
    matrix = chain.run_grf(tmp_path, "logru", cache=cache, tags=tags, progress=said.append)

    assert list(matrix.index) == tags
    assert "from cache" in " ".join(said)


def test_a_cache_for_other_snapshots_is_not_used(tmp_path):
    cache = tmp_path / "grf.csv"
    pd.DataFrame(np.ones((1, 3)), index=pd.Index(["2013_May"], name="snapshot")).to_csv(cache)

    said = []
    with pytest.raises(chain.ModulesUnavailableError):
        chain.run_grf(tmp_path, "logru", cache=cache, tags=["2014_May"], progress=said.append)

    assert "does not match" in " ".join(said)


def test_a_missing_grf_module_is_reported_not_crashed_on(tmp_path):
    with pytest.raises(chain.ModulesUnavailableError, match="cannot import the GRF module"):
        chain.run_grf(tmp_path, "logru", tags=["2014_May"], progress=lambda _: None)


def test_a_missing_fts_module_is_reported_not_crashed_on():
    matrix = pd.DataFrame(np.ones((5, 3)), index=[f"2014_{m}" for m in
                                                  ("May", "Jun", "Jul", "Aug", "Sep")])
    with pytest.raises(chain.ModulesUnavailableError, match="cannot import rpy2"):
        chain.run_fts(matrix, "model.coeff.R", progress=lambda _: None)


def test_a_matching_fts_cache_is_used(tmp_path):
    cache = tmp_path / "fts.csv"
    pd.DataFrame({"f1": [0.1], "f2": [0.2]}, index=[2014]).to_csv(cache)
    matrix = pd.DataFrame(np.ones((5, 3)), index=[f"2014_{m}" for m in
                                                 ("May", "Jun", "Jul", "Aug", "Sep")])

    said = []
    coefficients = chain.run_fts(matrix, "model.coeff.R", cache=cache, progress=said.append)

    assert len(coefficients) == 1
    assert "from cache" in " ".join(said)


def test_the_posterior_spread_is_cached_with_its_spatial_mean(tmp_path):
    out = tmp_path / "sd.csv"
    tags = ["2014_May", "2014_Jun"]

    said = []
    chain._write_posterior_spread(out, [np.full(3, 0.1), np.full(3, 0.2)],
                                  [0.05, 0.06], tags, said.append)

    frame = pd.read_csv(out, index_col=0)
    assert list(frame.index) == tags
    assert "__mean_sd__" in frame.columns
    assert "cached" in " ".join(said)


def test_nothing_is_cached_when_the_module_gave_no_spread(tmp_path):
    out = tmp_path / "sd.csv"

    said = []
    chain._write_posterior_spread(out, [], [], ["2014_May"], said.append)

    assert not out.exists()
    assert "no posterior spread" in " ".join(said)


def test_a_snapshot_keeps_only_the_points_the_mesh_can_take(tmp_path, vertices):
    # Five points: three usable, one far outside the mesh, one not a number.
    xy = np.array([[0.0, 0.0], [1000.0, 1000.0], [2000.0, 2000.0],
                   [500_000.0, 0.0], [3000.0, 3000.0]])
    np.savetxt(tmp_path / "2014_May_centroid.csv", xy, delimiter=",")
    np.savetxt(tmp_path / "2014_May_logru.csv", np.array([1.0, 2.0, 3.0, 4.0, np.nan]))

    dataset, used, total = chain._snapshot_dataset(
        tmp_path, "2014_May", "logru", "centroid", vertices, 1.5)

    assert total == 5
    assert used == 3
    assert dataset["v"].shape == (1, 3)
    np.testing.assert_allclose(dataset["x"].values, [0.0, 1000.0, 2000.0])


def test_a_snapshot_whose_files_disagree_is_refused(tmp_path, vertices):
    np.savetxt(tmp_path / "2014_May_centroid.csv", np.zeros((3, 2)), delimiter=",")
    np.savetxt(tmp_path / "2014_May_logru.csv", np.zeros(2))

    with pytest.raises(ValueError, match="coordinates against"):
        chain._snapshot_dataset(tmp_path, "2014_May", "logru", "centroid", vertices, 1.5)


def test_a_snapshot_that_misses_the_mesh_is_refused(tmp_path, vertices):
    far = np.full((4, 2), 500_000.0)
    np.savetxt(tmp_path / "2014_May_centroid.csv", far, delimiter=",")
    np.savetxt(tmp_path / "2014_May_logru.csv", np.ones(4))

    with pytest.raises(ValueError, match="usable points inside the mesh"):
        chain._snapshot_dataset(tmp_path, "2014_May", "logru", "centroid", vertices, 1.5)
