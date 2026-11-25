"""Tests for notebooks.api_helpers module."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from dtc_query_client.api_client import ApiClient
from shapely.geometry import shape

from dtc_is_notebook_helpers import api_helpers


@pytest.fixture()
def mock_file_content() -> bytes:
    return b"year,mass_balance\n2000,100\n2001,110\n"


def test_get_precomputed_mass_balance_dataset_url_valid():
    with (
        patch.object(api_helpers.os, "environ", {"DTC_API_TOKEN": "fake_token"}),
        patch.object(api_helpers.requests, "get") as mock_get,
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {"url": "https://fake-url.com/precomputed"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        for dataset in ["greenland", "antarctic", "greenland_and_antarctic"]:
            result = api_helpers.get_precomputed_mass_balance_dataset_url(dataset)
            assert result == "https://fake-url.com/precomputed"
            mock_get.assert_called_with(
                f"https://query.dtc-ice-sheets.org/mass-balance/{dataset}",
                timeout=10,
                headers=api_helpers.get_auth_headers(),
            )


def test_get_precomputed_mass_balance_dataset_url_invalid():
    with pytest.raises(ValueError, match="Unknown dataset: foo"):
        api_helpers.get_precomputed_mass_balance_dataset_url("foo")


def test_upload_mass_balance_csv(mock_file_content: bytes):
    with (
        patch.object(api_helpers.os, "environ", {"DTC_API_TOKEN": "fake_token"}),
        patch.object(api_helpers.requests, "post") as mock_post,
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {"url": "https://fake-url.com/dataset"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = api_helpers.upload_mass_balance_csv({"content": mock_file_content, "name": "AIS.csv"})
        assert result == "https://fake-url.com/dataset"
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["files"]["file"][0] == "AIS.csv"
        assert kwargs["headers"] == api_helpers.get_auth_headers()


def test_upload_mass_balance_csv_no_name(mock_file_content: bytes):
    with (
        patch.object(api_helpers.os, "environ", {"DTC_API_TOKEN": "fake_token"}),
        patch.object(api_helpers.requests, "post") as mock_post,
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {"url": "https://fake-url.com/dataset"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = api_helpers.upload_mass_balance_csv({"content": mock_file_content})
        assert result == "https://fake-url.com/dataset"
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["files"]["file"][0] == "custom_data.csv"
        assert kwargs["headers"] == api_helpers.get_auth_headers()


def test_upload_mass_balance_csv_invalid_type():
    with pytest.raises(ValueError, match="file_upload_value must contain 'content' key with file data"):
        api_helpers.upload_mass_balance_csv({"not_content": b"invalid_data"})


@pytest.mark.parametrize("analysis_mode", ["global", "annual"])
def test_run_selrem_module_success(analysis_mode):
    with (
        patch.object(
            api_helpers.os,
            "environ",
            {"DTC_API_TOKEN": "fake_token"},
        ),
        patch.object(api_helpers.requests, "post") as mock_post,
        patch.object(api_helpers.requests, "get") as mock_get,
        patch.object(api_helpers.xr, "open_dataset") as mock_open_dataset,
        patch.object(api_helpers.time, "sleep"),
    ):
        # Mock POST response (job submission)
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = {"job_id": "fake_job_id"}
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response

        # Mock GET response (job polling)
        # First call: job running, Second call: job succeeded
        mock_get_response_running = MagicMock()
        mock_get_response_running.json.return_value = {"status": "Running"}
        mock_get_response_running.raise_for_status.return_value = None

        mock_get_response_succeeded = MagicMock()
        mock_get_response_succeeded.json.return_value = {
            "status": "Succeeded",
            "outputs": {"main": {"output_path": "s3://bucket/path/to/zarr"}},
        }
        mock_get_response_succeeded.raise_for_status.return_value = None

        mock_get.side_effect = [mock_get_response_running, mock_get_response_succeeded]

        # Mock xarray.open_dataset
        mock_ds = MagicMock()
        mock_open_dataset.return_value = mock_ds

        result = api_helpers.run_selrem_module(
            vmb_url="s3://bucket/path/to/vmb",
            scale=1.23,
            start_year=2000,
            end_year=2005,
            analysis_mode=analysis_mode,
        )
        assert result == mock_ds
        mock_post.assert_called_once()
        assert mock_get.call_count == 2
        mock_open_dataset.assert_called_once_with("s3://bucket/path/to/zarr", engine="zarr")


def test_run_selrem_module_job_failed():
    with (
        patch.object(api_helpers.os, "environ", {"DTC_API_TOKEN": "fake_token"}),
        patch.object(api_helpers.requests, "post") as mock_post,
        patch.object(api_helpers.requests, "get") as mock_get,
        patch.object(api_helpers.time, "sleep"),
    ):
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = {"job_id": "fake_job_id"}
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response

        mock_get_response_failed = MagicMock()
        mock_get_response_failed.json.return_value = {"status": "Failed"}
        mock_get_response_failed.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response_failed

        with pytest.raises(RuntimeError, match="Job fake_job_id failed or was cancelled"):
            api_helpers.run_selrem_module(
                vmb_url="s3://bucket/path/to/vmb",
                scale=1.0,
                start_year=2000,
                end_year=2001,
            )


@pytest.fixture
def mock_api_client():
    return MagicMock(spec=ApiClient)


# --------------------------------------------------------------------------
# Submitting a job
# --------------------------------------------------------------------------

@pytest.fixture
def fake_hydropower_api(monkeypatch):
    """Stand in for the HydropowerApi, recording what it was built with and sent."""
    seen = {}

    class FakeApi:
        def __init__(self, api_client):
            seen["constructed_with"] = api_client

        async def _submit(self, request):
            seen["request"] = request
            return SimpleNamespace(job_id="job-1")

        submit_grf_job = submit_fts_job = _submit

    monkeypatch.setattr(api_helpers.dtc_query_client.api, "HydropowerApi", FakeApi)
    return seen


@pytest.mark.parametrize(
    ("submit", "api_class", "method", "output_key"),
    [
        (lambda c: api_helpers.run_grf_job(c, "UC1_WATSON_RUNOFF_INPUT_MAR", progress=lambda _: None),
         "HydropowerApi", "submit_grf_job", api_helpers.GRF_OUTPUT_KEY),
        (lambda c: api_helpers.run_fts_job(c, "https://example.org/grf.zarr", progress=lambda _: None),
         "HydropowerApi", "submit_fts_job", api_helpers.FTS_OUTPUT_KEY),
        (lambda c: api_helpers.download_dataset(c, "UC1_WATSON_DISCHARGE_GAUGE", "2006-01-01", "2026-01-01",
                                        progress=lambda _: None),
         "GenericApi", "download_data", api_helpers.DOWNLOAD_OUTPUT_KEY),
    ],
    ids=["grf", "fts", "download"],
)
def test_run_job_submits_and_waits_through_the_given_client(monkeypatch, mock_api_client, submit, api_class,
                                                             method, output_key):
    seen = {}

    class FakeApi:
        def __init__(self, api_client):
            seen["constructed_with"] = api_client

        async def _submit(self, request):
            seen["request"] = request
            return SimpleNamespace(job_id="job-1")

    setattr(FakeApi, method, FakeApi._submit)

    async def fake_wait(api_client, job_id, **kwargs):  # noqa: ANN003
        seen["awaited_with"], seen["job_id"], seen["wait_kwargs"] = api_client, job_id, kwargs
        return SimpleNamespace(outputs={output_key: {"output_path": "out.zarr"}})

    monkeypatch.setattr(api_helpers.dtc_query_client.api, api_class, FakeApi)
    monkeypatch.setattr(api_helpers, "wait_for_job", fake_wait)

    assert asyncio.run(submit(mock_api_client)) == "out.zarr"
    assert seen["constructed_with"] is mock_api_client
    assert seen["awaited_with"] is mock_api_client
    assert seen["job_id"] == "job-1"
    assert seen["wait_kwargs"] == {"max_job_missing_delay": api_helpers.MAX_JOB_MISSING_DELAY}


# --------------------------------------------------------------------------
# Waiting on a job
# --------------------------------------------------------------------------

def test_run_grf_job_returns_the_output_path_of_a_succeeded_job(monkeypatch, mock_api_client, fake_hydropower_api):
    finished = SimpleNamespace(outputs={"grf-module": {"output_path": "somewhere.zarr"}})

    async def fake_wait(client, job_id, **kwargs):   # noqa: ANN001, ARG001, ANN003
        return finished

    monkeypatch.setattr(api_helpers, "wait_for_job", fake_wait)
    path = asyncio.run(api_helpers.run_grf_job(mock_api_client, "UC1_WATSON_RUNOFF_INPUT_MAR",
                                       progress=lambda _: None))
    assert path == "somewhere.zarr"


def test_run_grf_job_raises_when_the_job_did_not_succeed(monkeypatch, mock_api_client, fake_hydropower_api):
    async def fake_wait(client, job_id, **kwargs):   # noqa: ANN001, ARG001, ANN003
        raise RuntimeError("Job failed")

    monkeypatch.setattr(api_helpers, "wait_for_job", fake_wait)
    with pytest.raises(RuntimeError, match="Job failed"):
        asyncio.run(api_helpers.run_grf_job(mock_api_client, "UC1_WATSON_RUNOFF_INPUT_MAR",
                                    progress=lambda _: None))


# --------------------------------------------------------------------------
# Fetching a published dataset
# --------------------------------------------------------------------------

def test_the_spatial_filter_is_built_in_a_form_the_endpoint_accepts():
    """
    The polygon field is a one-of wrapper, and handing it a plain dict leaves it unbound.

    That failure is silent until the request is serialised, so it is worth pinning here
    rather than discovering it against the deployed endpoint.
    """
    wrapped = api_helpers.dtc_query_client.models.Epsg4326PolygonJson.from_dict(api_helpers.WHOLE_WORLD)

    assert wrapped.to_dict() == api_helpers.WHOLE_WORLD


_SQUARE = {"type": "Polygon",
           "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]}


@pytest.mark.parametrize(
    ("polygon", "expected"),
    [(None, api_helpers.WHOLE_WORLD), (_SQUARE, _SQUARE), (shape(_SQUARE), _SQUARE)],
    ids=["default", "geojson-dict", "shapely"],
)
def test_download_dataset_accepts_the_polygon_as_geojson_or_shapely(monkeypatch, mock_api_client, polygon,
                                                                    expected):
    seen = {}

    class FakeApi:
        def __init__(self, api_client):  # noqa: ARG002
            pass

        async def download_data(self, request):
            seen["request"] = request
            return SimpleNamespace(job_id="job-1")

    async def fake_wait(client, job_id, **kwargs):   # noqa: ANN001, ARG001, ANN003
        return SimpleNamespace(outputs={api_helpers.DOWNLOAD_OUTPUT_KEY: {"output_path": "out.zarr"}})

    monkeypatch.setattr(api_helpers.dtc_query_client.api, "GenericApi", FakeApi)
    monkeypatch.setattr(api_helpers, "wait_for_job", fake_wait)
    asyncio.run(api_helpers.download_dataset(mock_api_client, "UC1_WATSON_DISCHARGE_GAUGE", "2006-01-01",
                                     "2026-01-01", polygon=polygon, progress=lambda _: None))

    sent = seen["request"].epsg4326_polygon_json.to_dict()
    # shapely hands coordinates back as tuples; compare the numbers, not the containers.
    assert sent["type"] == expected["type"]
    np.testing.assert_allclose(np.asarray(sent["coordinates"]), np.asarray(expected["coordinates"]))
