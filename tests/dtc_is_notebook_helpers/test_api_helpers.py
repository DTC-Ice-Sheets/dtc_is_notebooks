"""Tests for notebooks.api_helpers module."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from unittest.mock import MagicMock, patch

import pytest

from dtc_is_notebook_helpers import api_helpers


@pytest.fixture()
def mock_file_content() -> bytes:
    return b"year,mass_balance\n2000,100\n2001,110\n"


def test_get_precomputed_mass_balance_dataset_url_valid():
    with (
        patch.object(api_helpers.os, "environ", {"DTC_API_PASSWORD": "fake_password"}),
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
                timeout=600,
                headers=api_helpers.get_auth_headers(),
            )


def test_get_precomputed_mass_balance_dataset_url_invalid():
    with pytest.raises(ValueError, match="Unknown dataset: foo"):
        api_helpers.get_precomputed_mass_balance_dataset_url("foo")


def test_upload_mass_balance_csv(mock_file_content: bytes):
    with (
        patch.object(api_helpers.os, "environ", {"DTC_API_PASSWORD": "fake_password"}),
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
        patch.object(api_helpers.os, "environ", {"DTC_API_PASSWORD": "fake_password"}),
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
            {"DTC_API_PASSWORD": "fake_password"},
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
        patch.object(api_helpers.os, "environ", {"DTC_API_PASSWORD": "fake_password"}),
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

        with pytest.raises(RuntimeError, match="SELREM job fake_job_id failed or was cancelled"):
            api_helpers.run_selrem_module(
                vmb_url="s3://bucket/path/to/vmb",
                scale=1.0,
                start_year=2000,
                end_year=2001,
            )
