"""Tests for the API token authentication widget."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from dtc_query_client import ApiClient, ApiResponse, Configuration
from dtc_query_client.exceptions import ApiException

from dtc_is_notebook_helpers import authentication_widget

VALID_CREDENTIAL = "test-token"


def _client() -> ApiClient:
    return ApiClient(Configuration(host=authentication_widget.API_HOST, access_token=VALID_CREDENTIAL))


def _response(status_code: int, raw_data: bytes = b"") -> ApiResponse:
    return ApiResponse(status_code=status_code, headers=None, data=None, raw_data=raw_data)


def test_check_api_token_returns_api_response(monkeypatch: pytest.MonkeyPatch):
    client = MagicMock()
    expected_response = _response(200)
    api = MagicMock()
    api.get_mass_balance_with_http_info = AsyncMock(return_value=expected_response)
    monkeypatch.setattr(authentication_widget.dtc_query_client, "SeaLevelResponseApi", lambda api_client: api)

    result = asyncio.run(authentication_widget.check_api_token(client))

    assert result is expected_response
    api.get_mass_balance_with_http_info.assert_awaited_once_with("greenland", _request_timeout=5.0)


def test_check_api_token_converts_api_exception_with_response(monkeypatch: pytest.MonkeyPatch):
    api = MagicMock()
    api.get_mass_balance_with_http_info = AsyncMock(
        side_effect=ApiException(status=401, body='{"detail": "Invalid token"}', reason="Unauthorized")
    )
    monkeypatch.setattr(authentication_widget.dtc_query_client, "SeaLevelResponseApi", lambda api_client: api)

    result = asyncio.run(authentication_widget.check_api_token(MagicMock()))

    assert result.status_code == 401
    assert result.raw_data == b'{"detail": "Invalid token"}'


def test_check_api_token_reraises_api_exception_without_response(monkeypatch: pytest.MonkeyPatch):
    api = MagicMock()
    api.get_mass_balance_with_http_info = AsyncMock(side_effect=ApiException(status=None, reason="network error"))
    monkeypatch.setattr(authentication_widget.dtc_query_client, "SeaLevelResponseApi", lambda api_client: api)

    with pytest.raises(ApiException, match="network error"):
        asyncio.run(authentication_widget.check_api_token(MagicMock()))


@pytest.mark.parametrize(
    ("raw_data", "status_code", "expected"),
    [
        (b'{"detail": "Invalid token"}', 401, "Invalid token"),
        (b"Unauthorized", 401, "Unauthorized"),
        (b"", 503, "HTTP 503"),
        (b"\xff", 400, "\ufffd"),
        (b'["not", "an", "object"]', 400, '["not", "an", "object"]'),
    ],
)
def test_response_message(raw_data: bytes, status_code: int, expected: str):
    assert authentication_widget.response_message(_response(status_code, raw_data)) == expected


def test_submit_without_token_shows_validation_message(monkeypatch: pytest.MonkeyPatch):
    widget = authentication_widget.ApiTokenWidget()
    monkeypatch.setattr(authentication_widget, "ApiClient", MagicMock())

    widget.submit()

    assert widget.status.busy is False
    assert "Please paste an API token before submitting." in widget.status.value
    authentication_widget.ApiClient.assert_not_called()


def test_submit_checks_token_and_authenticates(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        authentication_widget,
        "check_api_token",
        AsyncMock(return_value=_response(200)),
    )
    monkeypatch.delenv(authentication_widget.DTC_API_TOKEN_ENV_NAME, raising=False)
    widget = authentication_widget.ApiTokenWidget()
    widget.token_input.value = f"  {VALID_CREDENTIAL}  "

    widget.submit()

    assert widget.is_authenticated is True
    assert widget.token == VALID_CREDENTIAL
    assert isinstance(widget.dtc_client, ApiClient)
    assert widget.dtc_client.configuration.access_token == VALID_CREDENTIAL
    assert widget._authenticated.is_set()
    assert authentication_widget.os.environ[authentication_widget.DTC_API_TOKEN_ENV_NAME] == VALID_CREDENTIAL
    assert widget.status.busy is False
    assert "API token is valid" in widget.status.value


def test_submit_schedules_check_on_running_event_loop(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        authentication_widget,
        "check_api_token",
        AsyncMock(return_value=_response(200)),
    )
    widget = authentication_widget.ApiTokenWidget()
    widget.token_input.value = VALID_CREDENTIAL

    async def submit_and_wait() -> None:
        widget.submit()
        assert widget._check_task is not None
        await widget._check_task

    asyncio.run(submit_and_wait())

    assert widget.is_authenticated is True
    assert widget.token == VALID_CREDENTIAL


def test_submit_closes_previous_client_before_checking(monkeypatch: pytest.MonkeyPatch):
    previous_client = _client()
    previous_close = AsyncMock()
    monkeypatch.setattr(previous_client, "close", previous_close)
    monkeypatch.setattr(
        authentication_widget,
        "check_api_token",
        AsyncMock(return_value=_response(200)),
    )
    widget = authentication_widget.ApiTokenWidget()
    widget.dtc_client = previous_client
    widget.token_input.value = VALID_CREDENTIAL

    widget.submit()

    previous_close.assert_awaited_once()
    assert isinstance(widget.dtc_client, ApiClient)
    assert widget.dtc_client.configuration.access_token == VALID_CREDENTIAL


def test_invalid_token_discards_client_and_shows_response(monkeypatch: pytest.MonkeyPatch):
    client = _client()
    close = AsyncMock()
    monkeypatch.setattr(client, "close", close)
    monkeypatch.setattr(
        authentication_widget,
        "check_api_token",
        AsyncMock(return_value=_response(401, b'{"detail": "Invalid token"}')),
    )
    widget = authentication_widget.ApiTokenWidget()
    widget.dtc_client = client

    asyncio.run(widget._check_token("bad-token", None))

    assert widget.is_authenticated is False
    assert widget.dtc_client is None
    assert widget._authenticated.is_set() is False
    close.assert_awaited_once()
    assert "401: Invalid token" in widget.status.value


def test_timeout_discards_client_and_shows_timeout(monkeypatch: pytest.MonkeyPatch):
    client = _client()
    close = AsyncMock()
    monkeypatch.setattr(client, "close", close)
    monkeypatch.setattr(
        authentication_widget,
        "check_api_token",
        AsyncMock(side_effect=asyncio.TimeoutError),
    )
    widget = authentication_widget.ApiTokenWidget()
    widget.dtc_client = client

    asyncio.run(widget._check_token("token", None))

    assert widget.dtc_client is None
    close.assert_awaited_once()
    assert "The API did not respond within 5 s" in widget.status.value


def test_request_api_token_displays_widget_and_checks_environment_token(monkeypatch: pytest.MonkeyPatch):
    display = MagicMock()
    monkeypatch.setattr(authentication_widget, "display", display)
    monkeypatch.setattr(
        authentication_widget,
        "check_api_token",
        AsyncMock(return_value=_response(200)),
    )
    monkeypatch.setenv(authentication_widget.DTC_API_TOKEN_ENV_NAME, f"  {VALID_CREDENTIAL}  ")

    widget = authentication_widget.request_api_token()

    display.assert_called_once_with(widget)
    assert widget.token_input.value == VALID_CREDENTIAL
    assert widget.is_authenticated is True
    assert widget.token == VALID_CREDENTIAL


def test_request_api_token_does_not_submit_without_environment_token(monkeypatch: pytest.MonkeyPatch):
    display = MagicMock()
    monkeypatch.setattr(authentication_widget, "display", display)
    monkeypatch.delenv(authentication_widget.DTC_API_TOKEN_ENV_NAME, raising=False)
    check_api_token = AsyncMock()
    monkeypatch.setattr(authentication_widget, "check_api_token", check_api_token)

    widget = authentication_widget.request_api_token()

    display.assert_called_once_with(widget)
    assert widget.token_input.value == ""
    assert widget.is_authenticated is False
    check_api_token.assert_not_awaited()


def test_wait_until_authenticated_returns_when_already_authenticated(monkeypatch: pytest.MonkeyPatch):
    widget = authentication_widget.ApiTokenWidget()
    widget._authenticated.set()
    get_ipython = MagicMock(side_effect=AssertionError("IPython should not be queried"))
    monkeypatch.setattr(authentication_widget, "get_ipython", get_ipython)

    asyncio.run(widget.wait_until_authenticated())

    get_ipython.assert_not_called()


def test_wait_until_authenticated_awaits_event_without_kernel(monkeypatch: pytest.MonkeyPatch):
    widget = authentication_widget.ApiTokenWidget()
    monkeypatch.setattr(authentication_widget, "get_ipython", lambda: None)

    async def authenticate_after_wait_starts() -> None:
        waiting = asyncio.create_task(widget.wait_until_authenticated())
        await asyncio.sleep(0)
        widget._authenticated.set()
        await waiting

    asyncio.run(authenticate_after_wait_starts())
