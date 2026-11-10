"""Tests for the API token authentication widget."""

import asyncio
from contextlib import nullcontext
from unittest.mock import AsyncMock, MagicMock

import pytest
from dtc_query_client import ApiClient, ApiResponse, Configuration
from dtc_query_client.exceptions import ApiException

from dtc_is_notebook_helpers import authentication_widget

VALID_CREDENTIAL = "test-token"


@pytest.fixture(autouse=True)
def not_in_vscode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Show the regular widget, even when the tests run from a VS Code terminal."""
    monkeypatch.setattr(authentication_widget, "running_in_vscode", lambda: False)
    monkeypatch.setattr(authentication_widget, "running_in_colab", lambda: False)


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


def test_api_token_label_uses_theme_aware_text_color() -> None:
    assert "color:var(--jp-ui-font-color1,CanvasText)" in authentication_widget.LABEL_HTML


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
    assert widget.token_input.placeholder == authentication_widget.TOKEN_INPUT_PLACEHOLDER
    widget.token_input.value = f"  {VALID_CREDENTIAL}  "

    widget.submit()

    assert widget.is_authenticated is True
    assert widget.token == VALID_CREDENTIAL
    assert isinstance(widget.dtc_client, ApiClient)
    assert widget.dtc_client.configuration.access_token == VALID_CREDENTIAL
    assert widget._authenticated.is_set()
    assert widget.token_input.placeholder == authentication_widget.AUTHENTICATED_PLACEHOLDER
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
    assert widget.token_input.placeholder == authentication_widget.TOKEN_INPUT_PLACEHOLDER
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


def test_request_api_token_clears_environment_token_and_starts_unauthenticated(monkeypatch: pytest.MonkeyPatch):
    display = MagicMock()
    monkeypatch.setattr(authentication_widget, "display", display)
    check_api_token = AsyncMock(return_value=_response(200))
    monkeypatch.setattr(authentication_widget, "check_api_token", check_api_token)
    monkeypatch.setenv(authentication_widget.DTC_API_TOKEN_ENV_NAME, f"  {VALID_CREDENTIAL}  ")

    widget = authentication_widget.request_api_token()

    display.assert_called_once_with(widget)
    assert widget.token_input.value == ""
    assert widget.is_authenticated is False
    assert widget.token is None
    assert widget.dtc_client is None
    assert widget._authenticated.is_set() is False
    assert authentication_widget.DTC_API_TOKEN_ENV_NAME not in authentication_widget.os.environ
    check_api_token.assert_not_awaited()


def test_request_api_token_requires_new_submission_after_previous_authentication(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(authentication_widget, "display", MagicMock())
    check_api_token = AsyncMock(return_value=_response(200))
    monkeypatch.setattr(authentication_widget, "check_api_token", check_api_token)
    monkeypatch.delenv(authentication_widget.DTC_API_TOKEN_ENV_NAME, raising=False)
    previous_widget = authentication_widget.request_api_token()
    previous_widget.token_input.value = VALID_CREDENTIAL
    previous_widget.submit()
    assert previous_widget.is_authenticated is True
    assert authentication_widget.os.environ[authentication_widget.DTC_API_TOKEN_ENV_NAME] == VALID_CREDENTIAL
    check_api_token.reset_mock()

    widget = authentication_widget.request_api_token()

    assert widget is not previous_widget
    assert widget.token_input.value == ""
    assert widget.token is None
    assert widget.dtc_client is None
    assert widget._authenticated.is_set() is False
    assert authentication_widget.DTC_API_TOKEN_ENV_NAME not in authentication_widget.os.environ
    check_api_token.assert_not_awaited()


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


@pytest.mark.parametrize("status_code", [200, 401])
def test_request_api_token_checks_explicit_token(monkeypatch: pytest.MonkeyPatch, status_code: int):
    display = MagicMock()
    monkeypatch.setattr(authentication_widget, "display", display)
    check_api_token = AsyncMock(return_value=_response(status_code))
    monkeypatch.setattr(authentication_widget, "check_api_token", check_api_token)
    monkeypatch.setenv(authentication_widget.DTC_API_TOKEN_ENV_NAME, "previous-token")

    widget = authentication_widget.request_api_token(token=f"  {VALID_CREDENTIAL}  ")

    display.assert_called_once_with(widget)
    assert widget.token_input.value == ""  # check that the token is cleared from the widget
    assert "value" not in widget.token_input.get_state()
    check_api_token.assert_awaited_once()
    checked_client = check_api_token.call_args.args[0]
    assert checked_client.configuration.access_token == VALID_CREDENTIAL
    if status_code == 200:
        assert widget.is_authenticated is True
        assert widget.token == VALID_CREDENTIAL
        assert authentication_widget.os.environ[authentication_widget.DTC_API_TOKEN_ENV_NAME] == VALID_CREDENTIAL
    else:
        assert widget.is_authenticated is False
        assert widget.dtc_client is None
        assert authentication_widget.DTC_API_TOKEN_ENV_NAME not in authentication_widget.os.environ


@pytest.mark.parametrize("token", [None, "", "   "])
def test_request_api_token_with_empty_explicit_token_starts_fresh(monkeypatch: pytest.MonkeyPatch, token):
    monkeypatch.setattr(authentication_widget, "display", MagicMock())
    check_api_token = AsyncMock()
    monkeypatch.setattr(authentication_widget, "check_api_token", check_api_token)
    monkeypatch.setenv(authentication_widget.DTC_API_TOKEN_ENV_NAME, VALID_CREDENTIAL)

    widget = authentication_widget.request_api_token(token=token)

    assert widget.token_input.value == ""
    assert widget.is_authenticated is False
    assert authentication_widget.DTC_API_TOKEN_ENV_NAME not in authentication_widget.os.environ
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


@pytest.mark.parametrize(
    ("environ", "expected"),
    [({"VSCODE_CWD": "/home"}, True), ({"VSCODE_PID": "1", "HOME": "/home"}, True), ({"HOME": "/home"}, False)],
)
def test_running_in_vscode(monkeypatch: pytest.MonkeyPatch, environ: dict[str, str], expected: bool):
    monkeypatch.undo()
    monkeypatch.setattr(authentication_widget.os, "environ", environ)

    assert authentication_widget.running_in_vscode() is expected


@pytest.mark.parametrize(
    ("modules", "expected"),
    [({"google.colab": object()}, True), ({}, False)],
)
def test_running_in_colab(monkeypatch: pytest.MonkeyPatch, modules: dict[str, object], expected: bool):
    monkeypatch.undo()
    monkeypatch.setattr(authentication_widget.sys, "modules", modules)

    assert authentication_widget.running_in_colab() is expected


@pytest.mark.parametrize(
    ("in_vscode", "in_colab", "label"),
    [
        (True, False, authentication_widget.VSCODE_LABEL_HTML),
        (False, True, authentication_widget.COLAB_LABEL_HTML),
        (True, True, authentication_widget.COLAB_LABEL_HTML),
    ],
)
def test_native_prompt_widget_hides_token_input(
    monkeypatch: pytest.MonkeyPatch, in_vscode: bool, in_colab: bool, label: str
):
    monkeypatch.setattr(authentication_widget, "running_in_vscode", lambda: in_vscode)
    monkeypatch.setattr(authentication_widget, "running_in_colab", lambda: in_colab)

    widget = authentication_widget.ApiTokenWidget()

    assert widget.uses_native_prompt
    input_row = widget.children[0]
    assert list(input_row.children) == [widget.label]
    assert widget.label.value == label
    assert widget.children[1] is widget.status


@pytest.mark.parametrize(
    ("prompted_tokens", "status_codes", "authenticated"),
    [
        ([VALID_CREDENTIAL], [200], True),
        (["bad-token", VALID_CREDENTIAL], [401, 200], True),
        (["  "], [], False),
        (["bad-token", ""], [401], False),
    ],
)
def test_vscode_wait_prompts_until_authenticated_or_cancelled(
    monkeypatch: pytest.MonkeyPatch, prompted_tokens: list[str], status_codes: list[int], authenticated: bool
):
    monkeypatch.setattr(authentication_widget, "running_in_vscode", lambda: True)
    prompt = MagicMock(side_effect=prompted_tokens)
    monkeypatch.setattr(authentication_widget.getpass, "getpass", prompt)
    check_api_token = AsyncMock(side_effect=[_response(code) for code in status_codes])
    monkeypatch.setattr(authentication_widget, "check_api_token", check_api_token)
    monkeypatch.setattr(authentication_widget.os, "environ", {})
    clear_output = MagicMock()
    monkeypatch.setattr(authentication_widget, "clear_output", clear_output)
    display = MagicMock()
    monkeypatch.setattr(authentication_widget, "display", display)

    widget = authentication_widget.ApiTokenWidget()
    with nullcontext() if authenticated else pytest.raises(KeyboardInterrupt, match="API token entry cancelled"):
        asyncio.run(widget.wait_until_authenticated())

    assert prompt.call_count == len(prompted_tokens)
    assert check_api_token.await_count == len(status_codes)
    assert widget.is_authenticated is authenticated
    assert widget.token == (VALID_CREDENTIAL if authenticated else None)
    # Every answered prompt is cleared away and the widget shown again in its place.
    assert clear_output.call_count == len(prompted_tokens)
    display.assert_called_with(widget)
    assert display.call_count == len(prompted_tokens)
