"""Tests for the cell spinner and busy indicator helpers."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import ipywidgets as widgets
import pytest

from dtc_is_notebook_helpers import cell_spinner


@pytest.fixture(autouse=True)
def reset_cell_spinner():
    cell_spinner.disable_cell_spinner()
    yield
    cell_spinner.disable_cell_spinner()


def test_spinner_html_escapes_message() -> None:
    result = cell_spinner.spinner_html("<script>alert('x')</script>")

    assert "<script>" not in result
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in result
    assert 'class="nb-spinner__ring"' in result
    assert "color:var(--jp-ui-font-color1,CanvasText)" in result


def test_busy_indicator_message_uses_theme_aware_color() -> None:
    indicator = cell_spinner.BusyIndicator()

    indicator.stop("Ready")

    assert 'class="nb-spinner-message"' in indicator.value
    assert "color:var(--jp-ui-font-color1,CanvasText)" in indicator.value
    assert ">Ready</div>" in indicator.value


def test_output_widget_capturing_is_false_when_ipywidgets_is_not_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(cell_spinner.sys.modules, "ipywidgets", raising=False)

    assert cell_spinner._output_widget_capturing() is False


def test_cell_spinner_bundle_contains_html_and_plain_text() -> None:
    spinner = cell_spinner._CellSpinner(MagicMock(), "Loading data")

    assert spinner._bundle() == {
        "text/html": cell_spinner.spinner_html("Loading data"),
        "text/plain": "Loading data",
    }


def test_set_message_updates_visible_spinner_and_restores_default() -> None:
    spinner = cell_spinner._CellSpinner(MagicMock(), "Default")
    spinner._running = True
    spinner._handle = MagicMock()

    spinner.set_message("Updated")
    spinner._handle.update.assert_called_once_with(spinner._bundle(), raw=True)
    assert spinner.message == "Updated"

    spinner.set_message(None)

    assert spinner.message == "Default"
    assert spinner._handle.update.call_count == 2


def test_set_message_does_not_update_hidden_spinner() -> None:
    spinner = cell_spinner._CellSpinner(MagicMock(), "Default")

    spinner.set_message("Updated")

    assert spinner.message == "Updated"


def test_publish_hook_notifies_on_new_output_but_not_updates() -> None:
    spinner = cell_spinner._CellSpinner(MagicMock())
    original = MagicMock(return_value=None)
    spinner._on_output = MagicMock()
    publish = spinner._wrap_publish(original)

    assert publish({"text/plain": "result"}) is None
    publish({"text/plain": "result"}, update=True)

    assert original.call_count == 2
    spinner._on_output.assert_called_once_with()


@pytest.mark.parametrize("wait", [False, True])
def test_clear_output_hook_schedules_or_defers_spinner(wait: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cell_spinner, "_output_widget_capturing", lambda: False)
    spinner = cell_spinner._CellSpinner(MagicMock())
    spinner._running = True
    spinner._handle = MagicMock()
    spinner._schedule_move = MagicMock()
    original = MagicMock()

    spinner._wrap_clear_output(original)(wait)

    original.assert_called_once_with(wait)
    if wait:
        assert spinner._clear_pending is True
        assert spinner._handle is not None
        spinner._schedule_move.assert_not_called()
    else:
        assert spinner._handle is None
        spinner._schedule_move.assert_called_once_with(cell_spinner.MOVE_DEBOUNCE_S)


def test_clear_output_wait_before_initial_spinner_does_not_prevent_display(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cell_spinner, "_output_widget_capturing", lambda: False)
    shell = MagicMock()
    spinner = cell_spinner._CellSpinner(shell, "Loading")
    spinner._display_pub = shell.display_pub
    spinner._running = True
    spinner._wrap_clear_output(MagicMock())(wait=True)

    display = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(cell_spinner, "display", display)
    spinner._move()

    assert spinner._clear_pending is False
    display.assert_called_once_with(spinner._bundle(), raw=True, display_id=True)


def test_stale_timer_does_not_move_spinner_or_clear_current_timer() -> None:
    spinner = cell_spinner._CellSpinner(MagicMock())
    stale_timer = MagicMock()
    current_timer = MagicMock()
    spinner._timer = current_timer
    spinner._move = MagicMock()

    spinner._move_if_current(stale_timer)

    spinner._move.assert_not_called()
    assert spinner._timer is current_timer


def test_move_displays_spinner_in_current_output(monkeypatch: pytest.MonkeyPatch) -> None:
    shell = MagicMock()
    spinner = cell_spinner._CellSpinner(shell, "Loading")
    spinner._display_pub = shell.display_pub
    spinner._running = True
    handle = MagicMock()
    display = MagicMock(return_value=handle)
    monkeypatch.setattr(cell_spinner, "display", display)
    monkeypatch.setattr(cell_spinner, "_output_widget_capturing", lambda: False)

    spinner._move()

    display.assert_called_once_with(spinner._bundle(), raw=True, display_id=True)
    assert spinner._handle is handle
    assert spinner._local.active is False


def test_hide_clears_display_handle() -> None:
    spinner = cell_spinner._CellSpinner(MagicMock())
    handle = MagicMock()
    spinner._handle = handle

    spinner._hide()

    handle.update.assert_called_once_with({"text/html": "", "text/plain": ""}, raw=True)
    assert spinner._handle is None


def test_pre_and_post_run_cell_manage_timer_and_visible_spinner() -> None:
    spinner = cell_spinner._CellSpinner(MagicMock(), "Default")
    spinner._patch_streams = MagicMock()
    spinner._unpatch_streams = MagicMock()
    spinner._schedule_move = MagicMock()
    spinner._handle = MagicMock()
    timer = MagicMock()
    spinner._timer = timer

    spinner._pre_run_cell(MagicMock())

    assert spinner._running is True
    assert spinner._handle is None
    assert spinner.message == "Default"
    spinner._patch_streams.assert_called_once_with()
    spinner._schedule_move.assert_called_once_with(cell_spinner.SHOW_DELAY_S)

    spinner._handle = MagicMock()
    spinner._post_run_cell(None)

    assert spinner._running is False
    assert spinner._timer is None
    timer.cancel.assert_called_once_with()
    spinner._unpatch_streams.assert_called_once_with()


def test_enable_cell_spinner_without_kernel_reports_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cell_spinner, "get_ipython", lambda: None)

    cell_spinner.enable_cell_spinner()

    assert "not running in a Jupyter kernel" in capsys.readouterr().out
    assert cell_spinner._spinner is None


def test_enable_is_idempotent_and_disable_uninstalls(monkeypatch: pytest.MonkeyPatch) -> None:
    shell = MagicMock()
    shell.kernel = object()
    monkeypatch.setattr(cell_spinner, "get_ipython", lambda: shell)

    cell_spinner.enable_cell_spinner("First")
    spinner = cell_spinner._spinner
    cell_spinner.enable_cell_spinner("Second")

    assert cell_spinner._spinner is spinner
    assert spinner.default_message == "Second"
    shell.events.register.assert_any_call("pre_run_cell", spinner._pre_run_cell)
    shell.events.register.assert_any_call("post_run_cell", spinner._post_run_cell)

    cell_spinner.disable_cell_spinner()

    assert cell_spinner._spinner is None
    shell.events.unregister.assert_any_call("pre_run_cell", spinner._pre_run_cell)
    shell.events.unregister.assert_any_call("post_run_cell", spinner._post_run_cell)


def test_suppress_cell_spinner_restores_enabled_spinner(monkeypatch: pytest.MonkeyPatch) -> None:
    spinner = MagicMock()
    spinner.default_message = "Custom default"
    monkeypatch.setattr(cell_spinner, "_spinner", spinner)

    def disable() -> None:
        cell_spinner._spinner = None

    def enable(message: str = cell_spinner.DEFAULT_MESSAGE) -> None:
        cell_spinner._spinner = MagicMock(default_message=message)

    monkeypatch.setattr(cell_spinner, "disable_cell_spinner", disable)
    monkeypatch.setattr(cell_spinner, "enable_cell_spinner", enable)
    with cell_spinner.suppress_cell_spinner():
        assert cell_spinner._spinner is None

    assert cell_spinner._spinner is not None
    assert cell_spinner._spinner.default_message == "Custom default"


def test_suppress_cell_spinner_leaves_disabled_spinner_disabled() -> None:
    with cell_spinner.suppress_cell_spinner():
        assert cell_spinner._spinner is None

    assert cell_spinner._spinner is None


def test_set_message_is_noop_when_spinner_is_disabled() -> None:
    cell_spinner.set_message("Ignored")

    assert cell_spinner._spinner is None


def test_load_and_unload_ipython_extension_delegate_to_enable_and_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    enable = MagicMock()
    disable = MagicMock()
    monkeypatch.setattr(cell_spinner, "enable_cell_spinner", enable)
    monkeypatch.setattr(cell_spinner, "disable_cell_spinner", disable)

    cell_spinner.load_ipython_extension(MagicMock())
    cell_spinner.unload_ipython_extension(MagicMock())

    enable.assert_called_once_with()
    disable.assert_called_once_with()


def test_busy_indicator_start_update_and_stop_escape_messages() -> None:
    indicator = cell_spinner.BusyIndicator()

    indicator.start("<working>")
    assert indicator.busy is True
    assert "&lt;working&gt;" in indicator.value

    indicator.update("Still working")
    assert indicator.busy is True
    assert "Still working" in indicator.value

    indicator.stop("<done>")
    assert indicator.busy is False
    assert "&lt;done&gt;" in indicator.value

    indicator.stop()
    assert indicator.value == ""


def test_busy_indicator_update_shows_plain_message_when_idle() -> None:
    indicator = cell_spinner.BusyIndicator()

    indicator.update("Ready")

    assert indicator.busy is False
    assert indicator.value == (
        '<div class="nb-spinner-message" style="color:var(--jp-ui-font-color1,CanvasText)">Ready</div>'
    )


def test_busy_indicator_context_disables_widgets_and_stops_cleanly() -> None:
    indicator = cell_spinner.BusyIndicator()
    button = widgets.Button()

    with indicator("Processing", disable=[button]) as active_indicator:
        assert active_indicator is indicator
        assert indicator.busy is True
        assert button.disabled is True

    assert indicator.busy is False
    assert indicator.value == ""
    assert button.disabled is False


def test_busy_indicator_context_preserves_error_message_and_reraises() -> None:
    indicator = cell_spinner.BusyIndicator()
    button = widgets.Button()

    with pytest.raises(ValueError, match="bad input"):
        with indicator("Processing", disable=[button]):
            raise ValueError("bad input")

    assert indicator.busy is False
    assert "Error: bad input" in indicator.value
    assert button.disabled is False


@pytest.mark.parametrize("exception", [KeyboardInterrupt, asyncio.CancelledError])
def test_busy_indicator_context_stops_on_interruption(exception: type[BaseException]) -> None:
    indicator = cell_spinner.BusyIndicator()
    button = widgets.Button()

    with pytest.raises(exception):
        with indicator("Processing", disable=[button]):
            raise exception()

    assert indicator.busy is False
    assert indicator.value == ""
    assert button.disabled is False


def test_busy_indicator_context_preserves_message_set_by_body() -> None:
    indicator = cell_spinner.BusyIndicator()

    with indicator("Processing"):
        indicator.stop("Validation failed")

    assert indicator.busy is False
    assert "Validation failed" in indicator.value


def test_busy_indicator_track_wraps_function_and_restores_metadata() -> None:
    indicator = cell_spinner.BusyIndicator()
    button = widgets.Button()

    @indicator.track("Loading", disable=[button])
    def callback(value: int) -> int:
        assert indicator.busy is True
        assert button.disabled is True
        return value + 1

    assert callback(2) == 3
    assert callback.__name__ == "callback"
    assert indicator.busy is False
    assert button.disabled is False


def test_busy_indicator_requires_ipywidgets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cell_spinner, "_widgets", None)

    with pytest.raises(ImportError, match="requires ipywidgets"):
        cell_spinner.BusyIndicator()
