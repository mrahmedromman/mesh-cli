"""Live `mesh status`: the redraw loop and the key reader, tested without a real terminal."""

import os
import sys
import time
from pathlib import Path

import pytest

from mesh_cli import cli as cli_module
from mesh_cli import runner, watch


def write_state(folder, *statuses):
    services = {
        f"s{i}": {"status": status, "pid": None, "port": 8000 + i, "health": "/", "error": None}
        for i, status in enumerate(statuses)
    }
    runner.write_state(folder, {"controller_pid": 1, "services": services})


def test_the_view_is_redrawn_until_the_stack_stops(tmp_path):
    write_state(tmp_path, "pending", "pending")
    frames = []

    def wait(seconds):
        if len(frames) == 1:
            write_state(tmp_path, "stopping", "pending")
        elif len(frames) == 2:
            runner.clear_state(tmp_path)  # mesh up finished and removed its state file
        return False

    rows, quit_by_user = watch.poll(tmp_path, lambda r: frames.append([x.state for x in r]), wait)
    assert frames == [["pending", "pending"], ["stopping", "pending"]]
    assert quit_by_user is False and [r.state for r in rows] == ["stopping", "pending"]


def test_pressing_q_ends_the_view_and_says_so(tmp_path):
    write_state(tmp_path, "pending")
    rows, quit_by_user = watch.poll(tmp_path, lambda r: None, lambda seconds: True)
    assert quit_by_user is True and len(rows) == 1


def test_a_stack_that_is_already_gone_shows_nothing(tmp_path):
    frames = []
    rows, quit_by_user = watch.poll(tmp_path, frames.append, lambda seconds: False)
    assert frames == [] and rows == [] and quit_by_user is False


def test_after_the_stack_stops_only_failed_services_keep_their_state():
    rows = [
        runner.ServiceRow("a", "running", 5, 1, "HTTP 200"),
        runner.ServiceRow("b", "failed", None, 2, "-", "b exited with status 3"),
    ]
    final = watch.stopped(rows)
    assert [(r.state, r.pid, r.health) for r in final] == [("stopped", None, "-"), ("failed", None, "-")]
    assert final[1].error == "b exited with status 3"


@pytest.fixture
def keys():
    read, write = os.pipe()
    yield read, write
    os.close(read)
    try:
        os.close(write)
    except OSError:
        pass


def test_q_quits_in_either_case(keys):
    read, write = keys
    os.write(write, b"Q")
    assert watch.quit_pressed(read, 1.0) is True
    os.write(write, b"q")
    assert watch.quit_pressed(read, 1.0) is True


def test_other_keys_do_not_quit_and_the_wait_lasts_its_full_time(keys):
    read, write = keys
    os.write(write, b"xyz")
    began = time.monotonic()
    assert watch.quit_pressed(read, 0.2) is False
    assert time.monotonic() - began >= 0.19


def test_a_closed_input_quits_instead_of_spinning(keys):
    read, write = keys
    os.close(write)
    assert watch.quit_pressed(read, 1.0) is True


# --- when `mesh status` goes live ------------------------------------------------------------------------------


def stack_in(folder):
    write_state(folder, "pending")


def test_status_goes_live_only_in_a_real_terminal(tmp_path, monkeypatch, capsys):
    stack_in(tmp_path)
    live = []
    monkeypatch.setattr(watch, "watch", lambda log_dir: live.append(log_dir))
    monkeypatch.setattr(cli_module.ui, "fancy", lambda: True)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    assert cli_module.main(["status", "--log-dir", str(tmp_path)]) == 0
    assert live == [tmp_path]


def test_status_prints_once_when_input_or_output_is_not_a_terminal(tmp_path, monkeypatch, capsys):
    stack_in(tmp_path)
    monkeypatch.setattr(watch, "watch", lambda log_dir: pytest.fail("must not go live"))
    monkeypatch.setattr(cli_module.ui, "fancy", lambda: True)  # a terminal for output, but input is piped
    assert cli_module.main(["status", "--log-dir", str(tmp_path)]) == 0
    monkeypatch.setattr(cli_module.ui, "fancy", lambda: False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    assert cli_module.main(["status", "--log-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.count("pending") == 2  # printed once each time, not a live view


# --- what stays on the screen when the view ends ------------------------------------------------------------------


def run_watch_ending_with(monkeypatch, rows, quit_by_user):
    """Run the real watch() with the terminal parts replaced, and return what it printed after leaving the live view."""
    import contextlib
    import io
    import sys

    from rich.console import Console

    class FakeLive:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, *args, **kwargs):
            pass

    class FakeInput:
        def fileno(self):
            return 0

    printed = io.StringIO()
    monkeypatch.setattr(watch, "Live", FakeLive)
    monkeypatch.setattr(watch, "single_keys", lambda fd: contextlib.nullcontext())
    monkeypatch.setattr(watch, "poll", lambda *args, **kwargs: (rows, quit_by_user))
    monkeypatch.setattr(watch.ui, "console", lambda: Console(file=printed, force_terminal=False, width=100))
    monkeypatch.setattr(sys, "stdin", FakeInput())
    watch.watch(Path("unused"))
    return printed.getvalue()


def test_pressing_q_leaves_the_last_table_on_the_screen(monkeypatch):
    rows = [runner.ServiceRow("llm", "running", 4121, 8001, "HTTP 200", up=True),
            runner.ServiceRow("stt", "starting", 4188, 8004, "not answering")]
    out = run_watch_ending_with(monkeypatch, rows, quit_by_user=True)
    assert "llm" in out and "4121" in out and "stt" in out and "4188" in out  # the PIDs can still be read
    assert "running" in out and "starting" in out  # as they were, not rewritten as stopped
    assert "The stack has stopped." not in out


def test_when_the_stack_stops_the_table_stays_with_everything_marked_stopped(monkeypatch):
    rows = [runner.ServiceRow("llm", "running", 4121, 8001, "HTTP 200", up=True)]
    out = run_watch_ending_with(monkeypatch, rows, quit_by_user=False)
    assert "llm" in out and "stopped" in out and "The stack has stopped." in out


def test_nothing_is_printed_if_there_was_nothing_to_show(monkeypatch):
    assert run_watch_ending_with(monkeypatch, [], quit_by_user=True) == ""


def test_the_redraw_loop_passes_the_gpu_pids_to_every_frame(tmp_path):
    write_state(tmp_path, "pending")
    seen = []
    rows, _ = watch.poll(tmp_path, lambda r: seen.append(r[0].gpu_pids), lambda seconds: True, pids=lambda: {"s0": [4242]})
    assert seen == [(4242,)] and rows[0].gpu_known is True
