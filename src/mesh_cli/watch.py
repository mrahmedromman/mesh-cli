"""Live `mesh status`: redraws the status table until you press q or the stack stops.

The table fills the terminal on its alternate screen, which is redrawn whole on every refresh, so resizing the
window cannot leave stale rows behind. Only used at an interactive terminal; piped output prints the table once.
"""

from __future__ import annotations

import os
import select
import sys
import termios
import time
import tty
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.text import Text

from . import runner, settings, ui

QUIT_KEYS = b"qQ"


def poll(
    log_dir: Path,
    show: Callable[[list[runner.ServiceRow]], None],
    wait: Callable[[float], bool],
    interval: float = 1.0,
) -> tuple[list[runner.ServiceRow], bool]:
    """Draw the status again and again. Returns the last rows and whether the user quit (False: the stack stopped)."""
    rows: list[runner.ServiceRow] = []
    while True:
        state = runner.read_state(log_dir)
        if state is None:  # `mesh up` finished and removed its state file
            return rows, False
        rows = runner.service_rows(state)
        show(rows)
        if wait(interval):
            return rows, True


def stopped(rows: list[runner.ServiceRow]) -> list[runner.ServiceRow]:
    """The last rows once the stack is gone: everything is stopped, except what had failed (it keeps its reason)."""
    return [replace(r, state="failed" if r.state == "failed" else "stopped", pid=None, health="-") for r in rows]


def quit_pressed(fd: int, seconds: float) -> bool:
    """Wait up to `seconds` for keys. True when q was pressed, or the input was closed."""
    deadline = time.monotonic() + seconds
    while (left := deadline - time.monotonic()) > 0:
        ready, _, _ = select.select([fd], [], [], left)
        if not ready:
            break
        data = os.read(fd, 64)
        if not data or any(key in data for key in QUIT_KEYS):
            return True
    return False


@contextmanager
def single_keys(fd: int) -> Iterator[None]:
    """Read keys as they are pressed (no Enter needed), and put the terminal back afterwards. Ctrl+C still works."""
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def watch(log_dir: Path, interval: float = 1.0) -> None:
    con = ui.console()
    fd = sys.stdin.fileno()
    folder = settings.tidy(log_dir)

    def frame(rows: list[runner.ServiceRow]) -> Group:
        return Group(Text(f"mesh status · {folder}", style="bold"), ui.status_view(rows), Text("press q to quit", style="dim"))

    rows: list[runner.ServiceRow] = []
    quit_by_user = False
    try:
        with single_keys(fd), Live(console=con, screen=True, auto_refresh=False) as live:
            rows, quit_by_user = poll(log_dir, lambda r: live.update(frame(r), refresh=True), lambda s: quit_pressed(fd, s), interval)
    except KeyboardInterrupt:
        return
    if rows and not quit_by_user:  # leave the last picture on screen: the alternate screen has just been closed
        ui.print_status(stopped(rows), con)
        con.print("The stack has stopped.", style="bold")
