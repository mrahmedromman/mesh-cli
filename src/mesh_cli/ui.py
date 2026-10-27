"""Terminal presentation: colour, tables, memory bars and a live start-up line.

This is used only when stdout is an interactive terminal. When output is piped, redirected, sent to a dumb
terminal, or NO_COLOR is set, the commands print the plain text they always have, so logs, scripts and tests see
no escape codes and no box drawing.
"""

from __future__ import annotations

import os
import re
import sys
import threading

from rich import box
from rich.console import Console, Group, JustifyMethod
from rich.table import Table
from rich.text import Text

from . import runner
from .holders import GpuReport
from .planner import GpuBudget, Plan

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
BAR_WIDTH = 24
KIND_STYLE = {
    "mesh": "green",
    "container": "cyan",
    "desktop": "dim",
    "other": "yellow",
    "unattributed": "dim italic",
}


def _columns(table: Table, spec: tuple[tuple[str, JustifyMethod], ...]) -> None:
    for title, justify in spec:
        table.add_column(title, justify=justify)


def fancy() -> bool:
    """True when we should draw colour and tables: an interactive terminal that has not opted out."""
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    return sys.stdout.isatty()


def console() -> Console:
    return Console(highlight=False)


def bar(used: int, total: int, width: int = BAR_WIDTH) -> Text:
    """A fixed-width memory bar that turns yellow above 70% and red above 90%."""
    fraction = 0.0 if total <= 0 else min(max(used / total, 0.0), 1.0)
    filled = round(fraction * width)
    colour = "green" if fraction < 0.7 else "yellow" if fraction < 0.9 else "red"
    text = Text()
    text.append("█" * filled, style=colour)
    text.append("░" * (width - filled), style="grey37")
    return text


def _percent(used: int, total: int) -> str:
    return f"{round(100 * used / total)}%" if total > 0 else "-"


def style_line(line: str) -> Text:
    """Colour one progress line printed by `mesh up` / `mesh down`. The words are never changed, only decorated."""
    text = Text(line)
    stripped = line.strip()
    indent = line[: len(line) - len(line.lstrip())]
    if re.match(r"^\[\d+/\d+\] starting ", line):
        head, sep, log = line.partition("  (log:")
        text = Text(head, style="bold")
        if sep:
            text.append(sep + log, style="dim")
    elif stripped.endswith(" is up") or " is up  (" in stripped:
        name, _, rest = stripped.partition(" is up")
        text = Text(indent)
        text.append("✓ ", style="green")
        text.append(name, style="bold")
        text.append(" is up", style="green")
        text.append(rest, style="dim")
    elif stripped.startswith("All services are up"):
        text = Text(line, style="bold green")
    elif stripped.startswith(("warning:", "note:")):
        text = Text(line, style="yellow")
    elif stripped.startswith(("could not stop", "error:")) or " could not be stopped." in stripped:
        text = Text(line, style="red")
    elif stripped.startswith("warning") or "is still running" in stripped:
        text = Text(line, style="yellow")
    elif stripped.startswith("stopped "):
        text = Text(indent)
        text.append("✓ ", style="green")
        text.append(stripped, style="green")
    elif stripped.startswith("stopping ") or stripped.endswith("was already stopped"):
        text = Text(line, style="dim")
    elif stripped.startswith(("Stopping ", "Asking mesh up")):
        text = Text(line, style="bold")
    elif re.match(r"^All (\d+ )?services? stopped\.$", stripped):
        text = Text(line, style="bold green")
    return text


def print_plan(plan: Plan, budgets: list[GpuBudget], con: Console | None = None) -> None:
    con = con or console()
    con.print(Text(f"Plan for {len(plan.services)} services on {len(budgets)} GPUs", style="bold"))

    gpus = Table(box=box.SIMPLE_HEAD, pad_edge=False)
    _columns(gpus, (("GPU", "left"), ("Card", "left"), ("Total", "right"), ("Reserved", "right"), ("In use by others", "left")))
    for b in budgets:
        in_use = Text.assemble(bar(b.external_mib, b.total_mib), f"  {b.external_mib} MiB")
        gpus.add_row(str(b.index), b.name, f"{b.total_mib} MiB", f"{b.reserved_mib} MiB", in_use)
    con.print(gpus)

    services = Table(box=box.SIMPLE_HEAD, pad_edge=False)
    _columns(services, (("#", "right"), ("Service", "left"), ("Result", "left"), ("GPU", "right"),
                        ("Needs free", "right"), ("Free before", "right"), ("Will hold", "right")))
    for number, svc in enumerate(plan.services, 1):
        result = Text("✓ fits", style="green") if svc.ok else Text("✗ does not fit", style="bold red")
        for position, check in enumerate(svc.checks):
            needs = "unknown" if check.needed_mib is None else f"{check.needed_mib} MiB"
            holds = "unknown" if check.footprint_mib is None else f"{check.footprint_mib} MiB"
            free = Text(f"{check.free_before_mib} MiB", style="" if check.ok else "red")
            services.add_row(
                str(number) if position == 0 else "",
                svc.name if position == 0 else "",
                result if position == 0 else "",
                str(check.gpu), needs, free, holds,
            )
    con.print(services)
    for svc in plan.services:
        if svc.suggested_utilization is not None:
            con.print(Text(f"  ✗ {svc.name}: try --gpu-memory-utilization {svc.suggested_utilization:.2f} or less", style="red"))
        for note in svc.notes:
            con.print(Text(f"  note ({svc.name}): {note}", style="dim"))

    totals = {b.index: b.total_mib for b in budgets}
    after = Table(box=box.SIMPLE_HEAD, pad_edge=False, title="After all services start", title_justify="left")
    _columns(after, (("GPU", "left"), ("Memory", "left"), ("Free", "right")))
    for index, left in sorted(plan.free_after_mib.items()):
        total = totals.get(index, 0)
        after.add_row(str(index), Text.assemble(bar(total - left, total), f"  {_percent(total - left, total)}"), f"{left} MiB")
    con.print(after)
    for warning in plan.warnings:
        con.print(Text(f"⚠ {warning}", style="yellow"))


def print_holders(reports: list[GpuReport], con: Console | None = None) -> None:
    con = con or console()
    for report in reports:
        g = report.gpu
        title = Text.assemble((f"GPU {g.index}", "bold"), f"  {g.name}  ")
        title.append(bar(g.used_mib, g.total_mib))
        title.append(f"  {g.used_mib} / {g.total_mib} MiB used ({_percent(g.used_mib, g.total_mib)})")
        con.print(title)
        details = [f"{g.free_mib} MiB free"]
        if g.reserved_mib > 0:
            details.append(f"{g.reserved_mib} MiB reserved by the driver")
        if g.utilization_pct is not None:
            details.append(f"{g.utilization_pct}% utilization")
        con.print(Text("  " + " · ".join(details), style="dim"))
        if not report.holders:
            con.print(Text("  nothing is using this GPU's memory", style="dim"))
        else:
            table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2, 0, 2))
            table.add_column(justify="right")
            table.add_column()
            table.add_column()
            for h in report.holders:
                style = KIND_STYLE.get(h.kind, "")
                table.add_row(f"{h.mib} MiB", Text(h.label, style=f"bold {style}".strip()), Text(h.detail, style="dim" if h.kind == "unattributed" else ""))
            con.print(table)
        con.print()


STATE_STYLE = {
    "pending": ("○", "dim"),
    "starting": ("◐", "yellow"),
    "running": ("●", "green"),
    "failed": ("✗", "bold red"),
    "stopping": ("◐", "yellow"),
    "stopped": ("○", "dim"),
}


def status_view(rows: list[runner.ServiceRow]) -> Group:
    """The status table, then one line under it for each service that failed and why."""
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False)
    _columns(table, (("Service", "left"), ("State", "left"), ("PID", "right"), ("Port", "right"), ("Health", "left")))
    for row in rows:
        mark, style = STATE_STYLE.get(row.state, ("?", ""))
        table.add_row(
            row.name,
            Text(f"{mark} {row.state}", style=style),
            row.shown_pid,
            str(row.port),
            Text(row.health, style="green" if row.up else "dim" if row.health == "-" else "red"),
        )
    notes = [Text(f"  {r.name}: {r.error}  (see: mesh logs {r.name})", style="red") for r in rows if r.error]
    return Group(table, *notes)


def print_status(rows: list[runner.ServiceRow], con: Console | None = None) -> None:
    (con or console()).print(status_view(rows))


class StartupDisplay:
    """Prints `mesh up` progress lines and, while a service loads, one short live line: spinner, name and time.

    The live line is drawn here, not with rich's live display, because that display erases by counting the rows it
    drew. If the terminal is narrowed while a service loads, the line wraps and the count is wrong, which leaves
    stale rows behind. The line is kept short (no log text) so it does not wrap unless the window is tiny, and it
    is cleared with one escape code. The service's log is read with `mesh logs NAME -f`.
    """

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    CLEAR = "\r\x1b[2K"

    def __init__(self, con: Console | None = None) -> None:
        self.con = con or console()
        self._lock = threading.Lock()
        self._label: tuple[str, str] | None = None
        self._frame = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def out(self, line: str) -> None:
        with self._lock:
            self._erase()
            self.con.print(style_line(line))

    def progress(self, name: str | None, elapsed: float, last_line: str) -> None:
        if name is None:
            self.close()
            return
        minutes, seconds = divmod(int(elapsed), 60)
        with self._lock:
            self._label = (name, f"{minutes}:{seconds:02d}")
            if self._thread is None:
                self._stop.clear()
                self._thread = threading.Thread(target=self._animate, daemon=True)
                self._thread.start()
            self._draw()

    def close(self) -> None:
        thread = self._thread
        self._stop.set()
        if thread is not None:
            thread.join()
        with self._lock:
            self._thread = None
            self._label = None
            self._erase()

    def _animate(self) -> None:
        while not self._stop.wait(0.1):
            with self._lock:
                self._frame += 1
                self._draw()

    def _draw(self) -> None:
        if self._label is None:
            return
        name, clock = self._label
        spinner = self.FRAMES[self._frame % len(self.FRAMES)]
        self.con.file.write(f"{self.CLEAR}\x1b[36m{spinner}\x1b[0m \x1b[1m{name}\x1b[0m  {clock}")
        self.con.file.flush()

    def _erase(self) -> None:
        self.con.file.write(self.CLEAR)
        self.con.file.flush()
