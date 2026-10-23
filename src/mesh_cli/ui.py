"""Terminal presentation: colour, tables, memory bars and a live start-up line.

This is used only when stdout is an interactive terminal. When output is piped, redirected, sent to a dumb
terminal, or NO_COLOR is set, the commands print the plain text they always have, so logs, scripts and tests see
no escape codes and no box drawing.
"""

from __future__ import annotations

import os
import re
import sys

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

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
    for column, justify in (("GPU", "left"), ("Card", "left"), ("Total", "right"), ("Reserved", "right"), ("In use by others", "left")):
        gpus.add_column(column, justify=justify)
    for b in budgets:
        in_use = Text.assemble(bar(b.external_mib, b.total_mib), f"  {b.external_mib} MiB")
        gpus.add_row(str(b.index), b.name, f"{b.total_mib} MiB", f"{b.reserved_mib} MiB", in_use)
    con.print(gpus)

    services = Table(box=box.SIMPLE_HEAD, pad_edge=False)
    for column, justify in (("#", "right"), ("Service", "left"), ("Result", "left"), ("GPU", "right"),
                            ("Needs free", "right"), ("Free before", "right"), ("Will hold", "right")):
        services.add_column(column, justify=justify)
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
    for column, justify in (("GPU", "left"), ("Memory", "left"), ("Free", "right")):
        after.add_column(column, justify=justify)
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


def print_status(rows: list[tuple[str, bool, int, int, str]], con: Console | None = None) -> None:
    """rows: (name, running, pid, port, health text)."""
    con = con or console()
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False)
    for column, justify in (("Service", "left"), ("State", "left"), ("PID", "right"), ("Port", "right"), ("Health", "left")):
        table.add_column(column, justify=justify)
    for name, running, pid, port, health in rows:
        state = Text("● running", style="green") if running else Text("○ stopped", style="red")
        answering = running and not health.startswith("not")
        table.add_row(name, state, str(pid), str(port), Text(health, style="green" if answering else "red" if running else "dim"))
    con.print(table)


class StartupDisplay:
    """Prints `mesh up` progress lines and, while a service loads, one live line: spinner, time and last log line."""

    def __init__(self, con: Console | None = None) -> None:
        self.con = con or console()
        self._status = None

    def out(self, line: str) -> None:
        self.con.print(style_line(line))

    def progress(self, name: str | None, elapsed: float, last_line: str) -> None:
        if name is None:
            self.close()
            return
        minutes, seconds = divmod(int(elapsed), 60)
        text = Text.assemble((name, "bold"), f"  {minutes}:{seconds:02d}  ")
        room = max(self.con.width - len(text.plain) - 4, 10)
        log = ANSI.sub("", last_line).strip()
        text.append(log if len(log) <= room else log[: room - 1] + "…", style="dim")
        if self._status is None:
            self._status = self.con.status(text, spinner="dots")
            self._status.start()
        else:
            self._status.update(text)

    def close(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None
