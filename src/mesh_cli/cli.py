"""Command line entry point: mesh plan | up | down | status | logs."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from . import __version__, completion, holders, runner, ui
from .config import ConfigError, load_config
from .gpu import GpuError, parse_gpu_query, read_gpus
from .planner import GpuBudget, Plan, PlanError, make_plan
from .suggest import did_you_mean


class _Parser(argparse.ArgumentParser):
    """Adds a 'did you mean' to the unknown-command and invalid-value errors."""

    def error(self, message: str):
        found = re.search(r"invalid choice: '?(.+?)'? \(choose from (.+)\)", message)
        if found and "maybe you meant" not in message:  # newer Pythons add their own suggestion
            word = found.group(1)
            options = [o.strip().strip("'") for o in found.group(2).split(",")]
            options = [o for o in options if not o.startswith("_")]
            what = "command" if message.startswith("argument command") else "value"
            message = f"unknown {what} '{word}'.{did_you_mean(word, options)} Choose from: {', '.join(options)}"
        super().error(message)


def _parse_external(value: str) -> dict[int, int]:
    result: dict[int, int] = {}
    for part in value.split(","):
        gpu, _, mib = part.partition("=")
        if not gpu.strip().isdigit() or not mib.strip().isdigit():
            raise argparse.ArgumentTypeError(f"expected GPU=MiB pairs like 0=417,1=4, got {value!r}")
        result[int(gpu)] = int(mib)
    return result


def _line_count(value: str) -> int | None:
    """`-n all` means the whole file (None); otherwise a number of lines, 0 or more."""
    if value.lower() == "all":
        return None
    if not value.isdigit():
        raise argparse.ArgumentTypeError(f"expected a number of lines or 'all', got {value!r}")
    return int(value)


def _render(plan: Plan, budgets: list[GpuBudget]) -> str:
    lines = [f"Plan for {len(plan.services)} services on {len(budgets)} GPUs", ""]
    for b in budgets:
        lines.append(f"GPU {b.index}  {b.name}  total {b.total_mib} MiB, driver-reserved {b.reserved_mib}, external {b.external_mib} MiB")
    lines.append("")
    for number, svc in enumerate(plan.services, 1):
        status = "OK  " if svc.ok else "FAIL"
        lines.append(f"{number}. {svc.name:<12} {status}")
        for c in svc.checks:
            if c.needed_mib is None:
                lines.append(f"     GPU {c.gpu}: footprint unknown, {c.free_before_mib} MiB free before start")
            elif c.ok:
                lines.append(f"     GPU {c.gpu}: needs {c.needed_mib} MiB free, has {c.free_before_mib}; will hold {c.footprint_mib} MiB")
            else:
                lines.append(f"     GPU {c.gpu}: needs {c.needed_mib} MiB free but only {c.free_before_mib} MiB is free")
        if svc.suggested_utilization is not None:
            lines.append(f"     try --gpu-memory-utilization {svc.suggested_utilization:.2f} or less")
        for note in svc.notes:
            lines.append(f"     note: {note}")
    lines.append("")
    lines.append("After all services start:")
    for gi, left in sorted(plan.free_after_mib.items()):
        lines.append(f"  GPU {gi}: {left} MiB free")
    for warning in plan.warnings:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)


def _budgets(gpus, external: dict[int, int] | None) -> list[GpuBudget]:
    return [
        GpuBudget(
            g.index,
            g.total_mib,
            external.get(g.index, 0) if external is not None else g.used_mib,
            g.name,
            g.reserved_mib,
        )
        for g in gpus
    ]


MEMORY_ERRORS = ("free memory on device", "out of memory", "outofmemoryerror")


def _looks_like_memory_error(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in MEMORY_ERRORS)


def _show_holders(log_dir: Path, title: str, only: set[int] | None = None) -> None:
    """Print who holds GPU memory. Never raises: this is extra help on top of an error that is already shown."""
    try:
        reports = holders.gather(log_dir, only)
    except (GpuError, OSError) as exc:
        print(f"\n(could not read which processes hold GPU memory: {exc})")
        return
    if ui.fancy():
        con = ui.console()
        con.print(f"\n{title}\n", style="bold", markup=False)
        ui.print_holders(reports, con)
        return
    print(f"\n{title}\n\n{holders.render(reports)}")


def _cmd_gpu(args: argparse.Namespace) -> int:
    reports = holders.gather(Path(args.log_dir))
    if ui.fancy():
        ui.print_holders(reports)
    else:
        print(holders.render(reports))
    return 0


def _show_plan(plan: Plan, budgets: list[GpuBudget]) -> None:
    if ui.fancy():
        ui.print_plan(plan, budgets)
    else:
        print(_render(plan, budgets))


def _plan_summary(plan: Plan, budgets: list[GpuBudget]) -> str:
    free = ", ".join(f"GPU {i}: {mib} MiB" for i, mib in sorted(plan.free_after_mib.items()))
    return f"Plan: all {len(plan.services)} services fit on {len(budgets)} GPUs. Free once they are up: {free}."


def _plan_has_news(plan: Plan) -> bool:
    """True when the plan has something to read: a service that will not fit, a warning, a note, or an unknown size."""
    return bool(
        not plan.ok
        or plan.warnings
        or any(s.notes or any(c.needed_mib is None for c in s.checks) for s in plan.services)
    )


def _cmd_plan(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    gpus = parse_gpu_query(Path(args.gpus_csv).read_text()) if args.gpus_csv else read_gpus()
    if args.external is None:
        print("note: counting all memory currently in use as external; stop running services or pass --external")
    budgets = _budgets(gpus, args.external)
    plan = make_plan(config, budgets)
    _show_plan(plan, budgets)
    if args.external is None and not args.gpus_csv and any(b.external_mib > 0 for b in budgets):
        _show_holders(Path(args.log_dir), "Memory currently in use, by holder:", {b.index for b in budgets if b.external_mib > 0})
    return 0 if plan.ok else 1


def _cmd_up(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    log_dir = Path(args.log_dir)
    alive = runner.running_services(log_dir)
    if alive:
        print(f"error: already running ({', '.join(alive)}). Stop them first with: mesh down", file=sys.stderr)
        return 1
    try:
        gpus = read_gpus()
    except GpuError:
        gpus = None
        print("note: nvidia-smi not available; skipping the memory plan and memory measurements")
    snapshot = None
    if gpus is not None:
        budgets = _budgets(gpus, None)
        plan = make_plan(config, budgets)
        if args.plan or _plan_has_news(plan):
            _show_plan(plan, budgets)
        else:
            print(_plan_summary(plan, budgets))
        if not plan.ok and not args.force:
            print("\nThe plan does not fit. Fix the config, or start anyway with --force.")
            short = {c.gpu for s in plan.services for c in s.checks if not c.ok}
            _show_holders(log_dir, "Who is holding memory on the GPUs that are short:", short)
            return 1
        print()
        snapshot = lambda: {g.index: g.used_mib for g in read_gpus()}  # noqa: E731
    display = ui.StartupDisplay() if ui.fancy() else None
    try:
        if display:
            runner.run_up(config, log_dir, snapshot=snapshot, out=display.out, progress=display.progress)
        else:
            runner.run_up(config, log_dir, snapshot=snapshot)
    except runner.RunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if isinstance(exc, runner.ServiceExited) and _looks_like_memory_error(str(exc)):
            _show_holders(log_dir, "Who is holding GPU memory now (the services mesh started have been stopped):")
        return 1
    finally:
        if display:
            display.close()
    return 0


def _cmd_down(args: argparse.Namespace) -> int:
    out = ui.StartupDisplay().out if ui.fancy() else print
    if not runner.stop_from_state(Path(args.log_dir), out=out):
        print(f"nothing running (looked for {Path(args.log_dir) / runner.STATE_FILE})")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    state = runner.read_state(Path(args.log_dir))
    if not state:
        print("nothing running")
        return 0
    rows = []
    for name, info in state["services"].items():
        alive = runner.group_alive(info["pid"])
        code = runner.probe(info["port"], info["health"]) if alive else None
        health = f"HTTP {code}" if code is not None else "not answering"
        rows.append((name, alive, info["pid"], info["port"], health))
    if ui.fancy():
        ui.print_status(rows)
        return 0
    for name, alive, pid, port, health in rows:
        print(f"{name:<12} {'running' if alive else 'stopped':<8} pid {pid:<8} port {port:<6} {health}")
    return 0


def _cmd_logs(args: argparse.Namespace) -> int:
    path = runner.log_path(Path(args.log_dir), args.service)
    if not path.exists():
        known = runner.known_services(Path(args.log_dir))
        if known:
            print(f"error: no log for service '{args.service}'.{did_you_mean(args.service, known)} "
                  f"Known services: {', '.join(known)}", file=sys.stderr)
        else:
            print(f"error: no log file {path}. Start a stack with 'mesh up', or pass --log-dir if it uses another directory.",
                  file=sys.stderr)
        return 1
    try:
        with open(path, errors="replace") as handle:
            if args.lines is None:  # -n all: the whole file, then (with -f) keep reading from where it ended
                for chunk in iter(lambda: handle.read(65536), ""):
                    sys.stdout.write(chunk)
            else:
                sys.stdout.write(runner.tail_lines(path, args.lines))
                handle.seek(0, 2)
            sys.stdout.flush()
            if not args.follow:
                return 0
            while True:
                chunk = handle.read()
                if chunk:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                else:
                    time.sleep(0.5)
    except KeyboardInterrupt:
        return 0


def _cmd_completion(args: argparse.Namespace) -> int:
    print(completion.script(args.shell, build_parser()), end="")
    return 0


def _cmd_complete(args: argparse.Namespace) -> int:
    for name in runner.known_services(Path(args.log_dir)):
        print(name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="mesh", description="Plan and run multi-service GPU stacks.")
    parser.add_argument("--version", action="version", version=f"mesh {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    plan = sub.add_parser("plan", help="check whether the services fit on the GPUs, in order")
    plan.add_argument("config", help="path to models.yaml")
    plan.add_argument("--gpus-csv", help="saved output of: nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv")
    plan.add_argument("--external", type=_parse_external, help="memory used by other processes, e.g. 0=417,1=4 (MiB)")
    plan.add_argument("--log-dir", default="logs", help="where `mesh up` keeps its state (used to recognise your services)")
    plan.set_defaults(func=_cmd_plan)

    gpu = sub.add_parser("gpu", help="show who is holding memory on each GPU")
    gpu.add_argument("--log-dir", default="logs", help="where `mesh up` keeps its state (used to recognise your services)")
    gpu.set_defaults(func=_cmd_gpu)

    up = sub.add_parser("up", help="plan, then start every service in order with one log file each")
    up.add_argument("config", help="path to models.yaml")
    up.add_argument("--log-dir", default="logs")
    up.add_argument("--force", action="store_true", help="start even if the plan says a service will not fit")
    up.add_argument("--plan", action="store_true", help="always print the full plan (by default only a one-line summary when everything fits)")
    up.set_defaults(func=_cmd_up)

    for name, help_text, func in (
        ("down", "stop a stack started with `mesh up`", _cmd_down),
        ("status", "show running services", _cmd_status),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--log-dir", default="logs")
        p.set_defaults(func=func)

    logs = sub.add_parser("logs", help="show a service's log")
    logs.add_argument("service")
    logs.add_argument("-n", "--lines", type=_line_count, default=50, metavar="N|all",
                      help="how many of the last lines to show (default 50), or 'all' for the whole log")
    logs.add_argument("-f", "--follow", action="store_true", help="keep printing new lines until Ctrl+C")
    logs.add_argument("--log-dir", default="logs")
    logs.set_defaults(func=_cmd_logs)

    comp = sub.add_parser("completion", help="print a shell completion script (bash, zsh or fish)")
    comp.add_argument("shell", choices=completion.SHELLS)
    comp.set_defaults(func=_cmd_completion)

    # Used by the completion scripts; hidden from --help.
    hidden = sub.add_parser("_complete")
    hidden.add_argument("what", choices=["services"])
    hidden.add_argument("--log-dir", default="logs")
    hidden.set_defaults(func=_cmd_complete)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, GpuError, PlanError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
