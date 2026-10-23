"""Command line entry point: mesh plan | up | down | status | logs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__, runner
from .config import ConfigError, load_config
from .gpu import GpuError, parse_gpu_query, read_gpus
from .planner import GpuBudget, Plan, PlanError, make_plan


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


def _cmd_plan(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    gpus = parse_gpu_query(Path(args.gpus_csv).read_text()) if args.gpus_csv else read_gpus()
    if args.external is None:
        print("note: counting all memory currently in use as external; stop running services or pass --external")
    budgets = _budgets(gpus, args.external)
    plan = make_plan(config, budgets)
    print(_render(plan, budgets))
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
        print(_render(plan, budgets))
        if not plan.ok and not args.force:
            print("\nThe plan does not fit. Fix the config, or start anyway with --force.")
            return 1
        print()
        snapshot = lambda: {g.index: g.used_mib for g in read_gpus()}  # noqa: E731
    try:
        runner.run_up(config, log_dir, snapshot=snapshot)
    except runner.RunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_down(args: argparse.Namespace) -> int:
    if not runner.stop_from_state(Path(args.log_dir)):
        print("nothing running")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    state = runner.read_state(Path(args.log_dir))
    if not state:
        print("nothing running")
        return 0
    for name, info in state["services"].items():
        alive = runner.group_alive(info["pid"])
        code = runner.probe(info["port"], info["health"]) if alive else None
        health = f"HTTP {code}" if code is not None else "not answering"
        print(f"{name:<12} {'running' if alive else 'stopped':<8} pid {info['pid']:<8} port {info['port']:<6} {health}")
    return 0


def _cmd_logs(args: argparse.Namespace) -> int:
    path = runner.log_path(Path(args.log_dir), args.service)
    if not path.exists():
        print(f"error: no log file {path}", file=sys.stderr)
        return 1
    try:
        with open(path, "r", errors="replace") as handle:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mesh", description="Plan and run multi-service GPU stacks.")
    parser.add_argument("--version", action="version", version=f"mesh {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="check whether the services fit on the GPUs, in order")
    plan.add_argument("config", help="path to models.yaml")
    plan.add_argument("--gpus-csv", help="saved output of: nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv")
    plan.add_argument("--external", type=_parse_external, help="memory used by other processes, e.g. 0=417,1=4 (MiB)")
    plan.set_defaults(func=_cmd_plan)

    up = sub.add_parser("up", help="plan, then start every service in order with one log file each")
    up.add_argument("config", help="path to models.yaml")
    up.add_argument("--log-dir", default="logs")
    up.add_argument("--force", action="store_true", help="start even if the plan says a service will not fit")
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
