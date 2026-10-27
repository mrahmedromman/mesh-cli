"""Start services in order, gate each on its health check, keep one log file per service."""

from __future__ import annotations

import functools
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .config import Config, Service
from .settings import tidy

STATE_FILE = "state.json"


MIN_ADDED_MIB = 100  # a model on a GPU takes more than the CUDA context alone, which is already a few hundred MiB
OTHER_GROWTH_MIB = 500  # growth on a GPU the service did not list that is worth pointing out


class RunError(Exception):
    pass


class ServiceExited(RunError):
    pass


class StartTimeout(RunError):
    pass


def log_path(log_dir: Path, name: str) -> Path:
    return Path(log_dir) / f"{name}.log"


def tail_lines(path: Path, count: int) -> str:
    try:
        with open(path, errors="replace") as handle:
            return "".join(deque(handle, maxlen=count))
    except OSError:
        return ""


def last_line(path: Path, size: int = 4096) -> str:
    """The most recent non-empty line of a log, reading only its tail. A progress bar that redraws itself counts as one line per redraw."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - size))
            data = handle.read().decode(errors="replace")
    except OSError:
        return ""
    lines = [line for line in re.split(r"[\r\n]+", data) if line.strip()]
    return lines[-1] if lines else ""


def probe(port: int, path: str, timeout: float = 3.0) -> int | None:
    """HTTP status of the service, or None if nothing answers. Any HTTP response means it is alive."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
        return None


def linux_stamp(boot_id: str, stat: str) -> str:
    """The boot id and field 22 (start time in ticks since boot) of a /proc/PID/stat line.

    The command name is in parentheses and may contain spaces or parentheses, so the fields are counted after the
    last ")". Field 3 is the first one there, which puts field 22 at index 19.
    """
    return f"{boot_id.strip()}:{stat.rsplit(')', 1)[1].split()[19]}"


def process_start(pid: int) -> str | None:
    """A stamp for when a process began, or None if there is no such process.

    A pid can be handed to a different program after a crash or a reboot, so the state file records this stamp
    next to each pid, and mesh only trusts a pid whose stamp still matches. On Linux it is the boot id and the
    start time in ticks since boot, which neither a reboot nor a clock change can repeat. Elsewhere it is the start
    time from `ps`, in UTC so that the shell's time zone cannot change it.
    """
    try:
        return linux_stamp(Path("/proc/sys/kernel/random/boot_id").read_text(), Path(f"/proc/{pid}/stat").read_text())
    except (OSError, IndexError):
        pass
    try:
        result = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True, timeout=5,
                                env={**os.environ, "TZ": "UTC", "LC_ALL": "C"})
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def is_ours(pid: int, started: str | None) -> bool:
    """False only when `pid` now clearly belongs to a different process than the one recorded.

    If the process is gone, or nothing was recorded (a state file from 0.1.0), it is treated as ours: while a process
    group still has members its id cannot be given to anyone else.
    """
    if not started:
        return True
    now = process_start(pid)
    return now is None or now == started


def still_ours(info: dict) -> bool:
    """True if the process group recorded for a service exists and is the one mesh started."""
    pid = info.get("pid") or 0
    return pid > 0 and group_alive(pid) and is_ours(pid, info.get("started"))


def is_up(code: int | None) -> bool:
    """Whether a health probe's answer means the service is up.

    Any answer below 500 does (200, but also 401, 404 or 405), because the server is listening. A 5xx means it is
    reachable but not ready, for example 503 while a model is still loading, so mesh keeps waiting.
    """
    return code is not None and code < 500


def port_in_use(port: int) -> bool:
    """True if something is already listening on the port. Its answers would make a new service look healthy."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def group_alive(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_state(log_dir: Path) -> dict | None:
    try:
        return json.loads((Path(log_dir) / STATE_FILE).read_text())
    except (OSError, ValueError):
        return None


def write_state(log_dir: Path, state: dict) -> None:
    target = Path(log_dir) / STATE_FILE
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, target)


def mark(log_dir: Path, state: dict, name: str, **fields) -> None:
    """Change one service's entry in the state file, which `mesh status` reads while `mesh up` runs."""
    state["services"][name].update(fields)
    write_state(log_dir, state)


def clear_state(log_dir: Path) -> None:
    try:
        (Path(log_dir) / STATE_FILE).unlink()
    except OSError:
        pass


def known_services(log_dir: Path) -> list[str]:
    """Names of services mesh has started from this log directory: the ones in the state file plus any with a log."""
    names = set((read_state(log_dir) or {}).get("services", {}))
    try:
        names.update(p.stem for p in Path(log_dir).glob("*.log"))
    except OSError:
        pass
    return sorted(names)


def running_services(log_dir: Path) -> list[str]:
    state = read_state(log_dir) or {}
    return [name for name, info in state.get("services", {}).items() if still_ours(info)]


@dataclass(frozen=True)
class ServiceRow:
    """One line of `mesh status`."""

    name: str
    state: str  # pending | starting | running | failed | stopping
    pid: int | None
    port: int
    health: str  # "HTTP 200", "not answering", or "-" when there is no process to ask
    error: str = ""
    up: bool = False  # the answer counts as healthy (see is_up)


def service_rows(state: dict, probe_fn: Callable[..., int | None] = probe) -> list[ServiceRow]:
    """The services in the state file, in start order, with what is true now: is the process there, does it answer.

    The health checks run at the same time with a short timeout, so a few silent services cannot freeze the screen.
    """
    entries = list(state.get("services", {}).items())
    alive = {name: still_ours(info) for name, info in entries}
    with ThreadPoolExecutor(max_workers=max(1, len(entries))) as pool:
        codes = {name: pool.submit(probe_fn, info["port"], info["health"], 1.0) for name, info in entries if alive[name]}
    rows = []
    for name, info in entries:
        recorded = info.get("status", "running")  # a state file written by 0.1.0 has no status
        error = info.get("error") or ""
        if recorded in ("starting", "running") and not alive[name]:
            recorded, error = "failed", error or "the process is gone"
        code = codes[name].result() if name in codes else None
        health = "-" if name not in codes else (f"HTTP {code}" if code is not None else "not answering")
        if recorded == "starting" and is_up(code):  # `mesh up` records it at its next check; this is already true
            recorded = "running"
        rows.append(ServiceRow(name, recorded, info.get("pid"), info["port"], health, error, is_up(code)))
    return rows


def wrong_gpu_warning(svc: Service, before: dict[int, int], after: dict[int, int]) -> str | None:
    """A warning when a GPU the service lists gained almost no memory while it started, else None.

    `mesh` only plans with the GPUs the config lists; it is the command that decides which GPU the service uses. If
    the command does not match the list, the plan was wrong, and the memory readings show it.
    """
    added = {g: after[g] - before[g] for g in svc.gpus if g in before and g in after}
    low = [(g, d) for g, d in added.items() if d < MIN_ADDED_MIB]
    if not low:
        return None
    where = ", ".join(f"GPU {g} ({d:+d} MiB)" for g, d in low)
    grew = [f"GPU {g} grew by ~{after[g] - before[g]} MiB" for g in sorted(after)
            if g not in svc.gpus and g in before and after[g] - before[g] >= OTHER_GROWTH_MIB]
    text = f"{svc.name} added almost no memory on {where}" + (f"; {', '.join(grew)} meanwhile" if grew else "")
    ids = ",".join(str(g) for g in svc.gpus)
    pin = f"--gpus '\"device={ids}\"'" if "docker" in (svc.command or "") else f"CUDA_VISIBLE_DEVICES={ids}"
    plan = f"GPU {ids.replace(',', ', ')}" if len(svc.gpus) == 1 else f"GPUs {ids.replace(',', ', ')}"
    return f"{text}. The plan assumes {svc.name} uses {plan}: make its command use {'it' if len(svc.gpus) == 1 else 'them'} too, for example {pin}."


def start_service(svc: Service, log_dir: Path) -> subprocess.Popen:
    if not svc.command:
        raise RunError(f"service '{svc.name}' has no 'command' in the config")
    path = log_path(log_dir, svc.name)
    with open(path, "wb") as log:
        return subprocess.Popen(
            ["bash", "-c", svc.command],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # own process group, so stopping never touches mesh itself
        )


def wait_healthy(
    svc: Service,
    proc: subprocess.Popen,
    log_file: Path,
    interval: float = 2.0,
    probe_fn: Callable[[int, str], int | None] = probe,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[float, str], None] | None = None,
) -> None:
    began = clock()
    deadline = began + svc.start_timeout_s
    last_answer: int | None = None
    while True:
        code = proc.poll()
        if code is not None:
            raise ServiceExited(
                f"{svc.name} exited with status {code} before it became healthy. Last log lines:\n"
                + tail_lines(log_file, 20)
            )
        answer = probe_fn(svc.port, svc.health)
        if is_up(answer):
            return
        if answer is not None:
            last_answer = answer
        if clock() > deadline:
            seen = f" (last answer: HTTP {last_answer})" if last_answer is not None else ""
            raise StartTimeout(
                f"{svc.name} did not become healthy on port {svc.port}{svc.health} within {svc.start_timeout_s}s{seen}. "
                f"See {log_file}"
            )
        if progress is not None:
            progress(clock() - began, last_line(log_file))
        sleep(interval)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_process(pid: int, stop_command: str | None = None, grace: float = 30.0, proc: subprocess.Popen | None = None) -> str:
    """Stop a service's process group.

    Returns "gone" (it was already stopped), "stopped", "killed" (SIGKILL was needed) or "denied" (the process
    belongs to another user and could not be signalled).
    """
    if stop_command:
        try:
            subprocess.run(["bash", "-c", stop_command], capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            pass

    def gone() -> bool:
        if proc is not None:
            proc.poll()  # reap the child so the group can disappear
        return not group_alive(pid)

    if not pid or gone():  # pid 0: what was recorded is gone or now belongs to another program; signal nothing
        return "gone"
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "gone"
    except PermissionError:
        return "denied"
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if gone():
            return "stopped"
        time.sleep(0.2)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return "stopped"
    except PermissionError:
        return "denied"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not gone():
        time.sleep(0.1)
    return "killed"


def _count(total: int) -> str:
    return f"{total} service{'' if total == 1 else 's'}"


def _all_stopped(total: int) -> str:
    return "All services stopped." if total == 1 else f"All {total} services stopped."


def stop_all(
    entries: list[tuple[str, int, str | None, subprocess.Popen | None]],
    grace: float,
    out: Callable[[str], None],
) -> int:
    """Stop services in the order given, printing progress. Returns how many are still running afterwards."""
    total = len(entries)
    out(f"Stopping {_count(total)}...")
    stuck = 0
    for name, pid, stop_command, proc in entries:
        out(f"  stopping {name} ...")
        began = time.monotonic()
        result = stop_process(pid, stop_command, grace, proc)
        seconds = time.monotonic() - began
        if result == "gone":
            out(f"  {name} was already stopped")
        elif result == "killed":
            out(f"  stopped {name} (force-killed after {grace:g}s)")
        elif result == "denied":
            out(f"  could not stop {name}: permission denied (pid {pid} belongs to another user?)")
        else:
            out(f"  stopped {name}" + (f" ({seconds:.0f}s)" if seconds >= 1 else ""))
        if result == "denied":
            stuck += 1
        elif pid and group_alive(pid):
            stuck += 1
            out(f"  warning: {name} is still running (pid {pid})")
    out(f"{stuck} of {_count(total)} could not be stopped." if stuck else _all_stopped(total))
    return stuck


def _on_sigterm(signum, frame):  # noqa: ARG001
    raise KeyboardInterrupt


def _report(progress: Callable[[str | None, float, str], None], name: str, seconds: float, line: str) -> None:
    progress(name, seconds, line)


def _hold_through_interrupts(out: Callable[[str], None]) -> list:
    """Make Ctrl+C and SIGTERM harmless while services are being stopped. Returns what to put back afterwards."""

    def note(signum, frame):  # noqa: ARG001
        out("  still stopping, please wait...")

    return [(sig, signal.signal(sig, note)) for sig in (signal.SIGINT, signal.SIGTERM)]


def _say(message: str) -> None:
    print(message, flush=True)  # flush so progress shows up when output is piped to a file


def run_up(
    config: Config,
    log_dir: Path,
    snapshot: Callable[[], dict[int, int]] | None = None,
    hold: Callable[[], bool] | None = None,
    out: Callable[[str], None] = _say,
    health_interval: float = 2.0,
    grace: float = 30.0,
    progress: Callable[[str | None, float, str], None] | None = None,
) -> None:
    """Start every service in order. Blocks (while `hold()` is true, default forever), then stops everything.

    `progress(name, seconds, last_log_line)` is called while a service loads; `progress(None, 0, "")` when it stops.
    """
    missing = [s.name for s in config.services if not s.command]
    if missing:
        raise RunError(f"these services have no 'command' in the config: {', '.join(missing)}")
    taken = [f"port {s.port} ({s.name}) is already in use by another program" for s in config.services if port_in_use(s.port)]
    if taken:  # checked before anything starts: whatever answers there would pass for the new service
        raise RunError("; ".join(taken) + ". Stop it, or change the port in the config.")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    started: list[tuple[Service, subprocess.Popen]] = []
    state: dict = {
        "controller_pid": os.getpid(),
        "controller_started": process_start(os.getpid()),
        "services": {  # every service is listed from the start, in start order; pid is set once it has a process
            svc.name: {"status": "pending", "pid": None, "port": svc.port, "health": svc.health, "stop": svc.stop,
                       "command": svc.command, "log": str(log_path(log_dir, svc.name)), "error": None}
            for svc in config.services
        },
    }
    previous_handler = None
    if threading.current_thread() is threading.main_thread():
        previous_handler = signal.signal(signal.SIGTERM, _on_sigterm)
    total = len(config.services)
    try:
        write_state(log_dir, state)
        for number, svc in enumerate(config.services, 1):
            before = snapshot() if snapshot else None
            file = log_path(log_dir, svc.name)
            out(f"[{number}/{total}] starting {svc.name} on port {svc.port}  (log: {tidy(file)})")
            proc = start_service(svc, log_dir)
            started.append((svc, proc))
            mark(log_dir, state, svc.name, status="starting", pid=proc.pid, started=process_start(proc.pid))
            on_wait: Callable[[float, str], None] | None = None
            if progress is not None:
                on_wait = functools.partial(_report, progress, svc.name)
            try:
                wait_healthy(svc, proc, file, interval=health_interval, progress=on_wait)
            except RunError as exc:
                mark(log_dir, state, svc.name, status="failed", error=str(exc).splitlines()[0])
                raise
            finally:
                if progress:
                    progress(None, 0.0, "")
            mark(log_dir, state, svc.name, status="running")
            line = f"      {svc.name} is up"
            if before is not None and snapshot:
                after = snapshot()
                held = ", ".join(f"GPU {g}: ~{after[g] - before[g]} MiB" for g in svc.gpus if g in after and g in before)
                line += f"  (memory it added: {held})"
            out(line)
            if before is not None and snapshot:
                warning = wrong_gpu_warning(svc, before, after)
                if warning:
                    out(f"warning: {warning}")
        out("All services are up. Press Ctrl+C to stop everything.")
        reported: set[str] = set()
        while hold is None or hold():
            if hold is None:
                time.sleep(2.0)
            for svc, proc in started:
                if proc.poll() is not None and svc.name not in reported:
                    reported.add(svc.name)
                    mark(log_dir, state, svc.name, status="failed", error=f"exited with status {proc.returncode}")
                    out(f"warning: {svc.name} exited with status {proc.returncode}; see {log_path(log_dir, svc.name)}")
    except KeyboardInterrupt:
        out("")  # Ctrl+C leaves the cursor after "^C"
    finally:
        # Cleanup must finish: a second Ctrl+C (or a SIGTERM from `mesh down`) would otherwise abort it halfway and
        # leave services running with a stale state file.
        restore = _hold_through_interrupts(out) if threading.current_thread() is threading.main_thread() else []
        try:
            for entry in state["services"].values():
                if entry["status"] in ("starting", "running"):
                    entry["status"] = "stopping"
            try:
                write_state(log_dir, state)
            except OSError:
                pass  # a full disk must not stop the cleanup below
            if started:
                stop_all([(svc.name, proc.pid, svc.stop, proc) for svc, proc in reversed(started)], grace, out)
            clear_state(log_dir)
        finally:
            for sig, handler in restore:
                signal.signal(sig, handler)
            if previous_handler is not None:
                signal.signal(signal.SIGTERM, previous_handler)


def stop_from_state(
    log_dir: Path,
    out: Callable[[str], None] = _say,
    wait: float = 90.0,
    poll: float = 0.5,
    grace: float = 30.0,
) -> bool:
    """`mesh down`: stop a stack started by another `mesh up` process. Returns False if nothing was running."""
    state = read_state(log_dir)
    if not state:
        return False
    services = [(name, info) for name, info in state.get("services", {}).items() if info.get("pid")]  # in start order
    controller = state.get("controller_pid") or 0
    controller_alive = (
        controller not in (0, os.getpid()) and pid_alive(controller) and is_ours(controller, state.get("controller_started"))
    )
    if not services and not controller_alive:  # nothing was ever started, so there is nothing to stop
        clear_state(log_dir)
        return False
    if controller_alive:
        out(f"Asking mesh up (pid {controller}) to stop {_count(len(services))}...")
        try:
            os.kill(controller, signal.SIGTERM)  # let `mesh up` shut down gracefully and clear the state
        except ProcessLookupError:
            pass
        else:
            remaining = [name for name, _ in reversed(services)]
            pids = {name: info["pid"] for name, info in services}
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                finished = read_state(log_dir) is None
                for name in list(remaining):  # report each service as it disappears (they stop in reverse order)
                    if finished or not group_alive(pids[name]):
                        out(f"  stopped {name}")
                        remaining.remove(name)
                if finished:
                    out(_all_stopped(len(services)))
                    return True
                time.sleep(poll)
            out("mesh up did not finish in time; stopping the remaining services directly.")
            services = [(name, info) for name, info in services if name in remaining]
    # A pid that now belongs to another program is passed as 0, so only its stop command (e.g. `docker stop NAME`) runs.
    stop_all([(name, info["pid"] if is_ours(info["pid"], info.get("started")) else 0, info.get("stop"), None)
              for name, info in reversed(services)], grace, out)
    clear_state(log_dir)
    return True
