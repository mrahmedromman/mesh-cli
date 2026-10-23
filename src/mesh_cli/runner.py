"""Start services in order, gate each on its health check, keep one log file per service."""

from __future__ import annotations

import http.client
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Callable

from .config import Config, Service

STATE_FILE = "state.json"


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
        with open(path, "r", errors="replace") as handle:
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


def clear_state(log_dir: Path) -> None:
    try:
        (Path(log_dir) / STATE_FILE).unlink()
    except OSError:
        pass


def running_services(log_dir: Path) -> list[str]:
    state = read_state(log_dir) or {}
    return [name for name, info in state.get("services", {}).items() if group_alive(info["pid"])]


def start_service(svc: Service, log_dir: Path) -> subprocess.Popen:
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
    while True:
        code = proc.poll()
        if code is not None:
            raise ServiceExited(
                f"{svc.name} exited with status {code} before it became healthy. Last log lines:\n"
                + tail_lines(log_file, 20)
            )
        if probe_fn(svc.port, svc.health) is not None:
            return
        if clock() > deadline:
            raise StartTimeout(
                f"{svc.name} did not answer on port {svc.port}{svc.health} within {svc.start_timeout_s}s. "
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

    if gone():
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
        elif group_alive(pid):
            stuck += 1
            out(f"  warning: {name} is still running (pid {pid})")
    out(f"{stuck} of {_count(total)} could not be stopped." if stuck else _all_stopped(total))
    return stuck


def _on_sigterm(signum, frame):  # noqa: ARG001
    raise KeyboardInterrupt


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
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    started: list[tuple[Service, subprocess.Popen]] = []
    state: dict = {"controller_pid": os.getpid(), "services": {}}
    previous_handler = None
    if threading.current_thread() is threading.main_thread():
        previous_handler = signal.signal(signal.SIGTERM, _on_sigterm)
    total = len(config.services)
    try:
        for number, svc in enumerate(config.services, 1):
            before = snapshot() if snapshot else None
            file = log_path(log_dir, svc.name)
            out(f"[{number}/{total}] starting {svc.name} on port {svc.port}  (log: {file})")
            proc = start_service(svc, log_dir)
            started.append((svc, proc))
            state["services"][svc.name] = {"pid": proc.pid, "port": svc.port, "health": svc.health, "stop": svc.stop, "command": svc.command, "log": str(file)}
            write_state(log_dir, state)
            try:
                wait_healthy(
                    svc, proc, file, interval=health_interval,
                    progress=(lambda seconds, line, name=svc.name: progress(name, seconds, line)) if progress else None,
                )
            finally:
                if progress:
                    progress(None, 0.0, "")
            line = f"      {svc.name} is up"
            if before is not None and snapshot:
                after = snapshot()
                held = ", ".join(f"GPU {g}: ~{after[g] - before[g]} MiB" for g in svc.gpus if g in after and g in before)
                line += f"  (memory it added: {held})"
            out(line)
        out("All services are up. Press Ctrl+C to stop everything.")
        reported: set[str] = set()
        while hold is None or hold():
            if hold is None:
                time.sleep(2.0)
            for svc, proc in started:
                if proc.poll() is not None and svc.name not in reported:
                    reported.add(svc.name)
                    out(f"warning: {svc.name} exited with status {proc.returncode}; see {log_path(log_dir, svc.name)}")
    except KeyboardInterrupt:
        out("")  # Ctrl+C leaves the cursor after "^C"
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)
        if started:
            stop_all([(svc.name, proc.pid, svc.stop, proc) for svc, proc in reversed(started)], grace, out)
        clear_state(log_dir)


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
    services = list(state.get("services", {}).items())  # in start order
    controller = state.get("controller_pid")
    if controller and controller != os.getpid() and pid_alive(controller):
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
    stop_all([(name, info["pid"], info.get("stop"), None) for name, info in reversed(services)], grace, out)
    clear_state(log_dir)
    return True
