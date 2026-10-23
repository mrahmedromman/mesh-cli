"""Runner tests use throwaway http.server processes instead of real models."""

import os
import socket
import sys
from pathlib import Path

import pytest

from mesh_cli import runner
from mesh_cli.config import Config, Service


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def http_service(name, port, directory, **extra):
    return Service(
        name=name,
        port=port,
        gpus=[0],
        health="/",
        start_timeout_s=30,
        command=f"exec {sys.executable} -m http.server {port} --bind 127.0.0.1 --directory {directory}",
        **extra,
    )


def test_up_starts_in_order_writes_logs_and_cleans_up(tmp_path):
    a, b = free_port(), free_port()
    config = Config(services=[http_service("a", a, tmp_path), http_service("b", b, tmp_path)])
    lines, seen = [], {}

    def hold():
        seen["a"], seen["b"] = runner.probe(a, "/"), runner.probe(b, "/")
        seen["state"] = runner.read_state(tmp_path / "logs")
        return False

    runner.run_up(config, tmp_path / "logs", hold=hold, out=lines.append, health_interval=0.2, grace=5)

    assert seen["a"] == 200 and seen["b"] == 200
    assert set(seen["state"]["services"]) == {"a", "b"}
    assert (tmp_path / "logs" / "a.log").exists() and (tmp_path / "logs" / "b.log").exists()
    starting = [l for l in lines if "starting" in l]
    assert "a" in starting[0] and "b" in starting[1]
    assert runner.probe(a, "/") is None and runner.probe(b, "/") is None
    assert runner.read_state(tmp_path / "logs") is None


def test_service_that_dies_stops_the_ones_already_started(tmp_path):
    a = free_port()
    bad = Service(name="bad", port=free_port(), gpus=[0], command="echo boom; exit 3", start_timeout_s=30)
    config = Config(services=[http_service("a", a, tmp_path), bad])
    with pytest.raises(runner.ServiceExited) as excinfo:
        runner.run_up(config, tmp_path / "logs", out=lambda _: None, health_interval=0.2, grace=5)
    assert "status 3" in str(excinfo.value) and "boom" in str(excinfo.value)
    assert runner.probe(a, "/") is None


def test_timeout_stops_the_process(tmp_path):
    pidfile = tmp_path / "pid"
    slow = Service(name="slow", port=free_port(), gpus=[0], start_timeout_s=1, command=f"echo $$ > {pidfile}; exec sleep 60")
    with pytest.raises(runner.StartTimeout):
        runner.run_up(Config(services=[slow]), tmp_path / "logs", out=lambda _: None, health_interval=0.2, grace=5)
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_stop_command_runs(tmp_path):
    marker = tmp_path / "stopped"
    port = free_port()
    config = Config(services=[http_service("a", port, tmp_path, stop=f"touch {marker}")])
    runner.run_up(config, tmp_path / "logs", hold=lambda: False, out=lambda _: None, health_interval=0.2, grace=5)
    assert marker.exists()


def test_missing_command_is_reported(tmp_path):
    config = Config(services=[Service(name="x", port=1, gpus=[0])])
    with pytest.raises(runner.RunError, match="no 'command'"):
        runner.run_up(config, tmp_path / "logs", out=lambda _: None)


def test_snapshot_reports_memory_added(tmp_path):
    port = free_port()
    config = Config(services=[http_service("a", port, tmp_path)])
    readings = iter([{0: 100}, {0: 2500}])
    lines = []
    runner.run_up(config, tmp_path / "logs", snapshot=lambda: next(readings), hold=lambda: False,
                  out=lines.append, health_interval=0.2, grace=5)
    assert any("GPU 0: ~2400 MiB" in l for l in lines)


# --- shutdown progress messages -------------------------------------------------------------------------------


def test_up_prints_progress_while_stopping_in_reverse_order(tmp_path):
    a, b = free_port(), free_port()
    config = Config(services=[http_service("a", a, tmp_path), http_service("b", b, tmp_path)])
    lines = []
    runner.run_up(config, tmp_path / "logs", hold=lambda: False, out=lines.append, health_interval=0.2, grace=5)
    stopping = lines[lines.index("Stopping 2 services..."):]
    assert [l.strip().split(" (")[0] for l in stopping] == [
        "Stopping 2 services...",
        "stopping b ...",
        "stopped b",
        "stopping a ...",
        "stopped a",
        "All 2 services stopped.",
    ]


def test_single_service_completion_message(tmp_path):
    config = Config(services=[http_service("only", free_port(), tmp_path)])
    lines = []
    runner.run_up(config, tmp_path / "logs", hold=lambda: False, out=lines.append, health_interval=0.2, grace=5)
    assert "Stopping 1 service..." in lines and lines[-1] == "All services stopped."


def test_failed_start_also_reports_the_cleanup(tmp_path):
    bad = Service(name="bad", port=free_port(), gpus=[0], command="exit 3", start_timeout_s=30)
    config = Config(services=[http_service("a", free_port(), tmp_path), bad])
    lines = []
    with pytest.raises(runner.ServiceExited):
        runner.run_up(config, tmp_path / "logs", out=lines.append, health_interval=0.2, grace=5)
    assert "  stopped a" in lines and lines[-1].startswith("All 2 services stopped")


def start_group(command):
    import subprocess

    return subprocess.Popen(["bash", "-c", command], start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_stop_process_results(tmp_path):
    done = start_group("true")
    done.wait()
    assert runner.stop_process(done.pid, grace=1, proc=done) == "gone"

    polite = start_group("exec sleep 60")
    assert runner.stop_process(polite.pid, grace=5, proc=polite) == "stopped"

    stubborn = start_group("trap '' TERM; exec sleep 60")  # ignores SIGTERM, so it needs SIGKILL
    import time

    time.sleep(0.3)
    assert runner.stop_process(stubborn.pid, grace=1, proc=stubborn) == "killed"


def test_stop_all_says_when_a_service_was_force_killed_or_already_gone(tmp_path):
    gone = start_group("true")
    gone.wait()
    stubborn = start_group("trap '' TERM; exec sleep 60")
    import time

    time.sleep(0.3)
    lines = []
    stuck = runner.stop_all([("old", gone.pid, None, gone), ("stubborn", stubborn.pid, None, stubborn)], 1, lines.append)
    assert stuck == 0
    assert "  old was already stopped" in lines
    assert "  stopped stubborn (force-killed after 1s)" in lines
    assert lines[-1] == "All 2 services stopped."


def test_down_without_a_running_mesh_up_stops_each_service_with_progress(tmp_path):
    import subprocess

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    dead_controller = subprocess.Popen(["true"])
    dead_controller.wait()
    first, second = start_group("exec sleep 60"), start_group("exec sleep 60")
    import threading

    for child in (first, second):  # in real use these are orphans that init reaps; here the test must do it
        threading.Thread(target=child.wait, daemon=True).start()
    runner.write_state(log_dir, {
        "controller_pid": dead_controller.pid,
        "services": {"one": {"pid": first.pid}, "two": {"pid": second.pid}},
    })
    lines = []
    assert runner.stop_from_state(log_dir, out=lines.append, grace=5) is True
    assert [l.strip() for l in lines if "stopping" in l] == ["stopping two ...", "stopping one ..."]
    assert lines[-1] == "All 2 services stopped."
    assert runner.read_state(log_dir) is None
    assert not runner.group_alive(first.pid) and not runner.group_alive(second.pid)


def test_down_asks_a_running_mesh_up_to_stop_and_reports_each_service(tmp_path):
    import os
    import subprocess
    import time

    a, b = free_port(), free_port()
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "services:\n"
        f"  - {{name: a, port: {a}, gpus: [0], health: /, command: 'exec {sys.executable} -m http.server {a} --bind 127.0.0.1 --directory {tmp_path}'}}\n"
        f"  - {{name: b, port: {b}, gpus: [0], health: /, command: 'exec {sys.executable} -m http.server {b} --bind 127.0.0.1 --directory {tmp_path}'}}\n"
    )
    log_dir = tmp_path / "logs"
    src = str(Path(__file__).resolve().parents[1] / "src")
    up = subprocess.Popen(
        [sys.executable, "-m", "mesh_cli.cli", "up", str(cfg), "--log-dir", str(log_dir)],
        env={**os.environ, "PYTHONPATH": src}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not (runner.probe(a, "/") and runner.probe(b, "/")):
            time.sleep(0.2)
        assert runner.probe(a, "/") == 200 and runner.probe(b, "/") == 200
        lines = []
        assert runner.stop_from_state(log_dir, out=lines.append, poll=0.1) is True
        assert up.wait(timeout=30) == 0
    finally:
        if up.poll() is None:
            up.kill()
    assert lines[0].startswith("Asking mesh up (pid ") and lines[0].endswith("to stop 2 services...")
    assert [l for l in lines if l.startswith("  stopped")] == ["  stopped b", "  stopped a"]
    assert lines[-1] == "All 2 services stopped."
    output = up.stdout.read()
    assert "Stopping 2 services..." in output and "All 2 services stopped." in output
    assert runner.probe(a, "/") is None and runner.probe(b, "/") is None


def test_permission_denied_is_reported_not_raised(monkeypatch):
    child = start_group("exec sleep 60")
    real_killpg = os.killpg

    def deny(pid, sig):
        if sig == 0:
            return real_killpg(pid, sig)
        raise PermissionError

    monkeypatch.setattr(runner.os, "killpg", deny)
    lines = []
    stuck = runner.stop_all([("other-users-service", child.pid, None, None)], 1, lines.append)
    monkeypatch.undo()
    real_killpg(child.pid, 9)
    child.wait()
    assert stuck == 1
    assert any("could not stop other-users-service: permission denied" in l for l in lines)
    assert lines[-1] == "1 of 1 service could not be stopped."
