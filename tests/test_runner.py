"""Runner tests use throwaway http.server processes instead of real models."""

import os
import socket
import sys

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
