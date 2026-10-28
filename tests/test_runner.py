"""Runner tests use throwaway http.server processes instead of real models."""

import errno
import os
import signal
import socket
import subprocess
import sys
import threading
import time
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


def wait_for(condition, timeout=15.0):
    """Wait until something is true, instead of sleeping a fixed time that a slow machine can overrun."""
    import time

    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.02)


def start_stubborn(tmp_path):
    """A process that ignores SIGTERM. Returns once it has really installed the trap."""
    ready = tmp_path / f"ready-{len(list(tmp_path.glob('ready-*')))}"
    proc = start_group(f"trap '' TERM; touch {ready}; exec sleep 60")
    wait_for(ready.exists)
    return proc


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

    stubborn = start_stubborn(tmp_path)  # ignores SIGTERM, so it needs SIGKILL
    assert runner.stop_process(stubborn.pid, grace=1, proc=stubborn) == "killed"


def test_stop_all_says_when_a_service_was_force_killed_or_already_gone(tmp_path):
    gone = start_group("true")
    gone.wait()
    stubborn = start_stubborn(tmp_path)
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


def test_run_up_reports_progress_for_each_service_and_clears_it(tmp_path):
    events = []
    config = Config(services=[http_service("a", free_port(), tmp_path)])
    runner.run_up(config, tmp_path / "logs", hold=lambda: False, out=lambda _: None, health_interval=0.2, grace=5,
                  progress=lambda name, seconds, line: events.append(name))
    assert events[-1] is None
    assert set(events) <= {"a", None}


def test_a_second_ctrl_c_during_shutdown_does_not_abort_the_cleanup(tmp_path):
    import signal
    import subprocess
    import time

    port = free_port()
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "services:\n"
        f"  - {{name: a, port: {port}, gpus: [0], health: /, stop: 'touch {tmp_path}/stopping; sleep 1', "
        f"command: 'exec {sys.executable} -m http.server {port} --bind 127.0.0.1 --directory {tmp_path}'}}\n"
    )
    log_dir = tmp_path / "logs"
    src = str(Path(__file__).resolve().parents[1] / "src")
    up = subprocess.Popen(
        [sys.executable, "-m", "mesh_cli.cli", "up", str(cfg), "--log-dir", str(log_dir)],
        env={**os.environ, "PYTHONPATH": src}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not runner.probe(port, "/"):
            time.sleep(0.2)
        assert runner.probe(port, "/") == 200
        up.send_signal(signal.SIGINT)
        wait_for((tmp_path / "stopping").exists)  # the stop command has started, so cleanup is under way (it takes a second)
        up.send_signal(signal.SIGINT)
        up.send_signal(signal.SIGTERM)
        assert up.wait(timeout=30) == 0
    finally:
        if up.poll() is None:
            up.kill()
    output = up.stdout.read()
    assert "Traceback" not in output and "All services stopped." in output
    assert runner.probe(port, "/") is None
    assert runner.read_state(log_dir) is None


# --- the state file lists every service and tracks its status ---------------------------------------------------


def one_request_service(name, port, directory):
    """Answers a single request (the health check) and then exits, like a service that crashes after it is up."""
    code = (
        "import http.server as h; "
        f"s = h.HTTPServer(('127.0.0.1', {port}), h.SimpleHTTPRequestHandler); s.handle_request()"
    )
    return Service(name=name, port=port, gpus=[0], health="/", start_timeout_s=30, command=f"cd {directory}; exec {sys.executable} -c \"{code}\"")


def test_state_lists_every_service_before_the_first_one_starts(tmp_path):
    config = Config(services=[http_service("a", free_port(), tmp_path), http_service("b", free_port(), tmp_path)])
    seen = []

    def out(line):
        if "starting a" in line:
            seen.append(runner.read_state(tmp_path / "logs"))

    runner.run_up(config, tmp_path / "logs", hold=lambda: False, out=out, health_interval=0.2, grace=5)
    services = seen[0]["services"]
    assert list(services) == ["a", "b"]  # in start order
    assert [s["status"] for s in services.values()] == ["pending", "pending"]
    assert [s["pid"] for s in services.values()] == [None, None]
    assert services["a"]["port"] and services["a"]["health"] == "/"


def test_a_service_goes_from_pending_to_starting_to_running(tmp_path):
    slow = Service(name="a", port=free_port(), gpus=[0], health="/", start_timeout_s=30,
                   command=f"sleep 1; exec {sys.executable} -m http.server {{port}} --bind 127.0.0.1 --directory {tmp_path}")
    slow.command = slow.command.replace("{port}", str(slow.port))
    config = Config(services=[slow, http_service("b", free_port(), tmp_path)])
    states = {}

    def progress(name, seconds, line):
        if name == "a" and "loading" not in states:
            states["loading"] = runner.read_state(tmp_path / "logs")["services"]["a"]

    def out(line):
        if "starting b" in line:
            states["b_next"] = runner.read_state(tmp_path / "logs")["services"]

    def hold():
        states["done"] = runner.read_state(tmp_path / "logs")["services"]
        return False

    runner.run_up(config, tmp_path / "logs", hold=hold, out=out, progress=progress, health_interval=0.2, grace=5)
    assert states["loading"]["status"] == "starting" and states["loading"]["pid"]
    assert states["b_next"]["a"]["status"] == "running" and states["b_next"]["b"]["status"] == "pending"
    assert states["b_next"]["b"]["pid"] is None
    assert [s["status"] for s in states["done"].values()] == ["running", "running"]
    assert all(s["pid"] for s in states["done"].values())


def test_a_service_that_fails_to_start_is_recorded_as_failed_while_the_rest_stop(tmp_path):
    bad = Service(name="bad", port=free_port(), gpus=[0], command="echo boom; exit 3", start_timeout_s=30)
    config = Config(services=[http_service("a", free_port(), tmp_path), bad, http_service("later", free_port(), tmp_path)])
    seen = []

    def out(line):
        if line.startswith("Stopping"):
            seen.append(runner.read_state(tmp_path / "logs")["services"])

    with pytest.raises(runner.ServiceExited):
        runner.run_up(config, tmp_path / "logs", out=out, health_interval=0.2, grace=5)
    services = seen[0]
    assert services["bad"]["status"] == "failed" and "status 3" in services["bad"]["error"]
    assert services["a"]["status"] == "stopping"
    assert services["later"]["status"] == "pending" and services["later"]["pid"] is None


def test_a_service_that_exits_after_it_was_up_is_marked_failed(tmp_path):
    config = Config(services=[one_request_service("a", free_port(), tmp_path)])
    seen = {}

    def hold():
        service = runner.read_state(tmp_path / "logs")["services"]["a"]
        if service["status"] == "failed":
            seen["service"] = service
            return False
        time.sleep(0.1)
        return True

    runner.run_up(config, tmp_path / "logs", hold=hold, out=lambda _: None, health_interval=0.2, grace=5)
    assert "exited with status" in seen["service"]["error"]


def test_services_that_never_started_have_nothing_to_stop(tmp_path):
    runner.write_state(tmp_path, {"controller_pid": 999999, "services": {
        "waiting": {"status": "pending", "pid": None, "port": 1, "health": "/"},
    }})
    assert runner.running_services(tmp_path) == []
    lines = []
    assert not runner.stop_from_state(tmp_path, out=lines.append, grace=1)
    assert lines == [] and runner.read_state(tmp_path) is None


# --- the rows `mesh status` shows ---------------------------------------------------------------------------------


@pytest.fixture
def sleeper():
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    yield proc
    proc.kill()
    proc.wait()


def fake_probe(code=200):
    return lambda port, path, timeout=1.0: code


def test_rows_show_pending_services_with_a_dash_instead_of_a_process(tmp_path):
    state = {"services": {"a": {"status": "pending", "pid": None, "port": 8001, "health": "/"}}}
    [row] = runner.service_rows(state, probe_fn=fake_probe())
    assert (row.name, row.state, row.pid, row.port, row.health) == ("a", "pending", None, 8001, "-")


def test_rows_probe_the_services_that_have_a_process(sleeper):
    state = {"services": {
        "up": {"status": "running", "pid": sleeper.pid, "port": 8001, "health": "/"},
        "loading": {"status": "starting", "pid": sleeper.pid, "port": 8002, "health": "/"},
    }}
    rows = runner.service_rows(state, probe_fn=lambda port, path, timeout=1.0: 200 if port == 8001 else None)
    assert [(r.state, r.health) for r in rows] == [("running", "HTTP 200"), ("starting", "not answering")]


def test_a_running_service_whose_process_is_gone_shows_as_failed():
    state = {"services": {"a": {"status": "running", "pid": 2_000_000_000, "port": 1, "health": "/"}}}
    [row] = runner.service_rows(state, probe_fn=fake_probe())
    assert row.state == "failed" and row.health == "-" and "gone" in row.error


def test_a_state_file_from_0_1_0_has_no_status_and_counts_as_running(sleeper):
    state = {"services": {"a": {"pid": sleeper.pid, "port": 1, "health": "/"}}}
    [row] = runner.service_rows(state, probe_fn=fake_probe())
    assert row.state == "running"


def test_the_error_of_a_failed_service_is_kept():
    state = {"services": {"a": {"status": "failed", "pid": None, "port": 1, "health": "/", "error": "a exited with status 3"}}}
    [row] = runner.service_rows(state, probe_fn=fake_probe())
    assert row.state == "failed" and row.error == "a exited with status 3"


def test_services_are_probed_at_the_same_time(sleeper):
    barrier = threading.Barrier(4)  # one at a time would never get all four to the barrier

    def probe_together(port, path, timeout=1.0):
        barrier.wait(timeout=5)
        return 200

    state = {"services": {f"s{i}": {"status": "running", "pid": sleeper.pid, "port": i, "health": "/"} for i in range(4)}}
    assert [r.health for r in runner.service_rows(state, probe_fn=probe_together)] == ["HTTP 200"] * 4


def test_a_starting_service_shows_as_running_as_soon_as_it_answers(sleeper):
    state = {"services": {"a": {"status": "starting", "pid": sleeper.pid, "port": 1, "health": "/"}}}
    assert runner.service_rows(state, probe_fn=fake_probe(200))[0].state == "running"
    assert runner.service_rows(state, probe_fn=fake_probe(None))[0].state == "starting"


# --- a leftover state file must never make mesh touch a program it did not start ---------------------------------------

STRANGER = "someone-else:1"  # what the recorded identity looks like once a pid belongs to another process


def test_a_live_process_has_a_start_stamp_and_a_missing_one_has_none(sleeper):
    stamp = runner.process_start(sleeper.pid)
    assert stamp and stamp == runner.process_start(sleeper.pid)  # the same process gives the same stamp every time
    assert runner.process_start(2_000_000_000) is None


def test_is_ours_is_false_only_when_the_pid_clearly_belongs_to_another_process(sleeper):
    stamp = runner.process_start(sleeper.pid)
    assert runner.is_ours(sleeper.pid, stamp)
    assert runner.is_ours(sleeper.pid, None)  # a state file from 0.1.0 recorded no stamp
    assert not runner.is_ours(sleeper.pid, STRANGER)
    assert runner.is_ours(2_000_000_000, STRANGER)  # nothing there: the group, if any, is still ours


def test_up_records_who_it_started(tmp_path):
    config = Config(services=[http_service("a", free_port(), tmp_path)])
    seen = {}

    def hold():  # while the service is still running, so its stamp can be read
        state = runner.read_state(tmp_path / "logs")
        seen["state"] = state
        seen["service_now"] = runner.process_start(state["services"]["a"]["pid"])
        return False

    runner.run_up(config, tmp_path / "logs", hold=hold, out=lambda _: None, health_interval=0.2, grace=5)
    state = seen["state"]
    assert state["controller_started"] == runner.process_start(os.getpid())
    assert state["services"]["a"]["started"] and state["services"]["a"]["started"] == seen["service_now"]


def test_a_pid_taken_over_by_another_program_is_not_a_running_service(tmp_path, sleeper):
    runner.write_state(tmp_path, {"controller_pid": 999999, "services": {
        "llm": {"status": "running", "pid": sleeper.pid, "started": STRANGER, "port": 1, "health": "/"}}})
    assert runner.running_services(tmp_path) == []  # so `mesh up` is not refused for no reason
    [row] = runner.service_rows(runner.read_state(tmp_path), probe_fn=fake_probe())
    assert row.state == "failed" and row.health == "-"


def test_mesh_down_leaves_a_program_that_took_over_a_recorded_pid_alone(tmp_path, sleeper):
    runner.write_state(tmp_path, {"controller_pid": 999999, "services": {
        "llm": {"status": "running", "pid": sleeper.pid, "started": STRANGER, "port": 1, "health": "/"}}})
    lines = []
    assert runner.stop_from_state(tmp_path, out=lines.append, grace=1) is True
    assert sleeper.poll() is None  # not signalled
    assert "  llm was already stopped" in lines and lines[-1] == "All services stopped."
    assert runner.read_state(tmp_path) is None


def test_mesh_down_does_not_signal_a_program_that_took_over_mesh_ups_pid(tmp_path, sleeper):
    runner.write_state(tmp_path, {"controller_pid": sleeper.pid, "controller_started": STRANGER, "services": {
        "llm": {"status": "running", "pid": 2_000_000_000, "port": 1, "health": "/"}}})
    assert runner.stop_from_state(tmp_path, out=lambda _: None, grace=1, wait=1) is True
    assert sleeper.poll() is None
    assert runner.read_state(tmp_path) is None


def test_the_stop_command_still_runs_for_a_service_whose_pid_was_taken_over(tmp_path, sleeper):
    marker = tmp_path / "stopped"
    runner.write_state(tmp_path, {"controller_pid": 999999, "services": {
        "llm": {"status": "running", "pid": sleeper.pid, "started": STRANGER, "port": 1, "health": "/",
                "stop": f"touch {marker}"}}})  # e.g. `docker stop NAME`: the container may well still be running
    runner.stop_from_state(tmp_path, out=lambda _: None, grace=1)
    assert marker.exists() and sleeper.poll() is None


def test_a_genuine_orphan_is_still_stopped(tmp_path):
    orphan = start_group("exec sleep 60")
    threading.Thread(target=orphan.wait, daemon=True).start()
    runner.write_state(tmp_path, {"controller_pid": 999999, "services": {
        "llm": {"status": "running", "pid": orphan.pid, "started": runner.process_start(orphan.pid), "port": 1, "health": "/"}}})
    assert runner.stop_from_state(tmp_path, out=lambda _: None, grace=5) is True
    assert not runner.group_alive(orphan.pid)


def test_the_linux_stamp_reads_the_start_time_even_when_the_command_name_has_spaces_and_parentheses():
    # a real /proc/PID/stat line (field 22, the start time, is 987654); the name is "my (odd) prog"
    stat = "1234 (my (odd) prog) S 1 1234 1234 0 -1 4194560 100 0 0 0 1 2 0 0 20 0 1 0 987654 12345678 900 18446744073709551615"
    assert runner.linux_stamp("3f2a-boot\n", stat) == "3f2a-boot:987654"
    assert runner.linux_stamp("other-boot", stat) != runner.linux_stamp("3f2a-boot", stat)  # a reboot changes the stamp


def test_a_short_or_damaged_stat_line_is_not_a_stamp():
    with pytest.raises(IndexError):
        runner.linux_stamp("boot", "1234 (x) S 1")


# --- what counts as "up" ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("code", [200, 204, 301, 401, 404, 405])
def test_a_response_below_500_means_the_server_is_up(code):
    assert runner.is_up(code)


@pytest.mark.parametrize("code", [None, 500, 502, 503])
def test_no_answer_or_a_server_error_means_not_ready_yet(code):
    assert not runner.is_up(code)  # 503 is what servers answer while a model is still loading


def wait_with_answers(answers, timeout_s=100):
    """Run wait_healthy against a scripted series of probe answers, with a fake clock."""
    svc = Service(name="s", port=1, gpus=[0], start_timeout_s=timeout_s, health="/health")
    now, seen = [0.0], []
    replies = iter(answers)

    def probe_fn(port, path):
        seen.append(now[0])
        return next(replies, answers[-1])

    class Alive:
        def poll(self):
            return None

    runner.wait_healthy(svc, Alive(), Path("s.log"), interval=5, probe_fn=probe_fn, clock=lambda: now[0],
                        sleep=lambda s: now.__setitem__(0, now[0] + s))
    return seen


def test_wait_healthy_keeps_waiting_through_503_until_the_server_is_ready():
    assert wait_with_answers([None, 503, 503, 200]) == [0.0, 5.0, 10.0, 15.0]


def test_a_service_that_only_ever_answers_503_times_out_and_the_message_says_so(tmp_path):
    with pytest.raises(runner.StartTimeout, match=r"within 20s \(last answer: HTTP 503\)"):
        wait_with_answers([503], timeout_s=20)


def test_a_timeout_with_no_answer_at_all_does_not_mention_one():
    with pytest.raises(runner.StartTimeout) as excinfo:
        wait_with_answers([None], timeout_s=20)
    assert "last answer" not in str(excinfo.value)


def test_the_next_service_waits_for_a_server_that_answers_503_while_it_loads(tmp_path):
    port = free_port()
    script = tmp_path / "loading_server.py"
    script.write_text(
        "import http.server, time\n"
        "t0 = time.time()\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(503 if time.time() - t0 < 1.5 else 200)\n"
        "        self.end_headers()\n"
        "    def log_message(self, *args):\n"
        "        pass\n"
        f"http.server.HTTPServer(('127.0.0.1', {port}), H).serve_forever()\n"
    )
    slow = Service(name="a", port=port, gpus=[0], health="/", start_timeout_s=30, command=f"exec {sys.executable} {script}")
    config = Config(services=[slow, http_service("b", free_port(), tmp_path)])
    began, started_b = time.monotonic(), []
    runner.run_up(config, tmp_path / "logs", hold=lambda: False, health_interval=0.2, grace=5,
                  out=lambda line: started_b.append(time.monotonic() - began) if "starting b" in line else None)
    assert started_b[0] >= 1.4  # b did not start while a was still answering 503


def test_the_status_row_of_a_loading_service_is_not_up(sleeper):
    state = {"services": {"a": {"status": "starting", "pid": sleeper.pid, "port": 1, "health": "/"}}}
    [row] = runner.service_rows(state, probe_fn=fake_probe(503))
    assert (row.state, row.health, row.up) == ("starting", "HTTP 503", False)
    [row] = runner.service_rows(state, probe_fn=fake_probe(200))
    assert (row.state, row.up) == ("running", True)


# --- a port that something else already holds --------------------------------------------------------------------


def test_port_in_use_sees_a_listener_and_a_free_port():
    import socket

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        assert runner.port_in_use(listener.getsockname()[1])
    assert not runner.port_in_use(free_port())


def test_up_refuses_to_start_when_a_port_is_already_taken_and_starts_nothing(tmp_path):
    import socket

    config = Config(services=[http_service("a", free_port(), tmp_path), http_service("b", 1, tmp_path)])
    with socket.socket() as squatter:
        squatter.bind(("127.0.0.1", 0))
        squatter.listen()
        config.services[1].port = squatter.getsockname()[1]
        with pytest.raises(runner.RunError, match=r"port \d+ \(b\) is already in use"):
            runner.run_up(config, tmp_path / "logs", out=lambda _: None, health_interval=0.2, grace=5)
    assert not (tmp_path / "logs" / "a.log").exists() and runner.read_state(tmp_path / "logs") is None


# --- a service that does not use the GPU it was planned on ------------------------------------------------------------


def planned_on(*gpus, command="exec vllm serve m"):
    return Service(name="llm", port=8001, gpus=list(gpus), command=command)


def test_no_warning_when_the_service_added_memory_on_its_gpus():
    before, after = {0: 100, 1: 100}, {0: 100, 1: 6500}
    assert runner.wrong_gpu_warning(planned_on(1), before, after) is None


def test_a_listed_gpu_that_gained_nothing_is_reported_with_the_gpu_that_grew_instead():
    before, after = {0: 100, 1: 100}, {0: 16360, 1: 103}
    warning = runner.wrong_gpu_warning(planned_on(1), before, after)
    assert warning == (
        "llm added almost no memory on GPU 1 (+3 MiB); GPU 0 grew by ~16260 MiB meanwhile. "
        "The plan assumes llm uses GPU 1: make its command use it too, for example CUDA_VISIBLE_DEVICES=1."
    )


def test_a_listed_gpu_that_gained_nothing_is_reported_even_if_no_other_gpu_grew():
    warning = runner.wrong_gpu_warning(planned_on(1), {0: 100, 1: 100}, {0: 100, 1: 100})
    assert warning and "added almost no memory on GPU 1 (+0 MiB)" in warning and "grew" not in warning


def test_only_the_listed_gpus_that_gained_nothing_are_named():
    before, after = {0: 100, 1: 100, 2: 100}, {0: 8000, 1: 105, 2: 100}
    warning = runner.wrong_gpu_warning(planned_on(0, 1), before, after)
    assert warning and "on GPU 1 (+5 MiB)" in warning and "GPU 0 (" not in warning
    assert "CUDA_VISIBLE_DEVICES=0,1" in warning


def test_a_small_service_that_really_used_the_gpu_is_not_flagged():
    assert runner.wrong_gpu_warning(planned_on(0), {0: 100}, {0: 100 + runner.MIN_ADDED_MIB}) is None


def test_memory_that_went_down_counts_as_nothing_added():
    warning = runner.wrong_gpu_warning(planned_on(0), {0: 5000}, {0: 4950})
    assert warning and "GPU 0 (-50 MiB)" in warning


def test_a_docker_service_is_told_to_use_the_gpus_option_not_cuda_visible_devices():
    warning = runner.wrong_gpu_warning(planned_on(1, command="exec docker run --rm --name llm img"), {1: 0}, {1: 0})
    assert warning and "--gpus" in warning and "device=1" in warning and "CUDA_VISIBLE_DEVICES" not in warning


def test_gpus_missing_from_the_readings_are_ignored():
    assert runner.wrong_gpu_warning(planned_on(3), {0: 1}, {0: 9000}) is None


def test_up_prints_the_warning_under_the_is_up_line(tmp_path):
    port = free_port()
    config = Config(services=[http_service("a", port, tmp_path)])  # planned on GPU 0
    readings = iter([{0: 100, 1: 100}, {0: 100, 1: 16000}])
    lines = []
    runner.run_up(config, tmp_path / "logs", snapshot=lambda: next(readings), hold=lambda: False, out=lines.append,
                  health_interval=0.2, grace=5)
    up = next(i for i, l in enumerate(lines) if "a is up" in l)
    assert lines[up + 1].startswith("warning: a added almost no memory on GPU 0")


# --- a failed memory reading must not undo the start ------------------------------------------------------------------


def test_a_memory_reading_that_fails_does_not_abort_the_start(tmp_path):
    config = Config(services=[http_service("a", free_port(), tmp_path), http_service("b", free_port(), tmp_path)])
    lines = []
    runner.run_up(config, tmp_path / "logs", snapshot=lambda: None, hold=lambda: False, out=lines.append,
                  health_interval=0.2, grace=5)
    assert [l for l in lines if " is up" in l] == [
        "      a is up  (memory it added: unavailable)",
        "      b is up  (memory it added: unavailable)",
    ]
    assert not any(l.startswith("warning") for l in lines) and lines[-1] == "All 2 services stopped."


def test_the_next_service_is_measured_again_once_readings_work(tmp_path):
    config = Config(services=[http_service("a", free_port(), tmp_path), http_service("b", free_port(), tmp_path)])
    readings = iter([None, None, {0: 100}, {0: 6000}])  # a: both readings fail; b: both work
    lines = []
    runner.run_up(config, tmp_path / "logs", snapshot=lambda: next(readings), hold=lambda: False, out=lines.append,
                  health_interval=0.2, grace=5)
    ups = [l for l in lines if " is up" in l]
    assert ups == ["      a is up  (memory it added: unavailable)", "      b is up  (memory it added: GPU 0: ~5900 MiB)"]


def test_only_the_reading_after_the_start_failing_is_also_unavailable(tmp_path):
    config = Config(services=[http_service("a", free_port(), tmp_path)])
    readings = iter([{0: 100}, None])
    lines = []
    runner.run_up(config, tmp_path / "logs", snapshot=lambda: next(readings), hold=lambda: False, out=lines.append,
                  health_interval=0.2, grace=5)
    assert "      a is up  (memory it added: unavailable)" in lines


def test_a_service_on_a_gpu_missing_from_the_readings_says_unavailable_not_an_empty_list(tmp_path):
    config = Config(services=[http_service("a", free_port(), tmp_path)])  # planned on GPU 0
    lines = []
    runner.run_up(config, tmp_path / "logs", snapshot=lambda: {5: 100}, hold=lambda: False, out=lines.append,
                  health_interval=0.2, grace=5)
    assert "      a is up  (memory it added: unavailable)" in lines


# --- files that hold commands and logs are private to the user ----------------------------------------------------------


@pytest.fixture
def open_umask():
    """A umask that hides nothing, so a file is only private if mesh made it so."""
    old = os.umask(0)
    yield
    os.umask(old)


def mode(path) -> int:
    import stat

    return stat.S_IMODE(os.stat(path).st_mode)


def test_the_state_file_is_private(tmp_path, open_umask):
    runner.write_state(tmp_path, {"services": {}})
    assert mode(tmp_path / "state.json") == 0o600


def test_a_leftover_temporary_state_file_does_not_loosen_the_new_one(tmp_path, open_umask):
    (tmp_path / "state.tmp").write_text("old")
    os.chmod(tmp_path / "state.tmp", 0o644)
    runner.write_state(tmp_path, {"services": {}})
    assert mode(tmp_path / "state.json") == 0o600


def test_service_logs_are_private_and_an_older_loose_log_is_tightened(tmp_path, open_umask):
    svc = Service(name="a", port=1, gpus=[0], command="echo hello")
    (tmp_path / "a.log").write_text("from an earlier version")
    os.chmod(tmp_path / "a.log", 0o644)
    runner.start_service(svc, tmp_path).wait()
    assert mode(tmp_path / "a.log") == 0o600
    assert (tmp_path / "a.log").read_text() == "hello\n"  # truncated and rewritten, as before


def test_a_log_folder_created_by_up_is_private_and_an_existing_one_is_left_alone(tmp_path, open_umask):
    config = Config(services=[http_service("a", free_port(), tmp_path)])
    fresh = tmp_path / "fresh"
    runner.run_up(config, fresh, hold=lambda: False, out=lambda _: None, health_interval=0.2, grace=5)
    assert mode(fresh) == 0o700
    shared = tmp_path / "shared"
    shared.mkdir()
    os.chmod(shared, 0o755)
    runner.run_up(config, shared, hold=lambda: False, out=lambda _: None, health_interval=0.2, grace=5)
    assert mode(shared) == 0o755  # someone's own choice is not changed


# --- the two "is it still there" checks behave the same way ------------------------------------------------------------


@pytest.mark.parametrize("check, signal_function", [("pid_alive", "kill"), ("group_alive", "killpg")])
def test_alive_checks_say_no_for_a_missing_process_and_yes_when_it_belongs_to_someone_else(monkeypatch, check, signal_function):
    function = getattr(runner, check)

    def missing(pid, sig):
        raise ProcessLookupError

    def not_ours(pid, sig):
        raise PermissionError

    monkeypatch.setattr(runner.os, signal_function, missing)
    assert function(123) is False
    monkeypatch.setattr(runner.os, signal_function, not_ours)
    assert function(123) is True  # it exists, we just may not signal it
    monkeypatch.setattr(runner.os, signal_function, lambda pid, sig: None)
    assert function(123) is True


def test_alive_checks_agree_with_a_real_process(sleeper):
    assert runner.pid_alive(sleeper.pid) and runner.group_alive(sleeper.pid)
    assert not runner.pid_alive(2_000_000_000) and not runner.group_alive(2_000_000_000)


def test_progress_is_reported_with_the_service_name(tmp_path):
    seen = []
    config = Config(services=[http_service("a", free_port(), tmp_path)])
    slow = config.services[0]
    slow.command = f"sleep 1; {slow.command}"
    runner.run_up(config, tmp_path / "logs", hold=lambda: False, out=lambda _: None, health_interval=0.2, grace=5,
                  progress=lambda name, seconds, line: seen.append((name, isinstance(seconds, float), isinstance(line, str))))
    assert ("a", True, True) in seen and seen[-1] == (None, True, True)


# --- a service whose main process dies must not leave workers behind ----------------------------------------------------


def test_when_a_service_main_process_is_killed_what_is_left_of_it_is_stopped_too(tmp_path):
    """A vLLM service with tensor parallelism has worker processes in the same group; killed alone, they keep the GPU memory."""
    port, pidfile = free_port(), tmp_path / "worker.pid"
    svc = Service(name="a", port=port, gpus=[0], health="/", start_timeout_s=30,
                  command=f"sleep 300 & echo $! > {pidfile}; exec {sys.executable} -m http.server {port} --bind 127.0.0.1 --directory {tmp_path}")
    seen = {}

    def hold():
        if "leader" not in seen:
            seen["leader"] = runner.read_state(tmp_path / "logs")["services"]["a"]["pid"]
            seen["worker"] = int(pidfile.read_text())
            os.kill(seen["leader"], signal.SIGKILL)  # only the main process, as `kill -9 <pid>` does
            seen["until"] = time.monotonic() + 15
            return True
        if runner.pid_alive(seen["worker"]) and time.monotonic() < seen["until"]:
            time.sleep(0.1)
            return True
        seen["worker_alive_while_holding"] = runner.pid_alive(seen["worker"])
        return False

    lines = []
    runner.run_up(Config(services=[svc]), tmp_path / "logs", hold=hold, out=lines.append, health_interval=0.2, grace=5)
    assert seen["worker_alive_while_holding"] is False, "the worker was left running, still holding its memory"
    assert any(l.startswith("warning: a exited with status -9") for l in lines)
    assert "  stopped what was left of a" in lines


def test_a_service_that_leaves_nothing_behind_gets_no_extra_message(tmp_path):
    config = Config(services=[one_request_service("a", free_port(), tmp_path)])  # answers one request, then exits cleanly
    lines = []

    def hold():
        time.sleep(0.2)
        return not any(l.startswith("warning: a exited") for l in lines)

    runner.run_up(config, tmp_path / "logs", hold=hold, out=lines.append, health_interval=0.2, grace=5)
    assert not any("what was left" in l for l in lines)


# --- the PID column shows the process ids that hold the GPU memory ------------------------------------------------------


def row(**fields):
    defaults = dict(name="a", state="running", pid=100, port=1, health="HTTP 200")
    return runner.ServiceRow(**{**defaults, **fields})


def test_the_shown_pid_is_the_gpu_process_when_there_is_one():
    assert row(gpu_pids=(235930,), gpu_known=True).shown_pid == "235930"
    assert row(gpu_pids=(555362, 555363), gpu_known=True).shown_pid == "555362, 555363"
    assert row(gpu_pids=(1, 2, 3, 4, 5), gpu_known=True).shown_pid == "1, 2, 3, …"


def test_the_shown_pid_is_a_dash_until_the_service_has_a_gpu_process():
    assert row(state="starting", gpu_pids=(), gpu_known=True).shown_pid == "-"  # loading: the engine has not claimed the GPU yet
    assert row(state="pending", pid=None, gpu_known=True).shown_pid == "-"


def test_the_shown_pid_falls_back_to_the_process_mesh_started_only_when_nvidia_smi_cannot_be_read():
    assert row(gpu_known=False).shown_pid == "100"
    assert row(gpu_known=False, pid=None).shown_pid == "-"


def test_service_rows_attach_the_gpu_pids_of_each_service(sleeper):
    state = {"services": {
        "llm": {"status": "running", "pid": sleeper.pid, "port": 1, "health": "/"},
        "idle": {"status": "running", "pid": sleeper.pid, "port": 2, "health": "/"},
    }}
    llm, idle = runner.service_rows(state, probe_fn=fake_probe(), gpu_pids={"llm": [555362, 555363]})
    assert (llm.shown_pid, idle.shown_pid) == ("555362, 555363", "-")
    unknown = runner.service_rows(state, probe_fn=fake_probe(), gpu_pids=None)
    assert unknown[0].shown_pid == str(sleeper.pid)  # no nvidia-smi: the only id there is


# --- how long each service gets to stop before it is killed -------------------------------------------------------------


def stubborn_http_service(name, port, directory, stop_timeout_s):
    """Ignores SIGTERM, so it only goes away when it is killed once its stop timeout has run out."""
    return Service(name=name, port=port, gpus=[0], health="/", start_timeout_s=30, stop_timeout_s=stop_timeout_s,
                   command=f"trap '' TERM; exec {sys.executable} -m http.server {port} --bind 127.0.0.1 --directory {directory}")


def test_each_service_gets_its_own_stop_timeout(tmp_path):
    config = Config(services=[stubborn_http_service("a", free_port(), tmp_path, 1), http_service("b", free_port(), tmp_path)])
    lines = []
    began = time.monotonic()
    runner.run_up(config, tmp_path / "logs", hold=lambda: False, out=lines.append, health_interval=0.2)  # no grace argument
    assert "  stopped a (force-killed after 1s)" in lines
    assert time.monotonic() - began < 15  # not the default 30 seconds


def test_a_longer_stop_timeout_lets_a_slow_service_finish_instead_of_being_killed(tmp_path):
    marker = tmp_path / "finished"
    slow = Service(name="a", port=free_port(), gpus=[0], health="/", start_timeout_s=30, stop_timeout_s=10,
                   command=f"trap 'sleep 2; touch {marker}; exit 0' TERM; {sys.executable} -m http.server {{port}} --bind 127.0.0.1 --directory {tmp_path} & wait")
    slow.command = slow.command.replace("{port}", str(slow.port))
    lines = []
    runner.run_up(Config(services=[slow]), tmp_path / "logs", hold=lambda: False, out=lines.append, health_interval=0.2)
    assert marker.exists()  # it had time to finish its own shutdown
    assert not any("force-killed" in l for l in lines)


def test_the_grace_argument_still_overrides_the_per_service_value(tmp_path):
    config = Config(services=[stubborn_http_service("a", free_port(), tmp_path, 25)])
    lines = []
    runner.run_up(config, tmp_path / "logs", hold=lambda: False, out=lines.append, health_interval=0.2, grace=2)
    assert "  stopped a (force-killed after 2s)" in lines


def test_the_state_records_each_services_stop_timeout(tmp_path):
    config = Config(services=[http_service("a", free_port(), tmp_path, stop_timeout_s=7)])
    seen = {}

    def hold():
        seen["state"] = runner.read_state(tmp_path / "logs")
        return False

    runner.run_up(config, tmp_path / "logs", hold=hold, out=lambda _: None, health_interval=0.2, grace=5)
    assert seen["state"]["services"]["a"]["stop_timeout_s"] == 7


def test_mesh_down_uses_the_recorded_stop_timeout_when_mesh_up_is_gone(tmp_path):
    orphan = start_stubborn(tmp_path)
    threading.Thread(target=orphan.wait, daemon=True).start()  # in real use init reaps an orphan
    runner.write_state(tmp_path, {"controller_pid": 999999, "services": {
        "a": {"status": "running", "pid": orphan.pid, "started": runner.process_start(orphan.pid), "port": 1, "health": "/",
              "stop_timeout_s": 1}}})
    lines = []
    began = time.monotonic()
    assert runner.stop_from_state(tmp_path, out=lines.append) is True  # no grace argument either
    assert "  stopped a (force-killed after 1s)" in lines
    assert time.monotonic() - began < 15


def test_a_state_file_from_an_older_version_still_gets_the_default_timeout(tmp_path):
    orphan = start_group("exec sleep 60")
    threading.Thread(target=orphan.wait, daemon=True).start()
    runner.write_state(tmp_path, {"controller_pid": 999999, "services": {
        "a": {"status": "running", "pid": orphan.pid, "port": 1, "health": "/"}}})  # no stop_timeout_s recorded
    assert runner.stop_from_state(tmp_path, out=lambda _: None) is True
    assert not runner.group_alive(orphan.pid)


def test_mesh_down_waits_for_mesh_up_long_enough_to_cover_the_services_stop_timeouts():
    assert runner.mesh_up_wait(None, [30, 30]) == 90.0  # never less than the usual 90 seconds
    assert runner.mesh_up_wait(None, [120, 120, 120]) == 390.0  # their sum, plus 30 seconds to spare
    assert runner.mesh_up_wait(5.0, [120, 120, 120]) == 5.0  # an explicit value is used as given


def worker_running(pid):
    """True if the process exists and is not just a zombie waiting to be collected."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def test_what_a_dead_service_leaves_behind_is_stopped_within_its_own_timeout(tmp_path):
    port, pidfile = free_port(), tmp_path / "worker.pid"
    svc = Service(name="a", port=port, gpus=[0], health="/", start_timeout_s=30, stop_timeout_s=1,
                  command=f"trap '' TERM; sleep 300 & echo $! > {pidfile}; exec {sys.executable} -m http.server {port} --bind 127.0.0.1 --directory {tmp_path}")
    seen = {}

    def hold():
        if "leader" not in seen:
            seen["leader"] = runner.read_state(tmp_path / "logs")["services"]["a"]["pid"]
            seen["worker"] = int(pidfile.read_text())
            os.kill(seen["leader"], signal.SIGKILL)
            seen["until"] = time.monotonic() + 10  # far less than the default 30 seconds
            return True
        if worker_running(seen["worker"]) and time.monotonic() < seen["until"]:
            time.sleep(0.1)
            return True
        seen["still_there"] = worker_running(seen["worker"])
        return False

    runner.run_up(Config(services=[svc]), tmp_path / "logs", hold=hold, out=lambda _: None, health_interval=0.2)
    assert seen["still_there"] is False


def test_a_service_without_its_own_timeout_keeps_the_default_even_after_one_that_has_one(tmp_path):
    first, second = start_stubborn(tmp_path), start_stubborn(tmp_path)
    lines = []
    runner.stop_all([("a", first.pid, None, first), ("b", second.pid, None, second)], 2, lines.append, graces={"a": 1})
    assert "  stopped a (force-killed after 1s)" in lines
    assert "  stopped b (force-killed after 2s)" in lines  # not 1s, left over from a


# --- the memory is checked again right before each service starts ----------------------------------------------------


def test_recheck_runs_before_each_service_in_order(tmp_path):
    config = Config(services=[http_service("a", free_port(), tmp_path), http_service("b", free_port(), tmp_path)])
    seen = []
    runner.run_up(config, tmp_path / "logs", hold=lambda: False, out=lambda _: None, health_interval=0.2, grace=5,
                  recheck=lambda svc: seen.append(svc.name))
    assert seen == ["a", "b"]


def test_a_failed_recheck_stops_the_start_and_what_was_started(tmp_path):
    a, b = free_port(), free_port()
    config = Config(services=[http_service("a", a, tmp_path), http_service("b", b, tmp_path)])
    lines = []

    def recheck(svc):
        if svc.name == "b":
            raise runner.NotEnoughMemory("b would not fit right now", {0})

    logs = tmp_path / "logs"
    with pytest.raises(runner.NotEnoughMemory) as caught:
        runner.run_up(config, logs, hold=lambda: False, out=lines.append, health_interval=0.2, grace=5, recheck=recheck)
    assert caught.value.gpus == {0}
    assert "      a is up" in "\n".join(lines) and "Stopping 1 service..." in lines
    assert not runner.log_path(logs, "b").exists()  # b was never launched
    assert not runner.port_in_use(a) and runner.read_state(logs) is None  # a was stopped and the state removed


def test_a_failed_recheck_is_recorded_as_the_reason_while_the_stack_stops(tmp_path):
    config = Config(services=[http_service("a", free_port(), tmp_path)])
    logs = tmp_path / "logs"
    states = []

    def recheck(svc):
        raise runner.NotEnoughMemory("a would not fit right now: GPU 0 needs 9 MiB free but only 1 MiB is free", {0})

    original = runner.clear_state
    runner.clear_state = lambda d: (states.append(runner.read_state(d)), original(d))[1]
    try:
        with pytest.raises(runner.NotEnoughMemory):
            runner.run_up(config, logs, hold=lambda: False, out=lambda _: None, recheck=recheck)
    finally:
        runner.clear_state = original
    assert states[0]["services"]["a"]["status"] == "failed"
    assert states[0]["services"]["a"]["error"].startswith("a would not fit right now")


# --- a memory error anywhere in the log is found, not only in its last lines -----------------------------------------


def vllm_style_log(cause: str, lines_after: int = 60) -> str:
    """A start-up log where the real cause is far above the traceback that ends it."""
    return "\n".join(
        ["INFO starting", cause, *[f"(APIServer pid=1) traceback line {i}" for i in range(lines_after)],
         "RuntimeError: Engine core initialization failed. See root cause above."]
    ) + "\n"


def test_a_memory_error_far_above_the_last_lines_is_found(tmp_path):
    log = tmp_path / "svc.log"
    log.write_text(vllm_style_log("ValueError: Free memory on device cuda:0 (3/23 GiB) on startup is less than desired"))
    assert runner.log_has_memory_error(log)


def test_a_log_without_a_memory_error_is_not_one(tmp_path):
    log = tmp_path / "svc.log"
    log.write_text(vllm_style_log("KeyError: 'model'"))
    assert not runner.log_has_memory_error(log)


def test_the_memory_markers_are_matched_in_any_case(tmp_path):
    log = tmp_path / "svc.log"
    log.write_text("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB\n")
    assert runner.log_has_memory_error(log)


def test_a_missing_log_has_no_memory_error(tmp_path):
    assert not runner.log_has_memory_error(tmp_path / "nothing.log")


def test_only_the_end_of_a_huge_log_is_read(tmp_path):
    log = tmp_path / "svc.log"
    log.write_text("Free memory on device\n" + "x" * (runner.LOG_SCAN_BYTES + 10) + "\n")
    assert not runner.log_has_memory_error(log)  # the marker is older than what is scanned


def test_a_service_that_exits_carries_its_log_file(tmp_path):
    svc = Service(name="a", port=free_port(), gpus=[0], health="/", start_timeout_s=30, command="exit 3")
    log = tmp_path / "a.log"
    proc = subprocess.Popen(["bash", "-c", "exit 3"])
    proc.wait()
    with pytest.raises(runner.ServiceExited) as caught:
        runner.wait_healthy(svc, proc, log, interval=0.1)
    assert caught.value.log_file == log


# --- the error shown for a service that exited includes the cause, even when it is far above the end of the log --------


CAUSE = "ValueError: Free memory on device cuda:0 (3/23 GiB) on startup is less than desired GPU memory utilization"


def test_excerpt_adds_the_cause_found_above_the_last_lines(tmp_path):
    log = tmp_path / "svc.log"
    log.write_text(vllm_style_log(CAUSE))
    text = runner.failure_excerpt(log)
    assert text.index(CAUSE) < text.index("Last log lines:")
    assert text.endswith("RuntimeError: Engine core initialization failed. See root cause above.")
    assert "Cause found earlier in the log" in text


def test_excerpt_shows_only_the_last_lines_when_the_cause_is_among_them(tmp_path):
    log = tmp_path / "svc.log"
    log.write_text("\n".join(["INFO starting", *[f"line {i}" for i in range(50)], CAUSE]) + "\n")
    text = runner.failure_excerpt(log)
    assert "Cause found earlier" not in text and text.count(CAUSE) == 1 and text.startswith("Last log lines:")


def test_excerpt_uses_the_first_error_line_when_there_is_no_memory_error(tmp_path):
    log = tmp_path / "svc.log"
    log.write_text(vllm_style_log("ERROR 10-28 [core.py:1] EngineCore failed to start"))
    text = runner.failure_excerpt(log)
    assert text.index("EngineCore failed to start") < text.index("Last log lines:")


def test_excerpt_prefers_a_memory_error_over_an_earlier_unrelated_error(tmp_path):
    log = tmp_path / "svc.log"
    log.write_text("\n".join(["ERROR something harmless was retried", CAUSE,
                              *[f"traceback {i}" for i in range(40)], "RuntimeError: failed"]) + "\n")
    text = runner.failure_excerpt(log)
    assert text.splitlines()[1] == CAUSE


def test_excerpt_has_only_the_last_lines_when_nothing_looks_like_an_error(tmp_path):
    log = tmp_path / "svc.log"
    log.write_text("\n".join(f"just a line {i}" for i in range(100)) + "\n")
    text = runner.failure_excerpt(log)
    assert text.startswith("Last log lines:\njust a line 80") and text.endswith("just a line 99")


def test_excerpt_cuts_very_long_lines(tmp_path):
    log = tmp_path / "svc.log"
    log.write_text("\n".join([CAUSE + " " + "x" * 5000, *[f"line {i}" for i in range(40)]]) + "\n")
    cause = runner.failure_excerpt(log).splitlines()[1]
    assert len(cause) <= runner.LINE_WIDTH and cause.endswith("…")


def test_excerpt_of_a_missing_or_empty_log_does_not_fail(tmp_path):
    assert runner.failure_excerpt(tmp_path / "nothing.log") == "Last log lines:\n"
    empty = tmp_path / "empty.log"
    empty.write_text("")
    assert runner.failure_excerpt(empty) == "Last log lines:\n"


def test_a_service_that_exits_reports_the_cause_and_a_clean_first_line(tmp_path):
    svc = Service(name="a", port=free_port(), gpus=[0], health="/", start_timeout_s=30, command="exit 1")
    log = tmp_path / "a.log"
    log.write_text(vllm_style_log(CAUSE))
    proc = subprocess.Popen(["bash", "-c", "exit 1"])
    proc.wait()
    with pytest.raises(runner.ServiceExited) as caught:
        runner.wait_healthy(svc, proc, log, interval=0.1)
    message = str(caught.value)
    assert message.splitlines()[0] == "a exited with status 1 before it became healthy."
    assert CAUSE in message and "Last log lines:" in message


# --- the cause block shows the cause, not the traceback frames after it ----------------------------------------------


def block_of(tmp_path, cause_and_after):
    log = tmp_path / "svc.log"
    log.write_text("\n".join(["INFO starting", *cause_and_after, *[f"filler {i}" for i in range(40)], "RuntimeError: see above"]) + "\n")
    text = runner.failure_excerpt(log)
    return text[len("Cause found earlier in the log:\n"): text.index("\n...\nLast log lines:")].splitlines()


def test_the_cause_block_stops_before_a_traceback(tmp_path):
    block = block_of(tmp_path, ["(EngineCore pid=4) ERROR " + CAUSE, "(EngineCore pid=4) Traceback (most recent call last):",
                                '(EngineCore pid=4)   File "/usr/lib/python3.12/process.py", line 314, in _bootstrap'])
    assert block == ["(EngineCore pid=4) ERROR " + CAUSE]


def test_the_cause_block_keeps_plain_lines_that_continue_the_message(tmp_path):
    block = block_of(tmp_path, [CAUSE, "Decrease the utilization or free memory.", "Then try again.", "a fourth line"])
    assert block == [CAUSE, "Decrease the utilization or free memory.", "Then try again."]


def test_a_cause_that_starts_a_traceback_shows_the_exception_that_ends_it(tmp_path):
    block = block_of(tmp_path, ["Traceback (most recent call last):", '  File "/app/serve.py", line 9, in main',
                                "    load_model(path)", '  File "/app/load.py", line 3, in load_model', "    raise KeyError(name)",
                                "KeyError: 'model'"])
    assert block == ["Traceback (most recent call last):", "KeyError: 'model'"]


def test_a_traceback_without_an_ending_exception_shows_only_its_first_line(tmp_path):
    log = tmp_path / "svc.log"
    log.write_text("\n".join(["Traceback (most recent call last):", '  File "/app/serve.py", line 9, in main',
                              *[f"plain line {i}" for i in range(40)]]) + "\n")
    text = runner.failure_excerpt(log)
    assert text.startswith("Cause found earlier in the log:\nTraceback (most recent call last):\n...\nLast log lines:")


# --- one `mesh up` per state folder -----------------------------------------------------------------------------------


def test_a_second_run_up_in_the_same_folder_is_refused_and_leaves_the_first_alone(tmp_path):
    logs = tmp_path / "logs"
    release = threading.Event()
    first = threading.Thread(
        target=runner.run_up, args=(Config(services=[http_service("a", free_port(), tmp_path)]), logs),
        kwargs={"hold": lambda: not release.is_set(), "out": lambda _: None, "health_interval": 0.2, "grace": 5},
    )
    first.start()
    try:
        wait_for(lambda: ((runner.read_state(logs) or {}).get("services", {}).get("a") or {}).get("status") == "running", timeout=20)
        other = Config(services=[http_service("b", free_port(), tmp_path)])
        with pytest.raises(runner.RunError, match="another mesh up is already using this folder"):
            runner.run_up(other, logs, hold=lambda: False, out=lambda _: None, health_interval=0.2, grace=5)
        assert list(runner.read_state(logs)["services"]) == ["a"]  # the refused one wrote nothing
        assert not runner.log_path(logs, "b").exists()
    finally:
        release.set()
        first.join(60)


def test_the_folder_can_be_used_again_once_mesh_up_has_finished(tmp_path):
    logs = tmp_path / "logs"
    for name in ("a", "b"):
        runner.run_up(Config(services=[http_service(name, free_port(), tmp_path)]), logs, hold=lambda: False,
                      out=lambda _: None, health_interval=0.2, grace=5)


def test_the_lock_is_free_again_when_the_holder_dies_without_cleaning_up(tmp_path):
    code = "import sys, time; from pathlib import Path; from mesh_cli import runner; runner.take_lock(Path(sys.argv[1])); print('ready', flush=True); time.sleep(60)"
    holder = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "ready"
        with pytest.raises(runner.RunError, match=f"pid {holder.pid}"):
            runner.take_lock(tmp_path)
        holder.kill()  # no cleanup at all, like kill -9
        holder.wait()
        fd = runner.take_lock(tmp_path)
        assert fd is not None
        os.close(fd)
    finally:
        holder.kill()


def test_a_filesystem_that_cannot_lock_does_not_stop_mesh_up(tmp_path, monkeypatch):
    def cannot(fd, operation):
        raise OSError(errno.ENOLCK, "no locks available")

    monkeypatch.setattr(runner.fcntl, "flock", cannot)
    assert runner.take_lock(tmp_path) is None
    runner.run_up(Config(services=[http_service("a", free_port(), tmp_path)]), tmp_path / "logs", hold=lambda: False,
                  out=lambda _: None, health_interval=0.2, grace=5)


def test_each_process_writes_the_state_file_through_a_temporary_file_of_its_own(tmp_path, monkeypatch):
    sources = []
    real = os.replace
    with monkeypatch.context() as patched:
        patched.setattr(runner.os, "replace", lambda source, target: (sources.append(Path(source).name), real(source, target))[1])
        for pid in (111, 222):
            patched.setattr(runner.os, "getpid", lambda pid=pid: pid)
            runner.write_state(tmp_path, {"services": {}})
    assert sources[0] != sources[1] and "111" in sources[0] and "222" in sources[1]
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]  # nothing is left behind


def test_a_state_write_that_fails_leaves_no_temporary_file(tmp_path, monkeypatch):
    def broken(source, target):
        raise OSError("disk full")

    monkeypatch.setattr(runner.os, "replace", broken)
    with pytest.raises(OSError):
        runner.write_state(tmp_path, {"services": {}})
    assert list(tmp_path.iterdir()) == []


def test_a_live_controller_is_found_in_the_state_and_a_dead_or_recycled_one_is_not():
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        stamp = runner.process_start(sleeper.pid)
        assert runner.controller_alive({"controller_pid": sleeper.pid, "controller_started": stamp})
        assert not runner.controller_alive({"controller_pid": sleeper.pid, "controller_started": "another boot:1"})
    finally:
        sleeper.kill()
        sleeper.wait()
    assert not runner.controller_alive({"controller_pid": sleeper.pid, "controller_started": stamp})
    assert not runner.controller_alive({"controller_pid": os.getpid()})  # ourselves do not count
    assert not runner.controller_alive({}) and not runner.controller_alive(None)
