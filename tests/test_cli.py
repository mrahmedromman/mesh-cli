import json
import subprocess
import types
from pathlib import Path

import pytest
import yaml

from mesh_cli import cli as cli_module
from mesh_cli import holders, settings
from mesh_cli.cli import main
from mesh_cli.config import Config
from mesh_cli.gpu import Gpu, GpuError, parse_gpu_query
from mesh_cli.holders import GpuReport, Holder
from mesh_cli.planner import Plan, ServicePlan

FIXTURE = str(Path(__file__).parent / "fixtures" / "gpu_query_running.csv")
EXTERNAL = "0=417,1=4"


def test_plan_ok_for_example_config(capsys):
    code = main(["plan", "examples/dual-gpu.yaml", "--gpus-csv", FIXTURE, "--external", EXTERNAL])
    out = capsys.readouterr().out
    assert code == 0
    assert "llm" in out and "tts" in out
    assert "footprint unknown" not in out  # tts is measured now
    assert "warning: GPU 0 has only" in out  # GPU 0 ends under the 1024 MiB margin


def test_plan_fails_with_suggestion(tmp_path, capsys):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "services:\n"
        "  - {name: llm, port: 8001, gpus: [0, 1], gpu_memory_utilization: 0.6, measured_mib: {0: 16252, 1: 16252}}\n"
        "  - {name: stt, port: 8004, gpus: [0], gpu_memory_utilization: 0.7}\n"
    )
    code = main(["plan", str(cfg), "--gpus-csv", FIXTURE, "--external", EXTERNAL])
    out = capsys.readouterr().out
    assert code == 1
    # usable = 24564 - 485 reserved = 24079; 0.7 -> 16856; free = 24079 - 417 - 16252
    assert "needs 16856 MiB free but only 7410 MiB is free" in out
    assert "--gpu-memory-utilization 0.30" in out


def test_status_with_nothing_running(tmp_path, capsys):
    assert main(["status", "--log-dir", str(tmp_path)]) == 0
    assert "nothing running" in capsys.readouterr().out


def test_logs_prints_the_tail(tmp_path, capsys):
    (tmp_path / "llm.log").write_text("".join(f"line {i}\n" for i in range(100)))
    assert main(["logs", "llm", "-n", "3", "--log-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out == "line 97\nline 98\nline 99\n"


def test_logs_default_is_the_last_50_lines(tmp_path, capsys):
    (tmp_path / "llm.log").write_text("".join(f"line {i}\n" for i in range(100)))
    assert main(["logs", "llm", "--log-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "line 50" and out[-1] == "line 99" and len(out) == 50


def test_logs_n_all_prints_the_whole_file(tmp_path, capsys):
    text = "".join(f"line {i}\n" for i in range(200_000))  # larger than one read chunk
    (tmp_path / "llm.log").write_text(text)
    assert main(["logs", "llm", "-n", "all", "--log-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out == text


def test_logs_n_zero_prints_nothing(tmp_path, capsys):
    (tmp_path / "llm.log").write_text("a\nb\n")
    assert main(["logs", "llm", "-n", "0", "--log-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out == ""


def test_logs_n_rejects_other_words(tmp_path, capsys):
    (tmp_path / "llm.log").write_text("a\n")
    with pytest.raises(SystemExit) as excinfo:
        main(["logs", "llm", "-n", "lots", "--log-dir", str(tmp_path)])
    assert excinfo.value.code == 2
    assert "number of lines or 'all'" in capsys.readouterr().err


def test_logs_for_unknown_service(tmp_path, capsys):
    assert main(["logs", "nope", "--log-dir", str(tmp_path)]) == 1
    assert "no log file" in capsys.readouterr().err


def fake_report():
    gpu = Gpu(0, "GPU-0", "Test GPU", 24564, 100, 24000, 5)
    return [GpuReport(gpu, [Holder("other", "python", "user alice · pid 9 · python x (not started by mesh)", 100)])]


def write_cfg(tmp_path, text):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(text)
    return str(cfg)


def test_gpu_command_prints_the_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(holders, "gather", lambda log_dir, only=None, proc=None: fake_report())
    assert main(["gpu", "--log-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "GPU 0  Test GPU" in out and "5% utilization" in out and "python" in out


def test_up_shows_holders_when_the_plan_does_not_fit(tmp_path, monkeypatch, capsys):
    gpus = parse_gpu_query(Path(FIXTURE).read_text())  # GPU 0 has 23187 MiB in use
    monkeypatch.setattr(cli_module, "read_gpus", lambda: gpus)
    calls = []
    monkeypatch.setattr(holders, "gather", lambda log_dir, only=None, proc=None: calls.append(only) or fake_report())
    cfg = write_cfg(tmp_path, "services:\n  - {name: big, port: 8001, gpus: [0], gpu_memory_utilization: 0.9, command: 'true'}\n")
    assert main(["up", cfg, "--log-dir", str(tmp_path / "logs")]) == 1
    out = capsys.readouterr().out
    assert "Who is holding memory on the GPUs that are short" in out and "python" in out
    assert calls == [{0}]


def idle_gpus():
    return [Gpu(0, "GPU-0", "Test GPU", 24564, 100, 24000)]


def test_up_shows_holders_after_a_memory_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "read_gpus", idle_gpus)
    calls = []
    monkeypatch.setattr(holders, "gather", lambda log_dir, only=None, proc=None: calls.append(only) or fake_report())

    def failing_up(*args, **kwargs):
        raise cli_module.runner.ServiceExited("svc exited with status 1 before it became healthy. Last log lines:\nValueError: Free memory on device cuda:0 (9/23 GiB) on startup is less than desired")

    monkeypatch.setattr(cli_module.runner, "run_up", failing_up)
    cfg = write_cfg(tmp_path, "services:\n  - {name: svc, port: 8001, gpus: [0], gpu_memory_utilization: 0.1, command: 'true'}\n")
    assert main(["up", cfg, "--log-dir", str(tmp_path / "logs")]) == 1
    captured = capsys.readouterr()
    assert "Free memory on device" in captured.err
    assert "Who is holding GPU memory now" in captured.out
    assert calls == [None]


def test_up_does_not_show_holders_for_other_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "read_gpus", idle_gpus)
    calls = []
    monkeypatch.setattr(holders, "gather", lambda log_dir, only=None, proc=None: calls.append(only) or fake_report())

    def failing_up(*args, **kwargs):
        raise cli_module.runner.ServiceExited("svc exited with status 1 before it became healthy. Last log lines:\nKeyError: 'model'")

    monkeypatch.setattr(cli_module.runner, "run_up", failing_up)
    cfg = write_cfg(tmp_path, "services:\n  - {name: svc, port: 8001, gpus: [0], gpu_memory_utilization: 0.1, command: 'true'}\n")
    assert main(["up", cfg, "--log-dir", str(tmp_path / "logs")]) == 1
    assert calls == []
    assert "Who is holding" not in capsys.readouterr().out


def test_gpu_command_reports_a_gpu_error(tmp_path, monkeypatch, capsys):
    def broken(log_dir, only=None, proc=None):
        raise GpuError("nvidia-smi not found")

    monkeypatch.setattr(holders, "gather", broken)
    assert main(["gpu", "--log-dir", str(tmp_path)]) == 2
    assert "nvidia-smi not found" in capsys.readouterr().err


def test_up_still_fails_cleanly_if_the_holder_lookup_breaks(tmp_path, monkeypatch, capsys):
    gpus = parse_gpu_query(Path(FIXTURE).read_text())
    monkeypatch.setattr(cli_module, "read_gpus", lambda: gpus)

    def broken(log_dir, only=None, proc=None):
        raise GpuError("docker is not allowed")

    monkeypatch.setattr(holders, "gather", broken)
    cfg = write_cfg(tmp_path, "services:\n  - {name: big, port: 8001, gpus: [0], gpu_memory_utilization: 0.9, command: 'true'}\n")
    assert main(["up", cfg, "--log-dir", str(tmp_path / "logs")]) == 1  # the plan failure is still the result
    out = capsys.readouterr().out
    assert "The plan does not fit" in out and "could not read which processes hold GPU memory" in out


def test_plan_lists_holders_only_for_live_nvidia_smi_reads(tmp_path, monkeypatch, capsys):
    gpus = parse_gpu_query(Path(FIXTURE).read_text())
    monkeypatch.setattr(cli_module, "read_gpus", lambda: gpus)
    calls = []
    monkeypatch.setattr(holders, "gather", lambda log_dir, only=None, proc=None: calls.append(only) or fake_report())
    cfg = write_cfg(tmp_path, "services:\n  - {name: a, port: 1, gpus: [0], measured_mib: {0: 10}}\n")
    main(["plan", cfg, "--log-dir", str(tmp_path)])
    assert calls == [{0, 1}]
    calls.clear()
    main(["plan", cfg, "--gpus-csv", FIXTURE, "--log-dir", str(tmp_path)])  # saved output: no live processes to read
    assert calls == []


def test_bad_config_returns_2(tmp_path, capsys):
    cfg = tmp_path / "models.yaml"
    cfg.write_text("services: []\n")
    assert main(["plan", str(cfg), "--gpus-csv", FIXTURE]) == 2
    assert "error:" in capsys.readouterr().err


def run_up_with_fake_start(tmp_path, monkeypatch, capsys, extra=(), services="  - {name: svc, port: 8001, gpus: [0], gpu_memory_utilization: 0.1, measured_mib: {0: 2500}, command: 'true'}\n"):
    monkeypatch.setattr(cli_module, "read_gpus", idle_gpus)
    monkeypatch.setattr(cli_module.runner, "run_up", lambda *a, **k: None)
    cfg = write_cfg(tmp_path, "services:\n" + services)
    assert main(["up", cfg, "--log-dir", str(tmp_path / "logs"), *extra]) == 0
    return capsys.readouterr().out


def test_up_prints_one_line_when_everything_fits(tmp_path, monkeypatch, capsys):
    out = run_up_with_fake_start(tmp_path, monkeypatch, capsys)
    assert "Plan: the service fits on 1 GPU. Free once it is up: GPU 0: " in out
    assert "After all services start" not in out


def test_up_plan_flag_prints_the_full_plan(tmp_path, monkeypatch, capsys):
    out = run_up_with_fake_start(tmp_path, monkeypatch, capsys, extra=("--plan",))
    assert "After all services start" in out and "Plan: all" not in out


def test_up_prints_the_full_plan_when_there_is_a_warning(tmp_path, monkeypatch, capsys):
    # 0.9 of the card leaves less than the 1024 MiB margin free, so the plan has a warning worth reading
    out = run_up_with_fake_start(tmp_path, monkeypatch, capsys, services="  - {name: svc, port: 8001, gpus: [0], gpu_memory_utilization: 0.95, measured_mib: {0: 23200}, command: 'true'}\n")
    assert "warning: GPU 0 has only" in out and "Plan: all" not in out


# --- the folder for logs and state is remembered, so commands work from any directory ----------------------------


def save_stack(folder):
    """A stack's files as `mesh up` leaves them: one log and a state file."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "llm.log").write_text("hello from llm\n")
    (folder / "state.json").write_text(
        '{"controller_pid": 1, "services": {"llm": {"pid": 1, "port": 8001, "health": "/v1/models"}}}'
    )


def test_status_and_logs_work_from_another_directory(tmp_path, monkeypatch, capsys):
    save_stack(tmp_path / "deck")
    settings.save_folder(tmp_path / "deck")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert main(["status"]) == 0
    assert "llm" in capsys.readouterr().out
    assert main(["logs", "llm"]) == 0
    assert capsys.readouterr().out == "hello from llm\n"


def test_up_asks_where_to_keep_files_the_first_time(tmp_path, monkeypatch, capsys):
    chosen = tmp_path / "deck"
    monkeypatch.setattr(settings, "interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": str(chosen))
    monkeypatch.setattr(cli_module, "read_gpus", idle_gpus)
    seen = []
    monkeypatch.setattr(cli_module.runner, "run_up", lambda config, log_dir, **k: seen.append(log_dir))
    cfg = write_cfg(tmp_path, "services:\n  - {name: svc, port: 8001, gpus: [0], measured_mib: {0: 2500}, command: 'true'}\n")
    assert main(["up", cfg]) == 0
    assert seen == [chosen] and settings.saved_folder() == chosen
    assert main(["up", cfg]) == 0  # the second time nobody is asked
    assert seen == [chosen, chosen]


def test_a_folder_that_cannot_be_used_is_an_error_not_a_crash(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setattr(settings, "interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": str(blocker / "inside"))
    monkeypatch.setattr(cli_module, "read_gpus", idle_gpus)
    cfg = write_cfg(tmp_path, "services:\n  - {name: svc, port: 8001, gpus: [0], measured_mib: {0: 1}, command: 'true'}\n")
    assert main(["up", cfg]) == 2
    assert "cannot create" in capsys.readouterr().err


def test_status_lists_services_that_have_not_started_yet(tmp_path, capsys):
    (tmp_path / "state.json").write_text(
        '{"controller_pid": 1, "services": {'
        '"llm": {"status": "pending", "pid": null, "port": 8001, "health": "/v1/models", "error": null},'
        '"stt": {"status": "failed", "pid": null, "port": 8004, "health": "/v1/models", "error": "stt exited with status 3"}}}'
    )
    assert main(["status", "--log-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].split() == ["llm", "pending", "pid", "-", "port", "8001", "-"]
    assert out[1].split()[:2] == ["stt", "failed"]
    assert "stt: stt exited with status 3  (see: mesh logs stt)" in out[2]


def test_plan_says_when_a_service_would_not_fit_once_it_is_running(tmp_path, capsys):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "services:\n"
        "  - {name: chat, port: 8001, gpus: [0], gpu_memory_utilization: 0.55, overhead_mib: 1500}\n"
        "  - {name: embeddings, port: 8003, gpus: [0], gpu_memory_utilization: 0.35, overhead_mib: 1500}\n"
    )
    code = main(["plan", str(cfg), "--gpus-csv", FIXTURE, "--external", EXTERNAL])
    out = capsys.readouterr().out
    assert code == 1
    assert "will hold 9928 MiB but only 8918 MiB is free" in out
    assert "--gpu-memory-utilization 0.30" in out


def test_up_reports_a_port_that_something_else_holds(tmp_path, monkeypatch, capsys):
    import socket

    monkeypatch.setattr(cli_module, "read_gpus", idle_gpus)
    with socket.socket() as squatter:
        squatter.bind(("127.0.0.1", 0))
        squatter.listen()
        port = squatter.getsockname()[1]
        cfg = write_cfg(tmp_path, f"services:\n  - {{name: svc, port: {port}, gpus: [0], measured_mib: {{0: 1}}, command: 'true'}}\n")
        assert main(["up", cfg, "--log-dir", str(tmp_path / "logs")]) == 1
    assert f"port {port} (svc) is already in use" in capsys.readouterr().err


def test_the_memory_reading_used_by_up_gives_none_instead_of_raising(monkeypatch):
    monkeypatch.setattr(cli_module, "read_gpus", idle_gpus)
    assert cli_module._used_memory() == {0: 100}

    def broken():
        raise GpuError("nvidia-smi failed: timed out")

    monkeypatch.setattr(cli_module, "read_gpus", broken)
    assert cli_module._used_memory() is None


def follow_with(tmp_path, monkeypatch, capsys, between):
    """Run `mesh logs llm -f` for as many waits as `between` has steps; each step changes the log, the last one ends it."""
    steps = iter(between)

    def fake_sleep(seconds):
        try:
            next(steps)()
        except StopIteration:
            raise KeyboardInterrupt from None

    monkeypatch.setattr(cli_module.time, "sleep", fake_sleep)
    assert main(["logs", "llm", "-f", "--log-dir", str(tmp_path)]) == 0
    return capsys.readouterr().out


def test_following_a_log_shows_what_is_appended(tmp_path, monkeypatch, capsys):
    log = tmp_path / "llm.log"
    log.write_text("one\ntwo\n")

    def append():
        with open(log, "a") as handle:
            handle.write("three\n")

    assert follow_with(tmp_path, monkeypatch, capsys, [append]) == "one\ntwo\nthree\n"


def test_following_a_log_that_is_started_over_shows_the_new_log_from_its_first_line(tmp_path, monkeypatch, capsys):
    log = tmp_path / "llm.log"
    log.write_text("old line one\nold line two\nold line three\n")

    def start_over():  # what the next `mesh up` does: the same file, emptied and written again
        with open(log, "w") as handle:
            handle.write("new one\n")

    assert follow_with(tmp_path, monkeypatch, capsys, [start_over]) == (
        "old line one\nold line two\nold line three\nnew one\n"
    )


def test_following_keeps_working_after_the_log_was_started_over(tmp_path, monkeypatch, capsys):
    log = tmp_path / "llm.log"
    log.write_text("old old old old\n")

    def start_over():
        log.write_text("new\n")

    def append():
        with open(log, "a") as handle:
            handle.write("more\n")

    assert follow_with(tmp_path, monkeypatch, capsys, [start_over, append]) == "old old old old\nnew\nmore\n"


def set_service_pid(folder, pid):
    runner_state = {"controller_pid": 1, "services": {"llm": {"status": "running", "pid": pid, "port": 1, "health": "/"}}}
    (folder / "state.json").write_text(__import__("json").dumps(runner_state))


def test_following_shows_a_restarted_log_from_its_first_line_even_if_it_is_already_longer_than_the_old_one(tmp_path, monkeypatch, capsys):
    log = tmp_path / "llm.log"
    log.write_text("old\n")  # a short old log: the new one will be longer than this by the time anyone looks
    set_service_pid(tmp_path, 111)

    def restart():  # the stack was started again: new process id, log started over and already longer than before
        log.write_text("new line one\nnew line two\nnew line three\n")
        set_service_pid(tmp_path, 222)

    assert follow_with(tmp_path, monkeypatch, capsys, [restart]) == "old\nnew line one\nnew line two\nnew line three\n"


def test_following_does_not_start_over_while_the_same_process_keeps_writing(tmp_path, monkeypatch, capsys):
    log = tmp_path / "llm.log"
    log.write_text("one\n")
    set_service_pid(tmp_path, 111)

    def append():
        with open(log, "a") as handle:
            handle.write("two\n")

    assert follow_with(tmp_path, monkeypatch, capsys, [append, append]) == "one\ntwo\ntwo\n"


def test_following_copes_with_the_state_file_being_gone_between_two_runs(tmp_path, monkeypatch, capsys):
    log = tmp_path / "llm.log"
    log.write_text("old old old\n")
    set_service_pid(tmp_path, 111)

    def stopped():  # `mesh up` ended and removed its state file
        (tmp_path / "state.json").unlink()

    def started_again():
        log.write_text("fresh start\nsecond line\n")
        set_service_pid(tmp_path, 222)

    assert follow_with(tmp_path, monkeypatch, capsys, [stopped, started_again]) == "old old old\nfresh start\nsecond line\n"


def test_status_shows_the_gpu_process_ids_when_nvidia_smi_can_be_read(tmp_path, monkeypatch, capsys):
    save_stack(tmp_path)  # one service, llm, whose recorded (launcher) pid is 1
    monkeypatch.setattr(cli_module.holders, "lookup_service_pids", lambda log_dir: {"llm": [235930]})
    assert main(["status", "--log-dir", str(tmp_path)]) == 0
    line = capsys.readouterr().out.splitlines()[0].split()
    assert line[:4] == ["llm", "running", "pid", "235930"]


def test_status_shows_the_process_mesh_started_when_nvidia_smi_cannot_be_read(tmp_path, monkeypatch, capsys):
    save_stack(tmp_path)

    def unreadable(log_dir):
        raise GpuError("nvidia-smi not found")

    monkeypatch.setattr(cli_module.holders, "lookup_service_pids", unreadable)
    assert main(["status", "--log-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.splitlines()[0].split()[:4] == ["llm", "running", "pid", "1"]


# --- the memory is checked again right before each service starts ----------------------------------------------------

TIGHT = "  - {name: svc, port: 8001, gpus: [0], gpu_memory_utilization: 0.5, command: 'true'}\n"


def recheck_passed_to_run_up(tmp_path, monkeypatch, extra=()):
    """Run `mesh up` with run_up faked, and return the recheck callable it was given (None if it got none)."""
    monkeypatch.setattr(cli_module, "read_gpus", idle_gpus)
    given = []
    monkeypatch.setattr(cli_module.runner, "run_up", lambda config, log_dir, **k: given.append(k.get("recheck")))
    cfg = write_cfg(tmp_path, "services:\n" + TIGHT)
    assert main(["up", cfg, "--log-dir", str(tmp_path / "logs"), *extra]) == 0
    return given[0]


def test_up_passes_a_recheck_to_run_up(tmp_path, monkeypatch):
    assert recheck_passed_to_run_up(tmp_path, monkeypatch) is cli_module._recheck_memory


def test_force_starts_without_a_recheck(tmp_path, monkeypatch):
    assert recheck_passed_to_run_up(tmp_path, monkeypatch, extra=("--force",)) is None


def service_of(text):
    return Config.model_validate(yaml.safe_load("services:\n" + text)).services[0]


def test_recheck_passes_while_the_memory_is_free(monkeypatch):
    monkeypatch.setattr(cli_module, "read_gpus", idle_gpus)  # 24000 MiB free, the service needs about 12000
    cli_module._recheck_memory(service_of(TIGHT))


def test_recheck_refuses_when_something_took_the_memory(monkeypatch):
    monkeypatch.setattr(cli_module, "read_gpus", lambda: [Gpu(0, "GPU-0", "Test GPU", 24564, 17000, 7000)])
    with pytest.raises(cli_module.runner.NotEnoughMemory) as caught:
        cli_module._recheck_memory(service_of(TIGHT))
    message = str(caught.value)
    assert "svc would not fit right now, so it was not started" in message
    assert "GPU 0 needs 12000 MiB free but only 7000 MiB is free" in message
    assert "--force" in message and caught.value.gpus == {0}


def test_recheck_counts_the_memory_the_service_will_hold_not_only_its_start_need(monkeypatch):
    # it could start (needs 2457 MiB free) but would then hold 20000 MiB, which is more than is free
    monkeypatch.setattr(cli_module, "read_gpus", lambda: [Gpu(0, "GPU-0", "Test GPU", 24564, 15000, 9000)])
    svc = service_of("  - {name: svc, port: 8001, gpus: [0], gpu_memory_utilization: 0.1, measured_mib: {0: 20000}, command: 'true'}\n")
    with pytest.raises(cli_module.runner.NotEnoughMemory, match="needs 20000 MiB free but only 9000 MiB is free"):
        cli_module._recheck_memory(svc)


def test_recheck_is_skipped_when_nvidia_smi_cannot_be_read(monkeypatch):
    def broken():
        raise GpuError("nvidia-smi failed")

    monkeypatch.setattr(cli_module, "read_gpus", broken)
    cli_module._recheck_memory(service_of(TIGHT))


def test_recheck_is_skipped_when_a_listed_gpu_has_disappeared(monkeypatch):
    monkeypatch.setattr(cli_module, "read_gpus", lambda: [Gpu(1, "GPU-1", "Test GPU", 24564, 100, 24000)])
    cli_module._recheck_memory(service_of(TIGHT))  # the service lists GPU 0, which is gone: nothing to compare


def test_up_reports_a_refused_service_and_shows_who_holds_the_memory(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "read_gpus", idle_gpus)
    shown = []
    monkeypatch.setattr(holders, "gather", lambda log_dir, only=None, proc=None: shown.append(only) or fake_report())

    def refusing_up(config, log_dir, **k):
        raise cli_module.runner.NotEnoughMemory("svc would not fit right now, so it was not started: GPU 0 needs 1 MiB free but only 0 MiB is free.", {0})

    monkeypatch.setattr(cli_module.runner, "run_up", refusing_up)
    cfg = write_cfg(tmp_path, "services:\n" + TIGHT)
    assert main(["up", cfg, "--log-dir", str(tmp_path / "logs")]) == 1
    captured = capsys.readouterr()
    assert "svc would not fit right now" in captured.err
    assert "Who is holding memory on the GPUs that are short" in captured.out
    assert shown == [{0}]


def test_up_shows_holders_when_the_memory_error_is_far_above_the_end_of_the_log(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "read_gpus", idle_gpus)
    calls = []
    monkeypatch.setattr(holders, "gather", lambda log_dir, only=None, proc=None: calls.append(only) or fake_report())
    log = tmp_path / "svc.log"
    log.write_text("ValueError: Free memory on device cuda:0 (3/23 GiB) on startup is less than desired\n"
                   + "".join(f"(APIServer pid=1) traceback line {i}\n" for i in range(60))
                   + "RuntimeError: Engine core initialization failed. See root cause above.\n")

    def failing_up(*args, **kwargs):  # the 20 lines in the message do not show the cause, the log file does
        raise cli_module.runner.ServiceExited("svc exited with status 1 before it became healthy.\nRuntimeError: Engine core initialization failed", log_file=log)

    monkeypatch.setattr(cli_module.runner, "run_up", failing_up)
    cfg = write_cfg(tmp_path, "services:\n  - {name: svc, port: 8001, gpus: [0], gpu_memory_utilization: 0.1, command: 'true'}\n")
    assert main(["up", cfg, "--log-dir", str(tmp_path / "logs")]) == 1
    assert "Who is holding GPU memory now" in capsys.readouterr().out
    assert calls == [None]


# --- wording: plurals, and what --force really does -------------------------------------------------------------------


def test_the_plan_summary_uses_the_right_plurals():
    one = cli_module._plan_summary(types.SimpleNamespace(services=[1], free_after_mib={0: 5}), [1])
    many = cli_module._plan_summary(types.SimpleNamespace(services=[1, 2], free_after_mib={0: 5, 1: 6}), [1, 2])
    assert one == "Plan: the service fits on 1 GPU. Free once it is up: GPU 0: 5 MiB."
    assert many == "Plan: all 2 services fit on 2 GPUs. Free once they are up: GPU 0: 5 MiB, GPU 1: 6 MiB."


def test_the_plain_plan_header_uses_the_right_plurals():
    gpu = [cli_module.GpuBudget(0, 24000, 0, "Card", 500)]
    one = Plan([], {0: 1}, [])
    assert cli_module._render(one, gpu).startswith("Plan for 0 services on 1 GPU\n")
    assert cli_module._render(Plan([ServicePlan("a", [])], {0: 1}, []), gpu).startswith("Plan for 1 service on 1 GPU\n")


def test_force_is_not_offered_as_a_way_to_make_it_fit(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "read_gpus", lambda: [Gpu(0, "GPU-0", "Test GPU", 24564, 20000, 4000)])
    cfg = write_cfg(tmp_path, "services:\n" + TIGHT)
    assert main(["up", cfg, "--log-dir", str(tmp_path / "logs")]) == 1
    out = capsys.readouterr().out
    assert "Fix the config or free the memory." in out
    assert "--force skips this check, but the service will probably still fail." in out
    assert "start anyway" not in out


def test_the_recheck_message_says_what_force_does(monkeypatch):
    monkeypatch.setattr(cli_module, "read_gpus", lambda: [Gpu(0, "GPU-0", "Test GPU", 24564, 17000, 7000)])
    with pytest.raises(cli_module.runner.NotEnoughMemory) as caught:
        cli_module._recheck_memory(service_of(TIGHT))
    message = str(caught.value)
    assert "Free the memory (the table below shows who holds it) or lower gpu_memory_utilization." in message
    assert "--force skips this check, but the service will probably still fail." in message
    assert "start anyway" not in message


def test_the_help_texts_describe_force_and_status():
    parser = cli_module.build_parser()
    commands = next(a for a in parser._actions if hasattr(a, "choices") and a.choices).choices

    def words(text):  # help is wrapped to the terminal width
        return " ".join(text.split())

    top = words(parser.format_help())
    assert "live view of every service" in top and "show running services" not in top
    up = words(commands["up"].format_help())
    assert "skip the memory checks" in up and "usually still fails" in up
    assert "name of a service in your config" in words(commands["logs"].format_help())


# --- a second `mesh up` while one is still starting --------------------------------------------------------------------


def state_with_controller(logs, pid, started):
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "state.json").write_text(json.dumps({"controller_pid": pid, "controller_started": started, "services": {
        "svc": {"status": "pending", "pid": None, "port": 8001, "health": "/"}}}))


def test_up_refuses_while_another_mesh_up_is_still_starting(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "read_gpus", idle_gpus)
    monkeypatch.setattr(cli_module.runner, "run_up", lambda *a, **k: pytest.fail("must not start anything"))
    sleeper = subprocess.Popen(["sleep", "30"])  # stands for the other `mesh up`, which has no service with a process yet
    try:
        logs = tmp_path / "logs"
        state_with_controller(logs, sleeper.pid, cli_module.runner.process_start(sleeper.pid))
        cfg = write_cfg(tmp_path, "services:\n" + TIGHT)
        assert main(["up", cfg, "--log-dir", str(logs)]) == 1
        err = capsys.readouterr().err
        assert "another mesh up" in err and f"pid {sleeper.pid}" in err and "mesh down" in err
    finally:
        sleeper.kill()
        sleeper.wait()


def test_up_goes_ahead_when_the_recorded_mesh_up_is_gone(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_module, "read_gpus", idle_gpus)
    started = []
    monkeypatch.setattr(cli_module.runner, "run_up", lambda config, log_dir, **k: started.append(log_dir))
    gone = subprocess.Popen(["true"])
    gone.wait()
    logs = tmp_path / "logs"
    state_with_controller(logs, gone.pid, "an earlier boot:1")
    cfg = write_cfg(tmp_path, "services:\n" + TIGHT)
    assert main(["up", cfg, "--log-dir", str(logs)]) == 0 and started == [logs]


# --- exit codes of `mesh down` and `mesh up` when something could not be stopped ----------------------------------------


def run_down(monkeypatch, tmp_path, result):
    monkeypatch.setattr(cli_module.runner, "stop_from_state", lambda *a, **k: result)
    return main(["down", "--log-dir", str(tmp_path / "logs")])


def test_down_exits_one_and_says_so_when_a_service_could_not_be_stopped(tmp_path, monkeypatch, capsys):
    assert run_down(monkeypatch, tmp_path, 2) == 1
    err = capsys.readouterr().err
    assert "2 services could not be stopped" in err and "mesh down" in err


def test_down_exits_zero_when_everything_stopped_or_nothing_was_running(tmp_path, monkeypatch, capsys):
    assert run_down(monkeypatch, tmp_path, 0) == 0
    assert run_down(monkeypatch, tmp_path, None) == 0
    assert "nothing running" in capsys.readouterr().out


def test_up_exits_one_when_the_clean_up_left_a_service_running(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "read_gpus", idle_gpus)
    cfg = write_cfg(tmp_path, "services:\n" + TIGHT)
    monkeypatch.setattr(cli_module.runner, "run_up", lambda *a, **k: 1)
    assert main(["up", cfg, "--log-dir", str(tmp_path / "logs")]) == 1
    assert "could not be stopped" in capsys.readouterr().err
    monkeypatch.setattr(cli_module.runner, "run_up", lambda *a, **k: 0)
    assert main(["up", cfg, "--log-dir", str(tmp_path / "logs")]) == 0
