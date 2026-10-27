from pathlib import Path

import pytest

from mesh_cli import cli as cli_module
from mesh_cli import holders, settings
from mesh_cli.cli import main
from mesh_cli.gpu import Gpu, GpuError, parse_gpu_query
from mesh_cli.holders import GpuReport, Holder

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
    assert "Plan: all 1 services fit on 1 GPUs. Free once they are up: GPU 0: " in out
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
