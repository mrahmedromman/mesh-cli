from pathlib import Path

from mesh_cli.cli import main

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


def test_logs_for_unknown_service(tmp_path, capsys):
    assert main(["logs", "nope", "--log-dir", str(tmp_path)]) == 1
    assert "no log file" in capsys.readouterr().err


def test_bad_config_returns_2(tmp_path, capsys):
    cfg = tmp_path / "models.yaml"
    cfg.write_text("services: []\n")
    assert main(["plan", str(cfg), "--gpus-csv", FIXTURE]) == 2
    assert "error:" in capsys.readouterr().err
