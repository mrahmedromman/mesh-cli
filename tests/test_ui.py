"""The terminal presentation. Output is captured from a recording console, so no real terminal is needed."""

import io
import sys

from rich.console import Console

from mesh_cli import holders, runner, ui
from mesh_cli.config import Config, Service
from mesh_cli.gpu import Gpu
from mesh_cli.planner import GpuBudget, make_plan


def recording():
    return Console(record=True, force_terminal=True, width=110, highlight=False)


def text_of(con):
    return con.export_text()


def test_fancy_needs_a_terminal_and_respects_no_color(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    assert ui.fancy()
    monkeypatch.setenv("NO_COLOR", "1")
    assert not ui.fancy()
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("TERM", "dumb")
    assert not ui.fancy()
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    assert not ui.fancy()


def test_bar_fills_in_proportion_and_changes_colour():
    assert ui.bar(0, 100, 10).plain == "░" * 10
    assert ui.bar(50, 100, 10).plain == "█" * 5 + "░" * 5
    assert ui.bar(100, 100, 10).plain == "█" * 10
    assert ui.bar(500, 100, 10).plain == "█" * 10  # over-full is capped
    assert ui.bar(1, 0, 10).plain == "░" * 10  # no total, no crash
    styles = [ui.bar(u, 100, 10).spans[0].style for u in (10, 80, 95)]
    assert styles == ["green", "yellow", "red"]


def test_style_line_keeps_the_words_and_adds_marks():
    cases = {
        "[1/2] starting llm on port 8001  (log: logs/llm.log)": "[1/2] starting llm on port 8001  (log: logs/llm.log)",
        "      llm is up  (memory it added: GPU 0: ~100 MiB)": "      ✓ llm is up  (memory it added: GPU 0: ~100 MiB)",
        "  stopped llm (3s)": "  ✓ stopped llm (3s)",
        "  stopping llm ...": "  stopping llm ...",
        "All 2 services stopped.": "All 2 services stopped.",
        "warning: llm exited with status 1; see logs/llm.log": "warning: llm exited with status 1; see logs/llm.log",
    }
    for line, expected in cases.items():
        assert ui.style_line(line).plain == expected


def test_style_line_colours_failures_red_and_success_green():
    assert ui.style_line("All 2 services stopped.").style == "bold green"
    assert ui.style_line("1 of 2 services could not be stopped.").style == "red"
    assert ui.style_line("  could not stop x: permission denied (pid 1 belongs to another user?)").style == "red"
    assert ui.style_line("warning: x exited").style == "yellow"


def test_plan_view_shows_each_service_the_memory_and_a_suggestion():
    budgets = [GpuBudget(0, 24564, 417, "NVIDIA GeForce RTX 4090", 485)]
    config = Config(services=[
        Service(name="llm", port=8001, gpus=[0], gpu_memory_utilization=0.55, overhead_mib=1500),
        Service(name="embed", port=8003, gpus=[0], gpu_memory_utilization=0.5, overhead_mib=500),
    ])
    plan = make_plan(config, budgets)
    con = recording()
    ui.print_plan(plan, budgets, con)
    out = text_of(con)
    assert "Plan for 2 services on 1 GPU" in out and "1 GPUs" not in out
    assert "NVIDIA GeForce RTX 4090" in out and "✓ fits" in out and "✗ does not fit" in out
    assert "try --gpu-memory-utilization" in out
    assert "After all services start" in out and "MiB" in out


def sample_gpu():
    return Gpu(0, "GPU-0", "NVIDIA GeForce RTX 4090", 24564, 23187, 892, 21)


def test_holders_view_lists_largest_first_with_bar_and_utilization():
    report = holders.GpuReport(sample_gpu(), [
        holders.Holder("mesh", "llm", "mesh service · 1 process", 16252),
        holders.Holder("desktop", "desktop", "graphics apps: Xorg · 3 processes", 417),
        holders.Holder("unattributed", "unattributed", "used memory that no listed process accounts for", 88),
    ])
    con = recording()
    ui.print_holders([report], con)
    out = text_of(con)
    assert "GPU 0" in out and "23187 / 24564 MiB used (94%)" in out
    assert "21% utilization" in out and "892 MiB free" in out
    assert out.index("llm") < out.index("desktop") < out.index("unattributed")
    assert "█" in out


def test_holders_view_for_an_idle_gpu():
    con = recording()
    ui.print_holders([holders.GpuReport(Gpu(0, "u", "Card", 1000, 0, 1000), [])], con)
    assert "nothing is using this GPU's memory" in text_of(con)


def test_status_view_lists_every_service_with_its_state():
    con = recording()
    ui.print_status([
        runner.ServiceRow("llm", "running", 100, 8001, "HTTP 200"),
        runner.ServiceRow("stt", "starting", 101, 8004, "not answering"),
        runner.ServiceRow("tts", "pending", None, 8005, "-"),
        runner.ServiceRow("emb", "failed", None, 8003, "-", "emb exited with status 3"),
    ], con)
    out = text_of(con)
    assert "● running" in out and "◐ starting" in out and "○ pending" in out and "✗ failed" in out
    assert "HTTP 200" in out and "not answering" in out
    assert out.index("llm") < out.index("stt") < out.index("tts") < out.index("emb")  # start order
    assert "emb: emb exited with status 3  (see: mesh logs emb)" in out
    tts_line = next(line for line in out.splitlines() if "tts" in line)
    assert " - " in tts_line  # no process yet, so a dash instead of a pid


def test_startup_display_shows_one_live_line_and_clears_it():
    con = recording()
    display = ui.StartupDisplay(con)
    display.progress("llm", 75, "\x1b[32mLoading safetensors shards 3/8\x1b[0m")
    assert display._thread is not None
    display.progress("llm", 80, "x" * 500)  # a very long log line must not break the line
    display.progress(None, 0, "")
    assert display._thread is None
    display.out("      llm is up")
    assert "✓ llm is up" in text_of(con)


def test_live_line_is_short_and_cleared_with_one_code():
    con = Console(file=io.StringIO(), force_terminal=True, width=110, highlight=False)
    display = ui.StartupDisplay(con)
    display._label = ("llm", "0:24")
    display._draw()
    frame = con.file.getvalue().split(display.CLEAR)[1]
    assert ui.ANSI.sub("", frame) == "⠋ llm  0:24"  # no log text, so it cannot wrap unless the window is tiny
    display._label = None
    display._erase()
    assert con.file.getvalue().endswith(display.CLEAR)


# --- runner hooks the display relies on -----------------------------------------------------------------------


def test_last_line_reads_the_tail_and_treats_carriage_returns_as_lines(tmp_path):
    log = tmp_path / "a.log"
    log.write_bytes(b"first\nsecond\n\n")
    assert runner.last_line(log) == "second"
    log.write_bytes(b"x" * 10000 + b"\nLoading 1/8\rLoading 2/8\rLoading 3/8\n")
    assert runner.last_line(log) == "Loading 3/8"
    assert runner.last_line(tmp_path / "missing.log") == ""
    log.write_bytes(b"")
    assert runner.last_line(log) == ""


class FakeProc:
    def poll(self):
        return None


def test_wait_healthy_reports_elapsed_time_and_the_last_log_line(tmp_path):
    log = tmp_path / "s.log"
    log.write_text("starting\nloading weights 40%\n")
    svc = Service(name="s", port=1, gpus=[0], start_timeout_s=100)
    now = [0.0]
    answers = iter([None, None, 200])
    seen = []
    runner.wait_healthy(
        svc, FakeProc(), log, interval=5,
        probe_fn=lambda port, path: next(answers),
        clock=lambda: now[0],
        sleep=lambda s: now.__setitem__(0, now[0] + s),
        progress=lambda seconds, line: seen.append((seconds, line)),
    )
    assert seen == [(0.0, "loading weights 40%"), (5.0, "loading weights 40%")]


def test_status_view_shows_the_gpu_process_ids_in_the_pid_column():
    con = recording()
    ui.print_status([
        runner.ServiceRow("nemotron", "running", 233844, 8003, "HTTP 200", up=True, gpu_pids=(235930,), gpu_known=True),
        runner.ServiceRow("qwen", "running", 193964, 8001, "HTTP 200", up=True, gpu_pids=(231394, 231395), gpu_known=True),
        runner.ServiceRow("loading", "starting", 300, 8002, "not answering", gpu_known=True),
    ], con)
    out = text_of(con)
    assert "235930" in out and "231394, 231395" in out
    assert "233844" not in out and "193964" not in out  # the launcher's pid is not what is shown
    loading = next(line for line in out.splitlines() if "loading" in line)
    assert " - " in loading and "300" not in loading
