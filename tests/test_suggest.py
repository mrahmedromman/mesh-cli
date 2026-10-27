"""'Did you mean' hints and shell completion."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mesh_cli import completion, runner, settings, suggest
from mesh_cli.cli import build_parser, main
from mesh_cli.config import ConfigError, load_config

SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_closest_and_did_you_mean():
    assert suggest.closest("stauts", ["plan", "status", "logs"]) == "status"
    assert suggest.closest("zzzz", ["plan", "status"]) is None
    assert suggest.did_you_mean("lm", ["llm", "stt"]) == " Did you mean 'llm'?"
    assert suggest.did_you_mean("zzzz", ["llm"]) == ""


def test_unknown_command_suggests_the_closest(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["stauts"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "unknown command 'stauts'. Did you mean 'status'?" in err
    assert "_complete" not in err  # the hidden helper is not offered


def test_invalid_value_for_completion_suggests_a_shell(capsys):
    with pytest.raises(SystemExit):
        main(["completion", "zhs"])
    assert "Did you mean 'zsh'?" in capsys.readouterr().err


def test_logs_for_a_mistyped_service_lists_the_known_ones(tmp_path, capsys):
    (tmp_path / "llm.log").write_text("x")
    (tmp_path / "stt.log").write_text("x")
    assert main(["logs", "lm", "--log-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "no log for service 'lm'. Did you mean 'llm'?" in err and "Known services: llm, stt" in err


def test_logs_with_no_logs_at_all_points_at_the_log_dir(tmp_path, capsys):
    assert main(["logs", "llm", "--log-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "no log file" in err and "--log-dir" in err


def test_known_services_combines_state_and_logs(tmp_path):
    (tmp_path / "b.log").write_text("x")
    runner.write_state(tmp_path, {"controller_pid": 1, "services": {"a": {"pid": 1}}})
    assert runner.known_services(tmp_path) == ["a", "b"]
    assert runner.known_services(tmp_path / "missing") == []


def test_config_file_typo_suggests_the_file(tmp_path):
    (tmp_path / "models.yaml").write_text("services: []")
    with pytest.raises(ConfigError, match="Did you mean 'models.yaml'"):
        load_config(tmp_path / "model.yaml")


def test_unknown_config_field_suggests_the_real_one(tmp_path):
    cfg = tmp_path / "m.yaml"
    cfg.write_text("services:\n  - {name: a, port: 1, gpus: [0], gpu_memory_utilisation: 0.5}\n")
    with pytest.raises(ConfigError) as excinfo:
        load_config(cfg)
    assert "gpu_memory_utilisation: Extra inputs are not permitted. Did you mean 'gpu_memory_utilization'?" in str(excinfo.value)


def test_unknown_field_without_a_close_match_has_no_hint(tmp_path):
    cfg = tmp_path / "m.yaml"
    cfg.write_text("services:\n  - {name: a, port: 1, gpus: [0], zzzzzz: 1}\n")
    with pytest.raises(ConfigError) as excinfo:
        load_config(cfg)
    assert "Extra inputs are not permitted\n" in str(excinfo.value) + "\n"


# --- completion -----------------------------------------------------------------------------------------------


def test_introspection_lists_visible_commands_and_their_options():
    top, commands = completion.introspect(build_parser())
    names = [c.name for c in commands]
    assert names == ["plan", "gpu", "up", "down", "status", "logs", "completion"]
    assert any("--version" in o.flags for o in top)
    logs = next(c for c in commands if c.name == "logs")
    assert logs.positional == "service"
    assert {"-n", "--lines"} in [set(o.flags) for o in logs.options]
    assert next(c for c in commands if c.name == "completion").choices == ["bash", "zsh", "fish"]


def test_scripts_mention_every_command():
    for shell in completion.SHELLS:
        text = completion.script(shell, build_parser())
        for name in ("plan", "up", "down", "status", "logs", "gpu", "completion"):
            assert name in text
        assert "_complete\n" not in text and "'_complete" not in text


def test_completion_command_prints_the_script(capsys):
    assert main(["completion", "bash"]) == 0
    assert capsys.readouterr().out.startswith("# bash completion for mesh")


def test_hidden_helper_prints_service_names(tmp_path, capsys):
    (tmp_path / "llm.log").write_text("x")
    assert main(["_complete", "services", "--log-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out == "llm\n"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_bash_script_is_valid_and_completes(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "mesh"
    wrapper.write_text(f'#!/bin/sh\nPYTHONPATH={SRC} exec {sys.executable} -c "import sys; from mesh_cli.cli import main; sys.exit(main())" "$@"\n')
    wrapper.chmod(0o755)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "llm.log").write_text("x")
    (logs / "stt.log").write_text("x")
    script = tmp_path / "mesh.bash"
    script.write_text(completion.script("bash", build_parser()))

    def complete(*words):
        program = (
            f'source "{script}"; COMP_WORDS=({" ".join(repr(w) for w in words)}); COMP_CWORD=$(( ${{#COMP_WORDS[@]}} - 1 )); '
            "_mesh; printf '%s\\n' \"${COMPREPLY[@]}\""
        )
        result = subprocess.run(["bash", "-c", program], capture_output=True, text=True,
                                env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}, cwd=tmp_path)
        assert result.returncode == 0, result.stderr
        return result.stdout.split()

    assert complete("mesh", "st") == ["status"]
    assert set(complete("mesh", "")) >= {"plan", "up", "down", "logs", "gpu", "completion", "--version"}
    assert complete("mesh", "completion", "") == ["bash", "zsh", "fish"]
    assert complete("mesh", "logs", "--fo") == ["--follow"]
    assert set(complete("mesh", "logs", "--log-dir", "logs", "")) == {"llm", "stt"}
    assert complete("mesh", "logs", "--log-dir", "logs", "l") == ["llm"]
    saved = tmp_path / "saved"
    saved.mkdir()
    (saved / "tts.log").write_text("x")
    settings.save_folder(saved)  # without --log-dir the remembered folder is used, wherever you are
    assert complete("mesh", "logs", "") == ["tts"]


@pytest.mark.skipif(shutil.which("zsh") is None, reason="needs zsh")
def test_zsh_script_has_valid_syntax(tmp_path):
    script = tmp_path / "mesh.zsh"
    script.write_text(completion.script("zsh", build_parser()))
    assert subprocess.run(["zsh", "-n", str(script)], capture_output=True, text=True).returncode == 0
