"""The folder where mesh keeps its logs and state, chosen once and remembered."""

from pathlib import Path

import pytest

from mesh_cli import settings


def no_prompt(*args):
    raise AssertionError("the user must not be asked here")


def answer(monkeypatch, text, terminal=True):
    monkeypatch.setattr(settings, "interactive", lambda: terminal)
    monkeypatch.setattr("builtins.input", lambda prompt="": text)


def test_without_a_saved_folder_it_falls_back_to_logs_in_the_current_directory():
    assert settings.log_dir(None) == Path("logs")


def test_a_folder_given_on_the_command_line_wins_and_is_not_saved(tmp_path):
    assert settings.log_dir(str(tmp_path / "x")) == tmp_path / "x"
    assert settings.saved_folder() is None


def test_the_saved_folder_is_used_from_any_directory(tmp_path, monkeypatch):
    settings.save_folder(tmp_path / "mine")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert settings.log_dir(None) == tmp_path / "mine"


def test_the_saved_folder_means_no_question(tmp_path, monkeypatch):
    settings.save_folder(tmp_path / "mine")
    monkeypatch.setattr(settings, "interactive", lambda: True)
    monkeypatch.setattr("builtins.input", no_prompt)
    assert settings.log_dir(None, ask=True) == tmp_path / "mine"


def test_first_run_suggests_mesh_deck_in_the_home_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    answer(monkeypatch, "")  # Enter accepts the suggestion
    folder = settings.log_dir(None, ask=True)
    assert folder == tmp_path / "mesh-deck" and folder.is_dir()
    assert settings.saved_folder() == folder
    out = capsys.readouterr().out
    assert "on this computer" in out and str(settings.settings_path()) in out


def test_first_run_accepts_another_folder(tmp_path, monkeypatch):
    chosen = tmp_path / "work" / "ai-logs"
    answer(monkeypatch, str(chosen))
    assert settings.log_dir(None, ask=True) == chosen
    assert chosen.is_dir() and settings.saved_folder() == chosen


def test_a_tilde_in_the_answer_means_the_home_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    answer(monkeypatch, "~/deck")
    assert settings.log_dir(None, ask=True) == tmp_path / "deck"


def test_nobody_is_asked_without_a_terminal(monkeypatch):
    monkeypatch.setattr(settings, "interactive", lambda: False)
    monkeypatch.setattr("builtins.input", no_prompt)
    assert settings.log_dir(None, ask=True) == Path("logs")
    assert settings.saved_folder() is None


def test_commands_that_only_read_never_ask(monkeypatch):
    monkeypatch.setattr(settings, "interactive", lambda: True)
    monkeypatch.setattr("builtins.input", no_prompt)
    assert settings.log_dir(None, ask=False) == Path("logs")


def test_a_damaged_settings_file_is_ignored(tmp_path):
    path = settings.settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert settings.saved_folder() is None
    path.write_text('{"folder": 5}')
    assert settings.saved_folder() is None


def test_a_folder_that_cannot_be_created_is_reported_and_not_saved(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    answer(monkeypatch, str(blocker / "inside"))
    with pytest.raises(settings.SettingsError, match="cannot create"):
        settings.log_dir(None, ask=True)
    assert settings.saved_folder() is None


def test_closing_the_question_with_ctrl_d_chooses_nothing(monkeypatch):
    monkeypatch.setattr(settings, "interactive", lambda: True)

    def eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    with pytest.raises(settings.SettingsError, match="no folder"):
        settings.log_dir(None, ask=True)
    assert settings.saved_folder() is None


def test_paths_under_the_home_folder_are_shown_with_a_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert settings.tidy(tmp_path / "mesh-deck" / "llm.log") == "~/mesh-deck/llm.log"
    assert settings.tidy(Path("/srv/other/llm.log")) == "/srv/other/llm.log"
