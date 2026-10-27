"""Where mesh keeps its logs and state: one folder, chosen once and remembered.

`mesh up` asks the first time it runs at a terminal and saves the answer in a small settings file, so `status`,
`down`, `logs` and `gpu` find the same files from any directory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEFAULT_FOLDER = "mesh-deck"  # inside the home folder
FALLBACK = Path("logs")  # when nothing is saved and nobody can be asked (the behaviour of 0.1.0)


class SettingsError(Exception):
    pass


def settings_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "mesh-deck.json"


def saved_folder() -> Path | None:
    try:
        folder = json.loads(settings_path().read_text()).get("folder")
    except (OSError, ValueError, AttributeError):
        return None
    return Path(folder) if isinstance(folder, str) and folder else None


def save_folder(folder: Path) -> None:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"folder": str(folder)}, indent=2) + "\n")


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def tidy(path: Path) -> str:
    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)


def _ask() -> Path:
    default = Path.home() / DEFAULT_FOLDER
    print("mesh keeps each service's log and its state in one folder on this computer,")
    print("so every mesh command works from any directory. Nothing is sent anywhere.")
    try:
        answer = input(f"Which folder should it use? [{tidy(default)}] ").strip()
    except (EOFError, KeyboardInterrupt) as exc:
        print()
        raise SettingsError("no folder chosen; run mesh up again, or pass --log-dir FOLDER") from exc
    folder = Path(os.path.abspath(os.path.expanduser(answer))) if answer else default
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SettingsError(f"cannot create {folder}: {exc.strerror}") from exc
    try:
        save_folder(folder)
    except OSError as exc:
        raise SettingsError(f"cannot save the choice to {settings_path()}: {exc.strerror}") from exc
    print(f"Using {folder}. Remembered in {settings_path()}; --log-dir overrides it.\n")
    return folder


def log_dir(chosen: str | None, ask: bool = False) -> Path:
    """The folder for logs and state: `--log-dir`, else the remembered folder, else (if `ask`, at a terminal) ask once."""
    if chosen:
        return Path(chosen)
    return saved_folder() or (_ask() if ask and interactive() else FALLBACK)
