# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- `mesh status` is now a live view in a terminal: every service of the stack is listed from the start, with its
  state (`pending`, `starting`, `running`, `failed`, `stopping`), its process id (`-` until it has one), its port
  and its health, refreshed every second. Press `q` to quit. When piped it prints once, as before.
- `state.json` now lists all services from the start and records each one's status.

### Fixed
- `mesh status`, `mesh down`, `mesh logs` and `mesh gpu` only worked from the folder where `mesh up` was run,
  because logs and state defaulted to `logs/` in the current directory. `mesh up` now asks once which folder to
  use (suggesting `~/mesh-deck`), remembers it in `~/.config/mesh-deck.json`, and every command uses it from any
  directory. `--log-dir` still overrides it. Without a terminal it never asks and falls back to `logs/` as before.

## [0.1.0] - 2026-10-26

First public release, published on PyPI as `mesh-deck` (the command is `mesh`).

### Added
- `mesh gpu`: shows each GPU's memory and utilization and who is holding the memory.
- Progress and completion messages when stopping services (`Ctrl+C` or `mesh down`).
- `mesh logs -n all` to print a whole log.
- `python -m mesh_cli` as an alternative to the `mesh` command.
- Terminal UI: colour, tables, memory bars and a live start-up line (spinner and time) when output is an
  interactive terminal. Piped output and `NO_COLOR` keep the plain text. Adds the `rich` dependency.
- "Did you mean" hints for mistyped commands, service names, config fields and config file names.
- `mesh completion bash|zsh|fish` prints a Tab-completion script; `mesh logs` completes service names.
- `mesh up` prints a one-line plan summary when everything fits, and the full plan only when there is something
  to read or with `--plan`.
- Type information is shipped with the package (`py.typed`).

### Fixed
- A second Ctrl+C (or `mesh down`) while `mesh up` was stopping services no longer aborts the cleanup and leaves
  services running.
