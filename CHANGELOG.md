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

### Added
- `mesh up` warns when a GPU a service lists gained almost no memory while it started, and names any other GPU
  that grew: the sign that the service's command is not using the GPU the plan assumed. The docs now explain
  that `gpus` only informs the plan and the command must pin the GPU itself.

### Fixed
- A failed `nvidia-smi` reading while `mesh up` was starting services (a timeout under load, for example) stopped
  the whole start, even though the reading only feeds the "memory it added" figure. The start now carries on and
  that line says `unavailable`.
- The health check treated any HTTP answer as "up", including `503` while a model was still loading, so the next
  service could start before the previous one had finished loading. Answers of 500 or more now count as not ready,
  and a timeout says what the service last answered.
- `mesh up` now refuses to start when another program is already listening on a service's port, instead of letting
  that program's answers make a service that never started look healthy.
- A leftover `state.json` (after a crash, `kill -9` or a reboot) could make `mesh down` stop an unrelated program
  that had been given the same process id, and could make `mesh up` refuse to start. `mesh` now records when each
  process began and only trusts a process id whose start time still matches. A service that is still running
  after `mesh up` was killed is still found and stopped. Programs that took over an id are left alone.
- `mesh plan` and `mesh up` said a service fitted when it could start but would hold more memory than was free
  (its `overhead_mib` or `measured_mib` pushed the GPU below zero). The plan now reports it as not fitting, and
  the suggested `--gpu-memory-utilization` allows for the overhead. When no smaller fraction can help (a
  `measured_mib` that is too big, or an overhead that leaves no room) it says so instead of suggesting one.
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
