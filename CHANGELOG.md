# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-10-26

First public release.

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
