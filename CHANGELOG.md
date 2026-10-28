# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- Two `mesh up` started at the same moment in the same folder were both allowed. Each one checked for a running stack
  before either had written anything, so both started, and the state file then listed only one of the two stacks, so
  `mesh down` and `mesh status` did not know about the other. On Linux one of them could also crash with
  `FileNotFoundError: state.tmp`, because both wrote the state file through the same temporary file. `mesh up` now
  locks the folder (`mesh.lock`) for as long as it runs and refuses to start when another one holds it, with the
  process id in the message. The system drops the lock when `mesh up` ends, however it ends, so a crash leaves none
  behind. Each process also writes its state file through a temporary file of its own. `mesh up` also refuses when
  the `mesh up` recorded in the state file is still running, which covers filesystems that cannot lock.
- `mesh down` said `1 of 1 service could not be stopped.`, then exited with code 0 and deleted the state file, so
  `mesh status` said "nothing running" while the process was still there, and a script that ran `mesh down && next-step`
  carried on. It now exits with code 1 and keeps the record of every service it could not stop (`mesh status` shows
  it as `stopping`, the next `mesh up` refuses to start on top of it, and `mesh down` tries again). The records of
  the services that did stop are removed. The clean-up at the end of `mesh up` does the same, and `mesh up` then
  exits with code 1. When the `mesh up` that `mesh down` asked to stop ends without finishing its clean-up,
  `mesh down` now stops what is left itself, at once, instead of waiting for it.
- Three mistakes in `models.yaml` failed late, or silently, and are now rejected when the config is loaded, with the
  field name in the message. `health: health` (no leading `/`) built an invalid address, so a healthy service never
  looked healthy and `mesh up` waited out the whole `start_timeout_s` (15 minutes by default) before reporting a
  timeout; `health` must now start with `/` and have no spaces. A service name such as `my/model` crashed `mesh up`
  with a Python traceback when it tried to open `my/model.log`; a name cannot now contain `/` or `\` or be `.` or
  `..`. `gpus: [0, 0]` counted the service's memory twice in the plan; a GPU can only be listed once.
- A file or folder problem, such as a `--log-dir` that cannot be created or a log that is not a file, ended in a long
  Python traceback. It is now one line, `error: Not a directory: /dev/null/nope`, with exit code 2. Piping
  `mesh logs NAME | head` also printed a `BrokenPipeError` traceback when `head` closed the pipe; it now ends quietly.

## [0.1.3] - 2026-10-28 (Latest)

### Changed
- The two messages that refuse to start a service no longer suggest `--force` as a way out. They now say to free the
  memory, or lower `gpu_memory_utilization`, and that `--force` only skips the check, so the service will probably
  still fail at its own start-up check. The `mesh up --help` text for `--force` says the same.
- `mesh --help` now describes `status` as a live view of every service (it still said "show running services"), and
  `mesh logs --help` explains the `service` argument.
- The "Cause found earlier in the log" block of a failed start-up no longer shows the traceback frames after the
  cause. It shows the cause line and the lines that continue its message. When the cause starts a traceback, it shows
  the exception that ends it.

### Fixed
- The plan said `1 services` and `1 GPUs` for a single service or a single GPU. It now says `1 service on 1 GPU`,
  and the one-line summary reads `Plan: the service fits on 1 GPU.`

## [0.1.2] - 2026-10-28

### Added
- `mesh up` now checks the GPU memory again right before each service starts. The plan is made once, when `mesh up`
  begins, and loading several models takes minutes, so another program may take memory in between. If the memory
  free at that moment is too little for the next service, `mesh up` refuses to start it, stops the services it
  already started, and shows how much is missing and who holds the memory. `--force` skips the check. Without
  `nvidia-smi` there is nothing to check against and it is skipped.

### Changed
- The error for a service that exited at start-up now shows its likely cause first when that is further up the log
  than the last 20 lines (the first line that mentions running out of GPU memory, else the first error line, with
  the lines after it). A vLLM log ends with "See root cause above" and the cause is far above it, so it was not
  visible. The first line of the message no longer ends with "Last log lines:".

### Fixed
- When a service exited at start-up, `mesh up` looked for a memory error only in the last 20 log lines. vLLM prints
  the real cause (`Free memory on device ...`) far above the traceback that ends the log, so the table of who holds
  the GPU memory was often not shown. It now searches the end of the whole log (the last 4 MiB).

## [0.1.1] - 2026-10-27

### Changed
- Pressing `q` in `mesh status` now leaves the last table on your terminal instead of clearing it, so the process
  ids can still be read and copied.
- `mesh gpu` now names the process ids behind each service or container (`mesh service · pid 235930`), the same ids
  `nvidia-smi` prints, so a service can be matched with its output. The PID column of `mesh status` shows the same
  ids, `-` until the service has a GPU process, instead of the process `mesh` started (for a Docker service the
  `docker run` client), which never matched `nvidia-smi`. Without `nvidia-smi` it still shows the process `mesh`
  started.
- The docs now say that the health check connects to `127.0.0.1`, so a service that listens only on a LAN address
  or on IPv6 is never seen and `mesh up` waits out `start_timeout_s`.
- `mesh status` is now a live view in a terminal: every service of the stack is listed from the start, with its
  state (`pending`, `starting`, `running`, `failed`, `stopping`), its process id (`-` until it has one), its port
  and its health, refreshed every second. Press `q` to quit. When piped it prints once, as before.
- `state.json` now lists all services from the start and records each one's status.

### Added
- `stop_timeout_s`, a per-service setting (default 30 seconds) for how long a service gets to stop before it is
  killed. Ctrl+C, `mesh down` and the clean-up of a service that died all use it; it is saved in the state file so
  `mesh down` honours it without your config, and `mesh down` waits for `mesh up` for at least as long as the
  services' timeouts add up to.
- `mesh up` warns when a GPU a service lists gained almost no memory while it started, and names any other GPU
  that grew: the sign that the service's command is not using the GPU the plan assumed. The docs now explain
  that `gpus` only informs the plan and the command must pin the GPU itself.

### Removed
- The `runtime` service field. It accepted `command` or `docker` but nothing read it; a Docker service is started
  by its `command` like any other. A config that still sets it now gets an error that says to delete the line.
  The Docker lines were also removed from `examples/dual-gpu.yaml`.

### Fixed
- When a service's main process died after it was up (for example `kill -9` on a vLLM server), `mesh up` only
  warned. Worker processes in the same process group kept running and kept the GPU memory, while `mesh status`
  showed the service as failed. `mesh up` now also stops what is left of that service.
- `mesh logs -f` went quiet after the stack was restarted: `mesh up` starts each log over, but the follower kept its
  old position and showed nothing until the new log grew past the old size. It now notices that the service was
  started again (its process id changes, or the file shrinks) and starts from the top of the new log.
- `state.json` copies each service's full command (which can contain tokens such as `HF_TOKEN`), and it and the
  logs were readable by other users on the machine. They are now written with mode `600`, and a folder that `mesh`
  creates gets mode `700`. An existing folder keeps its permissions.
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
