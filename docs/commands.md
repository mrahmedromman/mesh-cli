# Commands

Back to the [README](../README.md) · [Config reference](config.md) · [How it works](how-it-works.md)

| Command | What it does |
|---|---|
| `mesh plan CONFIG` | Simulates starting the services in order and reports which ones fit. Changes nothing. |
| `mesh up CONFIG` | Runs the plan, then starts each service, waits for its health check, and prints the memory it added. Blocks until Ctrl+C. |
| `mesh down` | Stops a stack started by `mesh up`, in reverse order. |
| `mesh status` | Shows each service's state, process id, port and health. |
| `mesh logs NAME` | Shows or follows a service's log. |
| `mesh gpu` | Shows each GPU's memory and utilization, and who is holding the memory. |
| `mesh --version` | Prints the version. |

## Options

| Option | Applies to | Meaning |
|---|---|---|
| `--gpus-csv FILE` | `plan` | Read GPUs from saved `nvidia-smi` output instead of running it. Handy for testing and for planning on another machine. |
| `--external 0=417,1=4` | `plan` | Memory (MiB) already used by other programs on each GPU. Without it, `plan` counts everything currently in use as external and prints a note. |
| `--force` | `up` | Start even if the plan says a service will not fit. |
| `--log-dir DIR` | `up`, `down`, `status`, `logs`, `gpu`, `plan` | Where logs and state live. Default `logs` in the current directory. Use the same value for every command. `gpu` and `plan` read it only to recognise your running services. |
| `-n N` | `logs` | Show the last `N` lines. Default 50. `-n 0` shows none (useful with `-f` to see only new lines). |
| `-n all` | `logs` | Show the whole log from the start, without needing to know how many lines it has. |
| `-f`, `--follow` | `logs` | After printing, keep showing new lines until Ctrl+C. |

`-n` needs a value (a number or `all`); a bare `-n` is an error.

## `mesh plan`

```bash
mesh plan models.yaml
mesh plan models.yaml --gpus-csv gpus.csv --external 0=417,1=4
```

Prints each GPU, then each service in order with what it needs and what is free, then the memory left on
each GPU after everything starts. When a service will not fit, it suggests a smaller
`--gpu-memory-utilization`. It warns if a GPU ends up with less free memory than `margin_mib`.

To make a `--gpus-csv` file:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv > gpus.csv
```

## `mesh up`

```bash
mesh up models.yaml
mesh up models.yaml --force
```

See [How it works](how-it-works.md#what-mesh-up-does) for the exact steps. In short: plan, start each service
in order, wait for its health check, report the memory it added, and stop everything on Ctrl+C.

## `mesh logs`

```bash
mesh logs llm                # the last 50 lines, then exit
mesh logs llm -n 200         # the last 200 lines, then exit
mesh logs llm -n all         # the whole log, then exit
mesh logs llm -f             # the last 50 lines, then follow live
mesh logs llm -n all -f      # the whole log, then follow live
mesh logs llm -n 0 -f        # only new lines from now on
```

## `mesh gpu`

```bash
mesh gpu
```

Prints one block per GPU: total memory, the part the driver reserves, used, free and utilization, then every
holder of the used memory, largest first:

```
GPU 0  NVIDIA GeForce RTX 4090  24564 MiB total · 485 reserved · 23187 used · 892 free · 21% utilization
    16252 MiB  llm          mesh service · 1 process
     6430 MiB  stt          mesh service · container "stt" · 1 process
      417 MiB  desktop      graphics apps: Xorg, gnome-shell, firefox · 6 processes
       88 MiB  unattributed used memory that no listed process accounts for
```

Each holder is one of:

- **A mesh service**, named as in your config. Docker services are recognised through their container.
- **A Docker container** that mesh did not start, by name (or the first 12 characters of its id if `docker ps`
  is not allowed).
- **desktop**: graphics apps such as the display server and browsers, grouped together.
- **Another program**, with its user, process id and command line. This is where an orphaned process left behind
  by a crash shows up.
- **unattributed**: memory in use that no listed process accounts for (driver overhead, or processes you are not
  allowed to see).

It only reports; it never stops or signals anything.

The same table is printed for you in three places, only for the GPUs involved:

- when `mesh up` finds the plan does not fit (the GPUs that are short);
- when a service exits during `mesh up` with a memory error in its log;
- under `mesh plan`, when it counts memory already in use as external (not with `--gpus-csv`).

Run it on the host, not inside a container: inside a container `nvidia-smi` cannot see the host's GPU processes.
It needs Linux, because it reads `/proc`. If a process exits while it runs, or a detail is unreadable, that part
is shown as unknown instead of failing.

## `mesh status` and `mesh down`

`mesh status` prints one line per service: `running` or `stopped`, the process id, the port, and the HTTP
status from its health path (or `not answering`).

`mesh down`, run from another terminal, asks the `mesh up` process to shut down gracefully, so the result is the
same as pressing Ctrl+C there. If that process is gone, it stops each recorded service itself.

### Shutdown messages

Stopping prints progress, one service at a time in reverse start order, and ends with a completion line. This is
what you see in the `mesh up` terminal after Ctrl+C:

```
Stopping 3 services...
  stopping tts ...
  stopped tts
  stopping stt ...
  stopped stt
  stopping llm ...
  stopped llm
All 3 services stopped.
```

`mesh down` prints its own progress in the terminal where you run it:

```
Asking mesh up (pid 44233) to stop 3 services...
  stopped tts
  stopped stt
  stopped llm
All 3 services stopped.
```

Other lines you may see:

- `stopped llm (12s)`: a service that took a second or more to stop shows how long it took.
- `stopped llm (force-killed after 30s)`: the service ignored the polite stop request for the full 30 seconds, so
  it was killed.
- `tts was already stopped`: the process had exited before shutdown started.
- `could not stop tts: permission denied ...`: the process belongs to another user. The last line then reads
  `1 of 3 services could not be stopped.`
- If `mesh up` does not finish within 90 seconds, `mesh down` says so and stops the remaining services itself.

The same messages appear when `mesh up` cleans up after a service fails to start.

## Terminal output

In an interactive terminal, `mesh` draws colour, tables and memory bars, and while a service loads during
`mesh up` it shows one live line: a spinner, the time so far and the last line of that service's log.

```
⠹ llm  2:14  Loading safetensors shards 3/8
```

The line disappears when the service is up and is replaced by `✓ llm is up`. Bars are green below 70% of a
GPU's memory, yellow up to 90% and red above that.

When output is not a terminal (piped, redirected, run from a script or CI), or when the `NO_COLOR` environment
variable is set, or `TERM=dumb`, `mesh` prints plain text with no escape codes. The words are the same in both
modes.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. |
| 1 | `plan`: a service will not fit. `up`: the plan does not fit (without `--force`), a service failed to start, or a stack is already running. `logs`: no such log file. |
| 2 | Bad config, unreadable `nvidia-smi` output, or a bad command line. |

## Where files go

Everything `mesh up` writes goes in the log directory (default `logs/` in the **current directory**):

- `logs/NAME.log`: one file per service, with stdout and stderr together. Each `mesh up` starts the file
  fresh, so a new run overwrites the previous log. Copy a log you want to keep.
- `logs/state.json`: which services are running, their process ids and ports. `mesh status`, `mesh down` and
  `mesh logs` read it, which is why they must be run from the same directory (or with the same `--log-dir`).
  It is removed when the stack stops.

Add `logs/` to your `.gitignore`.
