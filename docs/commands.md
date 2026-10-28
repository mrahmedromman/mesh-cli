# Commands

Back to the [README](../README.md) · [Config reference](config.md) · [How it works](how-it-works.md)

| Command | What it does |
|---|---|
| `mesh plan CONFIG` | Simulates starting the services in order and reports which ones fit. Changes nothing. |
| `mesh up CONFIG` | Runs the plan, then starts each service, waits for its health check, and prints the memory it added. Blocks until Ctrl+C. |
| `mesh down` | Stops a stack started by `mesh up`, in reverse order. |
| `mesh status` | A live view of every service: state, process id, port and health. Press `q` to quit. |
| `mesh logs NAME` | Shows or follows a service's log. |
| `mesh gpu` | Shows each GPU's memory and utilization, and who is holding the memory. |
| `mesh completion SHELL` | Prints a Tab-completion script for `bash`, `zsh` or `fish`. |
| `mesh --version` | Prints the version. |

## Options

| Option | Applies to | Meaning |
|---|---|---|
| `--gpus-csv FILE` | `plan` | Read GPUs from saved `nvidia-smi` output instead of running it. Handy for testing and for planning on another machine. |
| `--external 0=417,1=4` | `plan` | Memory (MiB) already used by other programs on each GPU. Without it, `plan` counts everything currently in use as external and prints a note. |
| `--force` | `up` | Skip the plan check and the check right before each service starts. It does not make the memory appear: a service that does not get it usually still fails at its own start-up check (vLLM stops with `Free memory on device ...`). Use it only when you know the memory reading is wrong. |
| `--plan` | `up` | Print the full plan. By default `up` prints one summary line when everything fits, and the full plan only when there is something to read (a service that does not fit, a warning, or a note). `mesh plan` always prints it in full. |
| `--log-dir DIR` | `up`, `down`, `status`, `logs`, `gpu`, `plan` | Where logs and state live. Default: the folder you chose the first time you ran `mesh up` (see [Where files go](#where-files-go)). Use the same value for every command of a stack. `gpu` and `plan` read it only to recognise your running services. |
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
mesh up models.yaml --plan     # also print the full plan when everything fits
```

See [How it works](how-it-works.md#what-mesh-up-does) for the exact steps. In short: plan, start each service
in order, wait for its health check, report the memory it added, and stop everything on Ctrl+C.

Right before each service starts, `mesh up` reads the GPUs again. Earlier services take a while to load, and another
program may have taken memory since the plan was made. If the memory free at that moment is too little for the next
service, `mesh up` does not start it: it stops the services it already started, prints how much is missing and who
holds the memory (the same table as `mesh gpu`), and exits with code 1. `--force` skips this check, but it does not
create the memory, so the service will usually fail at its own start-up check instead. `mesh up` cannot reserve
memory either: a program that grabs it after this check can still make a service fail while it starts. `mesh up` then
reports that failure and shows who holds the memory.

If a service's main process exits after it was up (it crashed, or someone ran `kill -9` on it), `mesh up` marks it
`failed`, prints a warning, and also stops what is left of its process group. A model server with several worker
processes (vLLM with tensor parallelism, for example) would otherwise keep its GPU memory although `mesh` reports it
as dead. The other services keep running.

## `mesh logs`

```bash
mesh logs llm                # the last 50 lines, then exit
mesh logs llm -n 200         # the last 200 lines, then exit
mesh logs llm -n all         # the whole log, then exit
mesh logs llm -f             # the last 50 lines, then follow live
mesh logs llm -n all -f      # the whole log, then follow live
mesh logs llm -n 0 -f        # only new lines from now on
```

If the stack is stopped and started again while you are following, `mesh up` starts the log over. `mesh logs -f`
notices the service has a new process and shows the new log from its first line.

## `mesh gpu`

```bash
mesh gpu
```

Prints one block per GPU: total memory, the part the driver reserves, used, free and utilization, then every
holder of the used memory, largest first:

```
GPU 0  NVIDIA GeForce RTX 4090  24564 MiB total · 485 reserved · 23187 used · 892 free · 21% utilization
    16252 MiB  llm          mesh service · pid 555362
     6430 MiB  stt          mesh service · container "stt" · pid 558799
      417 MiB  desktop      graphics apps: Xorg, gnome-shell, firefox · 6 processes
       88 MiB  unattributed used memory that no listed process accounts for
```

Each service or container line ends with the process ids that hold the memory on that GPU, the same ids
`nvidia-smi` prints (for a vLLM service these are its engine and worker processes). This is how to match
`nvidia-smi` with a service. `mesh status` shows the same ids in its PID column. Long lists are shortened (after
four here, after three in `mesh status`).

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
- when a service exits during `mesh up` and its log mentions running out of GPU memory, anywhere in the last few MiB of
  the log (the real cause is often far above the traceback that ends it);
- under `mesh plan`, when it counts memory already in use as external (not with `--gpus-csv`).

Run it on the host, not inside a container: inside a container `nvidia-smi` cannot see the host's GPU processes.
It needs Linux, because it reads `/proc`. If a process exits while it runs, or a detail is unreadable, that part
is shown as unknown instead of failing.

## `mesh status` and `mesh down`

`mesh status` lists every service of the stack, in start order, and keeps itself up to date every second. Press
`q` (or Ctrl+C) to leave it; the last table is then printed on your terminal, so the process ids stay where you can
copy them.

| Column | Meaning |
|---|---|
| State | `pending` (not started yet), `starting` (loading, waiting for its health check), `running`, `failed` (it exited, or did not become healthy in time) or `stopping`. |
| PID | The process ids that hold the service's GPU memory, the same ids `mesh gpu` and `nvidia-smi` show (for a vLLM service its engine and worker processes; for a Docker service the processes inside the container, not the `docker run` client). `-` until the service has any, which for vLLM is a few seconds after it starts. If `nvidia-smi` cannot be read, it shows the process `mesh` started instead. |
| Port | The port from your config. |
| Health | The HTTP status from its health path, `not answering`, or `-` when there is no process to ask. |

A failed service also gets a line under the table with the reason, and a pointer to `mesh logs NAME`. When the
stack stops while you are watching, the last table stays on screen with the services shown as stopped.

When the output or the input is not a terminal (a pipe, a script, CI), `mesh status` prints the table once and
exits, so it never waits for a key. With no stack running it prints `nothing running` and exits with code 0.

`mesh down`, run from another terminal, asks the `mesh up` process to shut down gracefully, so the result is the
same as pressing Ctrl+C there. If that process is gone, it stops each recorded service itself.

If a service could not be stopped (for example a process that belongs to another user, which you may not signal), `mesh down`
says so, exits with code 1, and keeps the record of that service. `mesh status` still shows it (as `stopping`), the
next `mesh up` refuses to start on top of it, and running `mesh down` again tries once more. The same holds when
`mesh up` itself ends after Ctrl+C with a service it could not stop: it exits with code 1 and keeps the record.

The state file records when each process began, next to its process id. After a crash or a reboot the same id may
belong to another program; `mesh` then ignores it: `mesh up` is not refused because of it, and `mesh down` does
not signal it (it still runs the service's `stop` command, such as `docker stop NAME`). A service that is really
still running, for example after `mesh up` was killed, is found and stopped as before.

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
- `stopped llm (force-killed after 30s)`: the service ignored the polite stop request for its whole
  [`stop_timeout_s`](config.md#stopping-a-service) (30 seconds unless you set another), so it was killed.
- `tts was already stopped`: the process had exited before shutdown started.
- `could not stop tts: permission denied ...`: the process belongs to another user. The last line then reads
  `1 of 3 services could not be stopped.`
- If `mesh up` does not finish stopping in time, `mesh down` says so and stops the remaining services itself. It
  waits 90 seconds, or longer when the services' stop timeouts add up to more.

The same messages appear when `mesh up` cleans up after a service fails to start.

## Suggestions and Tab completion

When you mistype something, `mesh` suggests the closest match instead of only failing:

```
$ mesh stauts
mesh: error: unknown command 'stauts'. Did you mean 'status'? Choose from: plan, gpu, up, down, status, logs, completion
$ mesh logs lm
error: no log for service 'lm'. Did you mean 'llm'? Known services: llm, stt, tts
```

The same applies to a config file name (`models.yml` for `models.yaml`) and to a misspelled field in the config
(`gpu_memory_utilisation` suggests `gpu_memory_utilization`).

To turn on Tab completion, add one line to your shell's startup file, then open a new terminal:

| Shell | Add this |
|---|---|
| bash | `eval "$(mesh completion bash)"` to `~/.bashrc` |
| zsh | `eval "$(mesh completion zsh)"` to `~/.zshrc`, after `compinit` |
| fish | `mesh completion fish > ~/.config/fish/completions/mesh.fish` |

It completes commands, options, config files and shell names. `mesh logs <Tab>` lists the services of the stack in
the current log directory (it honours `--log-dir` in bash and zsh; fish uses the remembered folder).

## Terminal output

In an interactive terminal, `mesh` draws colour, tables and memory bars, and while a service loads during
`mesh up` it shows one live line: a spinner and the time so far. Follow the service's log with `mesh logs NAME -f`.

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
| 1 | `plan`: a service will not fit. `up`: the plan does not fit (without `--force`), the memory ran short before a service started, a service failed to start, a stack is already running, or a service could not be stopped when it ended. `down`: a service could not be stopped. `logs`: no such log file. |
| 2 | Bad config, unreadable `nvidia-smi` output, or a bad command line. |

## Where files go

The first time you run `mesh up` in a terminal, it asks which folder to use and suggests `~/mesh-deck`. Press
Enter to accept it or type another folder. The choice is saved in `~/.config/mesh-deck.json` (or under
`$XDG_CONFIG_HOME`), so every command finds the same folder from any directory. Everything `mesh up` writes goes
there, on your own computer:

- `NAME.log`: one file per service, with stdout and stderr together. Each `mesh up` starts the file
  fresh, so a new run overwrites the previous log. Copy a log you want to keep.
- `state.json`: which services are running, their process ids and ports. `mesh status`, `mesh down` and
  `mesh logs` read it. It is removed when the stack stops.
- `mesh.lock`: held by `mesh up` while it runs, so that a second `mesh up` cannot start in the same folder. The
  file stays in the folder and can be ignored; it is the system's lock on it, which goes away when `mesh up` ends,
  not the file, that counts.

`state.json` holds each service's full command, which can contain tokens, and logs can hold prompts and paths. So
`mesh` creates the folder it uses readable only by you (mode `700`; folders above it keep the system default) and writes the files with mode `600`. A folder that
already exists, such as one you made yourself, keeps its permissions, and a log written by an older version is
tightened the next time `mesh up` starts that service.

`--log-dir FOLDER` uses a different folder for that command instead (give it to every command of the stack).
Delete the settings file to be asked again. When output is not a terminal (a script or CI), `mesh` never asks:
it uses the remembered folder, or `logs` in the current directory if nothing is saved. If your folder is inside a
git repository, add it to `.gitignore`.
