<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mrahmedromman/mesh-deck/main/docs/assets/logo-dark.svg">
    <img alt="mesh-deck" src="https://raw.githubusercontent.com/mrahmedromman/mesh-deck/main/docs/assets/logo-light.svg" width="320">
  </picture>
</p>

<p align="center"><b>Know your AI stack fits your GPUs before you start it.</b></p>

<p align="center">
  <a href="https://pypi.org/project/mesh-deck/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/mesh-deck"></a>
  <img alt="license: MIT" src="https://img.shields.io/badge/license-MIT-blue">
  <img alt="python: 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="status: early" src="https://img.shields.io/badge/status-early-orange">
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="https://github.com/mrahmedromman/mesh-deck/blob/main/docs/commands.md">Commands</a> ·
  <a href="https://github.com/mrahmedromman/mesh-deck/blob/main/docs/config.md">Config</a> ·
  <a href="https://github.com/mrahmedromman/mesh-deck/blob/main/docs/how-it-works.md">How it works</a> ·
  <a href="https://github.com/mrahmedromman/mesh-deck/blob/main/CONTRIBUTING.md">Contributing</a>
</p>

`mesh` plans and starts a stack of local AI services on shared GPUs, in the right order, with one log file per
service. A service is any command that listens on a port: a chat model, speech-to-text, embeddings, text-to-speech,
or anything else that uses GPU memory.

<p align="center">
  <img alt="mesh status showing four services: llm and stt running, tts starting, embed pending" src="https://raw.githubusercontent.com/mrahmedromman/mesh-deck/main/docs/assets/social-preview.png" width="720">
</p>

> **Early release.** Built and checked on one machine with two RTX 4090s, running vLLM services. It reads
> `nvidia-smi`, so it needs an NVIDIA GPU and a Unix-like system (tested on Linux). Reports from other GPUs and
> other engines are very welcome.

## Why

If you run several GPU services on one machine (vLLM is the case this was built for), you have probably hit
this: the second service crashes at startup with *"Free memory on device ... is less than desired GPU memory utilization"*. Each service
measures free memory when it starts, and the first one already took its share. Working out fractions that fit,
starting things in the right order, and finding which log holds the error is tedious to redo by hand.

## Features

- **Checks first.** Simulates GPU memory for your services in start order and tells you which will not fit,
  with a suggested fraction.
- **Starts in order.** Each service waits for the previous one's health check, so they never race for memory, and
  `mesh up` checks the free memory again right before each start.
- **Any runtime.** A service is a shell command: a venv, a conda environment or `docker run`.
- **One log per service.** `mesh logs llm -f` follows one; `-n all` prints the whole file.
- **Reports real memory.** After each start it prints how much GPU memory that service added.
- **Says who holds the memory.** `mesh gpu` shows each GPU's memory and utilization and names the holder: one of
  your services, a Docker container, the desktop, or another program. It prints this automatically when a start
  fails for lack of memory.
- **Clean shutdown.** One Ctrl+C (or `mesh down`) stops everything in reverse order, tells you as each service
  stops, and ends with a completion message.
- **Readable in a terminal.** Colour, tables, memory bars, and a live line while a model loads (a spinner and
  the time so far). When output is piped or `NO_COLOR` is set, it prints plain text.
- **Helpful errors and Tab completion.** A mistyped command, service name, config field or file gets a "did you
  mean" hint, and `mesh completion bash|zsh|fish` adds Tab completion, including your service names.
- **Just a YAML file.** No daemon, no proxy, nothing listening on a port.

## Other engines

The start order, health checks, logs, `mesh gpu` and shutdown work for any command. The memory check is built
around vLLM's `gpu_memory_utilization` fraction. For another engine (llama.cpp, Ollama, TGI, ComfyUI or your own
script), leave out the fraction and give the service `measured_mib` instead: run `mesh up` once, copy the
printed "memory it added" numbers, and the planner uses them. Set `health` to a path your engine answers on, since
the default is `/v1/models`. An answer of 500 or more (for example `503` while it loads) counts as not ready yet.
This has not been tested with engines other than vLLM yet; if you try one, please
[open an issue](https://github.com/mrahmedromman/mesh-deck/issues) with what worked.

## Install

Use a separate environment from the one that runs your models; `mesh` needs only `pydantic`, `pyyaml` and `rich`.

```bash
pipx install mesh-deck     # or: pip install mesh-deck
```

To run the latest unreleased code instead:

```bash
pipx install git+https://github.com/mrahmedromman/mesh-deck.git
```

## Quickstart

1. Write a `models.yaml` (see [examples/](https://github.com/mrahmedromman/mesh-deck/tree/main/examples/) for complete ones):

   ```yaml
   services:
     - name: chat
       port: 8001
       gpus: [0]
       gpu_memory_utilization: 0.55
       overhead_mib: 1500
       command: |
         source ~/venvs/vllm/bin/activate
         exec vllm serve /path/to/chat-model --port 8001 --gpu-memory-utilization 0.55

     - name: embeddings
       port: 8003
       gpus: [0]
       gpu_memory_utilization: 0.15
       overhead_mib: 500
       command: |
         source ~/venvs/vllm/bin/activate
         exec vllm serve /path/to/embedding-model --port 8003 --gpu-memory-utilization 0.15
   ```

2. Check that it fits, without starting anything:

   ```bash
   mesh plan models.yaml
   ```

3. Start it:

   ```bash
   mesh up models.yaml
   ```

4. In other terminals, from any directory:

   ```bash
   mesh status              # live view of every service; press q to quit
   mesh logs chat -f        # follow one service's log
   mesh down                # stop everything (Ctrl+C in the `mesh up` terminal does the same)
   ```

If the plan says a service will not fit, `mesh up` refuses to start unless you add `--force`. It also checks again
right before each service starts, in case another program took memory in the meantime. Output looks
like this (illustrative numbers):

```
2. embeddings   FAIL
     GPU 0: needs 3612 MiB free but only 2100 MiB is free
     try --gpu-memory-utilization 0.08 or less
```

The first time, run `mesh up` once and copy the printed "memory it added" numbers into `measured_mib`. The
planner is most accurate with measured numbers. See [Config](https://github.com/mrahmedromman/mesh-deck/blob/main/docs/config.md#getting-real-numbers-for-measured_mib).

### Where your files go

The first time you run `mesh up` in a terminal, it asks which folder to use for logs and state:

```
Which folder should it use? [~/mesh-deck]
```

Press Enter for the suggested folder in your home directory, or type another one. `mesh` creates it and
remembers your choice in `~/.config/mesh-deck.json`. Each service's log (`NAME.log`) and a small `state.json`
(which services are running) are saved there, as plain files on your own computer. Nothing is uploaded. Because
the folder is remembered, `mesh status`, `mesh down` and `mesh logs` work from any directory.

To use another folder for one run, add `--log-dir FOLDER` to every command of that stack. To be asked again,
delete the settings file. When output is not a terminal (a script or CI), `mesh` never asks: it uses the
remembered folder, or `logs` in the current directory if there is none.

### Try it without a GPU

`mesh plan` can read saved `nvidia-smi` output, so you can see what it does on any machine. Create two small files
(a pretend pair of 24 GB cards, and three services):

```bash
cat > gpus.csv <<'EOF'
index, uuid, name, memory.total [MiB], memory.used [MiB], memory.free [MiB]
0, GPU-00000000-0000-0000-0000-000000000000, NVIDIA GeForce RTX 4090, 24564 MiB, 417 MiB, 23662 MiB
1, GPU-00000000-0000-0000-0000-000000000001, NVIDIA GeForce RTX 4090, 24564 MiB, 4 MiB, 24078 MiB
EOF

cat > models.yaml <<'EOF'
services:
  - {name: chat,       port: 8001, gpus: [0], gpu_memory_utilization: 0.70, command: "echo chat"}
  - {name: embeddings, port: 8003, gpus: [0], gpu_memory_utilization: 0.30, command: "echo embeddings"}
  - {name: speech,     port: 8004, gpus: [1], gpu_memory_utilization: 0.30, command: "echo speech"}
EOF

mesh plan models.yaml --gpus-csv gpus.csv --external 0=417,1=4
```

The `embeddings` service does not fit next to `chat` on GPU 0, and the plan says how to fix it:

```
2. embeddings   FAIL
     GPU 0: needs 7224 MiB free but only 6806 MiB is free
     try --gpu-memory-utilization 0.28 or less
```

Change `embeddings` to `0.28` and run it again: it fits, with a warning that GPU 0 would have only 63 MiB to
spare. Nothing was started, and no GPU was needed.

## Commands

| Command | What it does |
|---|---|
| `mesh plan CONFIG` | Simulates starting the services in order and reports which ones fit. |
| `mesh up CONFIG` | Plans, starts each service in order, waits for its health check, and reports the memory it added. |
| `mesh down` | Stops a stack started by `mesh up`. |
| `mesh status` | A live view of every service: state, process id, port and health. Press `q` to quit. |
| `mesh logs NAME [-n N\|all] [-f]` | Shows or follows a service's log. |
| `mesh gpu` | Shows each GPU's memory and utilization, and who is holding the memory. |
| `mesh completion SHELL` | Prints a Tab-completion script for `bash`, `zsh` or `fish`. |

All options, `mesh logs` examples, exit codes and where files go: [docs/commands.md](https://github.com/mrahmedromman/mesh-deck/blob/main/docs/commands.md).

## How it works

The driver keeps back some GPU memory, and vLLM (the engine this was built around) measures free memory when each service starts. `mesh` subtracts
the reserved memory and anything other programs are using, then checks each service in order, reducing what is
free as services start. On two RTX 4090s the planned free memory matched the real `nvidia-smi` numbers to within
about 20 MiB once real footprints were entered. Details and limits: [docs/how-it-works.md](https://github.com/mrahmedromman/mesh-deck/blob/main/docs/how-it-works.md).

## Compared with

- **Docker Compose** can order containers with `depends_on` and healthchecks. It is the right tool if
  everything is a container. `mesh` also handles plain commands (a venv or conda environment) and checks
  GPU memory before starting.
- **[llama-swap](https://github.com/mostlygeek/llama-swap)** swaps models on demand behind one endpoint.
  `mesh` does not proxy requests or swap: it starts a fixed set of services that stay loaded.
- **A shell script** works fine for one machine. `mesh` replaces the parts that are tedious to redo: working
  out the fractions, waiting on health checks, separate logs, and a clean shutdown.

## Security

`command` and `stop` run through `bash` with your user's permissions, so a `models.yaml` is a script. Only run
configs you trust. `mesh` itself sends nothing anywhere: it runs `nvidia-smi`, starts your commands, and checks
`127.0.0.1` ports.

## Limitations

- NVIDIA GPUs and Unix-like systems only. Not tested on Windows, AMD or Intel GPUs.
- No request proxy or single endpoint; each service keeps its own port.
- It does not restart a service that crashes after startup; it prints a warning.
- `mesh gpu` needs Linux and reports a snapshot only: no live view, no per-process utilization, no killing of
  processes. It recognises Docker containers but not Podman or Kubernetes.
- Estimates for services nobody has measured are rough.
- Logs are overwritten on each `mesh up`, with no rotation.

## Contributing

Tests need no GPU: `pip install -e ".[dev]"` then `pytest`. The most useful contribution is real `nvidia-smi`
output from a GPU other than an RTX 4090, added as a test fixture. Bug reports and ideas are welcome as issues.
See [CONTRIBUTING.md](https://github.com/mrahmedromman/mesh-deck/blob/main/CONTRIBUTING.md) for the exact commands, what to include in a bug report, and what is in
scope.

## License

[MIT](https://github.com/mrahmedromman/mesh-deck/blob/main/LICENSE)
