<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img alt="mesh-cli" src="docs/assets/logo-light.svg" width="320">
  </picture>
</p>

<p align="center"><b>Know your AI stack fits your GPUs before you start it.</b></p>

<p align="center">
  <img alt="license: MIT" src="https://img.shields.io/badge/license-MIT-blue">
  <img alt="python: 3.12+" src="https://img.shields.io/badge/python-3.12%2B-blue">
  <img alt="status: early" src="https://img.shields.io/badge/status-early-orange">
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="docs/commands.md">Commands</a> ·
  <a href="docs/config.md">Config</a> ·
  <a href="docs/how-it-works.md">How it works</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

`mesh` plans and starts a stack of local AI services (vLLM, speech-to-text, embeddings, TTS) on shared GPUs,
in the right order, with one log file per service.

```
$ mesh up models.yaml
[1/4] starting llm on port 8001  (log: logs/llm.log)
      llm is up  (memory it added: GPU 0: ~16260 MiB, GPU 1: ~16260 MiB)
[2/4] starting stt on port 8004  (log: logs/stt.log)
      stt is up  (memory it added: GPU 0: ~6436 MiB)
[3/4] starting tts on port 8005  (log: logs/tts.log)
      tts is up  (memory it added: GPU 1: ~2353 MiB)
[4/4] starting embed on port 8003  (log: logs/embed.log)
      embed is up  (memory it added: GPU 1: ~3059 MiB)
All services are up. Press Ctrl+C to stop everything.
```

> **Early (0.0.1).** Built and checked on one machine with two RTX 4090s. It reads `nvidia-smi`, so it needs an
> NVIDIA GPU and a Unix-like system (tested on Linux). Reports from other GPUs are very welcome.

## Why

If you run several vLLM or Docker services on one machine, you have probably hit this: the second service
crashes at startup with *"Free memory on device ... is less than desired GPU memory utilization"*. Each service
measures free memory when it starts, and the first one already took its share. Working out fractions that fit,
starting things in the right order, and finding which log holds the error is tedious to redo by hand.

## Features

- **Checks first.** Simulates GPU memory for your services in start order and tells you which will not fit,
  with a suggested fraction.
- **Starts in order.** Each service waits for the previous one's health check, so they never race for memory.
- **Any runtime.** A service is a shell command: a venv, a conda environment or `docker run`.
- **One log per service.** `mesh logs llm -f` follows one; `-n all` prints the whole file.
- **Reports real memory.** After each start it prints how much GPU memory that service added.
- **Clean shutdown.** One Ctrl+C (or `mesh down`) stops everything in reverse order.
- **Just a YAML file.** No daemon, no proxy, nothing listening on a port.

## Install

A PyPI release is planned. Until then, install from source (use a separate environment from the one that
runs your models; `mesh` only needs `pydantic` and `pyyaml`):

```bash
git clone https://github.com/mrahmedromman/mesh-cli.git
cd mesh-cli
pipx install .            # or: pip install .
```

## Quickstart

1. Write a `models.yaml` (see [examples/](examples/) for complete ones):

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

4. In other terminals, from the same directory:

   ```bash
   mesh status              # what is running, and whether it answers
   mesh logs chat -f        # follow one service's log
   mesh down                # stop everything (Ctrl+C in the `mesh up` terminal does the same)
   ```

If the plan says a service will not fit, `mesh up` refuses to start unless you add `--force`. Output looks
like this (illustrative numbers):

```
2. embeddings   FAIL
     GPU 0: needs 3612 MiB free but only 2100 MiB is free
     try --gpu-memory-utilization 0.08 or less
```

The first time, run `mesh up` once and copy the printed "memory it added" numbers into `measured_mib`. The
planner is most accurate with measured numbers. See [Config](docs/config.md#getting-real-numbers-for-measured_mib).

## Commands

| Command | What it does |
|---|---|
| `mesh plan CONFIG` | Simulates starting the services in order and reports which ones fit. |
| `mesh up CONFIG` | Plans, starts each service in order, waits for its health check, and reports the memory it added. |
| `mesh down` | Stops a stack started by `mesh up`. |
| `mesh status` | Shows each service's state, process id, port and health. |
| `mesh logs NAME [-n N\|all] [-f]` | Shows or follows a service's log. |

All options, `mesh logs` examples, exit codes and where files go: [docs/commands.md](docs/commands.md).

## How it works

The driver keeps back some GPU memory, and vLLM measures free memory when each service starts. `mesh` subtracts
the reserved memory and anything other programs are using, then checks each service in order, reducing what is
free as services start. On two RTX 4090s the planned free memory matched the real `nvidia-smi` numbers to within
about 20 MiB once real footprints were entered. Details and limits: [docs/how-it-works.md](docs/how-it-works.md).

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
- It does not yet say which process is holding GPU memory when a service fails to start.
- Estimates for services nobody has measured are rough.
- Logs are overwritten on each `mesh up`, with no rotation.

## Contributing

Tests need no GPU: `pip install -e ".[dev]"` then `pytest`. The most useful contribution is real `nvidia-smi`
output from a GPU other than an RTX 4090, added as a test fixture. Bug reports and ideas are welcome as issues.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the exact commands, what to include in a bug report, and what is in
scope.

## License

[MIT](LICENSE)
