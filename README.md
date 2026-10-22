# mesh-cli

Plan and start a stack of local AI services (vLLM, speech-to-text, embeddings, TTS) on shared GPUs, in the
right order, with one log file per service.

If you have run several vLLM or Docker services on the same GPU, you have probably hit this:
the second one crashes at startup with *"Free memory on device ... is less than desired GPU memory
utilization"*, because each service measures free memory when it starts and the first one already
took its share. `mesh` works out whether your services fit **before** you launch, starts them one
at a time (each waits for the previous one to answer its health check), and keeps their logs apart.

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

> **Status: early (0.0.1).** Built and checked on one machine with two RTX 4090s. It reads `nvidia-smi`,
> so it needs an NVIDIA GPU and a Unix-like system (tested on Linux). Windows is not supported. Reports from other GPUs are very welcome.

## Install

```bash
pip install mesh-cli        # not published yet; for now:
git clone <this repo> && cd mesh-cli && pip install -e .
```

Use a separate virtual environment from the one that runs your models. `mesh` only needs `pydantic` and
`pyyaml`.

## Quickstart

1. Copy [examples/single-gpu.yaml](examples/single-gpu.yaml) to `models.yaml` and set your model paths.
2. Check that it fits:

   ```bash
   mesh plan models.yaml
   ```

3. Start everything:

   ```bash
   mesh up models.yaml
   ```

4. In other terminals:

   ```bash
   mesh status              # which services are running, and whether they answer
   mesh logs chat -f        # follow one service's log
   mesh down                # stop everything (Ctrl+C in the `mesh up` terminal does the same)
   ```

If the plan says a service will not fit, `mesh up` refuses to start unless you add `--force`:

```
2. stt      FAIL
     GPU 0: needs 16856 MiB free but only 7410 MiB is free
     try --gpu-memory-utilization 0.30 or less
```

## Commands

| Command | What it does |
|---|---|
| `mesh plan CONFIG` | Simulates starting the services in order and reports which ones fit. Exit code 1 if one does not. |
| `mesh up CONFIG` | Runs the plan, then starts each service, waits for its health check, and prints the memory it added. Blocks until Ctrl+C. |
| `mesh down` | Stops a stack started by `mesh up`, in reverse order. |
| `mesh status` | Shows each service's process, port and health. |
| `mesh logs NAME [-n N] [-f]` | Shows or follows `logs/NAME.log`. |

`plan` also accepts `--gpus-csv FILE` (saved `nvidia-smi` output, handy for testing) and
`--external 0=417,1=4` (memory in use by other programs, in MiB). All commands take `--log-dir`.

## models.yaml

See [examples/single-gpu.yaml](examples/single-gpu.yaml) and [examples/dual-gpu.yaml](examples/dual-gpu.yaml).

| Field | Meaning |
|---|---|
| `name`, `port`, `gpus` | Required. `gpus` is a list of GPU indexes the service uses. |
| `command` | Shell snippet run with `bash -c` (needed for `up`). End it with `exec <program>`. Activate a venv or conda env on the first lines if you need one. |
| `stop` | Optional command run first when stopping, e.g. `docker stop NAME`. |
| `gpu_memory_utilization` | vLLM-style fraction of each GPU. Used to check that the service can start. |
| `measured_mib` | Memory the service really held per GPU, e.g. `{0: 16252}`. Best source of truth. `mesh up` prints it for you. |
| `overhead_mib` | Extra memory beyond the fraction, used only when nothing is measured yet. |
| `health` | Path polled until the service answers. Default `/v1/models`. Any HTTP response counts. |
| `start_timeout_s` | How long to wait for the health check. Default 900. |
| `margin_mib` (top level) | Free memory that should remain on each GPU after everything starts. Default 1024. |

## How the planner decides

1. The driver keeps back some memory (about 485 MiB on a 4090). `nvidia-smi` counts it in the total, but
   vLLM does not: its log showed 23.51 GiB for a card `nvidia-smi` calls 24564 MiB. Fractions apply to
   total minus that reserved amount.
2. A vLLM service needs `gpu_memory_utilization x usable` MiB **free** when it starts.
3. After it starts it holds `measured_mib` if you gave it, otherwise the requested amount plus `overhead_mib`.
4. Services are checked in the order listed. Each one reduces the free memory for the next, and a service
   that does not fit is not counted.
5. Memory used by other programs (your desktop, other jobs) is read from `nvidia-smi` and subtracted first.

On the machine this was built on, planned free memory after all four services matched the real
`nvidia-smi` numbers to within about 20 MiB.

## How it compares

- **Docker Compose** can order containers with `depends_on` and healthchecks. It is the right tool if
  everything is a container. `mesh` also handles plain commands (a venv or conda environment) and checks
  GPU memory before starting.
- **[llama-swap](https://github.com/mostlygeek/llama-swap)** swaps models on demand behind one endpoint.
  `mesh` does not proxy requests or swap: it starts a fixed set of services that stay loaded.
- A plain shell script works fine for one machine. This replaces the parts that are tedious to redo:
  working out the fractions, waiting on health checks, separate logs, and a clean shutdown.

## Not done yet

- No request proxy or single endpoint; each service keeps its own port.
- AMD and Intel GPUs, Windows.
- It does not restart a service that crashes after startup; it prints a warning.
- Estimates for models nobody has measured are rough. Run once, then copy the printed memory into `measured_mib`.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests need no GPU. `tests/fixtures/` holds real `nvidia-smi` output. To help on a different GPU, send the
output of:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

(remove the `uuid` values if you prefer) as a pull request with a test.

## License

MIT
