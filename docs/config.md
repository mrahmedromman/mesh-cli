# models.yaml reference

Back to the [README](../README.md) · [Commands](commands.md) · [How it works](how-it-works.md)

Two complete examples: [examples/single-gpu.yaml](../examples/single-gpu.yaml) (a chat model and an embedding
model on one card) and [examples/dual-gpu.yaml](../examples/dual-gpu.yaml) (four services on two cards).

```yaml
margin_mib: 1024              # optional, top level

services:
  - name: chat
    port: 8001
    gpus: [0]
    gpu_memory_utilization: 0.55
    overhead_mib: 1500
    health: /v1/models
    command: |
      source ~/venvs/vllm/bin/activate
      exec vllm serve /path/to/model --port 8001 --gpu-memory-utilization 0.55
```

Unknown fields are rejected with the field name, so a typo such as `gpu_mem_util` is reported instead of
silently ignored. Service names and ports must be unique.

## Service fields

| Field | Required | Meaning |
|---|---|---|
| `name` | yes | Used in output, log file names (`NAME.log`) and `mesh logs NAME`. |
| `port` | yes | Port polled for the health check. Must be unique across services. |
| `gpus` | yes | List of GPU indexes the service uses, as `nvidia-smi` numbers them. This is what the plan assumes; your `command` must really use these GPUs (see [Choosing GPUs](#choosing-gpus-for-a-plain-command)). |
| `command` | for `up` | Shell snippet run with `bash -c`. See [Writing a command](#writing-a-command). Not needed for `plan`. |
| `stop` | no | Command run first when stopping, e.g. `docker stop NAME`. Recommended for Docker services. |
| `gpu_memory_utilization` | no | vLLM-style fraction (above 0, up to 1) of each GPU. Used to check the service can start. |
| `measured_mib` | no | Memory the service really held on each GPU, e.g. `{0: 16252, 1: 16252}`. The best source of truth. |
| `overhead_mib` | no | Extra memory beyond the fraction, used only for a GPU with no `measured_mib`. Default 0. |
| `health` | no | Path polled until the service answers. Default `/v1/models`. Any HTTP response counts as alive. |
| `start_timeout_s` | no | Seconds to wait for the health check. Default 900. |

`measured_mib` may only list GPUs that are in `gpus`.

A service needs either `gpu_memory_utilization` or `measured_mib` to be planned. With neither, `mesh plan`
counts its memory as 0 and says so.

## Using an engine other than vLLM

`gpu_memory_utilization` is vLLM's idea of a memory budget. Other engines do not have it, so for them:

- Leave out `gpu_memory_utilization` and set `measured_mib` (see below). Without a fraction, the planner uses your measured numbers as the footprint.
- Set `health` to a path the engine answers on. The default, `/v1/models`, suits OpenAI-compatible servers.
- A service with only `measured_mib` gets no "try --gpu-memory-utilization" hint when it does not fit, because there is no fraction to lower.

A measured-only service is planned correctly (checked with `mesh plan`), but no engine other than vLLM has been run end to end yet.

## Top-level fields

| Field | Meaning |
|---|---|
| `services` | The list of services, in the order they should start. At least one. |
| `margin_mib` | Free memory that should remain on each GPU after everything starts. Default 1024. `plan` warns below it. |

## Getting real numbers for `measured_mib`

1. Start with `gpu_memory_utilization` and a guess for `overhead_mib` (1500 is a reasonable first guess for a
   vLLM service; it came from one machine, so treat it as a guess).
2. Run `mesh up`. After each service is healthy it prints `memory it added: GPU 0: ~16260 MiB`.
3. Put those numbers in `measured_mib` and delete `overhead_mib`.

The printed figure is the change in memory used while that service started, so it is slightly off if something
else was using the GPU at the same moment. If `nvidia-smi` cannot be read at that moment, the line says
`memory it added: unavailable` and the start carries on; run `mesh up` again for a figure.

## Writing a command

`command` is run as `bash -c <command>`, in its own process group, with output written to `NAME.log` in your mesh folder (see [Where files go](commands.md#where-files-go)).

- **End with `exec`.** `exec vllm serve ...` replaces the shell with the program, so Ctrl+C and `mesh down`
  reach the real process.
- **Activate environments on the first lines**: `source ~/venvs/vllm/bin/activate` or `conda activate my-env`.
  For conda you may need `source ~/miniconda3/etc/profile.d/conda.sh` before `conda activate`.
- **Docker services:** use `exec docker run --rm --name NAME ...` and add `stop: docker stop NAME`.
- **Multi-line commands** work with a YAML block (`command: |`); lines ending in `\` continue as in a shell.

### Choosing GPUs for a plain command

`gpus` only tells the plan which GPUs a service uses. `mesh` does not choose the GPU for your command, so the
command has to match, for example `CUDA_VISIBLE_DEVICES=1 exec vllm serve ...` for `gpus: [1]`. Without it most
programs use GPU 0, and the plan is wrong. On a machine with different GPU models, also set
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, because otherwise CUDA can number the GPUs differently from `nvidia-smi`.

After each service is up, `mesh up` compares the memory before and after. If a GPU you listed gained almost nothing
(under 100 MiB) it prints a warning, and names any other GPU that grew meanwhile:

```
warning: llm added almost no memory on GPU 1 (+3 MiB); GPU 0 grew by ~16260 MiB meanwhile. The plan assumes llm uses GPU 1: make its command use it too, for example CUDA_VISIBLE_DEVICES=1.
```

The warning is a hint, not an error: the stack keeps starting. Another program that starts at the same moment can
also move the numbers.

### Choosing GPUs for a container

`--gpus '"device=1"'` shows the container only GPU 1, which becomes device 0 inside it. **Do not also set
`CUDA_VISIBLE_DEVICES=1`** in that container: it hides the only GPU, and many programs then silently run on
the CPU. You can check with:

```bash
docker exec NAME python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

It should print `True 1`.

## Health paths

A service counts as started when its `health` path returns an HTTP response below 500 (200, 401, 404 and 405
all count), because that proves the server is listening. A 5xx answer, most often `503` while a model is still
loading, means "reachable but not ready", so `mesh` keeps waiting and starts the next service only after the
answer changes. If the wait runs out, the error says what the service last answered. Use `/v1/models` for
OpenAI-compatible servers and `/docs` for FastAPI apps. If a service only answers on a different route, set
`health` to that route.

Before anything starts, `mesh up` also checks that no other program is already listening on a service's port. A
leftover process there would answer the health check and make a service that never started look healthy, so
`mesh up` stops with a message naming the port and the service.
