# How it works

Back to the [README](../README.md) · [Commands](commands.md) · [Config reference](config.md)

## The problem

vLLM checks free GPU memory when a service **starts**, against the fraction you asked for. If two services
start at once, or one starts after another has taken its share, the later one fails with an error like:

```
Free memory on device cuda:0 (9.62/23.51GiB) on startup is less than desired GPU memory utilization (0.7, 16.46 GiB).
```

The fix is to start services one at a time with fractions that fit. `mesh` works out whether they fit before
you launch.

## How the planner decides

1. **Usable memory.** The driver keeps back some memory (about 485 MiB on an RTX 4090). `nvidia-smi` counts it in
   the total, but vLLM does not: its log showed 23.51 GiB for a card `nvidia-smi` calls 24564 MiB. Fractions
   apply to total minus that reserved amount. `mesh` finds the reserved amount as `total - used - free` from
   `nvidia-smi`.
2. **Start check.** A vLLM service needs `gpu_memory_utilization x usable` MiB **free** on each of its GPUs when
   it starts.
3. **After it starts,** it holds `measured_mib` if you gave one, otherwise the requested amount plus
   `overhead_mib`.
4. **It must fit both ways.** A service fits only if the free memory covers what it needs to start *and* what it
   holds once running. A service can pass the start check and still hold more than is free, because of
   `overhead_mib` or a large `measured_mib`; the plan then reports it as not fitting. It suggests the largest
   `--gpu-memory-utilization` that would fit, allowing for `overhead_mib`. If the measured memory alone is too
   big, or the overhead leaves no room, no smaller fraction helps, and the plan says so instead of suggesting one.
5. **In order.** Services are checked in the order listed. Each one that starts reduces the free memory for the
   next. A service that does not fit is reported and not counted.
6. **Other programs.** Memory used by other programs (your desktop, other jobs) is read from `nvidia-smi` and
   subtracted first, or taken from `--external`.
7. **Margin.** `plan` warns if a GPU ends with less free memory than `margin_mib`.

When a service fails the check, `mesh` suggests the largest fraction (rounded down to two decimals) that would
fit at that point in the order.

### How accurate is it?

On the machine this was built on (two RTX 4090s, four services), the planned free memory after everything
started was within about 20 MiB of the real `nvidia-smi` numbers, once real footprints were entered as
`measured_mib`. For a service nobody has measured, the estimate is only the requested fraction plus your
`overhead_mib` guess, and it can be off by more. Run once, then copy the printed numbers into `measured_mib`.

Behaviour on other GPUs, drivers and vLLM versions is not yet known. See
[Contributing](../README.md#contributing) to help with that.

## What `mesh up` does

1. Reads `nvidia-smi` and runs the plan. When everything fits it prints one summary line (the full plan with
   `--plan`, or when the plan has a warning or note). If a service will not fit it stops, unless `--force` is
   given. Without `nvidia-smi` it skips the plan and the memory numbers and says so.
2. Refuses to start if a previous stack from this log directory is still running (run `mesh down` first).
3. Refuses to start if another program is already listening on any service's port, and names the port.
4. Right before each service starts, reads `nvidia-smi` again and checks that the memory free at that moment is
   enough for it. The plan in step 1 was made when `mesh up` began, and earlier services take a while to load, so
   another program may have taken memory since. If it is short, `mesh up` refuses to start that service, stops the
   ones it already started, and says who holds the memory. `--force` skips this check, but it does not create the memory,
   so the service will usually fail at its own start-up check instead. `mesh up` cannot reserve memory:
   something that grabs it after this check still makes the service fail at startup, as described in step 7.
5. After starting a service it polls the `health` path until it answers with a status below 500 (a `503` while a
   model loads means "not ready yet"), or the service exits, or `start_timeout_s` passes.
6. When a service is healthy it prints how much GPU memory it added.
7. If a service exits or times out before it is healthy, it prints that service's last 20 log lines. When the likely
   cause is further up the log (vLLM prints `Free memory on device ...` far above the traceback that ends it), those
   few lines are shown first. Then it stops every service already started and exits with code 1.
8. When all are up it waits. If a service dies later, it prints a warning with the log path, stops what is left
   of its process group, and leaves the others running; it does not restart it.
9. On Ctrl+C, SIGTERM, or `mesh down`, it stops services in reverse order: first the service's `stop` command if
   it has one, then SIGTERM to its process group, waiting up to the service's
   [`stop_timeout_s`](config.md#stopping-a-service) (30 seconds by default), then SIGKILL. It prints a line as
   each service stops and a final "All N services stopped." (see
   [Shutdown messages](commands.md#shutdown-messages)).

## How `mesh gpu` finds the owner of GPU memory

`nvidia-smi` lists GPU processes by process id, with names like `VLLM::EngineCore`. Two services can show up with
the same name, so the name alone does not say whose memory it is. `mesh` works out the owner like this:

1. **Mesh services.** `mesh up` starts each service in its own process group and records it in `state.json`. A
   GPU process belongs to a service if it is in that group, or if the service's process is one of its ancestors.
2. **Docker containers.** For a process that is not in a mesh group, `mesh` reads `/proc/<pid>/cgroup`. A
   Docker container shows up there as `docker-<id>.scope` or `/docker/<id>`. `docker ps` turns the id into a
   name. If that name appears in a service's `command` or `stop` as `--name NAME` or `docker stop NAME`, the
   memory is attributed to that service.
3. **Desktop.** Processes that `nvidia-smi` marks as graphics (type G) are grouped as `desktop`.
4. **Everything else** is shown with its user, process id and command line.
5. **Unattributed.** Memory used minus everything listed above.

Processes come from `nvidia-smi --query-compute-apps` (compute processes) and `nvidia-smi -q -x` (which also
lists graphics processes). If the XML cannot be read, graphics apps are simply not itemised.

Some limits: container matching relies on the container name appearing in your config, as above. Only Docker is
recognised, not Podman or Kubernetes. The graphics-process parsing was written against the documented XML layout
and has not yet been checked against real output from many drivers.

## What it does not do

- It does not proxy requests or give one endpoint for all services. Each service keeps its own port. Tools such
  as llama-swap and LiteLLM already do routing well.
- It does not swap models on demand. The services you list stay loaded.
- It does not restart a crashed service.
- It does not kill processes. `mesh gpu` only reports who is holding memory.
- It does not show live, updating utilization or per-process utilization; tools such as nvitop do that well.
- NVIDIA GPUs only, through `nvidia-smi`. Not tested on Windows.
