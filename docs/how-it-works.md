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
4. **In order.** Services are checked in the order listed. Each one that starts reduces the free memory for the
   next. A service that does not fit is reported and not counted.
5. **Other programs.** Memory used by other programs (your desktop, other jobs) is read from `nvidia-smi` and
   subtracted first, or taken from `--external`.
6. **Margin.** `plan` warns if a GPU ends with less free memory than `margin_mib`.

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

1. Reads `nvidia-smi`, runs the plan, and prints it. If a service will not fit it stops, unless `--force` is
   given. Without `nvidia-smi` it skips the plan and the memory numbers and says so.
2. Refuses to start if a previous stack from this log directory is still running (run `mesh down` first).
3. Starts services one at a time, in the order listed. Each runs as `bash -c <command>` in its own process group.
4. After starting a service it polls the `health` path until **any** HTTP response arrives, or the service
   exits, or `start_timeout_s` passes.
5. When a service is healthy it prints how much GPU memory it added.
6. If a service exits or times out before it is healthy, it prints that service's last 20 log lines, stops every
   service already started, and exits with code 1.
7. When all are up it waits. If a service dies later, it prints a warning with the log path and leaves the
   others running; it does not restart it.
8. On Ctrl+C, SIGTERM, or `mesh down`, it stops services in reverse order: first the service's `stop` command if
   it has one, then SIGTERM to its process group, waiting up to 30 seconds, then SIGKILL. It prints a line as
   each service stops and a final "All N services stopped." (see
   [Shutdown messages](commands.md#shutdown-messages)).

## What it does not do

- It does not proxy requests or give one endpoint for all services. Each service keeps its own port. Tools such
  as llama-swap and LiteLLM already do routing well.
- It does not swap models on demand. The services you list stay loaded.
- It does not restart a crashed service.
- It does not yet report which process is holding GPU memory when a start fails.
- NVIDIA GPUs only, through `nvidia-smi`. Not tested on Windows.
