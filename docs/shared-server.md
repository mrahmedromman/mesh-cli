# Using mesh-deck on a shared server

Back to the [README](../README.md) · [Commands](commands.md) · [Config reference](config.md) · [How it works](how-it-works.md)

mesh-deck helps people who share GPUs start their services without out-of-memory crashes. It does not decide who may
use the server: that is an agreement between people (a chat, a calendar, a small claim page). This page says what
`mesh` does, and does not do, when more than one person or terminal is involved.

## What mesh does, and what it does not

It does:

- **Plan from the memory that is free right now.** `mesh plan` and `mesh up` read `nvidia-smi`, and memory held by
  anyone else counts as "in use by others".
- **Check again right before each service starts,** and refuse if something took the memory in the meantime.
- **Show who holds the memory.** `mesh gpu` names each holder: your services, a Docker container, the desktop, or
  another program with its user name.
- **Refuse a second `mesh up` in the same folder.** Each `mesh up` locks its state folder (`mesh.lock`) while it runs.

It does not queue people, reserve memory, schedule, move a model to another GPU, or stop other people's processes.

## Which setup is yours?

| Setup | State folder | What keeps people from colliding |
|---|---|---|
| One person, one account | `~/mesh-deck` | The folder lock: two terminals cannot both start a stack. |
| Several people, each with an account | Each account has its own `~/mesh-deck` | The GPU's real memory, read from `nvidia-smi`. |
| Several people, one shared account | One shared `~/mesh-deck` | The folder (one stack at a time), plus whatever you agree between yourselves. |

### One person, one account

Run `mesh up` in one terminal, and `mesh status`, `mesh logs` and `mesh down` from any other, in any folder. A second
`mesh up` in the same folder is refused, either with `another mesh up is already using this folder (pid N)` or with
`already running (...)`, depending on whether the first one has started a service yet.

### Separate accounts

- **Settings and state are per account.** The remembered folder (`~/.config/mesh-deck.json`) and the state folder are
  in each person's home, so each person's `mesh status` shows only their own stack.
- **The GPU is shared.** Memory used by another account counts as "in use by others" in your plan, and the check
  before each start catches someone who started a service in between.
- **`mesh gpu` shows other accounts' processes** with their user name, process id and command line. A Docker
  container shows under its container name, without a user.
- **`mesh down` stops only what your own state folder lists.** Another account's stack is not in it. To free memory
  that someone else holds, ask them.

### One shared account

- **Everyone shares one folder and one state.** Only one stack can run at a time: a second `mesh up` is refused
  until the first is stopped.
- **Anyone can stop it.** `mesh status` and `mesh down` from any terminal act on that one stack, so agree who has the
  server before you start.
- **`mesh gpu` cannot say which person holds the memory,** because every process belongs to the same account. Put the
  person in the name instead, for example `docker run --name alice-llm`, and the table shows it.
- **`mesh` does not queue.** Use a chat, a calendar, or a small claim and queue page. Agree who holds the server, for
  how long, and that the holder runs `mesh down` when finished.
- **Separate stacks at the same time** need separate folders (`--log-dir ~/mesh-deck/alice`). Then `mesh` no longer
  stops a second start by itself, and only the check of the free memory protects you. Use it only if you want that.

## Good habits, in every setup

- Run `mesh plan` before `mesh up`.
- Stop your stack with `mesh down` (or Ctrl+C in the `mesh up` terminal) when you are finished.
- After a crash, run `mesh gpu`: a leftover process from a stack shows up there with the memory it still holds.

## Messages you may see

| Message | What it means |
|---|---|
| `another mesh up is already using this folder (pid N)` | Another `mesh up` (yours or someone else's) is starting or running in this folder. Wait for it, or stop it with `mesh down`. |
| `already running (llm, stt)` | A stack from this folder is running. Stop it with `mesh down` first. |
| `NAME would not fit right now, so it was not started` | Something took GPU memory after the plan was made. The table under it shows who holds it. Free the memory or lower `gpu_memory_utilization`; `--force` only skips the check. |
| `N services could not be stopped` | The processes belong to someone else, or would not stop. `mesh down` exits with code 1 and keeps their records, so `mesh status` still shows them and running `mesh down` again tries once more. |

## Limits

- On a network drive (some NFS home folders), file locks may not work. `mesh` then relies on the check of the `mesh up`
  recorded in the state file, which still refuses a second one that starts later. Two that start in the same instant
  could both get through.
- `mesh gpu` needs Linux and `nvidia-smi`, and shows a snapshot only.
