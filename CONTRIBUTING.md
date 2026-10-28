# Contributing to mesh-deck

Thanks for helping. This project is small and early, so the most useful contributions are real-world data and
bug reports. Please read [How it works](docs/how-it-works.md) first, especially "What it does not do", so you
know what is in scope. By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## The most useful thing you can send: output from your GPU

The planner has only been checked on two RTX 4090s. If you have a different GPU, driver or vLLM version,
real output from it is worth more than code.

1. Run these two commands, ideally with some services running:

   ```bash
   nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv
   nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
   ```

   If you can, also run `nvidia-smi -q -x` (it lists graphics processes such as the desktop) and keep only the
   `<gpu>` blocks with their `<uuid>` and `<processes>`.
2. Replace the `uuid` values with dummy ones (for example `GPU-00000000-0000-0000-0000-000000000000`).
3. Open a pull request that adds the output to `tests/fixtures/` and a test in `tests/test_gpu.py` or
   `tests/test_planner.py` that parses it. If the planned memory is wrong for your card, say so in the pull
   request. That is a bug we want to know about.

## Health paths for other engines

The [table of health paths](docs/config.md#health-paths-for-common-engines) was put together from documentation and
third-party guides. If you run another engine, or find a path that is wrong for your version, open an issue with the
engine, its version, the path, and what `curl -i` showed while the model was loading and once it was ready.

## Reporting a bug

Open an issue with:

- `mesh --version`, your OS, GPU model(s), driver version, and the engine you run (for example vLLM and its version).
- The command you ran and the full output.
- Your `models.yaml` **with private paths and model names removed**.
- The two `nvidia-smi` outputs above, from just before the failure if you can.

For a service that fails to start, also include the last lines of its log (`NAME.log` in your mesh folder, `~/mesh-deck` by default).

## Development setup

```bash
git clone https://github.com/mrahmedromman/mesh-deck.git
cd mesh-deck
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Before opening a pull request, also run the style and type checks that CI runs:

```bash
ruff check src tests
mypy
```

The tests need no GPU and no model. Runner tests start throwaway `http.server` processes, so they take a few
seconds.

### Layout

| Path | What it holds |
|---|---|
| `src/mesh_cli/config.py` | `models.yaml` loading and validation |
| `src/mesh_cli/gpu.py` | `nvidia-smi` output parsing |
| `src/mesh_cli/holders.py` | working out who holds GPU memory (`mesh gpu`) |
| `src/mesh_cli/planner.py` | the memory simulation |
| `src/mesh_cli/runner.py` | starting, health-checking, logging and stopping services |
| `src/mesh_cli/suggest.py` | "did you mean" hints |
| `src/mesh_cli/completion.py` | bash, zsh and fish completion scripts, generated from the argument parser |
| `src/mesh_cli/ui.py` | colour, tables, memory bars and the live start-up line (terminal only) |
| `src/mesh_cli/cli.py` | the `mesh` command |
| `tests/` | tests, with real `nvidia-smi` output in `tests/fixtures/` |
| `examples/`, `docs/` | example configs and the documentation |

## Pull requests

- **Keep them small** and about one thing.
- **Add or update a test** for any behaviour change. A bug fix should come with a test that fails without it.
- **Run `pytest`, `ruff check src tests` and `mypy`** before opening the pull request.
- **Add a line to `CHANGELOG.md`** under "Unreleased" for anything a user would notice.
- **Update the docs** if you change a command, an option or a config field: `README.md` and the pages in `docs/`.
- **Keep private details out.** Examples and fixtures must not contain your hostnames, home paths, model names
  or GPU UUIDs. The examples use generic names (`llm`, `stt`, `tts`, `embed`) on purpose.
- Follow the style of the surrounding code: type hints, short functions, comments only where the reason is not
  obvious.

## Scope

mesh checks GPU memory, starts services in order and keeps their logs. It deliberately does not proxy requests,
swap models on demand or replace a process manager. Tools such as llama-swap, LiteLLM and Docker Compose
already do those well. If you are unsure whether an idea fits, open an issue to ask before writing code.

Ideas that would fit well:

- Support for AMD GPUs (a parser for `rocm-smi`) and other vendors.
- Recognising Podman and Kubernetes containers in `mesh gpu`.
- Checking that other engines (not only vLLM) are planned correctly.
- Exporting a config for another tool, such as llama-swap.

## License

By contributing you agree that your contributions are licensed under the [MIT License](LICENSE).
