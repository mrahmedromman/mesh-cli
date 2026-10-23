# Contributing to mesh-cli

Thanks for helping. This project is small and early, so the most useful contributions are real-world data and
bug reports. Please read [How it works](docs/how-it-works.md) first, especially "What it does not do", so you
know what is in scope.

## The most useful thing you can send: output from your GPU

The planner has only been checked on two RTX 4090s. If you have a different GPU, driver or vLLM version,
real output from it is worth more than code.

1. Run these two commands, ideally with some services running:

   ```bash
   nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv
   nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
   ```

2. Replace the `uuid` values with dummy ones (for example `GPU-00000000-0000-0000-0000-000000000000`).
3. Open a pull request that adds the output to `tests/fixtures/` and a test in `tests/test_gpu.py` or
   `tests/test_planner.py` that parses it. If the planned memory is wrong for your card, say so in the pull
   request. That is a bug we want to know about.

## Reporting a bug

Open an issue with:

- `mesh --version`, your OS, GPU model(s), driver version, and the engine you run (for example vLLM and its version).
- The command you ran and the full output.
- Your `models.yaml` **with private paths and model names removed**.
- The two `nvidia-smi` outputs above, from just before the failure if you can.

For a service that fails to start, also include the last lines of its `logs/NAME.log`.

## Development setup

```bash
git clone https://github.com/mrahmedromman/mesh-cli.git
cd mesh-cli
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The tests need no GPU and no model. Runner tests start throwaway `http.server` processes, so they take a few
seconds.

### Layout

| Path | What it holds |
|---|---|
| `src/mesh_cli/config.py` | `models.yaml` loading and validation |
| `src/mesh_cli/gpu.py` | `nvidia-smi` output parsing |
| `src/mesh_cli/planner.py` | the memory simulation |
| `src/mesh_cli/runner.py` | starting, health-checking, logging and stopping services |
| `src/mesh_cli/cli.py` | the `mesh` command |
| `tests/` | tests, with real `nvidia-smi` output in `tests/fixtures/` |
| `examples/`, `docs/` | example configs and the documentation |

## Pull requests

- **Keep them small** and about one thing.
- **Add or update a test** for any behaviour change. A bug fix should come with a test that fails without it.
- **Run `pytest`** before opening the pull request.
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
- Reporting which process is holding GPU memory when a service fails to start.
- Checking that other engines (not only vLLM) are planned correctly.
- Exporting a config for another tool, such as llama-swap.

## License

By contributing you agree that your contributions are licensed under the [MIT License](LICENSE).
