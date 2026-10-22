import pytest

from mesh_cli.config import ConfigError, load_config


def write(tmp_path, text):
    path = tmp_path / "models.yaml"
    path.write_text(text)
    return path


def test_example_config_loads():
    config = load_config("examples/dual-gpu.yaml")
    assert [s.name for s in config.services] == ["llm", "stt", "tts", "embed"]
    assert config.services[0].measured_mib == {0: 16252, 1: 16252}


def test_single_gpu_example_loads_and_fits_a_24gb_card():
    from mesh_cli.planner import GpuBudget, make_plan

    config = load_config("examples/single-gpu.yaml")
    assert [s.name for s in config.services] == ["chat", "embeddings"]
    assert all(s.command for s in config.services)
    # a 24 GB card with a little desktop use
    plan = make_plan(config, [GpuBudget(0, 24564, 300, "24GB GPU", 485)])
    assert plan.ok
    assert plan.warnings == []


def test_utilization_must_be_a_fraction(tmp_path):
    path = write(tmp_path, "services:\n  - {name: a, port: 1, gpus: [0], gpu_memory_utilization: 1.5}\n")
    with pytest.raises(ConfigError, match="gpu_memory_utilization"):
        load_config(path)


def test_duplicate_ports_rejected(tmp_path):
    path = write(tmp_path, "services:\n  - {name: a, port: 8000, gpus: [0]}\n  - {name: b, port: 8000, gpus: [0]}\n")
    with pytest.raises(ConfigError, match="duplicate service port"):
        load_config(path)


def test_measured_gpu_must_be_in_gpus(tmp_path):
    path = write(tmp_path, "services:\n  - {name: a, port: 1, gpus: [0], measured_mib: {1: 100}}\n")
    with pytest.raises(ConfigError, match="not in gpus"):
        load_config(path)


def test_unknown_field_rejected(tmp_path):
    path = write(tmp_path, "services:\n  - {name: a, port: 1, gpus: [0], gpu_mem_util: 0.5}\n")
    with pytest.raises(ConfigError, match="gpu_mem_util"):
        load_config(path)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "nope.yaml")
