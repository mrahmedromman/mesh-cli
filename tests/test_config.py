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


def test_the_removed_runtime_field_is_explained_not_just_rejected(tmp_path):
    path = write(tmp_path, "services:\n  - {name: a, port: 1, gpus: [0], runtime: docker}\n")
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    message = str(excinfo.value)
    assert "services.0.runtime" in message and "removed" in message and "delete the line" in message
    assert "command" in message  # a Docker service is started by its command


def test_the_examples_do_not_use_the_removed_field():
    import pathlib

    for example in pathlib.Path("examples").glob("*.yaml"):
        assert "runtime:" not in example.read_text(), example


def test_stop_timeout_defaults_to_30_seconds_and_can_be_set_per_service(tmp_path):
    path = write(tmp_path, "services:\n"
                           "  - {name: a, port: 1, gpus: [0]}\n"
                           "  - {name: b, port: 2, gpus: [0], stop_timeout_s: 120}\n")
    a, b = load_config(path).services
    assert (a.stop_timeout_s, b.stop_timeout_s) == (30, 120)


@pytest.mark.parametrize("value", [0, -5])
def test_stop_timeout_must_be_positive(tmp_path, value):
    path = write(tmp_path, f"services:\n  - {{name: a, port: 1, gpus: [0], stop_timeout_s: {value}}}\n")
    with pytest.raises(ConfigError, match="stop_timeout_s"):
        load_config(path)


# --- mistakes that used to fail late or silently are rejected when the config is loaded --------------------------------


def one_service(tmp_path, **fields):
    text = ", ".join(f"{key}: {value}" for key, value in {"name": "a", "port": 1, "gpus": "[0]", **fields}.items())
    return write(tmp_path, f"services:\n  - {{{text}}}\n")


@pytest.mark.parametrize("health", ["health", "v1/models", "http://127.0.0.1:8000/health", "' /health'", "'/health '", "'/he alth'", "/état", "'/v1–models'"])
def test_health_must_be_a_path_that_starts_with_a_slash_and_has_no_spaces(tmp_path, health):
    with pytest.raises(ConfigError, match=r"services\.0\.health: .*health must be a path that starts with /"):
        load_config(one_service(tmp_path, health=health))


@pytest.mark.parametrize("health", ["/", "/health", "/v1/models", "/health?ready=1", "/docs", "/%C3%A9tat"])
def test_ordinary_health_paths_are_accepted(tmp_path, health):
    assert load_config(one_service(tmp_path, health=f'"{health}"')).services[0].health == health


@pytest.mark.parametrize("name", ['"my/model"', "'a\\b'", '"."', '".."', '"../x"'])
def test_a_service_name_must_work_as_a_file_name(tmp_path, name):
    with pytest.raises(ConfigError, match=r"services\.0\.name: .*cannot contain / or \\ or be . or \.\."):
        load_config(one_service(tmp_path, name=name))


@pytest.mark.parametrize("name", ["qwen3.5", '"whisper v3"', "llm-1", ".hidden", "üñí"])
def test_ordinary_service_names_are_accepted(tmp_path, name):
    assert load_config(one_service(tmp_path, name=name)).services[0].name == name.strip('"')


def test_a_gpu_listed_twice_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match=r"services\.0\.gpus: .*lists GPU 0 more than once"):
        load_config(one_service(tmp_path, gpus="[0, 0]"))


@pytest.mark.parametrize("gpus", ["[0]", "[0, 1]", "[1, 0]"])
def test_each_gpu_once_is_fine(tmp_path, gpus):
    assert load_config(one_service(tmp_path, gpus=gpus)).services[0].gpus == eval(gpus)
