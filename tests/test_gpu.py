from pathlib import Path

import pytest

from mesh_cli.gpu import GpuError, parse_compute_apps, parse_gpu_query

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_real_gpu_query():
    gpus = parse_gpu_query((FIXTURES / "gpu_query_running.csv").read_text())
    assert [g.index for g in gpus] == [0, 1]
    assert gpus[0].name == "NVIDIA GeForce RTX 4090"
    assert (gpus[0].total_mib, gpus[0].used_mib, gpus[0].free_mib) == (24564, 23187, 892)
    # used + free is ~485 MiB short of total: the driver keeps that back
    assert (gpus[0].reserved_mib, gpus[1].reserved_mib) == (485, 482)
    assert gpus[1].uuid == "GPU-00000000-0000-0000-0000-000000000001"


def test_parse_real_compute_apps():
    apps = parse_compute_apps((FIXTURES / "compute_apps_running.csv").read_text())
    assert len(apps) == 4
    assert (apps[0].pid, apps[0].process_name, apps[0].used_mib) == (555362, "VLLM::Worker_TP0", 16252)


def test_empty_output_is_an_error():
    with pytest.raises(GpuError):
        parse_gpu_query("index, uuid, name, memory.total [MiB], memory.used [MiB], memory.free [MiB]\n")
