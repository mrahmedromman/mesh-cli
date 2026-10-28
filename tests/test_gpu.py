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


def test_parse_utilization_column():
    gpus = parse_gpu_query((FIXTURES / "gpu_query_util.csv").read_text())
    assert [g.utilization_pct for g in gpus] == [21, 0]
    assert gpus[0].free_mib == 892


def test_utilization_is_unknown_without_the_column():
    gpus = parse_gpu_query((FIXTURES / "gpu_query_running.csv").read_text())
    assert [g.utilization_pct for g in gpus] == [None, None]


def test_parse_real_compute_apps():
    apps = parse_compute_apps((FIXTURES / "compute_apps_running.csv").read_text())
    assert len(apps) == 4
    assert (apps[0].pid, apps[0].process_name, apps[0].used_mib) == (555362, "VLLM::Worker_TP0", 16252)


def test_empty_output_is_an_error():
    with pytest.raises(GpuError):
        parse_gpu_query("index, uuid, name, memory.total [MiB], memory.used [MiB], memory.free [MiB]\n")


# --- a process memory that nvidia-smi cannot report ----------------------------------------------------------------------


COMPUTE_HEADER = "gpu_uuid, pid, process_name, used_gpu_memory [MiB]\n"


def test_a_process_memory_that_nvidia_smi_cannot_report_counts_as_zero():
    apps = parse_compute_apps(COMPUTE_HEADER + "GPU-0, 4121, python, [N/A]\nGPU-0, 4122, python, [Not Supported]\nGPU-0, 4123, python, 512 MiB\n")
    assert [(a.pid, a.used_mib) for a in apps] == [(4121, 0), (4122, 0), (4123, 512)]
    assert apps[0].process_name == "python" and apps[0].gpu_uuid == "GPU-0"


def test_a_process_row_with_missing_columns_is_a_clear_gpu_error():
    with pytest.raises(GpuError, match="unexpected nvidia-smi row"):
        parse_compute_apps(COMPUTE_HEADER + "GPU-0, 4121\n")


def test_lines_that_are_not_process_rows_are_skipped():
    assert parse_compute_apps(COMPUTE_HEADER + "No running processes found\n") == []
