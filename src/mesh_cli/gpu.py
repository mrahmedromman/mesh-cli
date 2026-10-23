"""Read GPU inventory from nvidia-smi (or from saved CSV output)."""

from __future__ import annotations

import csv
import io
import re
import subprocess
from dataclasses import dataclass

GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
    "--format=csv",
]


class GpuError(Exception):
    pass


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    name: str
    total_mib: int
    used_mib: int
    free_mib: int
    utilization_pct: int | None = None  # None when the saved output has no utilization column

    @property
    def reserved_mib(self) -> int:
        """Memory the driver keeps back: counted in total, but neither used nor free (~480 MiB on a 4090)."""
        return self.total_mib - self.used_mib - self.free_mib


@dataclass(frozen=True)
class ComputeApp:
    gpu_uuid: str
    pid: int
    process_name: str
    used_mib: int


def _mib(value: str) -> int:
    match = re.search(r"\d+", value)
    if not match:
        raise GpuError(f"cannot read a memory value from {value!r}")
    return int(match.group())


def _rows(text: str) -> list[list[str]]:
    return [row for row in csv.reader(io.StringIO(text), skipinitialspace=True) if row]


def parse_gpu_query(text: str) -> list[Gpu]:
    gpus = []
    for row in _rows(text):
        if not row[0].strip().isdigit():
            continue  # header line
        if len(row) < 6:
            raise GpuError(f"unexpected nvidia-smi row: {row}")
        utilization = int(re.search(r"\d+", row[6]).group()) if len(row) > 6 and re.search(r"\d+", row[6]) else None
        gpus.append(Gpu(int(row[0]), row[1], row[2], _mib(row[3]), _mib(row[4]), _mib(row[5]), utilization))
    if not gpus:
        raise GpuError("no GPUs found in nvidia-smi output")
    return gpus


def parse_compute_apps(text: str) -> list[ComputeApp]:
    apps = []
    for row in _rows(text):
        if not row[1].strip().isdigit():
            continue  # header line
        apps.append(ComputeApp(row[0], int(row[1]), row[2], _mib(row[3])))
    return apps


def read_gpus() -> list[Gpu]:
    try:
        result = subprocess.run(GPU_QUERY, capture_output=True, text=True, check=True, timeout=30)
    except FileNotFoundError as exc:
        raise GpuError("nvidia-smi not found; pass --gpus-csv with saved nvidia-smi output") from exc
    except subprocess.SubprocessError as exc:
        raise GpuError(f"nvidia-smi failed: {exc}") from exc
    return parse_gpu_query(result.stdout)
