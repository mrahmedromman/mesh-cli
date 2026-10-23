"""Simulate starting the services in order and report which ones will fit.

Rules, taken from observed vLLM behaviour:
  * The driver reserves some memory (~485 MiB on a 4090). nvidia-smi's total includes it, but vLLM's
    does not: its log showed 23.51 GiB for a card nvidia-smi calls 24564 MiB. Fractions apply to
    usable = total - reserved.
  * At startup a vLLM service needs free memory >= gpu_memory_utilization * usable on each GPU
    (the check that produced "Free memory on device ... is less than desired GPU memory utilization").
  * After startup it holds `measured_mib` if known, else the requested amount plus `overhead_mib`.
    On a 4090 a request of 0.6 held 16252 MiB, about 1.5 GiB more than the fraction.
Memory used by other processes (desktop, other jobs) is passed in as `external_mib`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import Config


class PlanError(Exception):
    pass


@dataclass(frozen=True)
class GpuBudget:
    index: int
    total_mib: int
    external_mib: int = 0
    name: str = ""
    reserved_mib: int = 0

    @property
    def usable_mib(self) -> int:
        return self.total_mib - self.reserved_mib


@dataclass(frozen=True)
class GpuCheck:
    gpu: int
    free_before_mib: int
    needed_mib: int | None
    footprint_mib: int | None
    ok: bool


@dataclass
class ServicePlan:
    name: str
    checks: list[GpuCheck]
    suggested_utilization: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)


@dataclass
class Plan:
    services: list[ServicePlan]
    free_after_mib: dict[int, int]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.services)


def make_plan(config: Config, gpus: list[GpuBudget]) -> Plan:
    by_index = {g.index: g for g in gpus}
    free = {g.index: g.usable_mib - g.external_mib for g in gpus}
    plans: list[ServicePlan] = []
    warnings: list[str] = []

    for svc in config.services:
        missing = [i for i in svc.gpus if i not in by_index]
        if missing:
            raise PlanError(f"service '{svc.name}' uses GPU {missing} but only GPUs {sorted(by_index)} were found")

        checks: list[GpuCheck] = []
        notes: list[str] = []
        suggestion: float | None = None
        for gi in svc.gpus:
            needed: int | None
            footprint: int | None
            total = by_index[gi].usable_mib
            measured = (svc.measured_mib or {}).get(gi)
            if svc.gpu_memory_utilization is not None:
                needed = math.ceil(svc.gpu_memory_utilization * total)
                footprint = measured if measured is not None else needed + svc.overhead_mib
            elif measured is not None:
                needed = footprint = measured
            else:
                needed = footprint = None
                notes.append(f"GPU {gi}: no gpu_memory_utilization or measured_mib, footprint unknown (counted as 0)")
            ok = needed is None or free[gi] >= needed
            if not ok and svc.gpu_memory_utilization is not None:
                fit = math.floor(max(free[gi], 0) / total * 100) / 100
                suggestion = fit if suggestion is None else min(suggestion, fit)
            checks.append(GpuCheck(gi, free[gi], needed, footprint, ok))

        plan = ServicePlan(svc.name, checks, suggestion, notes)
        if plan.ok:
            for check in checks:
                free[check.gpu] -= check.footprint_mib or 0
        plans.append(plan)

    for gi, left in sorted(free.items()):
        if left < config.margin_mib:
            warnings.append(f"GPU {gi} has only {left} MiB free after all services (margin is {config.margin_mib} MiB)")
    return Plan(plans, free, warnings)
