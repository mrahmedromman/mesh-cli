"""Work out who is holding GPU memory: mesh services, Docker containers, desktop apps or other programs.

Read-only: this module never signals or kills anything. It needs Linux (/proc) and nvidia-smi, and it should run
on the host, because inside a container nvidia-smi cannot see the host's GPU processes.
"""

from __future__ import annotations

import os
import pwd
import re
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from . import runner
from .gpu import Gpu, GpuError, parse_compute_apps, read_gpus

COMPUTE_QUERY = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"]
FULL_XML = ["nvidia-smi", "-q", "-x"]
CONTAINER_ID = re.compile(r"docker[-/]([0-9a-f]{64})")
MAX_COMMAND = 80


@dataclass(frozen=True)
class GpuProcess:
    gpu_uuid: str
    pid: int
    name: str
    used_mib: int
    graphics: bool = False


@dataclass(frozen=True)
class ServiceInfo:
    pid: int  # process group leader started by `mesh up`
    command: str = ""
    stop: str = ""


@dataclass(frozen=True)
class Holder:
    kind: str  # mesh | container | desktop | other | unattributed
    label: str
    detail: str
    mib: int


@dataclass(frozen=True)
class GpuReport:
    gpu: Gpu
    holders: list[Holder]


class ProcFs(Protocol):
    def cgroup(self, pid: int) -> str: ...
    def cmdline(self, pid: int) -> str: ...
    def user(self, pid: int) -> str | None: ...
    def pgid(self, pid: int) -> int | None: ...
    def ppid(self, pid: int) -> int | None: ...


class LinuxProcFs:
    """Reads /proc. Anything unreadable (a process that just exited, another user's) becomes empty/None."""

    def cgroup(self, pid: int) -> str:
        return self._read(f"/proc/{pid}/cgroup")

    def cmdline(self, pid: int) -> str:
        return self._read(f"/proc/{pid}/cmdline").replace("\0", " ").strip()

    def user(self, pid: int) -> str | None:
        try:
            uid = os.stat(f"/proc/{pid}").st_uid
        except OSError:
            return None
        try:
            return pwd.getpwuid(uid).pw_name
        except KeyError:
            return str(uid)

    def pgid(self, pid: int) -> int | None:
        try:
            return os.getpgid(pid)
        except OSError:
            return None

    def ppid(self, pid: int) -> int | None:
        stat = self._read(f"/proc/{pid}/stat")
        try:  # the command name is in parentheses and may contain spaces, so split after the last ")"
            return int(stat.rsplit(")", 1)[1].split()[1])
        except (IndexError, ValueError):
            return None

    @staticmethod
    def _read(path: str) -> str:
        try:
            with open(path, "rb") as handle:
                return handle.read().decode(errors="replace")
        except OSError:
            return ""


# --- reading nvidia-smi and docker -------------------------------------------------------------------------


def _run(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    except FileNotFoundError as exc:
        raise GpuError(f"{command[0]} not found") from exc
    except subprocess.SubprocessError as exc:
        raise GpuError(f"{command[0]} failed: {exc}") from exc
    return result.stdout


def parse_xml_processes(text: str) -> list[GpuProcess]:
    """Processes from `nvidia-smi -q -x`. This is the only source that lists graphics apps (Xorg, browsers)."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise GpuError(f"could not read nvidia-smi XML: {exc}") from exc
    found = []
    for gpu in root.iter("gpu"):
        uuid = (gpu.findtext("uuid") or "").strip()
        for info in gpu.iter("process_info"):
            pid = (info.findtext("pid") or "").strip()
            if not pid.isdigit():
                continue
            used = re.search(r"\d+", info.findtext("used_memory") or "")
            found.append(
                GpuProcess(
                    uuid,
                    int(pid),
                    (info.findtext("process_name") or "").strip(),
                    int(used.group()) if used else 0,
                    graphics="G" in (info.findtext("type") or ""),
                )
            )
    return found


def merge_processes(compute: list[GpuProcess], extra: list[GpuProcess]) -> list[GpuProcess]:
    """Compute processes come from the stable CSV query; add graphics-only ones from the XML."""
    seen = {(p.gpu_uuid, p.pid) for p in compute}
    return compute + [p for p in extra if p.graphics and (p.gpu_uuid, p.pid) not in seen]


def read_processes() -> list[GpuProcess]:
    apps = parse_compute_apps(_run(COMPUTE_QUERY))
    compute = [GpuProcess(a.gpu_uuid, a.pid, a.process_name, a.used_mib) for a in apps]
    try:
        extra = parse_xml_processes(_run(FULL_XML))
    except GpuError:
        extra = []  # graphics apps are then simply not itemised
    return merge_processes(compute, extra)


def read_container_names() -> dict[str, str]:
    """Container id -> name from `docker ps`. Empty if docker is missing or not allowed."""
    try:
        out = _run(["docker", "ps", "--no-trunc", "--format", "{{.ID}}\t{{.Names}}"])
    except GpuError:
        return {}
    names = {}
    for line in out.splitlines():
        cid, _, name = line.partition("\t")
        if cid.strip() and name.strip():
            names[cid.strip()] = name.strip()
    return names


def services_from_state(state: dict | None) -> dict[str, ServiceInfo]:
    services = (state or {}).get("services", {})
    return {
        name: ServiceInfo(int(info["pid"]), info.get("command") or "", info.get("stop") or "")
        for name, info in services.items()
        if info.get("pid") and runner.is_ours(info["pid"], info.get("started"))  # not yet started, or the pid is now another program's
    }


# --- attribution -------------------------------------------------------------------------------------------


def container_id(cgroup_text: str) -> str | None:
    """The Docker container id in a /proc/PID/cgroup file (cgroup v1 `/docker/ID`, or v2 `docker-ID.scope`)."""
    match = CONTAINER_ID.search(cgroup_text)
    return match.group(1) if match else None


def mentions_container(text: str, container: str) -> bool:
    """True if a command names the container: `--name NAME`, or `docker stop|kill|rm ... NAME`."""
    name = re.escape(container)
    if re.search(rf"--name[ =]{name}(?![\w.-])", text):
        return True
    return bool(re.search(rf"docker\s+(?:stop|kill|rm)\b[^\n]*?(?<![\w.-]){name}(?![\w.-])", text))


def _service_of_pid(pid: int, leaders: dict[int, str], proc: ProcFs) -> str | None:
    """A service owns a process if the process is in its process group or has its leader as an ancestor."""
    group = proc.pgid(pid)
    if group in leaders:
        return leaders[group]
    current: int | None = pid
    for _ in range(12):
        if current is None or current <= 1:
            return None
        if current in leaders:
            return leaders[current]
        current = proc.ppid(current)
    return None


def _short(command: str) -> str:
    return command if len(command) <= MAX_COMMAND else command[: MAX_COMMAND - 1] + "…"


def _processes(count: int) -> str:
    return f"{count} process{'' if count == 1 else 'es'}"


def _pids(members: list[tuple[int, str]]) -> str:
    """The process ids behind a holder, as nvidia-smi lists them, so a service can be matched with its output."""
    pids = sorted(pid for pid, _ in members)
    shown = ", ".join(str(pid) for pid in pids[:4])
    if len(pids) == 1:
        return f"pid {shown}"
    return f"pids {shown}" + (f", … ({_processes(len(pids))})" if len(pids) > 4 else "")


def _owner(
    p: GpuProcess, leaders: dict[int, str], services: dict[str, ServiceInfo], proc: ProcFs, containers: dict[str, str],
) -> tuple[str | None, str | None, str | None]:
    """The mesh service a GPU process belongs to (or None), and its Docker container id and name if it has one."""
    service = _service_of_pid(p.pid, leaders, proc)
    cid = container_id(proc.cgroup(p.pid))
    cname = next((n for i, n in containers.items() if cid and i.startswith(cid)), None) if cid else None
    if service is None and cname:
        service = next(
            (n for n, info in services.items() if mentions_container(f"{info.command}\n{info.stop}", cname)),
            None,
        )
    return service, cid, cname


def attribute(
    gpus: list[Gpu],
    processes: list[GpuProcess],
    services: dict[str, ServiceInfo],
    proc: ProcFs,
    containers: dict[str, str],
) -> list[GpuReport]:
    leaders = {info.pid: name for name, info in services.items()}
    reports = []
    for gpu in gpus:
        groups: dict[tuple[Any, ...], list] = {}  # owner key -> [label, detail, mib, pids, names]
        for p in (p for p in processes if p.gpu_uuid == gpu.uuid):
            service, cid, cname = _owner(p, leaders, services, proc, containers)
            key: tuple[Any, ...]
            if service is not None:
                key, kind, label = ("mesh", service), "mesh", service
                extra = f' · container "{cname}"' if cname else ""
                groups.setdefault(key, [kind, label, f"mesh service{extra}", 0, []])
            elif cid:
                key, kind, label = ("container", cname or cid[:12]), "container", cname or cid[:12]
                groups.setdefault(key, [kind, label, "Docker container, not started by mesh", 0, []])
            elif p.graphics:
                groups.setdefault(("desktop",), ["desktop", "desktop", "graphics apps", 0, []])
                key = ("desktop",)
            else:
                command = proc.cmdline(p.pid) or p.name
                user = proc.user(p.pid)
                key = ("other", p.pid)
                base = os.path.basename((command.split() or [p.name])[0]) or p.name
                who = f"user {user} · " if user else ""
                groups.setdefault(key, ["other", base, f"{who}pid {p.pid} · {_short(command)} (not started by mesh)", 0, []])
            entry = groups[key]
            entry[3] += p.used_mib
            entry[4].append((p.pid, os.path.basename((p.name.split() or [""])[0])))

        holders = []
        for kind, label, detail, mib, members in groups.values():
            if kind in ("mesh", "container"):
                detail += f" · {_pids(members)}"
            elif kind == "desktop":
                names = list(dict.fromkeys(n for _, n in members if n))
                shown = ", ".join(names[:4]) + (", …" if len(names) > 4 else "")
                detail += f": {shown} · {_processes(len(members))}"
            holders.append(Holder(kind, label, detail, mib))
        holders.sort(key=lambda h: -h.mib)
        leftover = gpu.used_mib - sum(h.mib for h in holders)
        if leftover > 0:
            holders.append(Holder("unattributed", "unattributed", "used memory that no listed process accounts for", leftover))
        reports.append(GpuReport(gpu, holders))
    return reports


def service_pids(
    services: dict[str, ServiceInfo], processes: list[GpuProcess], proc: ProcFs, containers: dict[str, str],
) -> dict[str, list[int]]:
    """The ids of the GPU processes of each service, over all its GPUs: the ids nvidia-smi lists for it."""
    leaders = {info.pid: name for name, info in services.items()}
    found: dict[str, set[int]] = {}
    for p in processes:
        service = _owner(p, leaders, services, proc, containers)[0]
        if service is not None:
            found.setdefault(service, set()).add(p.pid)
    return {name: sorted(pids) for name, pids in found.items()}


def lookup_service_pids(log_dir: Path, proc: ProcFs | None = None) -> dict[str, list[int]]:
    """Read the live system: which GPU processes belong to which mesh service. Raises GpuError if nvidia-smi fails."""
    apps = parse_compute_apps(_run(COMPUTE_QUERY))
    processes = [GpuProcess(a.gpu_uuid, a.pid, a.process_name, a.used_mib) for a in apps]
    services = services_from_state(runner.read_state(log_dir))
    containers = read_container_names() if services else {}
    return service_pids(services, processes, proc or LinuxProcFs(), containers)


class ServicePids:
    """`lookup_service_pids`, kept fresh in the background so a slow nvidia-smi can never freeze a live view.

    `get()` returns at once with the last answer. The first call waits a moment (`first_wait` seconds) for the first
    answer so that the first frame is already right. A failed lookup keeps the last good answer, and `None` means
    that nvidia-smi has not been readable yet.
    """

    def __init__(
        self,
        log_dir: Path,
        lookup: Callable[[Path], dict[str, list[int]]] | None = None,
        every: float = 2.0,
        first_wait: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._log_dir, self._lookup = log_dir, lookup or lookup_service_pids
        self._every, self._first_wait, self._clock = every, first_wait, clock
        self._result: dict[str, list[int]] | None = None
        self._asked: float | None = None
        self._thread: threading.Thread | None = None

    def get(self) -> dict[str, list[int]] | None:
        now = self._clock()
        busy = self._thread is not None and self._thread.is_alive()
        if not busy and (self._asked is None or now - self._asked >= self._every):
            self._asked = now
            self._thread = threading.Thread(target=self._refresh, daemon=True)
            self._thread.start()
            if self._result is None:
                self._thread.join(self._first_wait)
        return self._result

    def _refresh(self) -> None:
        try:
            self._result = self._lookup(self._log_dir)
        except Exception:  # noqa: BLE001 - a background thread must never print a traceback over the live view
            pass


def render(reports: list[GpuReport]) -> str:
    lines = []
    for report in reports:
        g = report.gpu
        head = f"GPU {g.index}  {g.name}  {g.total_mib} MiB total"
        if g.reserved_mib > 0:
            head += f" · {g.reserved_mib} reserved"
        head += f" · {g.used_mib} used · {g.free_mib} free"
        if g.utilization_pct is not None:
            head += f" · {g.utilization_pct}% utilization"
        lines.append(head)
        if not report.holders:
            lines.append("   nothing is using this GPU's memory")
        for h in report.holders:
            lines.append(f"  {h.mib:>7} MiB  {h.label:<12} {h.detail}")
        lines.append("")
    return "\n".join(lines).rstrip()


def gather(
    log_dir: Path,
    only: set[int] | None = None,
    proc: ProcFs | None = None,
) -> list[GpuReport]:
    """Read the live system and return a report per GPU (all GPUs, or just the indexes in `only`)."""
    gpus = [g for g in read_gpus() if only is None or g.index in only]
    processes = read_processes()
    services = services_from_state(runner.read_state(log_dir))
    return attribute(gpus, processes, services, proc or LinuxProcFs(), read_container_names())
