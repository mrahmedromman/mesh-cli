"""Who-holds-GPU-memory tests. The rig scenario uses the real numbers from the two-GPU test machine."""

from pathlib import Path

import pytest

from mesh_cli import holders
from mesh_cli.gpu import Gpu, GpuError, parse_compute_apps, parse_gpu_query
from mesh_cli.holders import GpuProcess, ServiceInfo, attribute, container_id, mentions_container, render

FIXTURES = Path(__file__).parent / "fixtures"
UUID0 = "GPU-00000000-0000-0000-0000-000000000000"
UUID1 = "GPU-00000000-0000-0000-0000-000000000001"
STT_ID = "a" * 64
EMBED_ID = "b" * 64


def cgroup_v2(cid: str) -> str:
    return f"0::/system.slice/docker-{cid}.scope\n"


class FakeProc:
    def __init__(self, groups=None, parents=None, cgroups=None, cmdlines=None, users=None):
        self.groups, self.parents = groups or {}, parents or {}
        self.cgroups, self.cmdlines, self.users = cgroups or {}, cmdlines or {}, users or {}

    def cgroup(self, pid): return self.cgroups.get(pid, "")
    def cmdline(self, pid): return self.cmdlines.get(pid, "")
    def user(self, pid): return self.users.get(pid)
    def pgid(self, pid): return self.groups.get(pid)
    def ppid(self, pid): return self.parents.get(pid)


def rig():
    gpus = parse_gpu_query((FIXTURES / "gpu_query_running.csv").read_text())
    compute = [GpuProcess(a.gpu_uuid, a.pid, a.process_name, a.used_mib) for a in parse_compute_apps((FIXTURES / "compute_apps_running.csv").read_text())]
    extra = holders.parse_xml_processes((FIXTURES / "nvidia_smi_processes.xml").read_text())
    services = {
        "llm": ServiceInfo(555000, "source ~/venvs/vllm/bin/activate\nexec vllm serve /path/to/llm"),
        "stt": ServiceInfo(556000, "exec docker run --rm --name stt --gpus all my-stt-image", "docker stop stt"),
        "embed": ServiceInfo(558000, "exec docker run --rm --name embed my-embed-image", "docker stop embed"),
    }
    proc = FakeProc(
        groups={555362: 555000, 555363: 555000},
        cgroups={558799: cgroup_v2(STT_ID), 562327: cgroup_v2(EMBED_ID)},
    )
    containers = {STT_ID: "stt", EMBED_ID: "embed"}
    return gpus, holders.merge_processes(compute, extra), services, proc, containers


def by_label(report):
    return {h.label: h for h in report.holders}


def test_rig_is_attributed_to_the_right_services():
    reports = attribute(*rig())
    gpu0, gpu1 = (by_label(r) for r in reports)
    assert {k: v.mib for k, v in gpu0.items()} == {"llm": 16252, "stt": 6430, "desktop": 417, "unattributed": 88}
    assert {k: v.mib for k, v in gpu1.items()} == {"llm": 16252, "embed": 3054, "desktop": 4, "unattributed": 25}
    assert gpu0["llm"].kind == "mesh" and gpu0["stt"].kind == "mesh"
    assert 'container "stt"' in gpu0["stt"].detail
    assert "6 processes" in gpu0["desktop"].detail and "Xorg" in gpu0["desktop"].detail


def test_holders_are_sorted_largest_first_with_unattributed_last():
    labels = [h.label for h in attribute(*rig())[0].holders]
    assert labels == ["llm", "stt", "desktop", "unattributed"]


def test_container_not_started_by_mesh():
    gpus = [Gpu(0, UUID0, "GPU", 24564, 3000, 21000)]
    procs = [GpuProcess(UUID0, 700, "python", 2500)]
    proc = FakeProc(cgroups={700: cgroup_v2(EMBED_ID)})
    (report,) = attribute(gpus, procs, {}, proc, {EMBED_ID: "webui"})
    holder = by_label(report)["webui"]
    assert holder.kind == "container" and "not started by mesh" in holder.detail and holder.mib == 2500


def test_unknown_container_shows_its_short_id():
    gpus = [Gpu(0, UUID0, "GPU", 24564, 100, 24000)]
    procs = [GpuProcess(UUID0, 700, "python", 100)]
    (report,) = attribute(gpus, procs, {}, FakeProc(cgroups={700: cgroup_v2(STT_ID)}), {})
    assert report.holders[0].label == "a" * 12


def test_other_process_gets_user_pid_and_command():
    gpus = [Gpu(0, UUID0, "GPU", 24564, 2300, 22000)]
    procs = [GpuProcess(UUID0, 900, "python", 2270)]
    proc = FakeProc(cmdlines={900: "python /home/alice/train.py --epochs 3"}, users={900: "alice"})
    (report,) = attribute(gpus, procs, {}, proc, {})
    holder = report.holders[0]
    assert holder.kind == "other" and holder.label == "python" and holder.mib == 2270
    assert "user alice" in holder.detail and "pid 900" in holder.detail and "train.py" in holder.detail
    assert "not started by mesh" in holder.detail


def test_long_commands_are_shortened():
    gpus = [Gpu(0, UUID0, "GPU", 24564, 100, 24000)]
    procs = [GpuProcess(UUID0, 900, "python", 100)]
    (report,) = attribute(gpus, procs, {}, FakeProc(cmdlines={900: "python " + "x" * 300}), {})
    assert len(report.holders[0].detail) < 160


def test_service_found_through_a_parent_process():
    gpus = [Gpu(0, UUID0, "GPU", 24564, 5000, 19000)]
    procs = [GpuProcess(UUID0, 800, "worker", 5000)]
    proc = FakeProc(groups={800: 800}, parents={800: 801, 801: 555000})  # own group, but its ancestor is the service
    (report,) = attribute(gpus, procs, {"llm": ServiceInfo(555000)}, proc, {})
    assert report.holders[0].label == "llm" and report.holders[0].kind == "mesh"


def test_no_unattributed_line_when_everything_is_accounted_for():
    gpus = [Gpu(0, UUID0, "GPU", 24564, 100, 24000)]
    procs = [GpuProcess(UUID0, 900, "python", 100)]
    (report,) = attribute(gpus, procs, {}, FakeProc(), {})
    assert [h.kind for h in report.holders] == ["other"]


def test_gpu_with_no_processes():
    gpus = [Gpu(0, UUID0, "GPU", 24564, 0, 24079)]
    (report,) = attribute(gpus, [], {}, FakeProc(), {})
    assert report.holders == []
    assert "nothing is using this GPU's memory" in render([report])


def test_container_id_from_cgroup_v1_and_v2():
    assert container_id(f"12:cpuset:/docker/{STT_ID}\n11:memory:/docker/{STT_ID}\n") == STT_ID
    assert container_id(cgroup_v2(EMBED_ID)) == EMBED_ID
    assert container_id("0::/user.slice/user-1000.slice/session-3.scope\n") is None
    assert container_id("") is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("exec docker run --rm --name stt --gpus all img", True),
        ("exec docker run --rm --name=stt img", True),
        ("docker stop stt", True),
        ("docker kill -s TERM stt", True),
        ("--served-model-name stt", False),
        ("exec docker run --rm --name stt-extra img", False),
        ("docker stop my-stt", False),
        ("", False),
    ],
)
def test_mentions_container(text, expected):
    assert mentions_container(text, "stt") is expected


def test_parse_xml_processes():
    procs = holders.parse_xml_processes((FIXTURES / "nvidia_smi_processes.xml").read_text())
    assert len(procs) == 11
    xorg = next(p for p in procs if p.pid == 8303 and p.gpu_uuid == UUID0)
    assert xorg.graphics and xorg.used_mib == 110
    assert not next(p for p in procs if p.pid == 555362).graphics


def test_merge_keeps_compute_entries_and_adds_only_graphics():
    compute = [GpuProcess(UUID0, 555362, "VLLM::Worker_TP0", 16252)]
    extra = holders.parse_xml_processes((FIXTURES / "nvidia_smi_processes.xml").read_text())
    merged = holders.merge_processes(compute, extra)
    assert sum(1 for p in merged if p.pid == 555362) == 1
    assert sum(1 for p in merged if p.graphics) == 7  # six on GPU 0, one on GPU 1


def test_bad_xml_is_an_error():
    with pytest.raises(GpuError):
        holders.parse_xml_processes("<nvidia_smi_log><gpu>")


def test_read_processes_falls_back_to_compute_only_if_xml_fails(monkeypatch):
    csv_text = (FIXTURES / "compute_apps_running.csv").read_text()

    def fake_run(command):
        if "-x" in command:
            raise GpuError("no xml")
        return csv_text

    monkeypatch.setattr(holders, "_run", fake_run)
    procs = holders.read_processes()
    assert len(procs) == 4 and not any(p.graphics for p in procs)


def test_read_container_names(monkeypatch):
    monkeypatch.setattr(holders, "_run", lambda command: f"{STT_ID}\tstt\n{EMBED_ID}\tembed\n\n")
    assert holders.read_container_names() == {STT_ID: "stt", EMBED_ID: "embed"}


def test_read_container_names_without_docker(monkeypatch):
    def fail(command):
        raise GpuError("docker not found")

    monkeypatch.setattr(holders, "_run", fail)
    assert holders.read_container_names() == {}


def test_services_from_state():
    state = {"services": {"llm": {"pid": 5, "command": "exec x", "stop": None}, "old": {"pid": 6}}}
    services = holders.services_from_state(state)
    assert services["llm"] == ServiceInfo(5, "exec x", "")
    assert services["old"] == ServiceInfo(6, "", "")
    assert holders.services_from_state(None) == {}


def test_services_from_state_ignores_a_pid_that_now_belongs_to_another_program():
    import os

    state = {"services": {"llm": {"pid": os.getpid(), "started": "someone-else:1"}}}  # a live process, but not the one mesh started
    assert holders.services_from_state(state) == {}


def test_linux_procfs_parses_parent_and_command_line(monkeypatch):
    files = {
        "/proc/9/stat": "9 (my (odd) name) S 456 9 9 0 -1 4194560",  # the name may contain spaces and parentheses
        "/proc/9/cmdline": "python\0train.py\0--epochs\x003\0",
        "/proc/10/stat": "",
    }
    monkeypatch.setattr(holders.LinuxProcFs, "_read", staticmethod(lambda path: files.get(path, "")))
    fs = holders.LinuxProcFs()
    assert fs.ppid(9) == 456
    assert fs.ppid(10) is None
    assert fs.cmdline(9) == "python train.py --epochs 3"
    assert fs.cmdline(404) == ""


def test_render_shows_utilization_and_holders():
    gpus = parse_gpu_query((FIXTURES / "gpu_query_util.csv").read_text())
    _, processes, services, proc, containers = rig()
    text = render(attribute(gpus, processes, services, proc, containers))
    first = text.splitlines()[0]
    assert first == "GPU 0  NVIDIA GeForce RTX 4090  24564 MiB total · 485 reserved · 23187 used · 892 free · 21% utilization"
    assert "   16252 MiB  llm" in text
    assert "0% utilization" in text
    assert "unattributed" in text


def test_render_omits_utilization_when_unknown():
    gpus = parse_gpu_query((FIXTURES / "gpu_query_running.csv").read_text())
    text = render(attribute(gpus, [], {}, FakeProc(), {}))
    assert "utilization" not in text
