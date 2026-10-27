"""Planner tests. Numbers come from a real dual RTX 4090 run (see tests/fixtures)."""

import pytest

from mesh_cli.config import Config, Service
from mesh_cli.planner import GpuBudget, PlanError, make_plan

# 24564 MiB per card; ~417 MiB of desktop apps on GPU 0 and ~4 MiB on GPU 1.
RIG = [GpuBudget(0, 24564, 417, "RTX 4090"), GpuBudget(1, 24564, 4, "RTX 4090")]

LLM = Service(name="llm", port=8001, gpus=[0, 1], gpu_memory_utilization=0.6, measured_mib={0: 16252, 1: 16252})


def config(*services):
    return Config(services=list(services))


def test_working_stack_fits_and_matches_real_free_memory():
    stt = Service(name="stt", port=8004, gpus=[0], gpu_memory_utilization=0.25, measured_mib={0: 6430})
    embed = Service(name="embed", port=8003, gpus=[1], gpu_memory_utilization=0.20, measured_mib={1: 3054})
    plan = make_plan(config(LLM, stt, embed), RIG)
    assert plan.ok
    # 24564 - 417 - 16252 - 6430
    assert plan.free_after_mib[0] == 1465
    # 24564 - 4 - 16252 - 3054
    assert plan.free_after_mib[1] == 5254
    assert plan.warnings == []


def test_stt_at_0_7_after_llm_is_predicted_to_fail():
    # The original failure: "Free memory on device cuda:0 ... less than desired GPU memory utilization (0.7, ...)"
    stt = Service(name="stt", port=8004, gpus=[0], gpu_memory_utilization=0.7)
    plan = make_plan(config(LLM, stt), RIG)
    assert not plan.ok
    failed = plan.services[1]
    check = failed.checks[0]
    assert (check.needed_mib, check.free_before_mib) == (17195, 7895)
    assert failed.suggested_utilization == 0.32


def test_prediction_matches_live_run_within_20_mib():
    # Observed with all four services running: nvidia-smi free = 892 MiB (GPU 0) and 2394 MiB (GPU 1).
    # Desktop-only usage while idle was 491 / 16 MiB; the driver reserves 485 / 482 MiB.
    gpus = [GpuBudget(0, 24564, 491, "RTX 4090", 485), GpuBudget(1, 24564, 16, "RTX 4090", 482)]
    stt = Service(name="stt", port=8004, gpus=[0], gpu_memory_utilization=0.25, measured_mib={0: 6430})
    tts = Service(name="tts", port=8005, gpus=[1], measured_mib={1: 2348})
    embed = Service(name="embed", port=8003, gpus=[1], gpu_memory_utilization=0.20, measured_mib={1: 3054})
    plan = make_plan(config(LLM, stt, tts, embed), gpus)
    assert plan.ok
    assert abs(plan.free_after_mib[0] - 892) <= 20
    assert abs(plan.free_after_mib[1] - 2394) <= 20


def test_fraction_applies_to_usable_memory_not_raw_total():
    # vLLM logged 23.51 GiB total for a 24564 MiB card: 0.7 of that is 16.46 GiB, not 16.8 GiB.
    gpus = [GpuBudget(0, 24564, 0, "RTX 4090", 485)]
    svc = Service(name="w", port=1, gpus=[0], gpu_memory_utilization=0.7)
    assert make_plan(config(svc), gpus).services[0].checks[0].needed_mib == 16856


def test_failed_service_does_not_consume_memory():
    stt = Service(name="stt", port=8004, gpus=[0], gpu_memory_utilization=0.7)
    plan = make_plan(config(LLM, stt), RIG)
    assert plan.free_after_mib[0] == 7895


def test_footprint_without_measurement_uses_overhead():
    llm = Service(name="llm", port=8001, gpus=[0, 1], gpu_memory_utilization=0.6, overhead_mib=1500)
    plan = make_plan(config(llm), RIG)
    # ceil(0.6 * 24564) = 14739, plus 1500
    assert plan.services[0].checks[0].footprint_mib == 16239
    assert plan.free_after_mib[0] == 24564 - 417 - 16239


def test_second_service_fails_when_first_leaves_too_little():
    big = Service(name="big", port=1, gpus=[0], gpu_memory_utilization=0.6)
    small = Service(name="small", port=2, gpus=[0], gpu_memory_utilization=0.5)
    plan = make_plan(config(big, small), RIG)
    assert [s.ok for s in plan.services] == [True, False]


def test_fractions_that_sum_under_the_free_memory_fit_in_either_order():
    a = Service(name="a", port=1, gpus=[0], gpu_memory_utilization=0.4)
    b = Service(name="b", port=2, gpus=[0], gpu_memory_utilization=0.5)
    assert make_plan(config(a, b), RIG).ok
    assert make_plan(config(b, a), RIG).ok


def test_unknown_footprint_is_reported():
    tts = Service(name="tts", port=8005, gpus=[1])
    plan = make_plan(config(tts), RIG)
    assert plan.ok
    assert "footprint unknown" in plan.services[0].notes[0]


def test_low_free_memory_warns():
    plan = make_plan(Config(services=[LLM], margin_mib=9000), RIG)
    assert any("GPU 0" in w for w in plan.warnings)


def test_unknown_gpu_is_an_error():
    svc = Service(name="x", port=1, gpus=[3], gpu_memory_utilization=0.1)
    with pytest.raises(PlanError, match="GPU \\[3\\]"):
        make_plan(config(svc), RIG)


# --- a service must fit once it is running, not only when it starts -----------------------------------------------

# usable = 24564 - 485 = 24079; the desktop uses 417, so 23662 is free before the first service.
CARD = [GpuBudget(0, 24564, 417, "RTX 4090", 485)]
CHAT = Service(name="chat", port=8001, gpus=[0], gpu_memory_utilization=0.55, overhead_mib=1500)  # holds 13244 + 1500


def embeddings(fraction=0.35, **extra):
    return Service(name="embeddings", port=8003, gpus=[0], gpu_memory_utilization=fraction, **extra)


def test_a_service_whose_footprint_does_not_fit_is_a_failure_even_if_it_can_start():
    # 8918 MiB are free after chat. 0.35 needs 8428 to start, so the start check passes, but with its overhead it
    # holds 9928 and would leave the card at -1010 MiB.
    plan = make_plan(config(CHAT, embeddings(overhead_mib=1500)), CARD)
    check = plan.services[1].checks[0]
    assert (check.free_before_mib, check.needed_mib, check.footprint_mib) == (8918, 8428, 9928)
    assert not plan.ok and not check.ok


def test_the_suggestion_leaves_room_for_the_overhead():
    plan = make_plan(config(CHAT, embeddings(overhead_mib=1500)), CARD)
    # (8918 - 1500) / 24079 = 0.308: 0.30 holds 7224 + 1500 = 8724, but 0.31 would hold 8965
    assert plan.services[1].suggested_utilization == 0.30


@pytest.mark.parametrize("overhead", [0, 500, 1500, 3000])
def test_following_the_suggestion_makes_the_plan_fit_with_memory_to_spare(overhead):
    first = make_plan(config(CHAT, embeddings(0.6, overhead_mib=overhead)), CARD)
    suggestion = first.services[1].suggested_utilization
    assert not first.ok and suggestion is not None
    second = make_plan(config(CHAT, embeddings(suggestion, overhead_mib=overhead)), CARD)
    assert second.ok and min(second.free_after_mib.values()) >= 0


def test_a_measured_footprint_larger_than_the_free_memory_fails_and_a_lower_fraction_is_not_suggested():
    svc = embeddings(0.30, measured_mib={0: 9000})  # starts fine (7224), but holds 9000 of 8918
    plan = make_plan(config(CHAT, svc), CARD)
    assert not plan.ok
    assert plan.services[1].suggested_utilization is None
    assert any("measured_mib" in note for note in plan.services[1].notes)


def test_a_measured_footprint_that_fits_but_a_fraction_that_does_not_still_gets_a_suggestion():
    svc = embeddings(0.40, measured_mib={0: 6000})  # needs 9632 to start, free is 8918, holds only 6000
    plan = make_plan(config(CHAT, svc), CARD)
    assert not plan.ok
    assert plan.services[1].suggested_utilization == 0.37  # 8918 / 24079


def test_overhead_alone_larger_than_the_free_memory_gets_no_suggestion():
    plan = make_plan(config(CHAT, embeddings(0.10, overhead_mib=9000)), CARD)
    assert not plan.ok
    assert plan.services[1].suggested_utilization is None
    assert any("overhead_mib" in note for note in plan.services[1].notes)


def test_a_suggestion_is_withheld_if_one_of_the_gpus_cannot_be_fixed_by_lowering_the_fraction():
    rig = [GpuBudget(0, 24564, 0, "RTX 4090", 485), GpuBudget(1, 24564, 0, "RTX 4090", 485)]
    svc = Service(name="both", port=1, gpus=[0, 1], gpu_memory_utilization=0.9, measured_mib={1: 30000})
    plan = make_plan(config(svc), rig)
    assert not plan.ok and plan.services[0].suggested_utilization is None


def test_a_service_that_fits_is_unchanged_by_the_stricter_check():
    plan = make_plan(config(CHAT, embeddings(0.25, overhead_mib=500)), CARD)  # holds 6020 + 500 of 8918
    assert plan.ok and plan.free_after_mib[0] == 8918 - 6520 and plan.warnings == []
