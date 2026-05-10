import time

from riprap_models.energy import EnergyResult, measure_energy


def test_measure_energy_returns_positive_duration_and_method():
    with measure_energy() as m:
        time.sleep(0.05)
    assert isinstance(m, EnergyResult)
    assert m.duration_s >= 0.04
    assert m.method in ("nvml", "rapl", "powermetrics", "estimated")
    assert m.joules >= 0
    assert m.peak_memory_mb > 0


def test_measure_energy_estimated_path():
    # Force the estimated branch by preferring a method we know is unavailable
    # in CI: nvml on a CPU runner returns None and falls through to estimate.
    with measure_energy(prefer="nvml") as m:
        time.sleep(0.02)
    # On a CI runner without an NVIDIA GPU the sampler bails and we drop to
    # estimated. On a runner that does have one, method stays "nvml" and the
    # joules figure is the integrated NVML reading. Either is correct here.
    assert m.method in ("nvml", "estimated")
    assert m.joules >= 0


def test_energy_result_to_dict_round_trip():
    r = EnergyResult(joules=1.234, duration_s=0.5, peak_memory_mb=42.0, method="estimated")
    d = r.to_dict()
    assert d["joules"] == 1.234
    assert d["method"] == "estimated"
    assert "notes" in d
