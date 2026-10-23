from trace.simulators.synthetic_physiology import SyntheticPhysiologySimulator


def test_simulator_branching_is_deterministic() -> None:
    initial = {
        "heart_rate": 100.0,
        "respiratory_rate": 22.0,
        "spo2": 94.0,
        "temperature": 38.0,
        "mean_arterial_pressure": 62.0,
        "lactate": 3.0,
        "creatinine": 1.4,
    }
    sim = SyntheticPhysiologySimulator()
    sim.reset(initial)
    branches = sim.branch([
        [{"iv_fluids": 0.0, "vasopressors": 0.0} for _ in range(3)],
        [{"iv_fluids": 1.0, "vasopressors": 0.5} for _ in range(3)],
    ])
    assert len(branches) == 2
    assert branches[1][-1]["mean_arterial_pressure"] > branches[0][-1]["mean_arterial_pressure"]

