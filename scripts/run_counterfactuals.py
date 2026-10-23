from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trace.evaluation.counterfactual import trajectory_mae
from trace.models.baseline import PersistenceModel
from trace.simulators.synthetic_physiology import SyntheticPhysiologySimulator


def main() -> None:
    checkpoint = Path("outputs/checkpoints/persistence_run")
    model = PersistenceModel.load(checkpoint)
    initial = dict(model.global_means)
    simulator = SyntheticPhysiologySimulator()
    simulator.reset(initial)
    actions = [
        [{"iv_fluids": 0.0, "vasopressors": 0.0} for _ in range(6)],
        [{"iv_fluids": 1.0, "vasopressors": 0.4} for _ in range(6)],
    ]
    targets = simulator.branch(actions)
    predictions = [model.rollout([initial], 6), model.rollout([initial], 6)]
    summary = {
        "branch_no_intervention_mae": trajectory_mae(predictions[0], targets[0]),
        "branch_intervention_mae": trajectory_mae(predictions[1], targets[1]),
    }
    out = Path("outputs/tables/counterfactual_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"counterfactual_summary": str(out), **summary}, indent=2))


if __name__ == "__main__":
    main()

